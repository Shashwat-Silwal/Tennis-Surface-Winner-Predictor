import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

st.set_page_config(page_title="🎾 Tennis Match Predictor", page_icon="🎾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
h1,h2,h3{font-family:'Bebas Neue',sans-serif;letter-spacing:2px;}
.stApp{background:#0d1117;color:#e6edf3;}
section[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d;}
.predict-box{background:#161b22;border:2px solid #f0c040;border-radius:12px;padding:1.5rem;text-align:center;margin:1rem 0;}
.predict-box h2{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:3px;color:#f0c040;margin:0;}
.prob-win{color:#3db554;font-size:3rem;font-family:'Bebas Neue',sans-serif;}
.prob-lose{color:#8b949e;font-size:3rem;font-family:'Bebas Neue',sans-serif;}
.section-title{font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:3px;color:#e6edf3;border-bottom:2px solid #f0c040;padding-bottom:0.3rem;margin-bottom:1rem;}
.info-box{background:#161b22;border-left:3px solid #f0c040;padding:0.8rem 1rem;border-radius:0 8px 8px 0;margin:0.5rem 0;font-size:0.88rem;color:#8b949e;}
.metric-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.5rem;}
.metric-card h4{color:#8b949e;font-size:0.7rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.2rem;}
.metric-card .val{font-size:1.5rem;font-family:'Bebas Neue',sans-serif;}
div[data-testid="stButton"] button{background:#f0c040;color:#0d1117;border:none;font-family:'Bebas Neue',sans-serif;letter-spacing:2px;font-size:1rem;padding:0.5rem 1.5rem;border-radius:6px;}
div[data-testid="stButton"] button:hover{background:#d4a800;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🎾 Tennis Match Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("", ["🏠 Home", "📂 Load & Train", "🔮 Predict Match", "📊 Model Performance"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("KNN & Logistic Regression · ATP 2016–2025")

for key in ["model_knn","model_lr","scaler","feature_cols","metrics","player_stats","knn_results_list"]:
    if key not in st.session_state:
        st.session_state[key] = None


def build_dataset(raw):
    df = raw.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")
    for col in ["WRank","LRank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for s in ["Clay","Hard","Grass"]:
        df[f"is_{s}"] = (df.get("Surface","Unknown") == s).astype(int) if "Surface" in df.columns else 0

    records = []
    for _, row in df.iterrows():
        records.append({"Date":row.get("Date"),"Player":row.get("Winner"),"won":1,"Surface":row.get("Surface","Unknown")})
        records.append({"Date":row.get("Date"),"Player":row.get("Loser"),"won":0,"Surface":row.get("Surface","Unknown")})
    pr = pd.DataFrame(records).sort_values("Date")
    pr["surf_pct_last10"] = pr.groupby(["Player","Surface"])["won"].transform(lambda x: x.shift(1).rolling(10,min_periods=1).mean())
    pr["overall_pct_last20"] = pr.groupby("Player")["won"].transform(lambda x: x.shift(1).rolling(20,min_periods=1).mean())

    w = pr[pr["won"]==1][["Date","Player","Surface","surf_pct_last10","overall_pct_last20"]].copy()
    w.columns = ["Date","Winner","Surface","w_surf","w_overall"]
    l = pr[pr["won"]==0][["Date","Player","Surface","surf_pct_last10","overall_pct_last20"]].copy()
    l.columns = ["Date","Loser","Surface","l_surf","l_overall"]

    merge_keys_w = ["Date","Winner","Surface"] if "Surface" in df.columns else ["Date","Winner"]
    merge_keys_l = ["Date","Loser","Surface"] if "Surface" in df.columns else ["Date","Loser"]
    df = df.merge(w, on=merge_keys_w, how="left")
    df = df.merge(l, on=merge_keys_l, how="left")

    # Symmetric: each match → 2 rows (winner perspective + loser perspective)
    p1_wins = df.assign(p1_rank=df["WRank"],p2_rank=df["LRank"],rank_diff=df["WRank"]-df["LRank"],
                        p1_surf=df["w_surf"],p2_surf=df["l_surf"],p1_overall=df["w_overall"],p2_overall=df["l_overall"],target=1)
    p2_wins = df.assign(p1_rank=df["LRank"],p2_rank=df["WRank"],rank_diff=df["LRank"]-df["WRank"],
                        p1_surf=df["l_surf"],p2_surf=df["w_surf"],p1_overall=df["l_overall"],p2_overall=df["w_overall"],target=0)
    combined = pd.concat([p1_wins,p2_wins],ignore_index=True)
    feat_cols = ["p1_rank","p2_rank","rank_diff","p1_surf","p2_surf","p1_overall","p2_overall","is_Clay","is_Hard","is_Grass"]
    combined = combined[feat_cols+["target"]].dropna()
    return combined, feat_cols


def get_player_stats(raw):
    records = []
    if "Date" in raw.columns:
        raw = raw.copy()
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    for _, row in raw.iterrows():
        for role, won in [("Winner",1),("Loser",0)]:
            p = row.get(role)
            if pd.notna(p):
                records.append({"Player":p,"won":won,"Surface":row.get("Surface","Unknown"),
                                 "Rank":pd.to_numeric(row.get("WRank" if role=="Winner" else "LRank",np.nan),errors="coerce"),
                                 "Date":row.get("Date",pd.NaT)})
    pr = pd.DataFrame(records)
    stats = pr.groupby("Player").agg(Matches=("won","count"),Wins=("won","sum"),LatestRank=("Rank","last")).reset_index()
    stats["WinPct"] = (stats["Wins"]/stats["Matches"]*100).round(1)
    surf_pct = pr.groupby(["Player","Surface"])["won"].mean().unstack(fill_value=np.nan)
    for s in ["Clay","Hard","Grass"]:
        col_name = f"{s}WinPct"
        if s in surf_pct.columns:
            stats = stats.merge(surf_pct[[s]].rename(columns={s:col_name}).reset_index(), on="Player", how="left")
        else:
            stats[col_name] = np.nan
    pr_s = pr.sort_values("Date")
    recent = pr_s.groupby("Player")["won"].apply(lambda x: x.tail(20).mean()*100).reset_index().rename(columns={"won":"RecentForm"})
    stats = stats.merge(recent, on="Player", how="left")
    return stats.sort_values("Matches", ascending=False)


# ── HOME ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="section-title">ATP TENNIS MATCH WINNER PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown("**Predict head-to-head match outcomes using KNN & Logistic Regression · ATP 2016–2025**")
    col1,col2 = st.columns([3,2])
    with col1:
        st.markdown("""
        ### How It Works
        Train ML models on 10 years of ATP match data, then predict **who will win** any head-to-head
        matchup — using only pre-match features (no data leakage).

        **Prediction features:**
        - 🏆 ATP Ranking of each player
        - 📊 Rank difference (P1 − P2)
        - 🌍 Recent surface win % (rolling last 10 matches)
        - 📈 Overall recent form (rolling last 20 matches)
        - 🟤 Surface type: Clay / Hard / Grass

        **Models trained:**
        - **KNN Classifier** — tested across k = 3, 5, 10, 20, 30, 45
        - **Logistic Regression** — balanced class weights
        """)
    with col2:
        st.markdown("""
        <div class="info-box">📁 Step 1 — Upload ATP yearly Excel files (2016–2025)</div>
        <div class="info-box">⚙️ Step 2 — Auto-engineer features & train models</div>
        <div class="info-box">🔮 Step 3 — Select two players + surface → get win probability</div>
        <div class="info-box">📊 Step 4 — Explore accuracy, F1 & confusion matrices</div>
        """, unsafe_allow_html=True)
        st.markdown("**No data leakage:** all rolling stats use `shift(1)` so each match only sees prior history.")


# ── LOAD & TRAIN ──────────────────────────────────────
elif page == "📂 Load & Train":
    st.markdown('<div class="section-title">LOAD DATA & TRAIN MODELS</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload ATP Excel files (2016.xlsx – 2025.xlsx)", type=["xlsx"], accept_multiple_files=True)

    if uploaded:
        dfs = []
        for f in uploaded:
            try:
                dfs.append(pd.read_excel(f))
                st.success(f"✅ {f.name} — {len(dfs[-1]):,} rows")
            except Exception as e:
                st.error(f"❌ {f.name}: {e}")

        if dfs:
            raw = pd.concat(dfs, ignore_index=True)
            st.info(f"Combined: **{len(raw):,} matches**")
            k_options = st.multiselect("k values to test (KNN)", [3,5,10,20,30,45], default=[3,5,10,20,30,45])

            if st.button("⚙️ Engineer Features & Train Models"):
                with st.spinner("Engineering features…"):
                    model_df, feat_cols = build_dataset(raw)
                    player_stats = get_player_stats(raw)
                st.success(f"✅ Training dataset: **{len(model_df):,} rows** | **{len(feat_cols)} features**")

                with st.spinner("Training models…"):
                    X = model_df[feat_cols]
                    y = model_df["target"]
                    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(X_train)
                    Xte = scaler.transform(X_test)

                    knn_rows, best_knn, best_f1 = [], None, -1
                    for k in k_options:
                        knn = KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
                        knn.fit(Xtr, y_train)
                        yp = knn.predict(Xte)
                        acc = accuracy_score(y_test, yp)
                        f1 = f1_score(y_test, yp)
                        knn_rows.append({"k":k,"Accuracy":round(acc,4),"F1":round(f1,4)})
                        if f1 > best_f1: best_f1,best_knn = f1,knn

                    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
                    lr.fit(Xtr, y_train)
                    yp_lr = lr.predict(Xte)

                st.session_state.model_knn = best_knn
                st.session_state.model_lr = lr
                st.session_state.scaler = scaler
                st.session_state.feature_cols = feat_cols
                st.session_state.player_stats = player_stats
                st.session_state.knn_results_list = knn_rows
                st.session_state.metrics = {
                    "knn_results": knn_rows,
                    "best_knn_k": max(knn_rows, key=lambda r:r["F1"])["k"],
                    "lr_acc": round(accuracy_score(y_test,yp_lr),4),
                    "lr_f1": round(f1_score(y_test,yp_lr),4),
                    "y_test": y_test, "y_pred_knn": best_knn.predict(Xte), "y_pred_lr": yp_lr,
                }

                kdf = pd.DataFrame(knn_rows)
                st.markdown("#### KNN Results by k")
                st.dataframe(kdf, use_container_width=True, hide_index=True)

                col1,col2 = st.columns(2)
                best_row = max(knn_rows, key=lambda r:r["F1"])
                with col1:
                    st.markdown(f'<div class="metric-card"><h4>Best KNN</h4><div class="val">k={best_row["k"]} · F1={best_row["F1"]}</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h4>Logistic Regression</h4><div class="val">Acc={st.session_state.metrics["lr_acc"]} · F1={st.session_state.metrics["lr_f1"]}</div></div>', unsafe_allow_html=True)
                st.success("✅ Done! Go to **🔮 Predict Match**.")
    else:
        st.info("👆 Upload your ATP Excel files to get started.")


# ── PREDICT MATCH ─────────────────────────────────────
elif page == "🔮 Predict Match":
    st.markdown('<div class="section-title">PREDICT MATCH WINNER</div>', unsafe_allow_html=True)
    if st.session_state.model_knn is None:
        st.warning("⚠️ Train the models first via **📂 Load & Train**.")
        st.stop()

    ps = st.session_state.player_stats
    players = sorted(ps["Player"].tolist())

    col1, col_vs, col2 = st.columns([5,1,5])
    with col1:
        st.markdown("### 🟡 Player 1")
        p1 = st.selectbox("Player 1", players, key="p1", label_visibility="collapsed")
    with col_vs:
        st.markdown("<br><br><br><div style='text-align:center;font-size:1.5rem;font-weight:bold;'>VS</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("### 🔵 Player 2")
        p2 = st.selectbox("Player 2", players, index=min(1,len(players)-1), key="p2", label_visibility="collapsed")

    surface = st.selectbox("🌍 Surface", ["Hard","Clay","Grass"])
    model_choice = st.radio("Model to use", ["KNN (Best k)","Logistic Regression","Both"], horizontal=True)

    # Player stat cards
    def fmt(val, pct=False, dec=1):
        if pd.isna(val): return "N/A"
        return f"{val*100:.{dec}f}%" if pct else f"{val:.{dec}f}"

    def show_card(player):
        r = ps[ps["Player"]==player]
        if r.empty: return
        r = r.iloc[0]
        sk = f"{surface}WinPct"
        rank = int(r["LatestRank"]) if pd.notna(r.get("LatestRank")) else "N/A"
        st.markdown(f"""
        <div class="metric-card">
        <b>{player}</b><br>
        Rank: <b>{rank}</b> &nbsp;|&nbsp; Win%: <b>{r.get("WinPct","N/A")}%</b><br>
        {surface} Win%: <b>{fmt(r.get(sk,np.nan), pct=True)}</b> &nbsp;|&nbsp; Recent Form: <b>{fmt(r.get("RecentForm",np.nan)/100 if pd.notna(r.get("RecentForm")) else np.nan, pct=True)}</b>
        </div>""", unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1: show_card(p1)
    with col2: show_card(p2)

    if st.button("🔮 Predict Winner"):
        if p1 == p2:
            st.error("Please select two different players.")
            st.stop()

        def get_feats(player, opponent, surface):
            r = ps[ps["Player"]==player]
            o = ps[ps["Player"]==opponent]
            sk = f"{surface}WinPct"
            pr = r["LatestRank"].values[0] if not r.empty and pd.notna(r["LatestRank"].values[0]) else 500.0
            or_ = o["LatestRank"].values[0] if not o.empty and pd.notna(o["LatestRank"].values[0]) else 500.0
            ps_ = r[sk].values[0] if not r.empty and pd.notna(r[sk].values[0]) else 0.5
            os_ = o[sk].values[0] if not o.empty and pd.notna(o[sk].values[0]) else 0.5
            pov = (r["RecentForm"].values[0]/100) if not r.empty and pd.notna(r["RecentForm"].values[0]) else 0.5
            oov = (o["RecentForm"].values[0]/100) if not o.empty and pd.notna(o["RecentForm"].values[0]) else 0.5
            return {"p1_rank":float(pr),"p2_rank":float(or_),"rank_diff":float(pr)-float(or_),
                    "p1_surf":float(ps_),"p2_surf":float(os_),"p1_overall":float(pov),"p2_overall":float(oov),
                    "is_Clay":int(surface=="Clay"),"is_Hard":int(surface=="Hard"),"is_Grass":int(surface=="Grass")}

        feats = get_feats(p1, p2, surface)
        X_in = pd.DataFrame([feats])[st.session_state.feature_cols]
        X_sc = st.session_state.scaler.transform(X_in)

        st.markdown("---")
        st.markdown("## 🎾 Prediction")

        def render_prediction(model, label):
            prob = model.predict_proba(X_sc)[0]
            pred = model.predict(X_sc)[0]
            p1_prob = prob[1]*100
            p2_prob = prob[0]*100
            winner = p1 if pred==1 else p2
            ca, cb = st.columns(2)
            with ca:
                css = "prob-win" if pred==1 else "prob-lose"
                crown = "🏆 " if pred==1 else ""
                st.markdown(f"""
                <div class="predict-box">
                <h2>{label} · {p1}</h2>
                <div class="{css}">{p1_prob:.1f}%</div>
                <div style="color:#8b949e;font-size:0.85rem;">{crown}Win probability</div>
                </div>""", unsafe_allow_html=True)
            with cb:
                css = "prob-win" if pred==0 else "prob-lose"
                crown = "🏆 " if pred==0 else ""
                st.markdown(f"""
                <div class="predict-box">
                <h2>{label} · {p2}</h2>
                <div class="{css}">{p2_prob:.1f}%</div>
                <div style="color:#8b949e;font-size:0.85rem;">{crown}Win probability</div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"**{label} → 🏆 {winner}** ({max(p1_prob,p2_prob):.1f}% confidence on {surface})")

        if model_choice in ["KNN (Best k)","Both"]:
            render_prediction(st.session_state.model_knn, "KNN")
        if model_choice in ["Logistic Regression","Both"]:
            render_prediction(st.session_state.model_lr, "Logistic Regression")

        # Feature comparison chart
        st.markdown("### Feature Comparison")
        sk = f"{surface}WinPct"
        labels = ["Rank Score\n(1/rank×1000)", f"{surface} Win%", "Recent Form%"]
        p1v = [1/(feats["p1_rank"]+1)*1000, feats["p1_surf"]*100, feats["p1_overall"]*100]
        p2v = [1/(feats["p2_rank"]+1)*1000, feats["p2_surf"]*100, feats["p2_overall"]*100]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9,4), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.bar(x-0.2, p1v, 0.38, label=p1, color="#f0c040", alpha=0.9)
        ax.bar(x+0.2, p2v, 0.38, label=p2, color="#4f8ef7", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(labels, color="#8b949e", fontsize=10)
        ax.tick_params(colors="#8b949e"); ax.set_ylabel("Score", color="#8b949e")
        ax.set_title(f"Head-to-Head Feature Comparison · {surface}", color="#e6edf3", fontsize=12)
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        st.pyplot(fig)


# ── MODEL PERFORMANCE ─────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown('<div class="section-title">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    if st.session_state.metrics is None:
        st.warning("⚠️ Train models first via **📂 Load & Train**.")
        st.stop()

    m = st.session_state.metrics
    kdf = pd.DataFrame(m["knn_results"])

    st.markdown("### KNN — Accuracy & F1 by k")
    fig, ax = plt.subplots(figsize=(9,4), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(kdf["k"], kdf["Accuracy"], color="#f0c040", marker="o", label="KNN Accuracy", linewidth=2)
    ax.plot(kdf["k"], kdf["F1"], color="#4f8ef7", marker="s", label="KNN F1", linewidth=2)
    ax.axhline(m["lr_acc"], color="#f0c040", linestyle=":", alpha=0.7, label="LR Accuracy")
    ax.axhline(m["lr_f1"], color="#4f8ef7", linestyle=":", alpha=0.7, label="LR F1")
    ax.set_xlabel("k", color="#8b949e"); ax.set_ylabel("Score", color="#8b949e")
    ax.set_title("KNN vs Logistic Regression", color="#e6edf3", fontsize=12)
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3"); ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    st.pyplot(fig)

    col1,col2 = st.columns(2)
    best = max(m["knn_results"], key=lambda r:r["F1"])
    with col1: st.markdown(f'<div class="metric-card"><h4>Best KNN</h4><div class="val">k={best["k"]} · Acc={best["Accuracy"]} · F1={best["F1"]}</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><h4>Logistic Regression</h4><div class="val">Acc={m["lr_acc"]} · F1={m["lr_f1"]}</div></div>', unsafe_allow_html=True)

    st.markdown("### Confusion Matrices")
    c1,c2 = st.columns(2)
    for col, yp, title, cmap in [(c1,m["y_pred_knn"],f'KNN k={m["best_knn_k"]}',"YlOrBr"),(c2,m["y_pred_lr"],"Logistic Regression","Blues")]:
        with col:
            fig2, ax2 = plt.subplots(figsize=(5,4), facecolor="#0d1117")
            ax2.set_facecolor("#161b22")
            sns.heatmap(confusion_matrix(m["y_test"],yp), annot=True, fmt="d", cmap=cmap, ax=ax2,
                        linewidths=0.5, linecolor="#0d1117",
                        xticklabels=["P2 Wins","P1 Wins"], yticklabels=["P2 Wins","P1 Wins"])
            ax2.set_title(title, color="#e6edf3"); ax2.set_ylabel("Actual",color="#8b949e"); ax2.set_xlabel("Predicted",color="#8b949e")
            ax2.tick_params(colors="#8b949e")
            st.pyplot(fig2)

    st.markdown("### Classification Report — Best KNN")
    report = classification_report(m["y_test"], m["y_pred_knn"], target_names=["P2 Wins","P1 Wins"], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
