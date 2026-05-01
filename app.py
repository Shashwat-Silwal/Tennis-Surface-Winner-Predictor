import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="🎾 ATP Match Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_players():
    with open("player_stats.json", "r") as f:
        return json.load(f)

artifacts    = load_model()
model        = artifacts["model"]
scaler       = artifacts["scaler"]
feat_cols    = artifacts["feat_cols"]
player_data  = load_players()
player_names = sorted(player_data.keys())

SURFACE_COLORS = {"Hard": "#4f8ef7", "Clay": "#e05a1e", "Grass": "#3db554"}

def predict(p1, p2, surface):
    d1, d2 = player_data[p1], player_data[p2]
    sk = f"{surface}WinPct"
    def safe(d, k, fb): v = d.get(k); return v if v is not None else fb
    p1_rank = safe(d1, "LatestRank", 500)
    p2_rank = safe(d2, "LatestRank", 500)
    feats = {
        "p1_rank": p1_rank, "p2_rank": p2_rank,
        "rank_diff": p1_rank - p2_rank,
        "p1_surf":    safe(d1, sk, 50) / 100,
        "p2_surf":    safe(d2, sk, 50) / 100,
        "p1_overall": safe(d1, "RecentForm", 50) / 100,
        "p2_overall": safe(d2, "RecentForm", 50) / 100,
        "is_Clay":  int(surface == "Clay"),
        "is_Hard":  int(surface == "Hard"),
        "is_Grass": int(surface == "Grass"),
    }
    X = pd.DataFrame([feats])[feat_cols]
    prob = model.predict_proba(scaler.transform(X))[0]
    return prob[1], prob[0]

def fmt_pct(v): return f"{v:.1f}%" if v is not None else "N/A"
def fmt_rank(v): return f"#{int(v)}" if v is not None else "N/A"
def safe_val(d, k, fb=0): v = d.get(k); return v if v is not None else fb

# ── Header ────────────────────────────────────────────
st.title("🎾 ATP Match Predictor")
st.caption("10 years of ATP data · 2016–2025 · Logistic Regression · 403 players")
st.divider()

# ── Player selectors ──────────────────────────────────
col_p1, col_vs, col_p2 = st.columns([5, 1, 5])
with col_p1:
    st.markdown("**Player 1**")
    p1 = st.selectbox("Player 1", player_names,
        index=player_names.index("Djokovic N.") if "Djokovic N." in player_names else 0,
        key="p1", label_visibility="collapsed")
with col_vs:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#484f58;font-size:1.2rem;font-weight:700;'>VS</div>",
                unsafe_allow_html=True)
with col_p2:
    st.markdown("**Player 2**")
    default_p2 = "Nadal R." if "Nadal R." in player_names else player_names[1]
    p2 = st.selectbox("Player 2", player_names,
        index=player_names.index(default_p2),
        key="p2", label_visibility="collapsed")

col_surf, col_btn = st.columns([3, 2])
with col_surf:
    surface = st.radio("Surface", ["Hard", "Clay", "Grass"], horizontal=True)
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡ Predict Winner", use_container_width=True)

st.divider()

# ── Player stat cards ─────────────────────────────────
d1 = player_data.get(p1, {})
d2 = player_data.get(p2, {})
surf_key = f"{surface}WinPct"
color = SURFACE_COLORS[surface]

col_c1, col_c2 = st.columns(2)

def render_card(col, player, data, surface, surf_key):
    with col:
        st.markdown(f"### {player}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATP Rank",        fmt_rank(data.get("LatestRank")))
        c2.metric("Career Win%",     fmt_pct(data.get("WinPct")))
        c3.metric(f"{surface} Win%", fmt_pct(data.get(surf_key)))
        c4.metric("Last 20 Form",    fmt_pct(data.get("RecentForm")))

render_card(col_c1, p1, d1, surface, surf_key)
render_card(col_c2, p2, d2, surface, surf_key)

# ── Prediction result ─────────────────────────────────
if predict_clicked:
    if p1 == p2:
        st.error("Please select two different players.")
    else:
        st.divider()
        p1_prob, p2_prob = predict(p1, p2, surface)
        winner    = p1 if p1_prob >= p2_prob else p2
        is_p1_win = p1_prob >= p2_prob

        # Winner banner
        st.markdown(
            f"<h2 style='text-align:center;color:{color};font-family:Bebas Neue,sans-serif;"
            f"letter-spacing:3px;'>🏆  {winner}  WINS</h2>",
            unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align:center;color:#8b949e;'>on {surface} · "
            f"{max(p1_prob, p2_prob)*100:.1f}% confidence</p>",
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability display using native Streamlit components
        col_l, col_r = st.columns(2)

        with col_l:
            crown = "🏆 " if is_p1_win else ""
            st.markdown(f"**{crown}{p1}**")
            st.progress(round(float(p1_prob), 2))
            win_color = "#3db554" if is_p1_win else "#8b949e"
            st.markdown(
                f"<p style='font-size:2rem;font-weight:700;color:{win_color};margin:0;'>"
                f"{p1_prob*100:.1f}%</p>",
                unsafe_allow_html=True)

        with col_r:
            crown = "🏆 " if not is_p1_win else ""
            st.markdown(f"**{crown}{p2}**")
            st.progress(round(float(p2_prob), 2))
            win_color = "#3db554" if not is_p1_win else "#8b949e"
            st.markdown(
                f"<p style='font-size:2rem;font-weight:700;color:{win_color};margin:0;'>"
                f"{p2_prob*100:.1f}%</p>",
                unsafe_allow_html=True)

        # Comparison bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        r1_rank = safe_val(d1, "LatestRank") or 500
        r2_rank = safe_val(d2, "LatestRank") or 500

        categories = ["Rank Score\n(1/rank×500)", f"{surface}\nWin%", "Recent\nForm%"]
        p1_vals = [round(1/r1_rank*500, 2), safe_val(d1, surf_key), safe_val(d1, "RecentForm")]
        p2_vals = [round(1/r2_rank*500, 2), safe_val(d2, surf_key), safe_val(d2, "RecentForm")]

        fig, ax = plt.subplots(figsize=(9, 3.8), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        x = np.arange(len(categories))
        ax.bar(x - 0.18, p1_vals, 0.35, color=color,    alpha=0.88, label=p1)
        ax.bar(x + 0.18, p2_vals, 0.35, color="#484f58", alpha=0.88, label=p2)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, color="#8b949e", fontsize=9)
        ax.tick_params(axis='y', colors="#8b949e")
        ax.set_title("Head-to-Head Feature Comparison", color="#e6edf3", fontsize=11, pad=10)
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        st.pyplot(fig)

st.caption("ATP match data 2016–2025 · Features: ATP ranking, surface win% (last 10), overall form (last 20)")
