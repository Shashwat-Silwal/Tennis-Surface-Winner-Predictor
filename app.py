import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

h1 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 4px; font-size: 2.8rem; }
h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; color: #8b949e; font-size: 1rem; }

.court-pill {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 1.2rem;
}
.pill-Clay  { background: #2a120610; color: #e05a1e; border: 1px solid #c24400; }
.pill-Hard  { background: #001d4410; color: #4f8ef7; border: 1px solid #003d88; }
.pill-Grass { background: #0a2e0010; color: #3db554; border: 1px solid #1a5c00; }

.player-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 0.6rem;
}
.player-card .name {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem;
    letter-spacing: 2px; margin-bottom: 0.5rem;
}
.player-card .stat-row { display: flex; gap: 1.4rem; flex-wrap: wrap; }
.player-card .stat { text-align: center; }
.player-card .stat .lbl { font-size: 0.65rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.player-card .stat .val { font-size: 1.15rem; font-weight: 600; }

.result-wrap {
    background: #161b22; border-radius: 16px; padding: 2rem;
    border: 2px solid #30363d; margin-top: 1.2rem;
}
.result-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
.result-name { font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem; letter-spacing: 1.5px; min-width: 180px; }
.bar-bg { flex: 1; background: #0d1117; border-radius: 8px; height: 28px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 8px; display: flex; align-items: center; padding-left: 10px;
            font-weight: 600; font-size: 0.95rem; transition: width 0.6s ease; }
.winner-crown { font-size: 1.4rem; margin-left: 0.4rem; }
.verdict { font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; letter-spacing: 2px; color: #8b949e; margin-top: 0.8rem; }
.verdict span { color: #f0c040; }

div[data-testid="stButton"] button {
    background: #f0c040; color: #0d1117; border: none;
    font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px;
    font-size: 1.05rem; padding: 0.6rem 2rem; border-radius: 8px; width: 100%;
}
div[data-testid="stButton"] button:hover { background: #d4a800; }

.divider { border: none; border-top: 1px solid #30363d; margin: 1.5rem 0; }
.footnote { color: #484f58; font-size: 0.75rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Load pre-trained model & player data ──────────────
@st.cache_resource
def load_model():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_players():
    with open("player_stats.json", "r") as f:
        return json.load(f)

artifacts = load_model()
model    = artifacts["model"]
scaler   = artifacts["scaler"]
feat_cols = artifacts["feat_cols"]
player_data = load_players()
player_names = sorted(player_data.keys())

SURFACE_COLORS = {"Hard": "#4f8ef7", "Clay": "#e05a1e", "Grass": "#3db554"}
SURFACE_PILL   = {"Hard": "pill-Hard", "Clay": "pill-Clay", "Grass": "pill-Grass"}


def predict(p1, p2, surface):
    d1 = player_data[p1]
    d2 = player_data[p2]
    surf_key = f"{surface}WinPct"

    def safe(d, k, fallback):
        v = d.get(k)
        return v if v is not None else fallback

    p1_rank    = safe(d1, "LatestRank", 500)
    p2_rank    = safe(d2, "LatestRank", 500)
    p1_surf    = safe(d1, surf_key, 0.5) / 100 if safe(d1, surf_key, None) is not None else 0.5
    p2_surf    = safe(d2, surf_key, 0.5) / 100 if safe(d2, surf_key, None) is not None else 0.5
    p1_overall = safe(d1, "RecentForm", 50) / 100
    p2_overall = safe(d2, "RecentForm", 50) / 100

    feats = {
        "p1_rank": p1_rank, "p2_rank": p2_rank,
        "rank_diff": p1_rank - p2_rank,
        "p1_surf": p1_surf, "p2_surf": p2_surf,
        "p1_overall": p1_overall, "p2_overall": p2_overall,
        "is_Clay": int(surface == "Clay"),
        "is_Hard": int(surface == "Hard"),
        "is_Grass": int(surface == "Grass"),
    }
    X = pd.DataFrame([feats])[feat_cols]
    X_sc = scaler.transform(X)
    prob = model.predict_proba(X_sc)[0]
    return prob[1], prob[0]   # p1_win_prob, p2_win_prob


def stat_block(label, value, color="#e6edf3"):
    return f"""
    <div class="stat">
      <div class="lbl">{label}</div>
      <div class="val" style="color:{color}">{value}</div>
    </div>"""

def fmt_pct(v):
    return f"{v:.1f}%" if v is not None else "N/A"

def fmt_rank(v):
    return f"#{int(v)}" if v is not None else "N/A"


# ── Header ────────────────────────────────────────────
st.markdown("<h1>🎾 ATP Match Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>10 years of ATP data · 2016–2025 · Logistic Regression</h3>", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────
col_p1, col_vs, col_p2 = st.columns([5, 1, 5])

with col_p1:
    st.markdown("**Player 1**")
    p1 = st.selectbox("", player_names, index=player_names.index("Djokovic N.") if "Djokovic N." in player_names else 0,
                      key="p1", label_visibility="collapsed")

with col_vs:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.3rem;color:#484f58;font-weight:700;'>VS</div>", unsafe_allow_html=True)

with col_p2:
    st.markdown("**Player 2**")
    default_p2 = "Nadal R." if "Nadal R." in player_names else (player_names[1] if len(player_names) > 1 else player_names[0])
    p2 = st.selectbox("", player_names, index=player_names.index(default_p2),
                      key="p2", label_visibility="collapsed")

col_surf, col_btn = st.columns([3, 2])
with col_surf:
    surface = st.radio("Surface", ["Hard", "Clay", "Grass"], horizontal=True)
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡ Predict Winner")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Player cards ──────────────────────────────────────
d1 = player_data.get(p1, {})
d2 = player_data.get(p2, {})
surf_key = f"{surface}WinPct"
color = SURFACE_COLORS[surface]

col_c1, col_c2 = st.columns(2)

def render_card(player, data, surface, surf_key, color):
    rank   = fmt_rank(data.get("LatestRank"))
    winpct = fmt_pct(data.get("WinPct"))
    spct   = fmt_pct(data.get(surf_key))
    form   = fmt_pct(data.get("RecentForm"))
    matches = data.get("Matches", 0)

    st.markdown(f"""
    <div class="player-card">
      <div class="name">{player}</div>
      <div class="stat-row">
        {stat_block("ATP Rank", rank, "#f0c040")}
        {stat_block("Career Win%", winpct)}
        {stat_block(f"{surface} Win%", spct, color)}
        {stat_block("Last 20 Form", form)}
        {stat_block("Matches", f"{matches:,}")}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_c1:
    render_card(p1, d1, surface, surf_key, color)
with col_c2:
    render_card(p2, d2, surface, surf_key, color)

# ── Prediction result ─────────────────────────────────
if predict_clicked:
    if p1 == p2:
        st.error("Please select two different players.")
    else:
        p1_prob, p2_prob = predict(p1, p2, surface)

        winner = p1 if p1_prob >= p2_prob else p2
        w_prob = max(p1_prob, p2_prob) * 100
        l_prob = min(p1_prob, p2_prob) * 100

        pill_cls = SURFACE_PILL[surface]

        def prob_bar(name, prob, is_winner, color):
            fill_color = color if is_winner else "#30363d"
            txt_color  = "#0d1117" if is_winner else "#8b949e"
            crown = " 🏆" if is_winner else ""
            pct = prob * 100
            return f"""
            <div class="result-row">
              <div class="result-name">{name}{crown}</div>
              <div class="bar-bg">
                <div class="bar-fill" style="width:{pct:.1f}%;background:{fill_color};color:{txt_color};">
                  {pct:.1f}%
                </div>
              </div>
            </div>"""

        p1_winner = p1_prob >= p2_prob
        st.markdown(f"""
        <div class="result-wrap">
          <span class="court-pill {pill_cls}">{surface.upper()}</span>
          {prob_bar(p1, p1_prob, p1_winner, color)}
          {prob_bar(p2, p2_prob, not p1_winner, color)}
          <div class="verdict">Prediction → <span>{winner}</span> wins with {w_prob:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # Feature radar / comparison chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        categories = ["Rank Score\n(1/rank)", f"{surface}\nWin%", "Recent\nForm%"]
        def safe_div(v): return 0 if v is None else v
        r1 = safe_div(d1.get("LatestRank")) or 500
        r2 = safe_div(d2.get("LatestRank")) or 500
        p1_vals = [
            1/r1 * 500,
            safe_div(d1.get(surf_key)),
            safe_div(d1.get("RecentForm")),
        ]
        p2_vals = [
            1/r2 * 500,
            safe_div(d2.get(surf_key)),
            safe_div(d2.get("RecentForm")),
        ]

        x = np.arange(len(categories))
        w = 0.35
        b1 = ax.bar(x - w/2, p1_vals, w, color=color, alpha=0.85, label=p1)
        b2 = ax.bar(x + w/2, p2_vals, w, color="#484f58", alpha=0.85, label=p2)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, color="#8b949e", fontsize=9)
        ax.tick_params(axis='y', colors="#8b949e")
        ax.set_title("Head-to-Head Feature Comparison", color="#e6edf3", fontsize=11, pad=10)
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.set_facecolor("#161b22")
        fig.patch.set_facecolor("#0d1117")

        st.pyplot(fig)

st.markdown("""
<div class="footnote">
ATP match data 2016–2025 · Logistic Regression (balanced) · 
Features: ATP ranking, surface win % (last 10), overall form (last 20) · 
403 players with ≥10 career matches
</div>
""", unsafe_allow_html=True)
