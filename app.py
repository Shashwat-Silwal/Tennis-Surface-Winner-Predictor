import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import streamlit.components.v1 as components

st.set_page_config(
    page_title="ATP Match Predictor",
    page_icon="🎾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 800px; }
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

artifacts   = load_model()
model       = artifacts["model"]
scaler      = artifacts["scaler"]
feat_cols   = artifacts["feat_cols"]
player_data = load_players()


def get_prediction(p1, p2, surface):
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
    return round(float(prob[1]), 4), round(float(prob[0]), 4)


# Build a clean player list with all needed stats for JS
def build_player_js(player_data):
    out = {}
    for name, d in player_data.items():
        out[name] = {
            "rank":    int(d["LatestRank"]) if d.get("LatestRank") else 999,
            "winPct":  round(d["WinPct"], 1) if d.get("WinPct") else 50.0,
            "Hard":    round(d["HardWinPct"] * 100, 1) if d.get("HardWinPct") else None,
            "Clay":    round(d["ClayWinPct"] * 100, 1) if d.get("ClayWinPct") else None,
            "Grass":   round(d["GrassWinPct"] * 100, 1) if d.get("GrassWinPct") else None,
            "form":    round(d["RecentForm"], 1) if d.get("RecentForm") else 50.0,
        }
    return out

player_js_data = build_player_js(player_data)
player_names_sorted = sorted(player_data.keys())

# Precompute all predictions for common surfaces so JS can call them instantly
# We'll do predictions server-side via Streamlit state, driven by a form submit
default_p1 = "Djokovic N." if "Djokovic N." in player_data else player_names_sorted[0]
default_p2 = "Nadal R." if "Nadal R." in player_data else player_names_sorted[1]

with st.form("predict_form"):
    col1, col_vs, col2 = st.columns([5,1,5])
    with col1:
        p1 = st.selectbox("Player 1", player_names_sorted,
            index=player_names_sorted.index(default_p1))
    with col_vs:
        st.markdown("<div style='text-align:center;padding-top:1.8rem;color:gray;font-weight:600;'>VS</div>",
                    unsafe_allow_html=True)
    with col2:
        p2 = st.selectbox("Player 2", player_names_sorted,
            index=player_names_sorted.index(default_p2))
    surface = st.radio("Surface", ["Hard", "Clay", "Grass"], horizontal=True)
    submitted = st.form_submit_button("⚡ Predict Winner", use_container_width=True)

# Compute prediction result
if submitted and p1 != p2:
    p1_prob, p2_prob = get_prediction(p1, p2, surface)
elif submitted and p1 == p2:
    st.error("Select two different players.")
    p1_prob, p2_prob = 0.5, 0.5
    p1, p2 = player_names_sorted[0], player_names_sorted[1]
else:
    p1_prob, p2_prob = get_prediction(p1, p2, surface)

# Pass everything to the interactive widget
d1 = player_data.get(p1, {})
d2 = player_data.get(p2, {})
surf_key = f"{surface}WinPct"

def sv(d, k, fb=None):
    v = d.get(k)
    return v if v is not None else fb

p1_stats = {
    "rank":    int(sv(d1,"LatestRank",999)),
    "winPct":  round(sv(d1,"WinPct",50),1),
    "surfPct": round(sv(d1,surf_key,0.5)*100,1),
    "form":    round(sv(d1,"RecentForm",50),1),
}
p2_stats = {
    "rank":    int(sv(d2,"LatestRank",999)),
    "winPct":  round(sv(d2,"WinPct",50),1),
    "surfPct": round(sv(d2,surf_key,0.5)*100,1),
    "form":    round(sv(d2,"RecentForm",50),1),
}

p1_rankScore = round(1/max(p1_stats["rank"],1)*500, 1)
p2_rankScore = round(1/max(p2_stats["rank"],1)*500, 1)

SURFACE_ACCENT = {"Hard": "#185FA5", "Clay": "#993C1D", "Grass": "#3B6D11"}
SURFACE_LIGHT  = {"Hard": "#B5D4F4", "Clay": "#F5C4B3", "Grass": "#C0DD97"}
accent = SURFACE_ACCENT[surface]
light  = SURFACE_LIGHT[surface]

is_p1_win   = p1_prob >= p2_prob
winner_name = p1 if is_p1_win else p2
confidence  = round(max(p1_prob, p2_prob)*100, 1)
p1_pct      = round(p1_prob*100, 1)
p2_pct      = round(p2_prob*100, 1)

html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: transparent; color: #e6edf3; }}

.cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }}
.pcard {{
  background: #161b22; border-radius: 12px;
  border: 1px solid #30363d; padding: 14px 16px;
  transition: border-color .3s;
}}
.pcard.winner {{ border-color: {accent}; border-width: 1.5px; }}
.pcard-name {{
  font-size: 15px; font-weight: 600; color: #e6edf3;
  margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
}}
.crown {{ color: {accent}; font-size: 14px; }}
.stats-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }}
.stat {{
  background: #0d1117; border-radius: 8px; padding: 8px 10px;
  border: 0.5px solid #30363d;
}}
.stat-lbl {{ font-size: 10px; color: #8b949e; margin-bottom: 4px; text-transform: uppercase; letter-spacing: .04em; }}
.stat-val {{ font-size: 14px; font-weight: 600; color: #e6edf3; }}
.stat-val.accent {{ color: {accent}; }}

.result-box {{
  background: #161b22; border-radius: 12px;
  border: 0.5px solid #30363d; padding: 20px;
}}
.verdict {{
  text-align: center; margin-bottom: 18px;
  padding-bottom: 16px; border-bottom: 0.5px solid #30363d;
}}
.verdict-winner {{ font-size: 20px; font-weight: 600; margin-bottom: 4px; }}
.verdict-winner span {{ color: {accent}; }}
.verdict-sub {{ font-size: 12px; color: #8b949e; }}
.verdict-sub b {{ color: {accent}; }}

.prob-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }}
.prob-item {{ }}
.prob-label {{
  display: flex; justify-content: space-between;
  font-size: 13px; color: #8b949e; margin-bottom: 6px;
}}
.prob-label .pct {{ font-weight: 600; color: #e6edf3; }}
.bar-track {{
  height: 8px; background: #30363d;
  border-radius: 4px; overflow: hidden;
}}
.bar-fill {{ height: 100%; border-radius: 4px; }}

.chart-section {{ }}
.chart-header {{
  display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 10px;
}}
.chart-title {{ font-size: 12px; color: #8b949e; }}
.chart-legend {{ display: flex; gap: 14px; font-size: 11px; color: #8b949e; }}
.leg-dot {{
  width: 10px; height: 10px; border-radius: 2px;
  display: inline-block; margin-right: 4px;
  vertical-align: middle;
}}
</style>
</head>
<body>

<div class="cards">
  <div class="pcard {'winner' if is_p1_win else ''}">
    <div class="pcard-name">
      {'<span class="crown">★</span>' if is_p1_win else ''}
      {p1}
    </div>
    <div class="stats-row">
      <div class="stat"><div class="stat-lbl">ATP rank</div><div class="stat-val">#{p1_stats['rank']}</div></div>
      <div class="stat"><div class="stat-lbl">Career win%</div><div class="stat-val">{p1_stats['winPct']}%</div></div>
      <div class="stat"><div class="stat-lbl">{surface} win%</div><div class="stat-val accent">{p1_stats['surfPct']}%</div></div>
      <div class="stat"><div class="stat-lbl">Last 20 form</div><div class="stat-val">{p1_stats['form']}%</div></div>
    </div>
  </div>
  <div class="pcard {'winner' if not is_p1_win else ''}">
    <div class="pcard-name">
      {'<span class="crown">★</span>' if not is_p1_win else ''}
      {p2}
    </div>
    <div class="stats-row">
      <div class="stat"><div class="stat-lbl">ATP rank</div><div class="stat-val">#{p2_stats['rank']}</div></div>
      <div class="stat"><div class="stat-lbl">Career win%</div><div class="stat-val">{p2_stats['winPct']}%</div></div>
      <div class="stat"><div class="stat-lbl">{surface} win%</div><div class="stat-val accent">{p2_stats['surfPct']}%</div></div>
      <div class="stat"><div class="stat-lbl">Last 20 form</div><div class="stat-val">{p2_stats['form']}%</div></div>
    </div>
  </div>
</div>

<div class="result-box">
  <div class="verdict">
    <div class="verdict-winner">★ <span>{winner_name}</span> wins</div>
    <div class="verdict-sub">on {surface} &nbsp;·&nbsp; <b>{confidence}% confidence</b></div>
  </div>

  <div class="prob-grid">
    <div class="prob-item">
      <div class="prob-label"><span>{p1}</span><span class="pct">{p1_pct}%</span></div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{p1_pct}%;background:{''+accent if is_p1_win else '#484f58'};"></div>
      </div>
    </div>
    <div class="prob-item">
      <div class="prob-label"><span>{p2}</span><span class="pct">{p2_pct}%</span></div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{p2_pct}%;background:{''+accent if not is_p1_win else '#484f58'};"></div>
      </div>
    </div>
  </div>

  <div class="chart-section">
    <div class="chart-header">
      <div class="chart-title">Head-to-head feature comparison</div>
      <div class="chart-legend">
        <span><span class="leg-dot" style="background:{accent}"></span>{p1}</span>
        <span><span class="leg-dot" style="background:#484f58"></span>{p2}</span>
      </div>
    </div>
    <div style="position:relative;height:180px;">
      <canvas id="cmpChart" role="img" aria-label="Bar chart comparing {p1} and {p2} on rank score, {surface} win%, and recent form"></canvas>
    </div>
  </div>
</div>

<p style="font-size:11px;color:#484f58;margin-top:12px;text-align:center;">
  ATP 2016–2025 · Logistic Regression · 403 players
</p>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
new Chart(document.getElementById('cmpChart'), {{
  type: 'bar',
  data: {{
    labels: ['Rank score', '{surface} win%', 'Recent form%'],
    datasets: [
      {{
        label: '{p1}',
        data: [{p1_rankScore}, {p1_stats['surfPct']}, {p1_stats['form']}],
        backgroundColor: '{accent}cc',
        borderColor: '{accent}',
        borderWidth: 1,
        borderRadius: 5,
      }},
      {{
        label: '{p2}',
        data: [{p2_rankScore}, {p2_stats['surfPct']}, {p2_stats['form']}],
        backgroundColor: '#48405899',
        borderColor: '#484f58',
        borderWidth: 1,
        borderRadius: 5,
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        titleColor: '#e6edf3',
        bodyColor: '#8b949e',
        callbacks: {{
          label: ctx => ' ' + ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1)
        }}
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#8b949e', font: {{ size: 11 }} }},
        grid: {{ color: 'rgba(48,54,61,0.6)' }}
      }},
      y: {{
        ticks: {{ color: '#8b949e', font: {{ size: 11 }} }},
        grid: {{ color: 'rgba(48,54,61,0.6)' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>
"""

components.html(html, height=680, scrolling=False)
