import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎾 ATP Tennis Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
}

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}

.metric-card h4 {
    color: #8b949e;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
}

.metric-card .value {
    font-size: 1.9rem;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 1px;
}

.court-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.court-clay { background: #c2440010; color: #e05a1e; border: 1px solid #c24400; }
.court-hard { background: #003d8810; color: #4f8ef7; border: 1px solid #003d88; }
.court-grass { background: #1a5c0010; color: #3db554; border: 1px solid #1a5c00; }

.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 3px;
    color: #e6edf3;
    border-bottom: 2px solid #f0c040;
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}

.info-box {
    background: #161b22;
    border-left: 3px solid #f0c040;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #8b949e;
}

div[data-testid="stButton"] button {
    background: #f0c040;
    color: #0d1117;
    border: none;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
    font-size: 1rem;
    padding: 0.5rem 1.5rem;
    border-radius: 6px;
}

div[data-testid="stButton"] button:hover {
    background: #d4a800;
    color: #0d1117;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🎾 ATP Tennis Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📋 Overview", "📂 Data Loading", "🔬 Research Questions",
     "📈 Regression Models", "🧩 Classification Models", "📊 Full Report"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project 2 · Data Science**  
Caleb Otic · Shashwat Silwal  
ATP Data 2016–2025
""")


# ─────────────────────────────────────────────
# Session state for data persistence
# ─────────────────────────────────────────────
if "atp_data" not in st.session_state:
    st.session_state.atp_data = None
if "surface_dataset" not in st.session_state:
    st.session_state.surface_dataset = None
if "rq1_data" not in st.session_state:
    st.session_state.rq1_data = None


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_player_info():
    """Scrape player age/country/rank from tennisabstract."""
    url = "https://tennisabstract.com/reports/atpRankings.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, "lxml")
        table = soup.find("table", {"id": "reportable"})
        if table is None:
            return None
        df = pd.read_html(str(table))[0]
        df['Birthdate'] = pd.to_datetime(df['Birthdate'], errors='coerce')
        today = pd.to_datetime("today")
        df['age'] = today.year - df['Birthdate'].dt.year
        df['age'] -= (
            (today.month < df['Birthdate'].dt.month) |
            ((today.month == df['Birthdate'].dt.month) & (today.day < df['Birthdate'].dt.day))
        )
        fixed = []
        for name in df['Player']:
            parts = name.split()
            fixed.append(parts[-1] + " " + parts[0][0] + ".")
        df['Player'] = fixed
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_sackmann_players():
    """Fetch height and handedness from JeffSackmann CSV."""
    try:
        players = pd.read_csv(
            "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_players.csv",
            header=None
        )
        players.columns = ["player_id", "name_first", "name_last", "hand", "dob", "ioc", "height", "wikidata_id"]
        fixed = []
        for i in range(len(players)):
            first = str(players.loc[i, "name_first"])
            last = str(players.loc[i, "name_last"])
            name = (last + " " + first[0] + ".") if (first != "nan" and last != "nan") else ""
            fixed.append(name)
        players["Player"] = fixed
        return players[["Player", "height", "hand"]]
    except Exception:
        return None


def engineer_features(df):
    """Engineer all features from the raw ATP dataset."""
    df = df.sort_values("Date").copy()

    # ── RQ1: first-round win % ──────────────────────────────
    first_round = df[df["Round"] == "1st Round"].copy()
    first_round = first_round.sort_values("Date")

    for player_col, win_col in [("Winner", 1), ("Loser", 0)]:
        first_round[f"_{player_col}_won"] = win_col

    melted = []
    for _, row in first_round.iterrows():
        melted.append({"Date": row["Date"], "Player": row["Winner"], "won": 1, "match_id": _})
        melted.append({"Date": row["Date"], "Player": row["Loser"], "won": 0, "match_id": _})
    melted_df = pd.DataFrame(melted).sort_values("Date")

    melted_df["first_round_win_pct_last20"] = (
        melted_df.groupby("Player")["won"]
        .transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    )
    melted_df["career_first_round_win_pct"] = (
        melted_df.groupby("Player")["won"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    winner_rq1 = melted_df[melted_df["won"] == 1][["Date", "Player", "first_round_win_pct_last20", "career_first_round_win_pct"]].copy()
    winner_rq1.columns = ["Date", "Winner", "winner_first_round_pct_last20", "winner_career_first_round_win_pct"]

    df = df.merge(winner_rq1, on=["Date", "Winner"], how="left")

    # ── RQ2 & RQ4: surface win % ────────────────────────────
    melted_surface = []
    for _, row in df.iterrows():
        melted_surface.append({"Date": row["Date"], "Player": row["Winner"], "won": 1, "Surface": row.get("Surface", "Unknown")})
        melted_surface.append({"Date": row["Date"], "Player": row["Loser"], "won": 0, "Surface": row.get("Surface", "Unknown")})
    ms = pd.DataFrame(melted_surface).sort_values("Date")

    ms["surface_win_pct_last10"] = (
        ms.groupby(["Player", "Surface"])["won"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    ms["career_surface_win_pct"] = (
        ms.groupby(["Player", "Surface"])["won"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    winner_surf = ms[ms["won"] == 1][["Date", "Player", "Surface", "surface_win_pct_last10", "career_surface_win_pct"]].copy()
    winner_surf.columns = ["Date", "Winner", "Surface", "winner_surface_pct_last10", "winner_career_surface_win_pct"]
    loser_surf = ms[ms["won"] == 0][["Date", "Player", "Surface", "surface_win_pct_last10", "career_surface_win_pct"]].copy()
    loser_surf.columns = ["Date", "Loser", "Surface", "loser_surface_win_pct_last10", "loser_career_surface_win_pct"]

    df = df.merge(winner_surf, on=["Date", "Winner", "Surface"], how="left")
    df = df.merge(loser_surf, on=["Date", "Loser", "Surface"], how="left")

    # ── RQ3: tiebreak ───────────────────────────────────────
    def check_tiebreak(row):
        for col in ["W1", "W2", "W3", "W4", "W5", "L1", "L2", "L3", "L4", "L5"]:
            if col in row and pd.notna(row[col]) and str(row[col]).strip() == "7":
                partner = "L" + col[1] if col.startswith("W") else "W" + col[1]
                if partner in row and pd.notna(row[partner]) and str(row[partner]).strip() == "6":
                    return 1
        return 0

    if "W1" in df.columns:
        df["had_tiebreak"] = df.apply(check_tiebreak, axis=1)
    else:
        df["had_tiebreak"] = 0

    # Rank difference
    if "WRank" in df.columns and "LRank" in df.columns:
        df["Difference"] = (pd.to_numeric(df["WRank"], errors="coerce") - pd.to_numeric(df["LRank"], errors="coerce")).abs()
    else:
        df["Difference"] = np.nan

    # ── RQ4: upset ──────────────────────────────────────────
    if "WRank" in df.columns and "LRank" in df.columns:
        df["is_upset"] = (pd.to_numeric(df["WRank"], errors="coerce") > pd.to_numeric(df["LRank"], errors="coerce")).astype(int)
    else:
        df["is_upset"] = 0

    # Age difference (if age cols available)
    if "age_winner" in df.columns and "age_loser" in df.columns:
        df["age_difference"] = (df["age_winner"] - df["age_loser"]).abs()

    return df


def calculate_regression_gof(ys, y_hat):
    y_mean = ys.mean()
    ss_total = np.sum((ys - y_mean) ** 2)
    ss_residual = np.sum((ys - y_hat) ** 2)
    ss_regression = np.sum((y_hat - y_mean) ** 2)
    r_square = ss_regression / ss_total if ss_total != 0 else 0
    rmse = np.sqrt(ss_residual / len(ys))
    return r_square, rmse


def color_for_surface(surface):
    m = {"Clay": "#e05a1e", "Hard": "#4f8ef7", "Grass": "#3db554"}
    return m.get(surface, "#8b949e")


# ─────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────
if page == "📋 Overview":
    st.markdown('<div class="section-title">ATP TENNIS DATA ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown("**Project 2 · Regression & Classification Models · 2016–2025**")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        This interactive dashboard presents the full analytical pipeline from Project 2.
        It allows you to **load data**, **engineer features**, **train models**, and **explore results**
        across four research questions.

        ### Research Questions
        | # | Question | Type |
        |---|----------|------|
        | RQ1 | Predict career first-round win % from recent form, age & rank | Regression |
        | RQ2 | Predict career surface win % from height, rank & recent surface form | Regression |
        | RQ3 | Predict if a match will have a tiebreak from rank difference & surface | Classification |
        | RQ4 | Predict whether a match results in an upset from age diff & recent surface form | Classification |

        ### Models Used
        - Simple, Multiple & Polynomial Linear Regression
        - kNN Regression (k=2–10)
        - kNN Classification (k=3,5,10,20,30,45)
        - Logistic Regression (standard & class-weight balanced)
        """)

    with col2:
        st.markdown("### Data Sources")
        st.markdown("""
        <div class="info-box">📁 ATP match records 2016–2025 (Excel files)</div>
        <div class="info-box">🌐 tennisabstract.com — Age, Country, Ranking</div>
        <div class="info-box">🐙 JeffSackmann GitHub — Height, Handedness</div>
        """, unsafe_allow_html=True)

        st.markdown("### Quick Start")
        st.markdown("""
        1. Go to **📂 Data Loading** and upload your Excel files
        2. Load player metadata from external sources
        3. Navigate through RQ sections to run models
        4. View the **📊 Full Report** for the complete summary
        """)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    ℹ️ <strong>Note:</strong> This app requires the ATP Excel files (2016.xlsx … 2025.xlsx) from Project 1.
    Upload them in the Data Loading page to get started. Player metadata is fetched live from public web sources.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: Data Loading
# ─────────────────────────────────────────────
elif page == "📂 Data Loading":
    st.markdown('<div class="section-title">DATA LOADING & PREPARATION</div>', unsafe_allow_html=True)

    st.markdown("#### Step 1 — Upload ATP Match Files")
    uploaded_files = st.file_uploader(
        "Upload your ATP Excel files (2016.xlsx to 2025.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        help="Upload the yearly ATP match data files from Project 1."
    )

    if uploaded_files:
        dfs = []
        with st.spinner("Reading uploaded files…"):
            for f in uploaded_files:
                try:
                    dfs.append(pd.read_excel(f))
                    st.success(f"✅ Loaded: {f.name} ({len(dfs[-1])} rows)")
                except Exception as e:
                    st.error(f"❌ Error reading {f.name}: {e}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            if "Date" in combined.columns:
                combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
            combined["Surface"] = combined["Surface"].fillna("Unknown") if "Surface" in combined.columns else "Unknown"

            st.success(f"✅ Combined dataset: **{len(combined):,} rows** | **{combined.shape[1]} columns**")
            st.dataframe(combined.head(10), use_container_width=True)

            st.markdown("#### Step 2 — Fetch Player Metadata")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🌐 Fetch from tennisabstract.com"):
                    with st.spinner("Scraping player info…"):
                        t1df = fetch_player_info()
                    if t1df is not None:
                        st.success(f"✅ Fetched {len(t1df)} players from tennisabstract")
                        st.session_state["t1df"] = t1df
                        st.dataframe(t1df.head(5))
                    else:
                        st.warning("⚠️ Could not fetch from tennisabstract. Continuing without age/country data.")
                        st.session_state["t1df"] = pd.DataFrame()

            with col2:
                if st.button("🐙 Fetch from JeffSackmann GitHub"):
                    with st.spinner("Downloading players CSV…"):
                        t2df = fetch_sackmann_players()
                    if t2df is not None:
                        st.success(f"✅ Fetched {len(t2df)} player records (height/hand)")
                        st.session_state["t2df"] = t2df
                        st.dataframe(t2df.head(5))
                    else:
                        st.warning("⚠️ Could not reach GitHub. Continuing without height/handedness data.")
                        st.session_state["t2df"] = pd.DataFrame()

            st.markdown("#### Step 3 — Merge & Engineer Features")
            if st.button("⚙️ Merge Data & Engineer Features"):
                with st.spinner("Merging datasets and engineering features…"):
                    t1df = st.session_state.get("t1df", pd.DataFrame())
                    t2df = st.session_state.get("t2df", pd.DataFrame())

                    if not t1df.empty and not t2df.empty:
                        new_info = pd.merge(t1df, t2df, on="Player", how="left")
                        winner_info = new_info.rename(columns={
                            "Rank": "Rank_winner", "Player": "Player_winner",
                            "Country": "Country_winner", "Birthdate": "Birthdate_winner",
                            "age": "age_winner", "height": "height_winner", "hand": "hand_winner"
                        })
                        loser_info = new_info.rename(columns={
                            "Rank": "Rank_loser", "Player": "Player_loser",
                            "Country": "Country_loser", "Birthdate": "Birthdate_loser",
                            "age": "age_loser", "height": "height_loser", "hand": "hand_loser"
                        })
                        merged = pd.merge(combined, winner_info, left_on="Winner", right_on="Player_winner", how="left")
                        merged = pd.merge(merged, loser_info, left_on="Loser", right_on="Player_loser", how="left")
                        if "WRank" not in merged.columns and "Rank_winner" in merged.columns:
                            merged["WRank"] = merged["Rank_winner"]
                            merged["LRank"] = merged["Rank_loser"]
                    else:
                        merged = combined.copy()
                        for col in ["age_winner", "age_loser", "height_winner", "height_winner", "Rank_winner", "Rank_loser"]:
                            if col not in merged.columns:
                                merged[col] = np.nan
                        if "WRank" in merged.columns:
                            merged["Rank_winner"] = pd.to_numeric(merged["WRank"], errors="coerce")
                            merged["Rank_loser"] = pd.to_numeric(merged["LRank"], errors="coerce")

                    engineered = engineer_features(merged)
                    st.session_state.atp_data = engineered
                    st.session_state.surface_dataset = engineered[engineered["Surface"].isin(["Clay", "Hard", "Grass"])].copy()

                st.success(f"✅ Feature engineering complete! Dataset shape: {st.session_state.atp_data.shape}")
                st.markdown("**Sample engineered features:**")
                show_cols = [c for c in ["Winner", "Loser", "Surface", "winner_first_round_pct_last20",
                                          "winner_career_first_round_win_pct", "had_tiebreak",
                                          "Difference", "is_upset"] if c in st.session_state.atp_data.columns]
                st.dataframe(st.session_state.atp_data[show_cols].dropna().head(10), use_container_width=True)
    else:
        st.info("👆 Upload your ATP Excel files above to begin.")


# ─────────────────────────────────────────────
# PAGE: Research Questions
# ─────────────────────────────────────────────
elif page == "🔬 Research Questions":
    st.markdown('<div class="section-title">RESEARCH QUESTIONS & FEATURE ENGINEERING</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Four Research Questions

    These questions guided all modelling decisions in Project 2.
    """)

    rqs = [
        ("RQ1", "Regression", "#f0c040",
         "How accurately can we predict player **1st round win percentage** based on their age, average first round wins in the last 20 matches, and rank?",
         ["winner_first_round_pct_last20 — Rolling win rate over last 20 first-round matches (shift=1)",
          "winner_career_first_round_win_pct — Expanding mean career first-round win rate (TARGET)"]),
        ("RQ2", "Regression", "#4f8ef7",
         "Can we predict a player's **overall career win percentage on a surface** using their height, rank, and recent form on that surface?",
         ["winner_surface_pct_last10 — Rolling win rate over last 10 matches per surface (shift=1)",
          "winner_career_surface_win_pct — Expanding mean career win rate per surface (TARGET)"]),
        ("RQ3", "Classification", "#3db554",
         "Can we predict if a match will have a **tiebreak** based on the rank difference between the two players?",
         ["had_tiebreak — 1 if any set ended 7-6, else 0 (TARGET)",
          "Difference — abs(WRank - LRank) — rank gap between players",
          "is_Clay / is_Hard / is_Grass — one-hot surface encoding"]),
        ("RQ4", "Classification", "#e05a1e",
         "Can we predict whether a match will result in an **'Upset'** based on the players' age difference and their last 10 matches win-loss record on that specific court surface?",
         ["is_upset — 1 if winner rank > loser rank (TARGET)",
          "age_difference — absolute age gap between players",
          "winner_surface_pct_last10 / loser_surface_win_pct_last10 — recent form (no data leakage)"]),
    ]

    for rq_id, rq_type, color, question, features in rqs:
        with st.expander(f"**{rq_id}** — {rq_type}", expanded=False):
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Type:** `{rq_type}`")
            st.markdown("**Features engineered:**")
            for f in features:
                st.markdown(f"- {f}")

    st.markdown("---")
    st.markdown("### Data Leakage Note (RQ4)")
    st.markdown("""
    <div class="info-box">
    ⚠️ Career surface win percentages (<code>winner_career_surface_win_pct</code>, <code>loser_career_surface_win_pct</code>)
    were initially used as predictors for RQ4 but caused <strong>data leakage</strong> — they were computed from all historical outcomes
    including the current match, meaning the model implicitly knew who won. These were replaced with
    <code>winner_surface_pct_last10</code> and <code>loser_surface_win_pct_last10</code>, which only use data from <em>prior</em> matches.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.atp_data is not None:
        df = st.session_state.atp_data
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card"><h4>Total Matches</h4><div class="value">{len(df):,}</div></div>""", unsafe_allow_html=True)
        with col2:
            surfaces = df["Surface"].value_counts() if "Surface" in df.columns else {}
            st.markdown(f"""<div class="metric-card"><h4>Surfaces</h4><div class="value">{len(surfaces)}</div></div>""", unsafe_allow_html=True)
        with col3:
            years = df["Date"].dt.year.nunique() if "Date" in df.columns else "N/A"
            st.markdown(f"""<div class="metric-card"><h4>Years Covered</h4><div class="value">{years}</div></div>""", unsafe_allow_html=True)
        with col4:
            upsets = df["is_upset"].sum() if "is_upset" in df.columns else "N/A"
            st.markdown(f"""<div class="metric-card"><h4>Upsets Recorded</h4><div class="value">{upsets:,}</div></div>""", unsafe_allow_html=True)
    else:
        st.info("ℹ️ Load data from the **📂 Data Loading** page first to see dataset statistics.")


# ─────────────────────────────────────────────
# PAGE: Regression Models
# ─────────────────────────────────────────────
elif page == "📈 Regression Models":
    st.markdown('<div class="section-title">REGRESSION MODELS</div>', unsafe_allow_html=True)

    if st.session_state.atp_data is None:
        st.warning("⚠️ Please load and engineer features first via the **📂 Data Loading** page.")
        st.stop()

    df = st.session_state.atp_data

    tab1, tab2 = st.tabs(["RQ1 — First-Round Win %", "RQ2 — Surface Win %"])

    # ── RQ1 ────────────────────────────────────────────────
    with tab1:
        st.markdown("**Target:** `winner_career_first_round_win_pct`  |  **Predictors:** recent first-round form, age, rank")

        req_cols = ["winner_first_round_pct_last20", "winner_career_first_round_win_pct"]
        rank_col = next((c for c in ["Rank_winner", "WRank"] if c in df.columns), None)
        age_col = "age_winner" if "age_winner" in df.columns else None

        if not all(c in df.columns for c in req_cols) or rank_col is None:
            st.error("Required columns not found. Ensure feature engineering was run successfully.")
        else:
            rq1_cols = [c for c in [*req_cols, rank_col, age_col] if c is not None]
            rq1_df = df[rq1_cols].dropna().copy()
            rq1_df[rank_col] = pd.to_numeric(rq1_df[rank_col], errors="coerce")
            rq1_df = rq1_df.dropna().copy()

            sample_size = st.slider("Sample size for regression", 500, min(10000, len(rq1_df)), min(3000, len(rq1_df)), 500, key="rq1_sample")
            rq1_df = rq1_df.sample(sample_size, random_state=42)

            if st.button("▶ Run RQ1 Regression Models", key="run_rq1"):
                results = []

                with st.spinner("Running models…"):
                    # Simple Linear Regression
                    m1 = smf.ols(
                        "winner_career_first_round_win_pct ~ winner_first_round_pct_last20",
                        data=rq1_df
                    ).fit()
                    results.append({"Model": "Simple Linear", "R²": round(m1.rsquared, 3), "Notes": "Recent form only"})

                    # Multiple Linear Regression
                    if age_col:
                        formula2 = f"winner_career_first_round_win_pct ~ winner_first_round_pct_last20 + {age_col} + {rank_col}"
                    else:
                        formula2 = f"winner_career_first_round_win_pct ~ winner_first_round_pct_last20 + {rank_col}"
                    m2 = smf.ols(formula2, data=rq1_df).fit()
                    results.append({"Model": "Multiple Linear", "R²": round(m2.rsquared, 3), "Notes": "Added age & rank"})

                    # Polynomial Regression
                    poly_formula = formula2 + f" + I({rank_col}**2)"
                    m3 = smf.ols(poly_formula, data=rq1_df).fit()
                    results.append({"Model": "Polynomial (Rank²)", "R²": round(m3.rsquared, 3), "Notes": "Non-linear rank effect"})

                    # kNN Regression
                    knn_cols = [c for c in ["winner_first_round_pct_last20", age_col, rank_col] if c]
                    X_knn = rq1_df[knn_cols]
                    y_knn = rq1_df["winner_career_first_round_win_pct"]
                    scaler = preprocessing.MinMaxScaler()
                    X_scaled = scaler.fit_transform(X_knn)

                    knn_results = []
                    for k in range(2, 11):
                        knn = KNeighborsRegressor(n_neighbors=k)
                        y_hat = knn.fit(X_scaled, y_knn).predict(X_scaled)
                        r2, rmse = calculate_regression_gof(y_knn.reset_index(drop=True), y_hat)
                        knn_results.append({"k": k, "R²": round(r2, 4), "RMSE": round(rmse, 4)})

                    best_knn = min(knn_results, key=lambda x: x["RMSE"])
                    results.append({"Model": f"kNN (k={best_knn['k']})", "R²": best_knn["R²"],
                                    "Notes": f"Best RMSE={best_knn['RMSE']}"})

                st.markdown("#### Model Comparison — RQ1")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                # kNN RMSE plot
                fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
                ax.set_facecolor("#161b22")
                knn_df = pd.DataFrame(knn_results)
                ax.plot(knn_df["k"], knn_df["RMSE"], color="#f0c040", marker="o", linewidth=2)
                ax.set_xlabel("k", color="#8b949e")
                ax.set_ylabel("RMSE", color="#8b949e")
                ax.set_title("RQ1: kNN Regression RMSE by k", color="#e6edf3", fontsize=12)
                ax.tick_params(colors="#8b949e")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#30363d")
                st.pyplot(fig)

                st.markdown("""
                **Interpretation:** A player's career first-round success cannot be well explained by recent form alone.
                Rank and age add meaningful signal, and the non-linear polynomial model confirms the relationship
                between ranking and win percentage is not linear. kNN achieves the lowest RMSE by leveraging local similarity.
                """)

    # ── RQ2 ────────────────────────────────────────────────
    with tab2:
        st.markdown("**Target:** `winner_career_surface_win_pct`  |  **Predictors:** recent surface form, height, rank")

        surf_df = st.session_state.surface_dataset
        if surf_df is None:
            st.error("Surface dataset not available. Re-run feature engineering.")
        else:
            req_cols2 = ["winner_career_surface_win_pct", "winner_surface_pct_last10"]
            rank_col2 = next((c for c in ["Rank_winner", "WRank"] if c in surf_df.columns), None)
            height_col = "height_winner" if "height_winner" in surf_df.columns else None

            if not all(c in surf_df.columns for c in req_cols2) or rank_col2 is None:
                st.error("Required columns not found.")
            else:
                rq2_cols = [c for c in [*req_cols2, rank_col2, height_col] if c]
                rq2_df = surf_df[rq2_cols].dropna().copy()
                if height_col:
                    rq2_df[height_col] = pd.to_numeric(rq2_df[height_col], errors="coerce")
                rq2_df[rank_col2] = pd.to_numeric(rq2_df[rank_col2], errors="coerce")
                rq2_df = rq2_df.dropna().copy()

                sample2 = st.slider("Sample size", 500, min(5000, len(rq2_df)), min(3000, len(rq2_df)), 500, key="rq2_sample")
                rq2_df = rq2_df.sample(sample2, random_state=42)

                if st.button("▶ Run RQ2 Regression Models", key="run_rq2"):
                    results2 = []
                    with st.spinner("Running models…"):
                        m1 = smf.ols("winner_career_surface_win_pct ~ winner_surface_pct_last10", data=rq2_df).fit()
                        results2.append({"Model": "Simple Linear", "R²": round(m1.rsquared, 3), "Notes": "Surface form only"})

                        if height_col:
                            f2 = f"winner_career_surface_win_pct ~ winner_surface_pct_last10 + {height_col} + {rank_col2}"
                        else:
                            f2 = f"winner_career_surface_win_pct ~ winner_surface_pct_last10 + {rank_col2}"
                        m2 = smf.ols(f2, data=rq2_df).fit()
                        results2.append({"Model": "Multiple Linear", "R²": round(m2.rsquared, 3), "Notes": "Added height & rank"})

                        f3 = f2 + f" + I({rank_col2}**2)"
                        m3 = smf.ols(f3, data=rq2_df).fit()
                        results2.append({"Model": "Polynomial (Rank²)", "R²": round(m3.rsquared, 3), "Notes": "Non-linear rank effect"})

                        knn2_cols = [c for c in ["winner_surface_pct_last10", height_col, rank_col2] if c]
                        X2 = rq2_df[knn2_cols]
                        y2 = rq2_df["winner_career_surface_win_pct"]
                        scaler2 = preprocessing.MinMaxScaler()
                        X2_scaled = scaler2.fit_transform(X2)

                        knn2_results = []
                        for k in range(2, 11):
                            knn = KNeighborsRegressor(n_neighbors=k)
                            y_hat = knn.fit(X2_scaled, y2).predict(X2_scaled)
                            r2, rmse = calculate_regression_gof(y2.reset_index(drop=True), y_hat)
                            knn2_results.append({"k": k, "R²": round(r2, 4), "RMSE": round(rmse, 4)})

                        best_knn2 = min(knn2_results, key=lambda x: x["RMSE"])
                        results2.append({"Model": f"kNN (k={best_knn2['k']})", "R²": best_knn2["R²"],
                                         "Notes": f"Best RMSE={best_knn2['RMSE']}"})

                    st.dataframe(pd.DataFrame(results2), use_container_width=True, hide_index=True)

                    fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
                    ax2.set_facecolor("#161b22")
                    knn2_df = pd.DataFrame(knn2_results)
                    ax2.plot(knn2_df["k"], knn2_df["RMSE"], color="#4f8ef7", marker="o", linewidth=2)
                    ax2.set_xlabel("k", color="#8b949e")
                    ax2.set_ylabel("RMSE", color="#8b949e")
                    ax2.set_title("RQ2: kNN Regression RMSE by k", color="#e6edf3", fontsize=12)
                    ax2.tick_params(colors="#8b949e")
                    for spine in ax2.spines.values():
                        spine.set_edgecolor("#30363d")
                    st.pyplot(fig2)


# ─────────────────────────────────────────────
# PAGE: Classification Models
# ─────────────────────────────────────────────
elif page == "🧩 Classification Models":
    st.markdown('<div class="section-title">CLASSIFICATION MODELS</div>', unsafe_allow_html=True)

    if st.session_state.atp_data is None:
        st.warning("⚠️ Please load and engineer features first via the **📂 Data Loading** page.")
        st.stop()

    df = st.session_state.atp_data
    surf_df = st.session_state.surface_dataset

    tab1, tab2 = st.tabs(["RQ3 — Tiebreak Prediction", "RQ4 — Upset Prediction"])

    # ── RQ3 ────────────────────────────────────────────────
    with tab1:
        st.markdown("**Target:** `had_tiebreak`  |  **Predictors:** rank difference, surface type")

        req_cols_rq3 = ["had_tiebreak", "Difference"]
        if not all(c in df.columns for c in req_cols_rq3):
            st.error("Required columns not found.")
        else:
            for surf in ["Clay", "Hard", "Grass"]:
                df[f"is_{surf}"] = (df["Surface"] == surf).astype(int)

            q3_data = df[["Difference", "is_Clay", "is_Hard", "is_Grass", "had_tiebreak"]].dropna()
            st.info(f"Dataset for RQ3: **{len(q3_data):,} rows**  |  Tiebreak rate: **{q3_data['had_tiebreak'].mean():.1%}**")

            k_options = [3, 5, 10, 20, 30, 45]
            selected_k = st.multiselect("Select k values to test", k_options, default=[3, 5, 10, 20], key="rq3_k")

            if st.button("▶ Run RQ3 Classification Models", key="run_rq3"):
                with st.spinner("Training models…"):
                    X = q3_data[["Difference", "is_Clay", "is_Hard", "is_Grass"]]
                    y = q3_data["had_tiebreak"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_test_sc = scaler.transform(X_test)

                    rows = []
                    y_preds = {}
                    for k in selected_k:
                        knn = KNeighborsClassifier(n_neighbors=k)
                        knn.fit(X_train_sc, y_train)
                        y_pred = knn.predict(X_test_sc)
                        y_preds[f"kNN k={k}"] = y_pred
                        rows.append({
                            "Model": f"kNN k={k}",
                            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                            "F1": round(f1_score(y_test, y_pred), 3),
                            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
                            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
                        })

                    # Logistic Regression
                    lr = LogisticRegression()
                    lr.fit(X_train_sc, y_train)
                    y_lr = lr.predict(X_test_sc)
                    y_preds["LR Standard"] = y_lr
                    rows.append({
                        "Model": "Logistic Regression",
                        "Accuracy": round(accuracy_score(y_test, y_lr), 3),
                        "F1": round(f1_score(y_test, y_lr, zero_division=0), 3),
                        "Precision": round(precision_score(y_test, y_lr, zero_division=0), 3),
                        "Recall": round(recall_score(y_test, y_lr, zero_division=0), 3),
                    })

                    # Balanced LR
                    lr_bal = LogisticRegression(class_weight="balanced")
                    lr_bal.fit(X_train_sc, y_train)
                    y_lr_bal = lr_bal.predict(X_test_sc)
                    y_preds["LR Balanced"] = y_lr_bal
                    rows.append({
                        "Model": "LR (Balanced)",
                        "Accuracy": round(accuracy_score(y_test, y_lr_bal), 3),
                        "F1": round(f1_score(y_test, y_lr_bal, zero_division=0), 3),
                        "Precision": round(precision_score(y_test, y_lr_bal, zero_division=0), 3),
                        "Recall": round(recall_score(y_test, y_lr_bal, zero_division=0), 3),
                    })

                res_df = pd.DataFrame(rows)
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                # Performance plot
                fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
                ax.set_facecolor("#161b22")
                knn_rows = [r for r in rows if "kNN" in r["Model"]]
                k_vals = [int(r["Model"].split("=")[1]) for r in knn_rows]
                for metric, color in [("Accuracy", "#f0c040"), ("F1", "#4f8ef7"), ("Precision", "#3db554"), ("Recall", "#e05a1e")]:
                    ax.plot(k_vals, [r[metric] for r in knn_rows], color=color, marker="o", label=f"kNN {metric}")
                    lr_val = next(r[metric] for r in rows if r["Model"] == "LR (Balanced)")
                    ax.axhline(lr_val, color=color, linestyle=":", alpha=0.7, label=f"LR Bal {metric}")
                ax.set_xlabel("k", color="#8b949e")
                ax.set_ylabel("Score", color="#8b949e")
                ax.set_title("RQ3: kNN vs Logistic Regression (Balanced)", color="#e6edf3")
                ax.tick_params(colors="#8b949e")
                ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#30363d")
                st.pyplot(fig)

                # Confusion matrices
                best_knn_name = max([r for r in rows if "kNN" in r["Model"]], key=lambda r: r["F1"])["Model"]
                y_best_knn = y_preds[best_knn_name]
                fig2, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
                for ax_, y_pred_, title_, cmap_ in [
                    (axes[0], y_best_knn, f"Confusion — {best_knn_name}", "Blues"),
                    (axes[1], y_lr_bal, "Confusion — LR (Balanced)", "Greens"),
                ]:
                    ax_.set_facecolor("#161b22")
                    sns.heatmap(confusion_matrix(y_test, y_pred_), annot=True, fmt="d", cmap=cmap_, ax=ax_,
                                linewidths=0.5, linecolor="#0d1117")
                    ax_.set_title(title_, color="#e6edf3")
                    ax_.set_ylabel("Actual", color="#8b949e")
                    ax_.set_xlabel("Predicted", color="#8b949e")
                    ax_.tick_params(colors="#8b949e")
                plt.tight_layout()
                st.pyplot(fig2)

                st.markdown("""
                **Key Finding (RQ3):** Neither model produces strong tiebreak predictions.
                Balanced Logistic Regression achieved the best F1 by penalising misclassification of the minority class.
                Standard LR collapsed to predicting "no tiebreak" for every match. Tiebreaks appear to be largely
                unpredictable from rank difference and surface type alone.
                """)

    # ── RQ4 ────────────────────────────────────────────────
    with tab2:
        st.markdown("**Target:** `is_upset`  |  **Predictors:** age difference, recent surface form (no leakage)")

        if surf_df is None:
            st.error("Surface dataset not available.")
        else:
            req_cols_rq4 = ["is_upset", "winner_surface_pct_last10", "loser_surface_win_pct_last10"]
            age_diff_available = "age_difference" in surf_df.columns

            if not all(c in surf_df.columns for c in req_cols_rq4):
                st.error("Required columns not found. Re-run feature engineering.")
            else:
                rq4_cols = ["age_difference", *req_cols_rq4] if age_diff_available else req_cols_rq4
                q4_data = surf_df[rq4_cols].dropna()

                sample4 = st.slider("Sample size (large = slower)", 5000, min(50000, len(q4_data)), min(20000, len(q4_data)), 5000, key="rq4_sample")
                q4_data = q4_data.sample(sample4, random_state=42).reset_index(drop=True)

                st.info(f"Dataset for RQ4: **{len(q4_data):,} rows**  |  Upset rate: **{q4_data['is_upset'].mean():.1%}**")

                pred_cols = [c for c in ["age_difference", "winner_surface_pct_last10", "loser_surface_win_pct_last10"] if c in q4_data.columns]
                k_options4 = [3, 5, 10, 20, 30, 45]
                selected_k4 = st.multiselect("Select k values to test", k_options4, default=[3, 10, 20], key="rq4_k")

                if st.button("▶ Run RQ4 Classification Models", key="run_rq4"):
                    with st.spinner("Training models (this may take a moment for large samples)…"):
                        X = q4_data[pred_cols]
                        y = q4_data["is_upset"]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        scaler4 = StandardScaler()
                        X_train_sc = scaler4.fit_transform(X_train)
                        X_test_sc = scaler4.transform(X_test)

                        rows4 = []
                        y_preds4 = {}

                        for k in selected_k4:
                            knn = KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
                            knn.fit(X_train_sc, y_train)
                            y_pred = knn.predict(X_test_sc)
                            y_preds4[f"kNN k={k}"] = y_pred
                            rows4.append({
                                "Model": f"kNN k={k}",
                                "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                                "F1": round(f1_score(y_test, y_pred), 3),
                                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
                                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
                            })

                        lr4 = LogisticRegression()
                        lr4.fit(X_train_sc, y_train)
                        y_lr4 = lr4.predict(X_test_sc)
                        y_preds4["LR"] = y_lr4
                        rows4.append({
                            "Model": "Logistic Regression",
                            "Accuracy": round(accuracy_score(y_test, y_lr4), 3),
                            "F1": round(f1_score(y_test, y_lr4), 3),
                            "Precision": round(precision_score(y_test, y_lr4, zero_division=0), 3),
                            "Recall": round(recall_score(y_test, y_lr4, zero_division=0), 3),
                        })

                    st.dataframe(pd.DataFrame(rows4), use_container_width=True, hide_index=True)

                    # Plot
                    knn_rows4 = [r for r in rows4 if "kNN" in r["Model"]]
                    if knn_rows4:
                        k_vals4 = [int(r["Model"].split("=")[1]) for r in knn_rows4]
                        fig4, ax4 = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
                        ax4.set_facecolor("#161b22")
                        for metric, color in [("Accuracy", "#f0c040"), ("F1", "#4f8ef7"), ("Precision", "#3db554"), ("Recall", "#e05a1e")]:
                            ax4.plot(k_vals4, [r[metric] for r in knn_rows4], color=color, marker="o", label=f"kNN {metric}")
                            lr_val = next(r[metric] for r in rows4 if r["Model"] == "Logistic Regression")
                            ax4.axhline(lr_val, color=color, linestyle=":", alpha=0.7)
                        ax4.set_xlabel("k", color="#8b949e")
                        ax4.set_ylabel("Score", color="#8b949e")
                        ax4.set_title("RQ4: kNN vs Logistic Regression", color="#e6edf3")
                        ax4.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3")
                        ax4.tick_params(colors="#8b949e")
                        for spine in ax4.spines.values():
                            spine.set_edgecolor("#30363d")
                        st.pyplot(fig4)

                    # Confusion matrix for best kNN
                    best4 = max([r for r in rows4 if "kNN" in r["Model"]], key=lambda r: r["F1"], default=None)
                    if best4:
                        y_best4 = y_preds4[best4["Model"]]
                        fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
                        for ax_, y_pred_, title_, cmap_ in [
                            (axes5[0], y_best4, f"Confusion — {best4['Model']}", "Blues"),
                            (axes5[1], y_lr4, "Confusion — Logistic Regression", "Oranges"),
                        ]:
                            ax_.set_facecolor("#161b22")
                            sns.heatmap(confusion_matrix(y_test, y_pred_), annot=True, fmt="d", cmap=cmap_, ax=ax_,
                                        linewidths=0.5, linecolor="#0d1117")
                            ax_.set_title(title_, color="#e6edf3")
                            ax_.set_ylabel("Actual", color="#8b949e")
                            ax_.set_xlabel("Predicted", color="#8b949e")
                            ax_.tick_params(colors="#8b949e")
                        plt.tight_layout()
                        st.pyplot(fig5)

                    st.markdown("""
                    **Key Finding (RQ4):** kNN clearly outperforms Logistic Regression for upset prediction.
                    It achieves strong recall (catching most actual upsets) while generating far fewer false positives.
                    Logistic Regression over-predicts upsets aggressively due to the non-linear nature of the problem.
                    kNN with k=20 was identified as the sweet spot balancing precision and recall.
                    """)


# ─────────────────────────────────────────────
# PAGE: Full Report
# ─────────────────────────────────────────────
elif page == "📊 Full Report":
    st.markdown('<div class="section-title">TENNIS ANALYTICS REPORT — 2016–2025</div>', unsafe_allow_html=True)
    st.markdown("**Authors:** Caleb Otic (25015998) & Shashwat Silwal (25006980) · 297.201 Data Science")

    st.markdown("---")
    st.markdown("## Executive Summary")
    st.markdown("""
    This report presents a data science analysis of ATP professional tennis matches spanning 2016 to 2025.
    Using regression and classification modelling, we investigated four research questions.

    | RQ | Area | Best Model | Key Metric |
    |----|------|------------|------------|
    | RQ1 | First-Round Win % | kNN Regression (k=7) | Lowest RMSE ≈ 0.1215 |
    | RQ2 | Surface Win % | Polynomial Regression | R² = 0.259; kNN RMSE ≈ 0.079 |
    | RQ3 | Tiebreak Prediction | LR (Balanced) | F1 ≈ 0.694 |
    | RQ4 | Upset Prediction | kNN k=20 | Recall = 0.865, precision >> LR |
    """)

    st.markdown("---")
    st.markdown("## Regression Findings")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### RQ1 — First-Round Win %")
        st.markdown("""
        | Model | R² |
        |-------|----|
        | Simple Linear | Low |
        | Multiple Linear | 0.147 |
        | Polynomial (Rank²) | **0.233** |
        | kNN (k=7) | **Lowest RMSE ≈ 0.1215** |

        Recent form alone is a weak predictor. Rank and age add meaningful signal. The non-linear relationship
        between ranking and first-round win rate is captured better by polynomial regression and kNN.
        """)

    with col2:
        st.markdown("### RQ2 — Surface Win %")
        st.markdown("""
        | Model | R² |
        |-------|----|
        | Simple Linear | 0.103 |
        | Multiple Linear | 0.238 |
        | Polynomial (Rank²) | **0.259** |
        | kNN (k=2) | **RMSE ≈ 0.079** |

        Surface-specific performance is moderately predictable when height and rank are included.
        Very low RMSE from kNN at k=2 suggests strong local patterns, though small k may indicate some overfitting.
        """)

    st.markdown("---")
    st.markdown("## Classification Findings")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### RQ3 — Tiebreak Prediction")
        st.markdown("""
        | Model | F1 | Accuracy |
        |-------|----|----------|
        | kNN k=3 (best kNN) | ~0.3 | ~0.60 |
        | LR Standard | ~0 | High |
        | **LR Balanced** | **~0.694** | ~0.59 |

        Standard LR collapsed to predicting "no tiebreak" for every match — class imbalance.
        Adding `class_weight='balanced'` fixed this, raising F1 to 0.694.
        Neither model is strong; tiebreaks appear close to random from pre-match features.
        """)

    with col4:
        st.markdown("### RQ4 — Upset Prediction")
        st.markdown("""
        | Model | F1 | Recall | Precision |
        |-------|----|--------|-----------|
        | kNN k=3 | High | 0.907 | Moderate |
        | **kNN k=20** | **Best overall** | 0.865 | Best |
        | Logistic Regression | Lower | Similar | Poor |

        kNN k=20 is clearly the best model. It catches the vast majority of actual upsets
        while generating roughly **half the false positives** of Logistic Regression.
        The non-linear nature of this problem favours kNN over a linear classifier.
        """)

    st.markdown("---")
    st.markdown("## Data Leakage Correction (RQ4)")
    st.markdown("""
    <div class="info-box">
    During development, <code>winner_career_surface_win_pct</code> and <code>loser_career_surface_win_pct</code>
    were initially used as predictors for RQ4. These produced suspiciously high accuracy — because they are
    computed from all historical outcomes including the current match, meaning the model implicitly knew who won.
    <br><br>
    These were replaced with <code>winner_surface_pct_last10</code> and <code>loser_surface_win_pct_last10</code>,
    which use only data from prior matches (via <code>shift(1)</code>). The resulting model produces honest,
    generalizable predictions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Limitations & Future Work")
    st.markdown("""
    - **Missing player data:** Supplementary data (height, handedness, demographics) was sourced from current ranking
      pages, so historical or inactive players often have missing attributes, reducing effective sample sizes.
    - **No in-match features:** Tiebreak prediction (RQ3) would likely benefit from in-match statistics
      (service dominance, game scores) not available in the dataset.
    - **Future improvement:** Finding richer historical player databases or APIs could significantly improve
      model coverage and accuracy across all research questions.
    """)

    st.markdown("---")
    st.markdown("""
    *ATP World Tour match data (2016–2025) · tennisabstract.com · JeffSackmann GitHub*
    """)
