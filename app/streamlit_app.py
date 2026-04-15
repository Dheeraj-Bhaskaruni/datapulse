"""DataPulse Analytics Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ingestion import DataLoader
from src.analysis.eda import AutoEDA
from src.analysis.statistical_tests import StatisticalTester
from src.analysis.segmentation import SegmentationAnalyzer
from src.features.player_features import PlayerFeatureGenerator
from src.features.market_features import MarketFeatureGenerator
from src.features.user_features import UserFeatureGenerator
from src.visualization.plots import PlotFactory
from src.pipeline.inference_pipeline import InferencePipeline, ModelNotFoundError

st.set_page_config(
    page_title="DataPulse Analytics",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container { padding-top: 1rem; max-width: 1400px; }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.85rem; margin: 0; font-weight: 500; }
    .metric-card h1 { color: #f1f5f9; font-size: 2rem; margin: 0.3rem 0; font-weight: 700; }
    .metric-card p { margin: 0; font-size: 0.8rem; }
    .metric-up { color: #10b981; }
    .metric-down { color: #ef4444; }

    .section-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
</style>
""", unsafe_allow_html=True)


def render_metric_card(col, title, value, delta="", delta_type="up"):
    delta_class = f"metric-{delta_type}" if delta else ""
    arrow = "+" if delta_type == "up" else "-"
    delta_html = f'<p class="{delta_class}">{arrow} {delta}</p>' if delta else ""
    col.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h1>{value}</h1>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    loader = DataLoader(str(ROOT / "data"))
    datasets = {}
    for name in ['players', 'contests', 'user_entries', 'market_odds', 'user_profiles']:
        try:
            datasets[name] = loader.get_sample_data(name)
        except Exception:
            pass
    return datasets


@st.cache_resource
def load_models():
    return InferencePipeline(model_path=str(ROOT / "models"))


with st.sidebar:
    st.markdown("# DataPulse")
    st.markdown("*Analytics Platform*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Overview", "Live Sports Data", "Data Explorer", "Player Analysis",
            "Risk Management", "Market Analysis", "Model Performance",
            "Predictions",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### Settings")
    theme = st.selectbox("Chart Theme", ["Dark", "Light"], index=0)
    auto_refresh = st.checkbox("Auto Refresh", value=False)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "DataPulse v1.0.0<br>2024</div>",
        unsafe_allow_html=True,
    )

datasets = load_data()
inference = load_models()


if page == "Overview":
    st.markdown('<div class="section-header">Dashboard Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    if 'players' in datasets:
        render_metric_card(c1, "Total Players", f"{len(datasets['players']):,}", "12", "up")
    if 'contests' in datasets:
        render_metric_card(c2, "Contests", f"{len(datasets['contests']):,}", "45", "up")
    if 'user_entries' in datasets:
        render_metric_card(c3, "Entries", f"{len(datasets['user_entries']):,}", "128", "up")
    if 'user_profiles' in datasets:
        render_metric_card(c4, "Users", f"{len(datasets['user_profiles']):,}", "18", "up")
    if 'market_odds' in datasets:
        render_metric_card(c5, "Markets", f"{len(datasets['market_odds']):,}", "67", "up")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    if 'players' in datasets:
        with col1:
            st.markdown("#### Fantasy Points Distribution")
            fig = PlotFactory.distribution_plot(datasets['players']['fantasy_points'], "Fantasy Points Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Points by Position")
            fig = PlotFactory.box_plot(datasets['players'], 'position', 'fantasy_points', "Fantasy Points by Position")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    if 'contests' in datasets:
        with col3:
            st.markdown("#### Contest Types")
            type_counts = datasets['contests']['contest_type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index, title="Contest Distribution",
                         color_discrete_sequence=PlotFactory.THEME['palette'])
            PlotFactory._apply_theme(fig)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    if 'user_profiles' in datasets:
        with col4:
            st.markdown("#### User Win Rate Distribution")
            fig = PlotFactory.distribution_plot(datasets['user_profiles']['win_rate'], "Win Rate Distribution")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Data Quality Summary")
    quality_data = []
    for name, df in datasets.items():
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        quality_data.append({
            'Dataset': name, 'Rows': f"{len(df):,}", 'Columns': df.shape[1],
            'Missing %': f"{missing_pct:.1f}%", 'Duplicates': df.duplicated().sum(),
            'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
        })
    st.dataframe(pd.DataFrame(quality_data), use_container_width=True, hide_index=True)


elif page == "Live Sports Data":
    st.markdown('<div class="section-header">Live Sports Data</div>', unsafe_allow_html=True)

    from src.data.live_feeds import OpenF1Client, CricketDataClient, BallDontLieClient, OddsAPIClient
    import os
    from dotenv import load_dotenv
    load_dotenv(str(ROOT / ".env"))

    api_status = {
        "F1 (OpenF1)": {"configured": True, "note": "No key needed"},
        "Cricket": {"configured": bool(os.environ.get("CRICKET_API_KEY")), "note": "cricketdata.org"},
        "NBA": {"configured": bool(os.environ.get("NBA_API_KEY")), "note": "balldontlie.io"},
        "Odds": {"configured": bool(os.environ.get("ODDS_API_KEY")), "note": "the-odds-api.com"},
    }

    status_cols = st.columns(4)
    for i, (name, info) in enumerate(api_status.items()):
        with status_cols[i]:
            if info["configured"]:
                st.success(f"**{name}** Connected")
            else:
                st.warning(f"**{name}** No key")
                st.caption(f"Free: {info['note']}")

    st.markdown("<br>", unsafe_allow_html=True)
    live_tab = st.tabs(["Formula 1", "Cricket", "NBA", "Live Odds"])

    with live_tab[0]:
        st.markdown("#### Formula 1 — Real-Time Data (OpenF1)")
        st.caption("Data from api.openf1.org — completely free, no API key needed")
        f1 = OpenF1Client()
        f1_action = st.selectbox("Select Data", [
            "Season Calendar", "Latest Drivers", "Latest Lap Times",
            "Race Stints & Tires", "Weather Data", "Race Control Messages",
        ], key="f1_action")

        if st.button("Fetch F1 Data", type="primary", key="f1_fetch"):
            with st.spinner("Fetching from OpenF1 API..."):
                try:
                    if f1_action == "Season Calendar":
                        df = f1.get_season_calendar(year=2025)
                        if df.empty:
                            df = f1.get_season_calendar(year=2024)
                        if not df.empty:
                            st.success(f"Fetched {len(df)} race weekends")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No calendar data available.")
                    elif f1_action == "Latest Drivers":
                        df = f1.get_latest_drivers()
                        if not df.empty:
                            st.success(f"Fetched {len(df)} driver records")
                            display_cols = [c for c in ['driver_number', 'broadcast_name', 'full_name',
                                                         'name_acronym', 'team_name', 'team_colour',
                                                         'country_code'] if c in df.columns]
                            if display_cols:
                                st.dataframe(df[display_cols].drop_duplicates(), use_container_width=True, hide_index=True)
                            if 'team_name' in df.columns:
                                team_counts = df.drop_duplicates(subset=['driver_number'])['team_name'].value_counts()
                                fig = px.bar(x=team_counts.index, y=team_counts.values, title="Drivers per Team",
                                             color_discrete_sequence=PlotFactory.THEME['palette'])
                                PlotFactory._apply_theme(fig)
                                st.plotly_chart(fig, use_container_width=True)
                    elif f1_action == "Latest Lap Times":
                        df = f1.get_laps(session_key="latest")
                        if not df.empty:
                            st.success(f"Fetched {len(df)} lap records")
                            display_cols = [c for c in ['driver_number', 'lap_number', 'lap_duration',
                                                         'duration_sector_1', 'duration_sector_2',
                                                         'duration_sector_3', 'is_pit_out_lap'] if c in df.columns]
                            st.dataframe(df[display_cols].head(100) if display_cols else df.head(100),
                                         use_container_width=True, hide_index=True)
                            if 'lap_duration' in df.columns and 'driver_number' in df.columns:
                                valid = df.dropna(subset=['lap_duration'])
                                if not valid.empty:
                                    avg = valid.groupby('driver_number')['lap_duration'].mean().sort_values()
                                    fig = px.bar(x=avg.index.astype(str), y=avg.values, title="Avg Lap Time by Driver",
                                                 labels={"x": "Driver #", "y": "Avg Lap Time (s)"},
                                                 color_discrete_sequence=[PlotFactory.THEME['primary']])
                                    PlotFactory._apply_theme(fig)
                                    st.plotly_chart(fig, use_container_width=True)
                    elif f1_action == "Race Stints & Tires":
                        df = f1.get_stints(session_key="latest")
                        if not df.empty:
                            st.success(f"Fetched {len(df)} stint records")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            if 'compound' in df.columns:
                                counts = df['compound'].value_counts()
                                fig = px.pie(values=counts.values, names=counts.index, title="Tire Compound Usage",
                                             color=counts.index,
                                             color_discrete_map={'SOFT': '#FF3333', 'MEDIUM': '#FFD700', 'HARD': '#FFFFFF',
                                                                 'INTERMEDIATE': '#00CC00', 'WET': '#0066FF'})
                                PlotFactory._apply_theme(fig)
                                st.plotly_chart(fig, use_container_width=True)
                    elif f1_action == "Weather Data":
                        df = f1.get_weather(session_key="latest")
                        if not df.empty:
                            st.success(f"Fetched {len(df)} weather records")
                            display_cols = [c for c in ['date', 'air_temperature', 'track_temperature',
                                                         'humidity', 'pressure', 'rainfall', 'wind_speed',
                                                         'wind_direction'] if c in df.columns]
                            st.dataframe(df[display_cols].tail(20) if display_cols else df.tail(20),
                                         use_container_width=True, hide_index=True)
                            if 'air_temperature' in df.columns and 'track_temperature' in df.columns:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(y=df['air_temperature'], mode='lines', name='Air Temp', line=dict(color='#3B82F6')))
                                fig.add_trace(go.Scatter(y=df['track_temperature'], mode='lines', name='Track Temp', line=dict(color='#EF4444')))
                                fig.update_layout(title="Temperature During Session", yaxis_title="Temperature (C)")
                                PlotFactory._apply_theme(fig)
                                st.plotly_chart(fig, use_container_width=True)
                    elif f1_action == "Race Control Messages":
                        df = f1.get_race_control(session_key="latest")
                        if not df.empty:
                            st.success(f"Fetched {len(df)} race control messages")
                            display_cols = [c for c in ['date', 'category', 'flag', 'message', 'driver_number'] if c in df.columns]
                            st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Error fetching F1 data: {str(e)}")

    with live_tab[1]:
        st.markdown("#### Cricket — Live Scores & Stats")
        cricket_key = os.environ.get("CRICKET_API_KEY", "")
        if not cricket_key:
            st.info("**Cricket API key not configured.** Get a free key at [cricketdata.org](https://cricketdata.org/) and add `CRICKET_API_KEY=your_key` to your `.env` file.")
            cricket_key_input = st.text_input("Cricket API Key", type="password", key="cricket_key_input")
            if cricket_key_input:
                cricket_key = cricket_key_input
        if cricket_key:
            cricket = CricketDataClient(api_key=cricket_key)
            cricket_action = st.selectbox("Select Data", ["Live / Current Matches", "Upcoming Matches", "Current Series", "Search Players"], key="cricket_action")
            if st.button("Fetch Cricket Data", type="primary", key="cricket_fetch"):
                with st.spinner("Fetching cricket data..."):
                    try:
                        if cricket_action == "Live / Current Matches":
                            df = cricket.get_current_matches()
                            if not df.empty:
                                st.success(f"Fetched {len(df)} current matches")
                                display_cols = [c for c in ['name', 'status', 'venue', 'match_type', 'team_1', 'team_2', 'score_1', 'score_2'] if c in df.columns]
                                st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)
                        elif cricket_action == "Current Series":
                            df = cricket.get_series()
                            if not df.empty:
                                st.dataframe(df, use_container_width=True, hide_index=True)
                        elif cricket_action == "Search Players":
                            player_name = st.text_input("Player Name", "Virat", key="cricket_player_search")
                            if player_name:
                                df = cricket.search_players(player_name)
                                if not df.empty:
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with live_tab[2]:
        st.markdown("#### NBA — Stats & Games")
        nba_key = os.environ.get("NBA_API_KEY", "")
        if not nba_key:
            st.info("**NBA API key not configured.** Get a free key at [balldontlie.io](https://www.balldontlie.io/) and add `NBA_API_KEY=your_key` to your `.env` file.")
            nba_key_input = st.text_input("NBA API Key", type="password", key="nba_key_input")
            if nba_key_input:
                nba_key = nba_key_input
        if nba_key:
            nba = BallDontLieClient(api_key=nba_key)
            nba_action = st.selectbox("Select Data", ["All Teams", "Search Players", "Recent Games", "Player Stats"], key="nba_action")
            if st.button("Fetch NBA Data", type="primary", key="nba_fetch"):
                with st.spinner("Fetching NBA data..."):
                    try:
                        if nba_action == "All Teams":
                            df = nba.get_teams()
                            if not df.empty:
                                st.dataframe(df, use_container_width=True, hide_index=True)
                        elif nba_action == "Search Players":
                            player_search = st.text_input("Player Name", "LeBron", key="nba_player_search")
                            if player_search:
                                df = nba.get_players(search=player_search)
                                if not df.empty:
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                        elif nba_action == "Recent Games":
                            from datetime import date, timedelta
                            yesterday = (date.today() - timedelta(days=1)).isoformat()
                            df = nba.get_games(dates=[yesterday, date.today().isoformat()])
                            if not df.empty:
                                st.dataframe(df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No recent games found.")
                        elif nba_action == "Player Stats":
                            player_id = st.number_input("Player ID", min_value=1, value=115, key="nba_player_id")
                            df = nba.get_stats(player_ids=[player_id], per_page=10)
                            if not df.empty:
                                st.dataframe(df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with live_tab[3]:
        st.markdown("#### Live Betting Odds — 40+ Sportsbooks")
        odds_key = os.environ.get("ODDS_API_KEY", "")
        if not odds_key:
            st.info("**Odds API key not configured.** Get a free key (500 req/mo) at [the-odds-api.com](https://the-odds-api.com/) and add `ODDS_API_KEY=your_key` to your `.env` file.")
            odds_key_input = st.text_input("Odds API Key", type="password", key="odds_key_input")
            if odds_key_input:
                odds_key = odds_key_input
        if odds_key:
            odds_client = OddsAPIClient(api_key=odds_key)
            col1, col2, col3 = st.columns(3)
            with col1:
                sport = st.selectbox("Sport", ["nba", "nfl", "mlb", "nhl", "epl", "cricket_ipl", "cricket_odi", "cricket_t20"], key="odds_sport")
            with col2:
                market = st.selectbox("Market", ["h2h", "spreads", "totals"], key="odds_market")
            with col3:
                region = st.selectbox("Region", ["us", "uk", "eu", "au"], key="odds_region")
            if st.button("Fetch Odds", type="primary", key="odds_fetch"):
                with st.spinner("Fetching live odds..."):
                    try:
                        df = odds_client.get_odds(sport=sport, markets=market, regions=region)
                        if not df.empty:
                            st.success(f"Fetched {len(df)} odds entries")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            if odds_client.remaining_requests:
                                st.caption(f"API requests remaining: {odds_client.remaining_requests}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


elif page == "Data Explorer":
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    dataset_name = st.selectbox("Select Dataset", list(datasets.keys()))
    df = datasets[dataset_name]

    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Statistics", "Correlations", "Distributions"])
    with tab1:
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)
        st.info(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns | Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with tab2:
        eda = AutoEDA(df)
        numeric_summary = df.describe().T
        if not numeric_summary.empty:
            st.dataframe(numeric_summary.round(3), use_container_width=True)
        missing = eda.get_missing_analysis()
        if missing['columns_with_missing']:
            st.markdown("##### Missing Values")
            miss_df = pd.DataFrame([{'Column': col, 'Count': info['count'], 'Percentage': f"{info['percentage']}%"}
                                     for col, info in missing['columns_with_missing'].items()])
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
    with tab3:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            fig = PlotFactory.correlation_heatmap(df, "Feature Correlations")
            st.plotly_chart(fig, use_container_width=True)
    with tab4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select Column", numeric_cols)
            fig = PlotFactory.distribution_plot(df[selected_col], f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{df[selected_col].mean():.2f}")
            col2.metric("Median", f"{df[selected_col].median():.2f}")
            col3.metric("Std Dev", f"{df[selected_col].std():.2f}")
            col4.metric("Skewness", f"{df[selected_col].skew():.2f}")


elif page == "Player Analysis":
    st.markdown('<div class="section-header">Player Analysis</div>', unsafe_allow_html=True)
    if 'players' not in datasets:
        st.error("Players dataset not found")
    else:
        players = datasets['players'].copy()
        col1, col2, col3 = st.columns(3)
        with col1:
            positions = ['All'] + sorted(players['position'].unique().tolist())
            pos_filter = st.selectbox("Position", positions)
        with col2:
            teams = ['All'] + sorted(players['team'].unique().tolist())
            team_filter = st.selectbox("Team", teams)
        with col3:
            salary_range = st.slider("Salary Range", int(players['salary'].min()), int(players['salary'].max()),
                                     (int(players['salary'].min()), int(players['salary'].max())))

        filtered = players.copy()
        if pos_filter != 'All':
            filtered = filtered[filtered['position'] == pos_filter]
        if team_filter != 'All':
            filtered = filtered[filtered['team'] == team_filter]
        filtered = filtered[(filtered['salary'] >= salary_range[0]) & (filtered['salary'] <= salary_range[1])]
        st.info(f"Showing {len(filtered)} of {len(players)} players")

        feat_gen = PlayerFeatureGenerator()
        enhanced = feat_gen.generate_all(filtered)

        tab1, tab2, tab3 = st.tabs(["Rankings", "Value Analysis", "Performance"])
        with tab1:
            display_cols = [c for c in ['name', 'team', 'position', 'fantasy_points', 'salary', 'games_played', 'consistency_score'] if c in enhanced.columns]
            ranked = enhanced[display_cols].sort_values('fantasy_points', ascending=False).head(30)
            st.dataframe(ranked, use_container_width=True, hide_index=True)
        with tab2:
            if 'points_per_dollar' in enhanced.columns:
                fig = PlotFactory.scatter_plot(enhanced, 'salary', 'fantasy_points', color='position', title="Salary vs Fantasy Points")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("##### Best Value Players")
                value_cols = [c for c in ['name', 'position', 'salary', 'fantasy_points', 'points_per_dollar', 'value_score'] if c in enhanced.columns]
                st.dataframe(enhanced[value_cols].sort_values('points_per_dollar', ascending=False).head(15), use_container_width=True, hide_index=True)
        with tab3:
            if 'consistency_score' in enhanced.columns:
                col1, col2 = st.columns(2)
                with col1:
                    fig = PlotFactory.distribution_plot(enhanced['consistency_score'], "Consistency Score Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if 'ceiling' in enhanced.columns and 'floor' in enhanced.columns:
                        fig = PlotFactory.scatter_plot(enhanced, 'floor', 'ceiling', color='position', title="Floor vs Ceiling")
                        st.plotly_chart(fig, use_container_width=True)


elif page == "Risk Management":
    st.markdown('<div class="section-header">Risk Management</div>', unsafe_allow_html=True)
    if 'user_profiles' not in datasets:
        st.error("User profiles dataset not found")
    else:
        profiles = datasets['user_profiles'].copy()
        feat_gen = UserFeatureGenerator()
        enhanced = feat_gen.generate_all(profiles)

        c1, c2, c3, c4 = st.columns(4)
        render_metric_card(c1, "Total Users", f"{len(enhanced):,}")
        if 'risk_level' in enhanced.columns:
            high_risk = (enhanced['risk_level'] == 'high').sum()
            render_metric_card(c2, "High Risk Users", str(high_risk), f"{high_risk/len(enhanced)*100:.1f}%", "down")
        if 'win_rate' in enhanced.columns:
            render_metric_card(c3, "Avg Win Rate", f"{enhanced['win_rate'].mean():.1%}")
        if 'roi' in enhanced.columns:
            render_metric_card(c4, "Avg ROI", f"{enhanced['roi'].mean():.1%}")

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Risk Profiles", "Segmentation", "Anomalies"])

        with tab1:
            if 'risk_level' in enhanced.columns:
                col1, col2 = st.columns(2)
                with col1:
                    risk_counts = enhanced['risk_level'].value_counts()
                    fig = px.pie(values=risk_counts.values, names=risk_counts.index, title="Risk Distribution",
                                 color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444'])
                    PlotFactory._apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if 'computed_risk_score' in enhanced.columns:
                        fig = PlotFactory.distribution_plot(enhanced['computed_risk_score'], "Risk Score Distribution")
                        st.plotly_chart(fig, use_container_width=True)
            if 'win_rate' in enhanced.columns and 'total_wagered' in enhanced.columns:
                fig = PlotFactory.scatter_plot(enhanced, 'win_rate', 'total_wagered',
                                              color='risk_level' if 'risk_level' in enhanced.columns else None,
                                              title="Win Rate vs Total Wagered")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            seg = SegmentationAnalyzer()
            numeric_cols = enhanced.select_dtypes(include=[np.number]).columns.tolist()
            safe_cols = [c for c in numeric_cols if c not in ['user_id']]
            if len(safe_cols) >= 2:
                n_clusters = st.slider("Number of Segments", 2, 8, 4)
                segmented, seg_metrics = seg.segment_kmeans(enhanced[safe_cols], n_clusters=n_clusters)
                st.metric("Silhouette Score", f"{seg_metrics['silhouette_score']:.4f}")
                sizes = seg_metrics['cluster_sizes']
                fig = px.bar(x=list(sizes.keys()), y=list(sizes.values()), title="Segment Sizes",
                             color_discrete_sequence=PlotFactory.THEME['palette'])
                PlotFactory._apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            from src.models.anomaly_detection import AnomalyDetectionModel
            numeric_cols = enhanced.select_dtypes(include=[np.number]).columns.tolist()
            safe_cols = [c for c in numeric_cols if c not in ['user_id']]
            if len(safe_cols) >= 2:
                anomaly_results = AnomalyDetectionModel.statistical_anomalies(enhanced, safe_cols[:5], z_threshold=3.0)
                if 'total_anomaly_flags' in anomaly_results.columns:
                    n_anomalies = (anomaly_results['total_anomaly_flags'] > 0).sum()
                    st.metric("Anomalous Users", n_anomalies)
                    if n_anomalies > 0:
                        anomalous = anomaly_results[anomaly_results['total_anomaly_flags'] > 0]
                        display_cols = [c for c in ['username', 'win_rate', 'total_wagered', 'net_profit', 'total_anomaly_flags'] if c in anomalous.columns]
                        if display_cols:
                            st.dataframe(anomalous[display_cols].head(20), use_container_width=True, hide_index=True)


elif page == "Market Analysis":
    st.markdown('<div class="section-header">Market Analysis</div>', unsafe_allow_html=True)
    if 'market_odds' not in datasets:
        st.error("Market odds dataset not found")
    else:
        odds = datasets['market_odds'].copy()
        feat_gen = MarketFeatureGenerator()
        enhanced = feat_gen.generate_all(odds)

        c1, c2, c3, c4 = st.columns(4)
        render_metric_card(c1, "Total Markets", f"{len(enhanced):,}")
        if 'ev_positive' in enhanced.columns:
            render_metric_card(c2, "+EV Markets", f"{enhanced['ev_positive'].mean() * 100:.1f}%")
        if 'beat_closing_line' in enhanced.columns:
            render_metric_card(c3, "Beat CLV", f"{enhanced['beat_closing_line'].mean() * 100:.1f}%")
        if 'significant_move' in enhanced.columns:
            render_metric_card(c4, "Sig. Line Moves", f"{enhanced['significant_move'].mean() * 100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Line Movement", "Expected Value", "Market Efficiency"])

        with tab1:
            if 'line_movement' in enhanced.columns:
                col1, col2 = st.columns(2)
                with col1:
                    fig = PlotFactory.distribution_plot(enhanced['line_movement'], "Line Movement Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if 'market_type' in enhanced.columns:
                        fig = PlotFactory.box_plot(enhanced, 'market_type', 'line_movement', "Line Movement by Market")
                        st.plotly_chart(fig, use_container_width=True)
        with tab2:
            if 'expected_value' in enhanced.columns:
                fig = PlotFactory.distribution_plot(enhanced['expected_value'], "Expected Value Distribution")
                st.plotly_chart(fig, use_container_width=True)
                if 'implied_probability' in enhanced.columns:
                    fig = PlotFactory.scatter_plot(enhanced, 'implied_probability', 'expected_value',
                                                  color='market_type' if 'market_type' in enhanced.columns else None,
                                                  title="Implied Probability vs Expected Value")
                    st.plotly_chart(fig, use_container_width=True)
        with tab3:
            if 'calibration_error' in enhanced.columns:
                st.metric("Mean Calibration Error", f"{enhanced['calibration_error'].mean():.4f}")
                fig = PlotFactory.distribution_plot(enhanced['calibration_error'], "Calibration Error Distribution")
                st.plotly_chart(fig, use_container_width=True)
            if 'prediction_correct' in enhanced.columns:
                st.metric("Market Prediction Accuracy", f"{enhanced['prediction_correct'].mean():.1%}")


elif page == "Model Performance":
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    import json

    model_info = {
        "Player Performance": {"path": "player_performance", "type": "Regression", "algorithm": "Gradient Boosting Regressor"},
        "Risk Scoring": {"path": "risk_scoring", "type": "Classification", "algorithm": "Random Forest Classifier"},
        "Market Predictor": {"path": "market_predictor", "type": "Classification", "algorithm": "Calibrated Gradient Boosting"},
        "Anomaly Detection": {"path": "anomaly_detection", "type": "Unsupervised", "algorithm": "Isolation Forest"},
    }

    c1, c2 = st.columns(2)
    render_metric_card(c1, "Models Loaded", str(len(inference.available_models())))
    render_metric_card(c2, "Available", ", ".join(inference.available_models()) if inference.available_models() else "None")
    st.markdown("<br>", unsafe_allow_html=True)

    for display_name, info in model_info.items():
        meta_path = ROOT / "models" / info["path"] / "metadata.json"
        with st.expander(f"**{display_name}** — {info['algorithm']}", expanded=True):
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                st.caption(f"Type: {info['type']} | Version: {meta.get('version', '1.0.0')}")
                cv = meta.get("cv_metrics", {})
                if cv:
                    cols = st.columns(min(len(cv), 5))
                    for i, (key, value) in enumerate(list(cv.items())[:5]):
                        cols[i % len(cols)].metric(key.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else str(value))
                features = meta.get("feature_names", [])
                if features:
                    st.caption(f"Features ({len(features)}): {', '.join(features[:8])}{'...' if len(features) > 8 else ''}")
            else:
                st.warning(f"Model not found at {meta_path}")

    st.markdown("---")
    st.caption("Models are pre-trained and shipped with the app. To retrain, run `make train` locally and redeploy.")


elif page == "Predictions":
    st.markdown('<div class="section-header">Real-Time Predictions</div>', unsafe_allow_html=True)
    available = inference.available_models()
    if not available:
        st.error("No trained models found. Run `make build` locally and redeploy.")
    else:
        st.caption(f"Models loaded: {', '.join(available)}")

    pred_type = st.selectbox("Prediction Type", ["Player Performance", "Risk Score", "Market Evaluation"])

    if pred_type == "Player Performance":
        st.markdown("#### Predict Player Fantasy Points")
        col1, col2, col3 = st.columns(3)
        with col1:
            points_avg = st.number_input("Points Avg", 0.0, 50.0, 20.0)
            assists_avg = st.number_input("Assists Avg", 0.0, 15.0, 5.0)
        with col2:
            rebounds_avg = st.number_input("Rebounds Avg", 0.0, 15.0, 7.0)
            games_played = st.number_input("Games Played", 0, 82, 60)
        with col3:
            salary = st.number_input("Salary", 3000, 12000, 7000)
            consistency = st.slider("Consistency Score", 0.0, 1.0, 0.7)

        if st.button("Predict", type="primary"):
            features = {"points_avg": points_avg, "assists_avg": assists_avg, "rebounds_avg": rebounds_avg,
                        "games_played": float(games_played), "salary": float(salary), "consistency_score": consistency}
            try:
                result = inference.predict_player(features)
                predicted_fp = result['prediction']
                st.success(f"Predicted Fantasy Points: **{predicted_fp:.1f}**")
                st.caption(f"Model version: {result.get('model_version', '1.0.0')}")
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = PlotFactory.gauge_chart(predicted_fp, "Predicted Fantasy Points", 80)
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    feat_df = pd.DataFrame({'Feature': list(features.keys()), 'Value': list(features.values())})
                    fig = px.bar(feat_df, x='Feature', y='Value', title="Input Features",
                                 color_discrete_sequence=[PlotFactory.THEME['primary']])
                    PlotFactory._apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
            except ModelNotFoundError as e:
                st.error(str(e))

    elif pred_type == "Risk Score":
        st.markdown("#### Evaluate User Risk")
        col1, col2 = st.columns(2)
        with col1:
            win_rate = st.slider("Win Rate", 0.0, 1.0, 0.5)
            total_wagered = st.number_input("Total Wagered ($)", 0, 500000, 10000)
            avg_entry_fee = st.number_input("Avg Entry Fee ($)", 0.0, 500.0, 25.0)
        with col2:
            total_entries = st.number_input("Total Entries", 0, 10000, 500)
            total_won = st.number_input("Total Won ($)", 0, 500000, 8000)
            net_profit = st.number_input("Net Profit ($)", -100000, 200000, -2000)

        if st.button("Score Risk", type="primary"):
            features = {"total_entries": float(total_entries), "win_rate": win_rate, "avg_entry_fee": avg_entry_fee,
                        "total_wagered": float(total_wagered), "total_won": float(total_won), "net_profit": float(net_profit)}
            try:
                result = inference.score_risk(features)
                risk_cat = result['risk_category']
                probas = result['probabilities']
                level_map = {0: "Low", 1: "Medium", 2: "High"}
                level = level_map.get(risk_cat, f"Category {risk_cat}")
                st.markdown(f"### Risk Level: :{('green' if risk_cat == 0 else 'orange' if risk_cat == 1 else 'red')}[**{level}**]")
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = PlotFactory.gauge_chart(max(probas) * 100, "Model Confidence (%)", 100)
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    prob_df = pd.DataFrame({'Class': [level_map.get(i, str(i)) for i in range(len(probas))],
                                            'Probability': [round(p, 4) for p in probas]})
                    fig = px.bar(prob_df, x='Class', y='Probability', title="Class Probabilities",
                                 color_discrete_sequence=[PlotFactory.THEME['primary']])
                    PlotFactory._apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
            except ModelNotFoundError as e:
                st.error(str(e))

    elif pred_type == "Market Evaluation":
        st.markdown("#### Evaluate Market Expected Value")
        col1, col2 = st.columns(2)
        with col1:
            odds = st.number_input("American Odds", -500, 500, -110)
        with col2:
            true_prob = st.slider("Your Estimated Probability", 0.0, 1.0, 0.55)

        if st.button("Evaluate", type="primary"):
            implied = 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)
            payout = odds / 100 if odds > 0 else 100 / abs(odds)
            ev = (true_prob * payout) - (1 - true_prob)
            edge = true_prob - implied

            c1, c2, c3 = st.columns(3)
            c1.metric("Implied Probability", f"{implied:.1%}")
            c2.metric("Expected Value", f"{ev:.4f}", delta="Positive" if ev > 0 else "Negative")
            c3.metric("Edge", f"{edge:.1%}", delta="Favorable" if edge > 0 else "Unfavorable")

            if ev > 0.02:
                st.success("Recommendation: BET - Positive expected value detected!")
            elif ev > 0:
                st.warning("Marginal edge - Proceed with caution.")
            else:
                st.error("Recommendation: PASS - Negative expected value.")

            st.markdown("##### Sensitivity Analysis")
            prob_range = np.linspace(max(0.01, true_prob - 0.2), min(0.99, true_prob + 0.2), 50)
            evs = [(p * payout) - (1 - p) for p in prob_range]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prob_range, y=evs, mode='lines', name='Expected Value', line=dict(color=PlotFactory.THEME['primary'])))
            fig.add_hline(y=0, line_dash="dash", line_color=PlotFactory.THEME['danger'])
            fig.add_vline(x=true_prob, line_dash="dash", line_color=PlotFactory.THEME['success'], annotation_text="Your estimate")
            fig.update_layout(title="EV vs Probability", xaxis_title="True Probability", yaxis_title="Expected Value")
            PlotFactory._apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
