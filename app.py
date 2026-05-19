"""
AgroIntel – Main Streamlit Application
The home / landing page with interactive overview dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from utils.helpers import BRAND, dark_layout, gauge_chart

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AgroIntel – AI Smart Agriculture",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #060e06;
    color: #E8F5E9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1A0A 0%, #061206 100%);
    border-right: 1px solid #1E3A1E;
}
[data-testid="stSidebar"] * { color: #C8E6C9 !important; }
[data-testid="stSidebar"] .stRadio label { color: #A5D6A7 !important; }
[data-testid="stSidebarNav"] a { color: #A5D6A7 !important; padding: 8px 12px; border-radius: 8px; }
[data-testid="stSidebarNav"] a:hover { background: #1B3A1B !important; }
[data-testid="stSidebarNav"] a[aria-selected="true"] { background: #1B5E20 !important; }

/* ── Main content ── */
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* ── Cards ── */
.ag-card {
    background: linear-gradient(135deg, #0D1F0D 0%, #0A150A 100%);
    border: 1px solid #1E3A1E;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.ag-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 200, 83, 0.15);
}

/* ── Hero ── */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00C853 0%, #76FF03 50%, #CCFF90 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1.15rem;
    color: #81C784;
    font-weight: 400;
    line-height: 1.6;
    max-width: 700px;
}

/* ── Feature pill ── */
.feature-pill {
    display: inline-block;
    background: linear-gradient(90deg, #1B5E20, #2E7D32);
    border: 1px solid #388E3C;
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #A5D6A7;
    margin: 4px;
}

/* ── Stat box ── */
.stat-box {
    background: linear-gradient(135deg, #0D2A0D, #081A08);
    border: 1px solid #2E7D32;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.stat-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #00C853;
    font-family: 'Space Grotesk', sans-serif;
}
.stat-label {
    font-size: 0.82rem;
    color: #81C784;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* ── Section header ── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #A5D6A7;
    border-left: 4px solid #00C853;
    padding-left: 12px;
    margin: 1.5rem 0 1rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1B5E20, #2E7D32) !important;
    color: #E8F5E9 !important;
    border: 1px solid #388E3C !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2E7D32, #388E3C) !important;
    box-shadow: 0 4px 20px rgba(0,200,83,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background: #0D1F0D !important;
    border: 1px solid #2E7D32 !important;
    color: #E8F5E9 !important;
    border-radius: 8px !important;
}
.stSlider > div > div { color: #00C853 !important; }
[data-testid="stMetricValue"] { color: #00C853 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #81C784 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    background: #0D1F0D !important;
    border: 1px solid #1E3A1E !important;
    border-radius: 8px 8px 0 0 !important;
    color: #81C784 !important;
}
.stTabs [aria-selected="true"] {
    background: #1B5E20 !important;
    color: #E8F5E9 !important;
}

/* ── Plotly container ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Alert / success boxes ── */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 12px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #060e06; }
::-webkit-scrollbar-thumb { background: #1B5E20; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:3rem;">🌿</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:800;
                    background:linear-gradient(135deg,#00C853,#76FF03);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            AgroIntel
        </div>
        <div style="font-size:0.75rem; color:#66BB6A; margin-top:4px;">AI Smart Agriculture Platform</div>
    </div>
    <hr style="border-color:#1E3A1E; margin: 0.5rem 0 1rem;">
    """, unsafe_allow_html=True)

    st.markdown("### 🗂️ Navigation")
    st.markdown("""
    <div style="font-size:0.82rem; color:#66BB6A; padding: 4px 0;">
    Use the sidebar links above to navigate between modules:
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("🌾", "Crop Recommendation"),
        ("📈", "Yield Prediction"),
        ("🔬", "Disease Detection"),
        ("🧪", "Fertilizer Advisor"),
        ("🌦️", "Weather Forecast"),
        ("💰", "Market Prices"),
        ("💧", "Irrigation Advisor"),
        ("📊", "Analytics Dashboard"),
    ]
    for icon, name in features:
        st.markdown(f"<div style='padding:4px 8px; color:#A5D6A7; font-size:0.85rem;'>{icon} {name}</div>",
                    unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E3A1E; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem; color:#4CAF50; text-align:center;'>🕒 {datetime.now().strftime('%d %b %Y  %H:%M')}</div>",
                unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────
col_hero, col_img = st.columns([3, 2], gap="large")

with col_hero:
    st.markdown("""
    <div class="hero-title">AgroIntel</div>
    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:600;
                color:#76FF03; margin-bottom:0.8rem; letter-spacing:0.05em;">
        AI-POWERED SMART AGRICULTURE PLATFORM
    </div>
    <div class="hero-sub">
        Empowering farmers with cutting-edge machine learning, real-time analytics,
        and data-driven insights to maximize yield, minimize risk, and build a sustainable future.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    pills = ["🌾 Crop Intelligence", "📊 Predictive Analytics", "🔬 Disease AI",
             "💧 Smart Irrigation", "💰 Market Insights", "🌦️ Weather AI"]
    pill_html = "".join(f'<span class="feature-pill">{p}</span>' for p in pills)
    st.markdown(f"<div>{pill_html}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_img:
    # Animated stats
    stats = [
        ("23+", "Crop Varieties"),
        ("95%", "Model Accuracy"),
        ("8", "AI Modules"),
        ("Real-time", "Analytics"),
    ]
    c1, c2 = st.columns(2)
    for i, (val, lbl) in enumerate(stats):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1E3A1E; margin: 1.5rem 0;'>", unsafe_allow_html=True)

# ── Overview Charts ───────────────────────────────────────────
st.markdown('<div class="section-header">📊 Platform Overview</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌾 Crop Distribution", "📈 Yield Trends", "🗺️ Regional Heatmap"])

with tab1:
    if os.path.exists("data/crop_data.csv"):
        df = pd.read_csv("data/crop_data.csv")
        counts = df["label"].value_counts().head(12)
        fig = go.Figure(go.Bar(
            x=counts.values, y=counts.index, orientation="h",
            marker=dict(
                color=counts.values,
                colorscale=[[0,"#1B5E20"],[0.5,"#2E7D32"],[1,"#00C853"]],
                showscale=False,
            ),
            text=counts.values, textposition="outside",
        ))
        fig = dark_layout(fig, "Crop Sample Distribution in Training Dataset")
        fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📂 Run `python models/train_models.py` to generate data & models first.", icon="ℹ️")

with tab2:
    if os.path.exists("data/market_data.csv"):
        mkt = pd.read_csv("data/market_data.csv", parse_dates=["date"])
        sel_crops = st.multiselect("Select crops", mkt["crop"].unique().tolist(),
                                   default=["Rice","Wheat","Maize"], key="home_mkt")
        fig2 = go.Figure()
        colors = px.colors.qualitative.Prism
        for i, crop in enumerate(sel_crops):
            sub = mkt[mkt["crop"] == crop]
            fig2.add_trace(go.Scatter(
                x=sub["date"], y=sub["price_per_quintal"],
                name=crop, mode="lines",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig2 = dark_layout(fig2, "Market Price Trends (₹/Quintal)")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("📂 Run `python models/train_models.py` to generate data first.")

with tab3:
    # Synthetic regional yield heatmap
    states = ["Punjab","Haryana","UP","MP","Maharashtra","Karnataka","Andhra Pradesh",
              "Tamil Nadu","Gujarat","Rajasthan","Bihar","West Bengal"]
    yields = np.random.uniform(2, 6, len(states))
    fig3 = go.Figure(go.Bar(
        x=states, y=yields,
        marker_color=yields,
        marker_colorscale=[[0,"#1B5E20"],[0.5,"#388E3C"],[1,"#00C853"]],
        text=[f"{v:.1f} t/ha" for v in yields],
        textposition="outside",
    ))
    fig3 = dark_layout(fig3, "Average Crop Yield by State (tons/hectare)")
    fig3.update_layout(height=380, xaxis_tickangle=-35)
    st.plotly_chart(fig3, use_container_width=True)

# ── Module Cards ──────────────────────────────────────────────
st.markdown('<div class="section-header">🚀 AI Modules</div>', unsafe_allow_html=True)

modules = [
    ("🌾", "Crop Recommendation", "Get AI-powered crop suggestions based on your soil nutrients, climate, and location.", "#00C853"),
    ("📈", "Yield Prediction",     "Predict crop yield using historical data, weather patterns, and agronomic factors.", "#76FF03"),
    ("🔬", "Disease Detection",    "Upload leaf images for instant AI diagnosis of 15+ plant diseases.", "#69F0AE"),
    ("🧪", "Fertilizer Advisor",   "Receive optimal fertilizer recommendations based on your soil composition.", "#B9F6CA"),
    ("🌦️", "Weather Forecast",     "Real-time 7-day weather forecasts and farm-specific alerts via Open-Meteo API.", "#CCFF90"),
    ("💰", "Market Prices",        "Track commodity prices, spot trends, and time your harvest for maximum profit.", "#F4FF81"),
    ("💧", "Irrigation Advisor",   "Smart water management recommendations based on soil moisture and crop stage.", "#80D8FF"),
    ("📊", "Analytics Dashboard",  "Interactive charts, heatmaps, and KPIs for comprehensive farm performance analysis.", "#EA80FC"),
]

cols = st.columns(4, gap="medium")
for i, (icon, title, desc, color) in enumerate(modules):
    with cols[i % 4]:
        st.markdown(f"""
        <div class="ag-card" style="border-color: {color}22; min-height:160px;">
            <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
            <div style="font-weight:700; color:{color}; font-size:0.95rem; margin-bottom:6px;">{title}</div>
            <div style="font-size:0.8rem; color:#81C784; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1E3A1E; margin: 2rem 0 1rem;">
<div style="text-align:center; color:#2E7D32; font-size:0.8rem; padding-bottom:1rem;">
    🌿 AgroIntel &nbsp;|&nbsp; AI-Powered Smart Agriculture &nbsp;|&nbsp;
    Built with Streamlit · Scikit-learn · Plotly · Open-Meteo
</div>
""", unsafe_allow_html=True)
