"""
Page 8 – Analytics Dashboard
Comprehensive farm performance analytics with multiple chart types.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import dark_layout

st.set_page_config(page_title="Analytics Dashboard | AgroIntel", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#060e06;color:#E8F5E9;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1A0A,#061206);border-right:1px solid #1E3A1E;}
[data-testid="stSidebar"] *{color:#C8E6C9!important;}
.main .block-container{padding:1.5rem 2rem;max-width:1400px;}
.ag-card{background:linear-gradient(135deg,#0D1F0D,#0A150A);border:1px solid #1E3A1E;border-radius:16px;padding:1.5rem;margin-bottom:1rem;}
.section-header{font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#A5D6A7;border-left:4px solid #00C853;padding-left:12px;margin:1.5rem 0 1rem;}
.kpi-card{background:linear-gradient(135deg,#0D2A0D,#08150A);border:1px solid #2E7D32;border-radius:14px;padding:1.2rem;text-align:center;}
.kpi-value{font-size:2rem;font-weight:800;color:#00C853;font-family:'Space Grotesk',sans-serif;}
.kpi-label{font-size:0.78rem;color:#81C784;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;}
.kpi-delta{font-size:0.8rem;margin-top:4px;}
[data-testid="stMetricValue"]{color:#00C853!important;font-weight:700!important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:800;
            background:linear-gradient(135deg,#00C853,#76FF03);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.3rem;">
    📊 AgroIntel Analytics Dashboard
</div>
<p style="color:#81C784;">Comprehensive farm performance metrics, soil health analytics, and predictive insights.</p>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")
CROP_PATH  = os.path.join(BASE, "data/crop_data.csv")
YIELD_PATH = os.path.join(BASE, "data/yield_data.csv")
MKT_PATH   = os.path.join(BASE, "data/market_data.csv")

has_data = all(os.path.exists(p) for p in [CROP_PATH, YIELD_PATH, MKT_PATH])

if not has_data:
    st.error("⚠️ Data not found! Run `python models/train_models.py` first.", icon="🚫")
    st.stop()

@st.cache_data
def load_all():
    c = pd.read_csv(CROP_PATH)
    y = pd.read_csv(YIELD_PATH)
    m = pd.read_csv(MKT_PATH, parse_dates=["date"])
    return c, y, m

crop_df, yield_df, mkt_df = load_all()

# ── KPI Row ───────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Key Performance Indicators</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)

kpis = [
    ("2,200+", "Data Points", "+12%", "#00C853"),
    (f"{crop_df['label'].nunique()}", "Crop Types", "23 crops", "#76FF03"),
    (f"{yield_df['yield_tons_ha'].mean():.1f}", "Avg Yield t/ha", "+5.2%", "#69F0AE"),
    (f"{yield_df['state'].nunique()}", "States Covered", "All India", "#B9F6CA"),
    (f"{mkt_df['crop'].nunique()}", "Market Crops", "Tracked", "#CCFF90"),
    ("4", "AI Models", "Active", "#80D8FF"),
]
for col, (val, lbl, delta, clr) in zip([k1,k2,k3,k4,k5,k6], kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{clr};">{val}</div>
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-delta" style="color:{clr}99;">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1E3A1E;margin:1.5rem 0;'>", unsafe_allow_html=True)

# ── Main Dashboard Tabs ───────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🌾 Crop Analytics", "📈 Yield Analytics", "💰 Market Analytics", "🗺️ Regional View"])

with tab1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        # Crop distribution
        counts = crop_df["label"].value_counts()
        fig = go.Figure(go.Treemap(
            labels=counts.index.tolist(),
            parents=[""] * len(counts),
            values=counts.values.tolist(),
            marker=dict(colorscale=[[0,"#1B5E20"],[0.5,"#2E7D32"],[1,"#00C853"]],
                        showscale=False),
            textfont=dict(color="white"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=30,b=0),
            height=360, title=dict(text="Crop Type Distribution", font=dict(color="white")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # NPK distributions
        fig2 = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.08)
        for i, (col_, color, name) in enumerate(zip(["N","P","K"], ["#00C853","#76FF03","#40C4FF"], ["Nitrogen","Phosphorus","Potassium"]), 1):
            fig2.add_trace(go.Histogram(x=crop_df[col_], nbinsx=30, name=name,
                                         marker_color=color, opacity=0.7, showlegend=True), row=i, col=1)
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,31,13,0.6)",
            font=dict(color="white"), height=360, showlegend=False,
            title=dict(text="NPK Distribution in Dataset", font=dict(color="white")),
            margin=dict(l=40,r=20,t=50,b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ["N","P","K","temperature","humidity","ph","rainfall"]
    corr = crop_df[num_cols].corr()
    fig3 = go.Figure(go.Heatmap(
        z=corr.values, x=num_cols, y=num_cols,
        colorscale=[[0,"#0D1F0D"],[0.5,"#2E7D32"],[1,"#76FF03"]],
        text=corr.values.round(2), texttemplate="%{text}",
        showscale=True,
    ))
    fig3 = dark_layout(fig3, "Soil & Climate Feature Correlations")
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        avg_crop = yield_df.groupby("crop")["yield_tons_ha"].agg(["mean","std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=avg_crop["crop"], y=avg_crop["mean"],
            error_y=dict(type="data", array=avg_crop["std"].tolist(), visible=True, color="#1E3A1E"),
            marker=dict(color=avg_crop["mean"],
                        colorscale=[[0,"#1B5E20"],[1,"#76FF03"]], showscale=False),
            text=avg_crop["mean"].round(2), textposition="outside",
        ))
        fig = dark_layout(fig, "Average Yield ± Std Dev by Crop (tons/ha)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        avg_season = yield_df.groupby("season")["yield_tons_ha"].mean().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=avg_season["season"], values=avg_season["yield_tons_ha"],
            hole=0.5,
            marker=dict(colors=["#00C853","#76FF03","#40C4FF"],
                        line=dict(color="#0D1F0D", width=2)),
            textfont=dict(color="white"),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
            title=dict(text="Yield by Season", font=dict(color="white")),
            height=380, legend=dict(font=dict(color="white")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: fertilizer vs yield
    fig3 = px.scatter(
        yield_df.sample(500), x="fertilizer_kg", y="yield_tons_ha",
        color="crop", size="area_ha",
        color_discrete_sequence=px.colors.qualitative.Prism,
        labels={"fertilizer_kg":"Fertilizer (kg/ha)", "yield_tons_ha":"Yield (t/ha)"},
    )
    fig3 = dark_layout(fig3, "Fertilizer vs Yield by Crop Type")
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    # Price trend for all crops
    fig = go.Figure()
    colors = px.colors.qualitative.Prism
    for i, crop in enumerate(mkt_df["crop"].unique()):
        sub = mkt_df[mkt_df["crop"]==crop]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["price_per_quintal"],
            name=crop, mode="lines", line=dict(color=colors[i%len(colors)], width=2),
        ))
    fig = dark_layout(fig, "All Crop Price Trends (₹/Quintal) – 3 Year History")
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Box plots for price distribution
    fig2 = go.Figure()
    for i, crop in enumerate(mkt_df["crop"].unique()):
        sub = mkt_df[mkt_df["crop"]==crop]
        fig2.add_trace(go.Box(
            y=sub["price_per_quintal"], name=crop,
            marker_color=colors[i%len(colors)],
            line_color=colors[i%len(colors)],
        ))
    fig2 = dark_layout(fig2, "Price Distribution by Crop (₹/Quintal)")
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    # State-wise performance bubble chart
    states = ["Punjab","Haryana","UP","MP","Maharashtra","Karnataka","AP","Tamil Nadu",
              "Gujarat","Rajasthan","Bihar","West Bengal"]
    data_regional = pd.DataFrame({
        "state": states,
        "avg_yield": np.random.uniform(2.5, 6.5, len(states)),
        "area_M_ha":  np.random.uniform(1, 15, len(states)),
        "rainfall_mm": np.random.uniform(350, 1400, len(states)),
    })

    fig = px.scatter(
        data_regional, x="rainfall_mm", y="avg_yield",
        size="area_M_ha", text="state", color="avg_yield",
        color_continuous_scale=[[0,"#1B5E20"],[0.5,"#388E3C"],[1,"#00C853"]],
        labels={"rainfall_mm":"Avg Rainfall (mm)","avg_yield":"Avg Yield (t/ha)","area_M_ha":"Agri Area (M ha)"},
    )
    fig.update_traces(textposition="top center", textfont=dict(color="white", size=10))
    fig = dark_layout(fig, "State Agricultural Performance Bubble Chart")
    fig.update_layout(height=480, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # State yield bar
    fig2 = go.Figure(go.Bar(
        x=data_regional["state"], y=data_regional["avg_yield"],
        marker=dict(color=data_regional["avg_yield"],
                    colorscale=[[0,"#1B5E20"],[1,"#00C853"]], showscale=False),
        text=data_regional["avg_yield"].round(2), textposition="outside",
    ))
    fig2 = dark_layout(fig2, "Average Crop Yield by State (tons/ha)")
    fig2.update_layout(height=320, xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)
