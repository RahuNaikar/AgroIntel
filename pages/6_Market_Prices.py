"""
Page 6 – Market Price Prediction & Trends
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import dark_layout

st.set_page_config(page_title="Market Prices | AgroIntel", page_icon="💰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#060e06;color:#E8F5E9;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1A0A,#061206);border-right:1px solid #1E3A1E;}
[data-testid="stSidebar"] *{color:#C8E6C9!important;}
.main .block-container{padding:1.5rem 2rem;max-width:1400px;}
.ag-card{background:linear-gradient(135deg,#0D1F0D,#0A150A);border:1px solid #1E3A1E;border-radius:16px;padding:1.5rem;margin-bottom:1rem;}
.section-header{font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#A5D6A7;border-left:4px solid #00C853;padding-left:12px;margin:1.5rem 0 1rem;}
.price-card{background:linear-gradient(135deg,#1A2A0D,#0D1A07);border:1px solid #2E7D32;border-radius:12px;padding:1rem;text-align:center;}
.stButton>button{background:linear-gradient(135deg,#7B1FA2,#6A1B9A)!important;color:#E8F5E9!important;border:1px solid #8E24AA!important;border-radius:10px!important;font-weight:600!important;}
[data-testid="stMetricValue"]{color:#F4FF81!important;font-weight:700!important;}
.stSelectbox div[data-baseweb="select"]{background:#0D1F0D!important;border:1px solid #2E7D32!important;color:#E8F5E9!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">💰 Market Price Intelligence</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#81C784;">Track commodity prices, spot market trends, and get AI-based sell timing recommendations to maximize your revenue.</p>', unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data/market_data.csv")

if not os.path.exists(DATA_PATH):
    st.error("⚠️ Market data not found! Run `python models/train_models.py` first.")
    st.stop()

@st.cache_data
def load_market():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

mkt = load_market()
crops = sorted(mkt["crop"].unique().tolist())

# ── Current price summary ─────────────────────────────────────
latest = mkt.groupby("crop").last().reset_index()
st.markdown('<div class="section-header">📊 Current Market Snapshot</div>', unsafe_allow_html=True)

cols = st.columns(len(crops))
for i, (_, row) in enumerate(latest.iterrows()):
    prev = mkt[mkt["crop"] == row["crop"]].iloc[-2]["price_per_quintal"]
    change = ((row["price_per_quintal"] - prev) / prev) * 100
    with cols[i]:
        st.markdown(f"""
        <div class="price-card">
            <div style="font-size:0.75rem;color:#81C784;">{row['crop']}</div>
            <div style="font-size:1.3rem;font-weight:800;color:#F4FF81;">₹{row['price_per_quintal']:,.0f}</div>
            <div style="font-size:0.7rem;color:{'#00C853' if change>=0 else '#EF5350'};">
                {'▲' if change>=0 else '▼'} {abs(change):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1E3A1E;margin:1.5rem 0;'>", unsafe_allow_html=True)

# ── Price trend charts ────────────────────────────────────────
col_chart, col_analysis = st.columns([3, 2], gap="large")

with col_chart:
    st.markdown('<div class="section-header">📈 Price Trend Analysis</div>', unsafe_allow_html=True)
    sel = st.multiselect("Select Crops to Compare", crops, default=["Rice","Wheat","Maize"])
    period = st.selectbox("Time Period", ["Last 3 Months","Last 6 Months","Last 1 Year","All Time"], index=2)
    period_days = {"Last 3 Months":90,"Last 6 Months":180,"Last 1 Year":365,"All Time":9999}[period]
    cutoff = mkt["date"].max() - timedelta(days=period_days)

    fig = go.Figure()
    colors = px.colors.qualitative.Prism
    for i, crop in enumerate(sel):
        sub = mkt[(mkt["crop"]==crop) & (mkt["date"]>=cutoff)].copy()
        # Moving average
        sub["ma4"] = sub["price_per_quintal"].rolling(4).mean()
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["price_per_quintal"], name=crop,
            mode="lines", line=dict(color=colors[i%len(colors)], width=1.5, dash="dot"),
            opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["ma4"], name=f"{crop} (4W MA)",
            mode="lines", line=dict(color=colors[i%len(colors)], width=2.5),
            showlegend=False,
        ))

    fig = dark_layout(fig, "Crop Prices (₹/Quintal) with 4-Week Moving Average")
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with col_analysis:
    st.markdown('<div class="section-header">🔍 Price Analysis</div>', unsafe_allow_html=True)
    crop_sel = st.selectbox("Analyze Crop", crops, key="analysis_crop")
    sub_a = mkt[mkt["crop"] == crop_sel].copy()
    sub_a["month"] = sub_a["date"].dt.month

    latest_price = sub_a["price_per_quintal"].iloc[-1]
    avg_price    = sub_a["price_per_quintal"].mean()
    max_price    = sub_a["price_per_quintal"].max()
    min_price    = sub_a["price_per_quintal"].min()
    trend_3m     = sub_a.tail(12)["price_per_quintal"]
    trend_pct    = ((trend_3m.iloc[-1] - trend_3m.iloc[0]) / trend_3m.iloc[0]) * 100

    st.metric("Current Price",  f"₹{latest_price:,.0f}", f"{trend_pct:+.1f}% (3M)")
    st.metric("Average Price",  f"₹{avg_price:,.0f}")
    st.metric("52W High",       f"₹{max_price:,.0f}")
    st.metric("52W Low",        f"₹{min_price:,.0f}")

    # Seasonality
    monthly = sub_a.groupby("month")["price_per_quintal"].mean().reset_index()
    best_month = monthly.loc[monthly["price_per_quintal"].idxmax(), "month"]
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    st.markdown(f"""
    <div class="ag-card" style="margin-top:1rem;">
        <div style="font-weight:700;color:#F4FF81;margin-bottom:0.5rem;">🎯 Best Selling Month</div>
        <div style="font-size:1.8rem;font-weight:800;color:#00C853;">{month_names[best_month]}</div>
        <div style="color:#81C784;font-size:0.85rem;">Historically highest prices for {crop_sel}</div>
    </div>
    """, unsafe_allow_html=True)

    # Monthly seasonality bar
    fig4 = go.Figure(go.Bar(
        x=[month_names[m] for m in monthly["month"]],
        y=monthly["price_per_quintal"],
        marker=dict(color=monthly["price_per_quintal"],
                    colorscale=[[0,"#1B5E20"],[0.5,"#388E3C"],[1,"#F4FF81"]], showscale=False),
    ))
    fig4 = dark_layout(fig4, f"{crop_sel} – Seasonal Price Pattern")
    fig4.update_layout(height=250, margin=dict(l=20,r=10,t=40,b=20))
    st.plotly_chart(fig4, use_container_width=True)

# ── Volatility heatmap ────────────────────────────────────────
st.markdown('<div class="section-header">📊 Price Volatility Heatmap</div>', unsafe_allow_html=True)
mkt_recent = mkt[mkt["date"] >= mkt["date"].max() - timedelta(days=365)]
pivot = mkt_recent.pivot_table(index=mkt_recent["date"].dt.strftime("%b %Y"),
                                columns="crop", values="price_per_quintal", aggfunc="mean")
fig5 = go.Figure(go.Heatmap(
    z=pivot.values, x=pivot.columns, y=pivot.index,
    colorscale=[[0,"#0D1F0D"],[0.5,"#2E7D32"],[1,"#F4FF81"]],
    showscale=True,
))
fig5 = dark_layout(fig5, "Average Monthly Prices by Crop (₹/Quintal)")
fig5.update_layout(height=380)
st.plotly_chart(fig5, use_container_width=True)
