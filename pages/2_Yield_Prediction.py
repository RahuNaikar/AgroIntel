"""
Page 2 – Yield Prediction
"""

import streamlit as st
import joblib, os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import BRAND, dark_layout

st.set_page_config(page_title="Yield Prediction | AgroIntel", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#060e06;color:#E8F5E9;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1A0A,#061206);border-right:1px solid #1E3A1E;}
[data-testid="stSidebar"] *{color:#C8E6C9!important;}
.main .block-container{padding:1.5rem 2rem;max-width:1400px;}
.ag-card{background:linear-gradient(135deg,#0D1F0D,#0A150A);border:1px solid #1E3A1E;border-radius:16px;padding:1.5rem;margin-bottom:1rem;}
.result-card{background:linear-gradient(135deg,#0D2A0D,#091A09);border:2px solid #76FF03;border-radius:20px;padding:2rem;text-align:center;margin:1rem 0;}
.section-header{font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#A5D6A7;border-left:4px solid #00C853;padding-left:12px;margin:1.5rem 0 1rem;}
.stButton>button{background:linear-gradient(135deg,#1B5E20,#2E7D32)!important;color:#E8F5E9!important;border:1px solid #388E3C!important;border-radius:10px!important;font-weight:600!important;}
[data-testid="stMetricValue"]{color:#76FF03!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:#81C784!important;}
.stNumberInput input,.stSlider,.stSelectbox div[data-baseweb="select"]{background:#0D1F0D!important;border:1px solid #2E7D32!important;color:#E8F5E9!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">📈 Crop Yield Prediction</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#81C784;">Predict expected crop yield (tons/hectare) using Gradient Boosting Regressor trained on diverse agronomic data.</p>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "..")
    m = joblib.load(os.path.join(base, "models/yield_model.pkl"))
    e = joblib.load(os.path.join(base, "models/yield_encoders.pkl"))
    return m, e

try:
    yield_model, encoders = load_models()
except:
    st.error("⚠️ Model not found! Run `python models/train_models.py` first.")
    st.stop()

CROPS  = ["Rice","Wheat","Maize","Cotton","Sugarcane","Soybean","Sunflower","Groundnut"]
STATES = ["Punjab","Haryana","UP","MP","Maharashtra","Karnataka","AP","Rajasthan"]
SEASONS = ["Kharif","Rabi","Zaid"]

col_left, col_right = st.columns([2, 3], gap="large")

with col_left:
    st.markdown('<div class="ag-card">', unsafe_allow_html=True)
    st.markdown("### 🌾 Crop & Location")
    crop   = st.selectbox("Crop", CROPS)
    state  = st.selectbox("State", STATES)
    season = st.selectbox("Season", SEASONS)

    st.markdown("### 📐 Farm Parameters")
    area        = st.number_input("Farm Area (Hectares)", 0.5, 100.0, 5.0, 0.5)
    rainfall    = st.slider("Seasonal Rainfall (mm)", 50.0, 2000.0, 600.0, 10.0)
    temperature = st.slider("Avg Temperature (°C)", 5.0, 50.0, 28.0, 0.5)
    fertilizer  = st.slider("Fertilizer Used (kg/ha)", 0.0, 500.0, 150.0, 5.0)
    pesticide   = st.slider("Pesticide Used (kg/ha)", 0.0, 20.0, 2.5, 0.1)

    predict_btn = st.button("📊 Predict Yield", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data/yield_data.csv")):
        ydf = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data/yield_data.csv"))
        avg_by_crop = ydf.groupby("crop")["yield_tons_ha"].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=avg_by_crop["crop"], y=avg_by_crop["yield_tons_ha"],
            marker=dict(color=avg_by_crop["yield_tons_ha"],
                        colorscale=[[0,"#1B5E20"],[1,"#76FF03"]], showscale=False),
            text=avg_by_crop["yield_tons_ha"].round(2), textposition="outside",
        ))
        fig = dark_layout(fig, "Average Yield by Crop (tons/ha)")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        avg_by_state = ydf.groupby("state")["yield_tons_ha"].mean().reset_index()
        fig2 = px.bar(avg_by_state, x="state", y="yield_tons_ha",
                      color="yield_tons_ha", color_continuous_scale=["#1B5E20","#00C853","#76FF03"])
        fig2 = dark_layout(fig2, "Average Yield by State (tons/ha)")
        fig2.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

if predict_btn:
    try:
        crop_enc   = encoders["crop"].transform([crop])[0]
        state_enc  = encoders["state"].transform([state])[0]
        season_enc = encoders["season"].transform([season])[0]
    except:
        crop_enc   = CROPS.index(crop)
        state_enc  = STATES.index(state)
        season_enc = SEASONS.index(season)

    X = np.array([[crop_enc, state_enc, season_enc, area, rainfall, temperature, fertilizer, pesticide]])
    pred = float(yield_model.predict(X)[0])
    total = pred * area

    st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yield (tons/ha)", f"{pred:.2f}")
    c2.metric("Total Yield",     f"{total:.1f} tons")
    c3.metric("Farm Area",       f"{area} ha")
    c4.metric("Season",          season)

    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:3.5rem;">🌾</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:800;color:#76FF03;">
            {pred:.2f} <span style="font-size:1rem;">tons/ha</span>
        </div>
        <div style="color:#81C784;margin-top:0.5rem;">Expected Yield for {crop} in {season} season, {state}</div>
        <div style="margin-top:1rem;color:#A5D6A7;">
            📦 Total production on <b>{area} ha</b> farm: <b style="color:#00C853;">{total:.1f} tons</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sensitivity analysis
    st.markdown('<div class="section-header">📉 Fertilizer Sensitivity Analysis</div>', unsafe_allow_html=True)
    fert_range = np.linspace(50, 450, 20)
    preds = []
    for fv in fert_range:
        Xv = np.array([[crop_enc, state_enc, season_enc, area, rainfall, temperature, fv, pesticide]])
        preds.append(float(yield_model.predict(Xv)[0]))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=fert_range, y=preds, mode="lines+markers",
        line=dict(color="#76FF03", width=3),
        marker=dict(color="#00C853", size=6),
        fill="tozeroy", fillcolor="rgba(118,255,3,0.08)",
    ))
    fig3.add_vline(x=fertilizer, line_dash="dash", line_color="#FF6B35",
                   annotation_text=f"Current: {fertilizer} kg/ha",
                   annotation_font_color="#FF6B35")
    fig3 = dark_layout(fig3, "How Fertilizer Amount Affects Yield")
    fig3.update_xaxes(title="Fertilizer (kg/ha)")
    fig3.update_yaxes(title="Yield (tons/ha)")
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)
