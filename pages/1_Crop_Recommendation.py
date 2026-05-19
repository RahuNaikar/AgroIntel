"""
Page 1 – Crop Recommendation
"""

import streamlit as st
import joblib, os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import BRAND, dark_layout, gauge_chart, soil_health_score, CROP_EMOJIS

st.set_page_config(page_title="Crop Recommendation | AgroIntel", page_icon="🌾", layout="wide")

# ── Shared CSS (import from main via st.markdown) ─────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#060e06;color:#E8F5E9;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1A0A,#061206);border-right:1px solid #1E3A1E;}
[data-testid="stSidebar"] *{color:#C8E6C9!important;}
.main .block-container{padding:1.5rem 2rem;max-width:1400px;}
.ag-card{background:linear-gradient(135deg,#0D1F0D,#0A150A);border:1px solid #1E3A1E;border-radius:16px;padding:1.5rem;margin-bottom:1rem;}
.result-card{background:linear-gradient(135deg,#0D2A0D,#091A09);border:2px solid #00C853;border-radius:20px;padding:2rem;text-align:center;margin:1rem 0;}
.section-header{font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#A5D6A7;border-left:4px solid #00C853;padding-left:12px;margin:1.5rem 0 1rem;}
.stButton>button{background:linear-gradient(135deg,#1B5E20,#2E7D32)!important;color:#E8F5E9!important;border:1px solid #388E3C!important;border-radius:10px!important;font-weight:600!important;}
.stSlider>div>div{color:#00C853!important;}
[data-testid="stMetricValue"]{color:#00C853!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:#81C784!important;}
.stNumberInput input,.stSelectbox div[data-baseweb="select"]{background:#0D1F0D!important;border:1px solid #2E7D32!important;color:#E8F5E9!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">🌾 Crop Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown("""
<p style="color:#81C784; font-size:0.95rem;">
Enter your soil parameters and environmental conditions to get AI-powered crop recommendations.
The model uses <b>Random Forest</b> trained on 2,200+ agricultural data points.
</p>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_crop_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "crop_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_crop_model()

if model is None:
    st.error("⚠️ Model not found! Run `python models/train_models.py` first.", icon="🚫")
    st.stop()

# ── Input Form ────────────────────────────────────────────────
col_form, col_right = st.columns([2, 3], gap="large")

with col_form:
    st.markdown('<div class="ag-card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Soil Nutrients")

    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.number_input("Nitrogen (N)", 0, 200, 80, help="kg/ha – Primary macronutrient")
    with c2:
        P = st.number_input("Phosphorus (P)", 0, 200, 40, help="kg/ha – Root development")
    with c3:
        K = st.number_input("Potassium (K)", 0, 250, 40, help="kg/ha – Disease resistance")

    st.markdown("### 🌡️ Environmental Conditions")
    temperature = st.slider("Temperature (°C)", 5.0, 50.0, 25.0, 0.5)
    humidity    = st.slider("Humidity (%)",      10.0, 100.0, 65.0, 0.5)
    ph          = st.slider("Soil pH",           3.0,  9.0,  6.5,  0.1)
    rainfall    = st.slider("Rainfall (mm)",     20.0, 400.0, 120.0, 5.0)

    predict_btn = st.button("🚀 Get Crop Recommendation", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Soil Health Gauges ────────────────────────────────────────
with col_right:
    score, interp = soil_health_score(N, P, K, ph)

    st.markdown(f"""
    <div class="ag-card">
        <div style="font-size:1rem;font-weight:700;color:#A5D6A7;margin-bottom:1rem;">🌱 Soil Health Score</div>
        <div style="text-align:center;">
            <div style="font-size:4rem;font-weight:800;color:#00C853;">{score}</div>
            <div style="font-size:1.2rem;color:#81C784;">{interp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(gauge_chart(N, "Nitrogen", 200), use_container_width=True)
        st.plotly_chart(gauge_chart(K, "Potassium", 250), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_chart(P, "Phosphorus", 200, "#76FF03"), use_container_width=True)
        st.plotly_chart(gauge_chart(humidity, "Humidity", 100, "#40C4FF"), use_container_width=True)

# ── Prediction ────────────────────────────────────────────────
if predict_btn:
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]

    # Top-5 probabilities
    proba = model.predict_proba(features)[0]
    classes = model.classes_
    top5_idx = proba.argsort()[-5:][::-1]
    top5 = [(classes[i], proba[i]*100) for i in top5_idx]

    emoji = CROP_EMOJIS.get(prediction, "🌱")

    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:5rem;">{emoji}</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:800;color:#00C853;">
            {prediction}
        </div>
        <div style="color:#81C784;font-size:1rem;margin-top:0.5rem;">
            Recommended crop based on your soil & climate conditions
        </div>
        <div style="margin-top:1rem;background:#0A1A0A;border-radius:12px;padding:0.8rem;">
            <span style="color:#76FF03;font-weight:700;font-size:1.1rem;">
                {top5[0][1]:.1f}% Confidence
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top-5 chart
    st.markdown('<div class="section-header">📊 Top 5 Crop Recommendations</div>', unsafe_allow_html=True)
    names = [f"{CROP_EMOJIS.get(c,'🌱')} {c}" for c, _ in top5]
    vals  = [v for _, v in top5]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=vals, colorscale=[[0,"#1B5E20"],[1,"#00C853"]], showscale=False),
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
    ))
    fig = dark_layout(fig, "Prediction Confidence (%)")
    fig.update_layout(height=280, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # Input radar chart
    st.markdown('<div class="section-header">🕸️ Parameter Radar</div>', unsafe_allow_html=True)
    categories = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH×10", "Rainfall/10"]
    vals_radar  = [N/200, P/200, K/250, temperature/50, humidity/100, ph/9, rainfall/400]
    fig2 = go.Figure(go.Scatterpolar(
        r=[v*100 for v in vals_radar] + [vals_radar[0]*100],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(0,200,83,0.15)",
        line=dict(color="#00C853", width=2),
    ))
    fig2.update_layout(
        polar=dict(
            bgcolor="#0D1F0D",
            radialaxis=dict(visible=True, range=[0,100], gridcolor="#1E3A1E", color="#81C784"),
            angularaxis=dict(gridcolor="#1E3A1E", color="#81C784"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        height=380,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)
