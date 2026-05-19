"""
Page 7 – Smart Irrigation Advisor
"""

import streamlit as st
import joblib, os, sys
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import dark_layout, gauge_chart

st.set_page_config(page_title="Irrigation Advisor | AgroIntel", page_icon="💧", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#060e06;color:#E8F5E9;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1A0A,#061206);border-right:1px solid #1E3A1E;}
[data-testid="stSidebar"] *{color:#C8E6C9!important;}
.main .block-container{padding:1.5rem 2rem;max-width:1400px;}
.ag-card{background:linear-gradient(135deg,#0D1F0D,#0A150A);border:1px solid #1E3A1E;border-radius:16px;padding:1.5rem;margin-bottom:1rem;}
.section-header{font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#A5D6A7;border-left:4px solid #00C853;padding-left:12px;margin:1.5rem 0 1rem;}
.water-card{background:linear-gradient(135deg,#0D1A2A,#091014);border:2px solid #80D8FF;border-radius:20px;padding:2rem;text-align:center;margin:1rem 0;}
.stButton>button{background:linear-gradient(135deg,#01579B,#0277BD)!important;color:#E8F5E9!important;border:1px solid #0288D1!important;border-radius:10px!important;font-weight:600!important;}
[data-testid="stMetricValue"]{color:#80D8FF!important;font-weight:700!important;}
.stSlider>div>div{color:#80D8FF!important;}
.stNumberInput input,.stSelectbox div[data-baseweb="select"]{background:#0D1F0D!important;border:1px solid #2E7D32!important;color:#E8F5E9!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">💧 Smart Irrigation Advisor</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#81C784;">Optimize water usage with AI-powered irrigation scheduling based on soil moisture, weather conditions, and crop growth stage.</p>', unsafe_allow_html=True)

@st.cache_resource
def load_irr_model():
    base = os.path.join(os.path.dirname(__file__), "..")
    m = joblib.load(os.path.join(base, "models/irrigation_model.pkl"))
    e = joblib.load(os.path.join(base, "models/irrigation_stage_enc.pkl"))
    return m, e

try:
    irr_model, stage_enc = load_irr_model()
except:
    st.error("⚠️ Irrigation model not found! Run `python models/train_models.py` first.")
    st.stop()

STAGES = ["Seedling","Vegetative","Flowering","Fruiting","Maturity"]

STAGE_INFO = {
    "Seedling":   {"water":"Low",   "desc":"Young roots, high sensitivity. Keep soil moist but not waterlogged.", "icon":"🌱"},
    "Vegetative": {"water":"Medium","desc":"Active growth. Regular irrigation needed for leaf & stem development.", "icon":"🌿"},
    "Flowering":  {"water":"High",  "desc":"Critical stage. Water stress can severely reduce fruit set.", "icon":"🌸"},
    "Fruiting":   {"water":"High",  "desc":"Fruit filling. Consistent moisture prevents blossom-end rot.", "icon":"🍅"},
    "Maturity":   {"water":"Low",   "desc":"Reduce irrigation 2 weeks before harvest to improve quality.", "icon":"🌾"},
}

col_form, col_right = st.columns([2, 3], gap="large")

with col_form:
    st.markdown('<div class="ag-card">', unsafe_allow_html=True)
    st.markdown("### 🌱 Crop Stage")
    crop_stage = st.selectbox("Growth Stage", STAGES)
    info = STAGE_INFO[crop_stage]
    st.markdown(f"""
    <div style="background:#0A1A0A;border:1px solid #1E3A1E;border-radius:10px;padding:0.8rem;margin:0.5rem 0;">
        <div style="font-size:1.5rem;">{info['icon']}</div>
        <div style="color:#80D8FF;font-weight:600;font-size:0.85rem;">Water Need: {info['water']}</div>
        <div style="color:#81C784;font-size:0.8rem;margin-top:0.3rem;">{info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌡️ Current Conditions")
    temperature   = st.slider("Temperature (°C)",   15.0, 48.0, 30.0, 0.5)
    humidity      = st.slider("Humidity (%)",        10.0, 100.0, 55.0, 1.0)
    soil_moisture = st.slider("Soil Moisture (%)",   0.0,  100.0, 30.0, 1.0)
    rainfall      = st.slider("Recent Rainfall (mm)", 0.0, 50.0, 5.0, 0.5)

    predict_btn = st.button("💧 Calculate Water Need", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(gauge_chart(soil_moisture, "Soil Moisture %", 100, "#80D8FF"), use_container_width=True)
        st.plotly_chart(gauge_chart(temperature, "Temperature °C", 50, "#FF6B35"), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_chart(humidity, "Humidity %", 100, "#76FF03"), use_container_width=True)
        st.plotly_chart(gauge_chart(rainfall, "Rainfall mm", 50, "#40C4FF"), use_container_width=True)

if predict_btn:
    try:
        stage_e = stage_enc.transform([crop_stage])[0]
    except:
        stage_e = STAGES.index(crop_stage)

    X = np.array([[temperature, humidity, soil_moisture, rainfall, stage_e]])
    water_need = float(irr_model.predict(X)[0])
    water_need = max(0, water_need)

    # Recommendation level
    if water_need < 5:
        level, lvl_color, advice = "Low", "#00C853", "No irrigation needed today. Monitor soil moisture."
    elif water_need < 15:
        level, lvl_color, advice = "Moderate", "#FFA000", "Light irrigation recommended. Apply in early morning."
    elif water_need < 30:
        level, lvl_color, advice = "High", "#FF6B35", "Significant irrigation required. Apply in 2 sessions."
    else:
        level, lvl_color, advice = "Critical", "#D32F2F", "Urgent irrigation needed! Crop stress risk is high."

    st.markdown(f"""
    <div class="water-card">
        <div style="font-size:3rem;">💧</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:2.5rem;font-weight:800;color:#80D8FF;">
            {water_need:.1f} <span style="font-size:1rem;">L/m²/day</span>
        </div>
        <div style="margin-top:0.5rem;">
            <span style="background:{lvl_color}22;border:1px solid {lvl_color};color:{lvl_color};
                         border-radius:20px;padding:4px 16px;font-weight:700;">
                {level} Need
            </span>
        </div>
        <div style="color:#81C784;font-size:0.9rem;margin-top:1rem;">{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Water Need",   f"{water_need:.1f} L/m²")
    c2.metric("Stage",        f"{info['icon']} {crop_stage}")
    c3.metric("Soil Moisture",f"{soil_moisture}%")
    c4.metric("Net Rainfall", f"{rainfall} mm")

    # Stage comparison
    st.markdown('<div class="section-header">📊 Water Need by Growth Stage</div>', unsafe_allow_html=True)
    stage_water = []
    for s in STAGES:
        try: se = stage_enc.transform([s])[0]
        except: se = STAGES.index(s)
        Xv = np.array([[temperature, humidity, soil_moisture, rainfall, se]])
        stage_water.append(max(0, float(irr_model.predict(Xv)[0])))

    colors_bar = ["#80D8FF" if s == crop_stage else "#1E5E7E" for s in STAGES]
    fig = go.Figure(go.Bar(
        x=STAGES, y=stage_water,
        marker_color=colors_bar,
        text=[f"{v:.1f}" for v in stage_water], textposition="outside",
    ))
    fig = dark_layout(fig, "Irrigation Requirement by Crop Growth Stage (L/m²/day)")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    # Weekly schedule
    st.markdown('<div class="section-header">📅 Suggested Weekly Irrigation Schedule</div>', unsafe_allow_html=True)
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    freq = max(1, int(7 * water_need / 50))
    scheduled = sorted(np.random.choice(range(7), min(freq, 7), replace=False))
    sched_water = [round(water_need * 7 / len(scheduled), 1) if i in scheduled else 0 for i in range(7)]

    fig2 = go.Figure(go.Bar(
        x=days, y=sched_water,
        marker_color=["#80D8FF" if w > 0 else "#1E3A1E" for w in sched_water],
        text=[f"{w} L" if w>0 else "Off" for w in sched_water], textposition="outside",
    ))
    fig2 = dark_layout(fig2, "Recommended Irrigation Schedule (L/m²)")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)
