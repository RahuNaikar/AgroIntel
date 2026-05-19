"""
Page 4 – Fertilizer Advisor
"""

import streamlit as st
import joblib, os, sys
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import dark_layout, gauge_chart

st.set_page_config(page_title="Fertilizer Advisor | AgroIntel", page_icon="🧪", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #060e06;
    color: #E8F5E9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1A0A, #061206);
    border-right: 1px solid #1E3A1E;
}
[data-testid="stSidebar"] * { color: #C8E6C9 !important; }

/* ── Main layout ── */
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* ── ALL widget labels (Temperature, Humidity, Moisture, etc.) ── */
label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stSlider"] label,
[data-testid="stSlider"] p,
[data-testid="stSlider"] span,
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] p,
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] p {
    color: #A5D6A7 !important;
    font-weight: 600 !important;
}

/* ── Slider track & ticks ── */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #66BB6A !important;
}
/* Current value bubble on slider */
[data-testid="stSlider"] [aria-valuetext] { color: #00C853 !important; }

/* ── Number input box ── */
.stNumberInput input {
    background: #0D1F0D !important;
    border: 1px solid #2E7D32 !important;
    color: #E8F5E9 !important;
    border-radius: 8px !important;
}
/* Stepper arrows */
.stNumberInput button { color: #A5D6A7 !important; background: #1B5E20 !important; }

/* ── Selectbox ── */
.stSelectbox div[data-baseweb="select"] {
    background: #0D1F0D !important;
    border: 1px solid #2E7D32 !important;
    border-radius: 8px !important;
}
.stSelectbox div[data-baseweb="select"] * { color: #E8F5E9 !important; }

/* ── Markdown headings inside cards ── */
h1, h2, h3, h4 { color: #A5D6A7 !important; }
p { color: #C8E6C9; }

/* ── Metric ── */
[data-testid="stMetricValue"] { color: #B9F6CA !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #81C784 !important; }

/* ── Cards ── */
.ag-card {
    background: linear-gradient(135deg, #0D1F0D, #0A150A);
    border: 1px solid #1E3A1E;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #A5D6A7;
    border-left: 4px solid #00C853;
    padding-left: 12px;
    margin: 1.5rem 0 1rem;
}
.fert-result {
    background: linear-gradient(135deg, #1A2A0D, #0D1A07);
    border: 2px solid #B9F6CA;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1B5E20, #2E7D32) !important;
    color: #E8F5E9 !important;
    border: 1px solid #388E3C !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 16px rgba(0,200,83,0.3) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">🧪 Fertilizer Recommendation Advisor</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#81C784;">Get precise fertilizer recommendations based on soil analysis, crop type, and environmental conditions using Decision Tree AI.</p>', unsafe_allow_html=True)

@st.cache_resource
def load_fert_model():
    base = os.path.join(os.path.dirname(__file__), "..")
    m = joblib.load(os.path.join(base, "models/fertilizer_model.pkl"))
    e = joblib.load(os.path.join(base, "models/fertilizer_encoders.pkl"))
    return m, e

try:
    fert_model, encoders = load_fert_model()
except:
    st.error("⚠️ Model not found! Run `python models/train_models.py` first.")
    st.stop()

SOIL_TYPES = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
CROP_TYPES = ["Wheat","Rice","Maize","Sugarcane","Cotton","Pulses","Groundnut","Soybean","Sunflower","Potato"]

FERTILIZER_INFO = {
    "Urea":                    {"npk":"46-0-0",  "desc":"High nitrogen source. Best for nitrogen-deficient soils and leafy growth.", "icon":"🟡"},
    "DAP":                     {"npk":"18-46-0", "desc":"Di-Ammonium Phosphate. Excellent for root development and phosphorus boost.", "icon":"🟤"},
    "MOP":                     {"npk":"0-0-60",  "desc":"Muriate of Potash. Improves disease resistance and fruit quality.", "icon":"🔴"},
    "NPK 10-26-26":            {"npk":"10-26-26","desc":"Balanced formula ideal for fruiting and grain-filling stage.", "icon":"🟢"},
    "NPK 12-32-16":            {"npk":"12-32-16","desc":"High phosphorus formula good for root vegetables and transplanting.", "icon":"🔵"},
    "SSP":                     {"npk":"0-16-0",  "desc":"Single Super Phosphate. Also provides sulfur and calcium.", "icon":"⚪"},
    "Ammonium Sulphate":       {"npk":"21-0-0",  "desc":"Nitrogen + sulfur for alkaline soils.", "icon":"🟠"},
    "Calcium Ammonium Nitrate":{"npk":"27-0-0",  "desc":"Fast-acting nitrogen. pH-neutral, good for most soils.", "icon":"🟣"},
}

col_form, col_right = st.columns([2, 3], gap="large")

with col_form:
    st.markdown('<div class="ag-card">', unsafe_allow_html=True)
    st.markdown("### 🌱 Soil & Crop Info")
    soil_type = st.selectbox("Soil Type", SOIL_TYPES)
    crop_type = st.selectbox("Crop Type", CROP_TYPES)

    st.markdown("### 🌡️ Environment")
    temperature = st.slider("Temperature (°C)", 10.0, 45.0, 28.0, 0.5)
    humidity    = st.slider("Humidity (%)",     10.0, 100.0, 60.0, 1.0)
    moisture    = st.slider("Soil Moisture (%)", 0.0, 100.0, 45.0, 1.0)

    st.markdown("### 🧪 Soil Nutrients (Current Levels)")
    nitrogen    = st.number_input("Nitrogen (N)",    0, 140, 50)
    phosphorous = st.number_input("Phosphorous (P)", 0, 145, 40)
    potassium   = st.number_input("Potassium (K)",   0, 205, 50)

    predict_btn = st.button("🧪 Get Fertilizer Recommendation", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # NPK gauge display
    g1, g2, g3 = st.columns(3)
    with g1: st.plotly_chart(gauge_chart(nitrogen,    "N Level", 140, "#76FF03"), use_container_width=True)
    with g2: st.plotly_chart(gauge_chart(phosphorous, "P Level", 145, "#40C4FF"), use_container_width=True)
    with g3: st.plotly_chart(gauge_chart(potassium,   "K Level", 205, "#FF6B35"), use_container_width=True)

    # Nutrient deficiency chart
    ideal      = {"N": 80, "P": 60, "K": 80}
    actual     = {"N": nitrogen, "P": phosphorous, "K": potassium}
    categories = list(ideal.keys())

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Ideal Level",   x=categories, y=list(ideal.values()),
                         marker_color="#1B5E20", opacity=0.6))
    fig.add_trace(go.Bar(name="Current Level", x=categories, y=list(actual.values()),
                         marker_color="#00C853"))
    fig = dark_layout(fig, "Soil Nutrient Levels vs Ideal")
    fig.update_layout(barmode="group", height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Radar for all conditions
    fig2 = go.Figure(go.Scatterpolar(
        r=[temperature/45*100, humidity, moisture, nitrogen/140*100, phosphorous/145*100, potassium/205*100],
        theta=["Temp", "Humidity", "Moisture", "Nitrogen", "Phosphorus", "Potassium"],
        fill="toself", fillcolor="rgba(185,246,202,0.1)",
        line=dict(color="#B9F6CA", width=2),
    ))
    fig2.update_layout(
        polar=dict(
            bgcolor="#0D1F0D",
            radialaxis=dict(visible=True, range=[0,100], gridcolor="#1E3A1E", color="#81C784"),
            angularaxis=dict(gridcolor="#1E3A1E", color="#81C784"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        showlegend=False, height=320, margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

if predict_btn:
    try:
        soil_enc = encoders["soil"].transform([soil_type])[0]
        crop_enc = encoders["crop"].transform([crop_type])[0]
    except:
        soil_enc = SOIL_TYPES.index(soil_type)
        crop_enc = CROP_TYPES.index(crop_type)

    X = np.array([[temperature, humidity, moisture, soil_enc, crop_enc, nitrogen, potassium, phosphorous]])
    pred_enc  = fert_model.predict(X)[0]
    fert_name = encoders["fert"].inverse_transform([pred_enc])[0]

    finfo = FERTILIZER_INFO.get(fert_name, {"npk":"N/A","desc":"Apply as recommended by local agronomist.","icon":"🌿"})

    st.markdown(f"""
    <div class="fert-result">
        <div style="font-size:3rem;">{finfo['icon']}</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:800;color:#B9F6CA;">
            {fert_name}
        </div>
        <div style="background:#0A1A0A;border-radius:10px;padding:0.6rem 1.2rem;display:inline-block;margin:0.8rem 0;">
            <span style="color:#76FF03;font-weight:700;">NPK Ratio: {finfo['npk']}</span>
        </div>
        <div style="color:#A5D6A7;font-size:0.9rem;line-height:1.6;max-width:500px;margin:0 auto;">
            {finfo['desc']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Soil Type",   soil_type)
    c2.metric("Crop",        crop_type)
    c3.metric("Recommended", fert_name)
