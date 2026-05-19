"""
AgroIntel – Shared Utilities
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Brand colors ──────────────────────────────────────────────
BRAND = {
    "primary":   "#00C853",
    "secondary": "#1B5E20",
    "accent":    "#76FF03",
    "dark":      "#0A0F0A",
    "card_bg":   "#0D1F0D",
    "border":    "#1E3A1E",
}

CROP_EMOJIS = {
    "Rice":"🌾","Wheat":"🌾","Maize":"🌽","Chickpea":"🫘","KidneyBeans":"🫘",
    "PigeonPeas":"🫘","MothBeans":"🫘","MungBean":"🫘","Blackgram":"🫘",
    "Lentil":"🫘","Pomegranate":"🍎","Banana":"🍌","Mango":"🥭","Grapes":"🍇",
    "Watermelon":"🍉","Muskmelon":"🍈","Apple":"🍎","Orange":"🍊","Papaya":"🍈",
    "Coconut":"🥥","Cotton":"🌿","Jute":"🌿","Coffee":"☕",
    "Sugarcane":"🎍","Soybean":"🫘","Sunflower":"🌻","Groundnut":"🥜",
}

DISEASE_INFO = {
    "Healthy": {
        "description": "No disease detected. Your plant appears healthy!",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None",
        "color": "#00C853"
    },
    "Bacterial Spot": {
        "description": "Dark, water-soaked spots on leaves/fruits. Caused by Xanthomonas bacteria.",
        "treatment": "Apply copper-based bactericides. Remove infected plant parts. Improve air circulation.",
        "severity": "Moderate",
        "color": "#FFA000"
    },
    "Early Blight": {
        "description": "Brown spots with concentric rings, yellowing leaves. Caused by Alternaria solani.",
        "treatment": "Apply fungicides (Mancozeb / Chlorothalonil). Ensure proper plant spacing.",
        "severity": "Moderate",
        "color": "#FFA000"
    },
    "Late Blight": {
        "description": "Dark, water-soaked lesions on leaves and stems. Rapid spread in cool, moist conditions.",
        "treatment": "Immediate fungicide application (Metalaxyl). Remove and destroy infected tissue.",
        "severity": "High",
        "color": "#D32F2F"
    },
    "Leaf Mold": {
        "description": "Pale greenish-yellow spots on upper leaf surface with olive-green mold below.",
        "treatment": "Improve ventilation. Apply fungicides. Avoid overhead irrigation.",
        "severity": "Low",
        "color": "#388E3C"
    },
    "Powdery Mildew": {
        "description": "White powdery fungal growth on leaves, stems, and flowers.",
        "treatment": "Apply sulfur-based fungicides or potassium bicarbonate. Improve air circulation.",
        "severity": "Moderate",
        "color": "#FFA000"
    },
    "Rust": {
        "description": "Orange-red pustules on leaves. Reduces photosynthesis significantly.",
        "treatment": "Apply triazole fungicides. Plant resistant varieties. Crop rotation.",
        "severity": "High",
        "color": "#D32F2F"
    },
    "Mosaic Virus": {
        "description": "Mottled yellow-green patterns on leaves. Spread by aphids.",
        "treatment": "No cure. Remove infected plants. Control aphid vectors. Use virus-free seeds.",
        "severity": "High",
        "color": "#D32F2F"
    },
}

def metric_card(label: str, value: str, delta: str = "", icon: str = "📊", color: str = "#00C853"):
    """Render a styled metric card using native Streamlit metric."""
    st.metric(label=f"{icon} {label}", value=value, delta=delta if delta else None)

def get_weather_icon(code: int) -> str:
    """Map WMO weather codes to emoji."""
    if code == 0:   return "☀️"
    if code in (1, 2): return "🌤️"
    if code == 3:   return "☁️"
    if code in range(45, 50): return "🌫️"
    if code in range(51, 68): return "🌧️"
    if code in range(71, 78): return "❄️"
    if code in range(80, 83): return "🌦️"
    if code in range(95, 100): return "⛈️"
    return "🌈"

def soil_health_score(N, P, K, ph) -> tuple[float, str]:
    """Return a 0-100 soil health score and interpretation."""
    n_score = min(N / 140 * 100, 100)
    p_score = min(P / 145 * 100, 100)
    k_score = min(K / 205 * 100, 100)
    ph_score = max(0, 100 - abs(ph - 6.5) * 25)
    score = (n_score + p_score + k_score + ph_score) / 4

    if score >= 75:   interp = "🟢 Excellent"
    elif score >= 50: interp = "🟡 Good"
    elif score >= 30: interp = "🟠 Fair"
    else:             interp = "🔴 Poor"
    return round(score, 1), interp

def gauge_chart(value: float, title: str, max_val: float = 100, color: str = "#00C853") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "white", "size": 14}},
        gauge={
            "axis":  {"range": [0, max_val], "tickcolor": "white"},
            "bar":   {"color": color},
            "bgcolor": "#0D1F0D",
            "bordercolor": "#1E3A1E",
            "steps": [
                {"range": [0, max_val*0.33],  "color": "#1B2E1B"},
                {"range": [max_val*0.33, max_val*0.66], "color": "#1F3A1F"},
                {"range": [max_val*0.66, max_val],      "color": "#224422"},
            ],
        },
        number={"font": {"color": "white"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20), height=200,
    )
    return fig

def dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,31,13,0.6)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#1E3A1E", zerolinecolor="#1E3A1E"),
        yaxis=dict(gridcolor="#1E3A1E", zerolinecolor="#1E3A1E"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
