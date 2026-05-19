"""
Page 5 – Weather Forecast (Open-Meteo free API – no key required)
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import dark_layout, get_weather_icon

st.set_page_config(page_title="Weather Forecast | AgroIntel", page_icon="🌦️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#060e06;color:#E8F5E9;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1A0A,#061206);border-right:1px solid #1E3A1E;}
[data-testid="stSidebar"] *{color:#C8E6C9!important;}
.main .block-container{padding:1.5rem 2rem;max-width:1400px;}
.ag-card{background:linear-gradient(135deg,#0D1F0D,#0A150A);border:1px solid #1E3A1E;border-radius:16px;padding:1.5rem;margin-bottom:1rem;}
.section-header{font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#A5D6A7;border-left:4px solid #00C853;padding-left:12px;margin:1.5rem 0 1rem;}
.weather-card{background:linear-gradient(135deg,#0D1A2A,#091014);border:1px solid #1E3A5E;border-radius:16px;padding:1rem;text-align:center;margin:0.3rem;}
.day-card{background:linear-gradient(135deg,#0D1F1A,#080F0D);border:1px solid #1E3A2E;border-radius:12px;padding:1rem;text-align:center;}
.stButton>button{background:linear-gradient(135deg,#0D47A1,#1565C0)!important;color:#E8F5E9!important;border:1px solid #1976D2!important;border-radius:10px!important;font-weight:600!important;}
[data-testid="stMetricValue"]{color:#80D8FF!important;font-weight:700!important;}
.stNumberInput input,.stSelectbox div[data-baseweb="select"]{background:#0D1F0D!important;border:1px solid #2E7D32!important;color:#E8F5E9!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">🌦️ Weather Forecast & Farm Alerts</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#81C784;">Real-time 7-day weather forecast powered by <b>Open-Meteo API</b> (free, no API key required). Location-specific farm alerts included.</p>', unsafe_allow_html=True)

# ── Location selector ─────────────────────────────────────────
LOCATIONS = {
    "New Delhi":    (28.6139, 77.2090),
    "Mumbai":       (19.0760, 72.8777),
    "Bangalore":    (12.9716, 77.5946),
    "Hyderabad":    (17.3850, 78.4867),
    "Chennai":      (13.0827, 80.2707),
    "Kolkata":      (22.5726, 88.3639),
    "Ahmedabad":    (23.0225, 72.5714),
    "Pune":         (18.5204, 73.8567),
    "Jaipur":       (26.9124, 75.7873),
    "Lucknow":      (26.8467, 80.9462),
    "Bhopal":       (23.2599, 77.4126),
    "Chandigarh":   (30.7333, 76.7794),
    "Amritsar":     (31.6340, 74.8723),
    "Nagpur":       (21.1458, 79.0882),
    "Custom":       None,
}

col_loc, col_options = st.columns([2, 2])
with col_loc:
    city = st.selectbox("📍 Select City", list(LOCATIONS.keys()))
with col_options:
    if city == "Custom":
        lat = st.number_input("Latitude",  -90.0, 90.0, 20.0)
        lon = st.number_input("Longitude", -180.0, 180.0, 77.0)
    else:
        lat, lon = LOCATIONS[city]
        st.metric("Latitude",  f"{lat}°N")
        st.metric("Longitude", f"{lon}°E")

fetch_btn = st.button("🌐 Fetch Weather Data", use_container_width=False)

@st.cache_data(ttl=1800)
def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"windspeed_10m_max,weathercode,uv_index_max,et0_fao_evapotranspiration"
        f"&hourly=temperature_2m,relativehumidity_2m,precipitation_probability,windspeed_10m"
        f"&timezone=auto&forecast_days=7"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

if fetch_btn or True:  # Auto-load for demo
    data = fetch_weather(lat, lon)

    if data:
        daily   = data["daily"]
        hourly  = data["hourly"]
        dates   = pd.to_datetime(daily["time"])
        t_max   = daily["temperature_2m_max"]
        t_min   = daily["temperature_2m_min"]
        rain    = daily["precipitation_sum"]
        wind    = daily["windspeed_10m_max"]
        codes   = daily["weathercode"]
        uv      = daily["uv_index_max"]
        et0     = daily["et0_fao_evapotranspiration"]

        # Today's summary
        st.markdown('<div class="section-header">🌤️ Today\'s Forecast</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🌡️ Max Temp",   f"{t_max[0]}°C")
        c2.metric("🌡️ Min Temp",   f"{t_min[0]}°C")
        c3.metric("🌧️ Rainfall",   f"{rain[0]} mm")
        c4.metric("💨 Wind",       f"{wind[0]} km/h")
        c5.metric("☀️ UV Index",   f"{uv[0]}")

        # 7-day cards
        st.markdown('<div class="section-header">📅 7-Day Forecast</div>', unsafe_allow_html=True)
        cols = st.columns(7)
        for i in range(7):
            with cols[i]:
                icon = get_weather_icon(codes[i])
                day_name = dates[i].strftime("%a\n%d %b")
                st.markdown(f"""
                <div class="day-card">
                    <div style="font-size:0.75rem;color:#81C784;margin-bottom:0.3rem;">{dates[i].strftime('%a')}<br>{dates[i].strftime('%d %b')}</div>
                    <div style="font-size:2rem;">{icon}</div>
                    <div style="color:#FF6B35;font-weight:700;font-size:0.9rem;">{t_max[i]}°</div>
                    <div style="color:#80D8FF;font-size:0.8rem;">{t_min[i]}°</div>
                    <div style="color:#76FF03;font-size:0.75rem;margin-top:0.3rem;">{rain[i]}mm</div>
                </div>
                """, unsafe_allow_html=True)

        # Charts
        st.markdown('<div class="section-header">📊 Weather Trends</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "🌧️ Rainfall & ET₀", "💨 Wind & UV"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=t_max, name="Max Temp",
                                     line=dict(color="#FF6B35", width=3), fill="tonexty"))
            fig.add_trace(go.Scatter(x=dates, y=t_min, name="Min Temp",
                                     line=dict(color="#80D8FF", width=3), fill="tozeroy",
                                     fillcolor="rgba(128,216,255,0.08)"))
            fig = dark_layout(fig, "Daily Temperature Range (°C)")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=dates, y=rain, name="Rainfall (mm)",
                                   marker_color="#76FF03", opacity=0.8))
            fig2.add_trace(go.Scatter(x=dates, y=et0, name="ET₀ (mm)",
                                       line=dict(color="#FFA000", width=2, dash="dot"),
                                       mode="lines+markers"))
            fig2 = dark_layout(fig2, "Rainfall vs Evapotranspiration (mm)")
            fig2.update_layout(height=320)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=dates, y=wind, name="Wind Speed (km/h)",
                                       line=dict(color="#B39DDB", width=3), fill="tozeroy",
                                       fillcolor="rgba(179,157,219,0.1)"))
            fig3.add_trace(go.Scatter(x=dates, y=uv, name="UV Index", yaxis="y2",
                                       line=dict(color="#FF8F00", width=2, dash="dash")))
            fig3.update_layout(yaxis2=dict(overlaying="y", side="right", color="#FF8F00"))
            fig3 = dark_layout(fig3, "Wind Speed & UV Index")
            fig3.update_layout(height=320)
            st.plotly_chart(fig3, use_container_width=True)

        # Farm Alerts
        st.markdown('<div class="section-header">🚨 Farm Advisory Alerts</div>', unsafe_allow_html=True)
        alerts = []
        for i in range(7):
            d = dates[i].strftime("%a %d %b")
            if rain[i] > 30:
                alerts.append(("⛈️", d, "Heavy rainfall expected", f"{rain[i]:.1f}mm – Delay irrigation. Check drainage.", "red"))
            elif rain[i] > 10:
                alerts.append(("🌧️", d, "Moderate rain", f"{rain[i]:.1f}mm – Reduce irrigation accordingly.", "orange"))
            if wind[i] > 40:
                alerts.append(("💨", d, "High winds", f"{wind[i]:.0f} km/h – Protect young plants. Avoid spraying.", "orange"))
            if uv[i] > 8:
                alerts.append(("☀️", d, "High UV Index", f"UV {uv[i]:.0f} – Avoid field work 11am–3pm.", "yellow"))
            if t_max[i] > 40:
                alerts.append(("🌡️", d, "Extreme Heat", f"{t_max[i]}°C – Irrigate early morning. Shade sensitive crops.", "red"))

        if alerts:
            for icon, day, title, msg, sev in alerts:
                colors = {"red":"#D32F2F","orange":"#FFA000","yellow":"#F9A825"}
                c = colors.get(sev, "#388E3C")
                st.markdown(f"""
                <div style="background:#0D1F0D;border-left:4px solid {c};border-radius:0 10px 10px 0;padding:0.8rem 1rem;margin-bottom:0.5rem;display:flex;align-items:center;gap:1rem;">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div>
                        <div style="color:{c};font-weight:700;font-size:0.85rem;">{day} – {title}</div>
                        <div style="color:#81C784;font-size:0.8rem;">{msg}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No critical weather alerts for the next 7 days. Favorable farming conditions ahead!")
    else:
        st.error("❌ Unable to fetch weather data. Please check your internet connection.")
