# 🌿 AgroIntel – AI-Powered Smart Agriculture Platform

> **AI-driven data science platform for farmers**: crop recommendations, yield prediction, disease detection, fertilizer advice, weather forecasting, market insights, and smart irrigation – all in one interactive dashboard.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train ML Models (run once)
```bash
cd "d:\AgroIntel Projet"
python models/train_models.py
```
This generates synthetic datasets in `data/` and trains 4 ML models saved in `models/`.

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

### 4. (Optional) Start Flask API
```bash
python backend/flask_api.py
```

---

## 🧩 Features

| Module | Description | Technology |
|--------|-------------|------------|
| 🌾 **Crop Recommendation** | AI crop suggestions based on N, P, K, climate | Random Forest |
| 📈 **Yield Prediction** | Forecast yield from farm & weather data | Gradient Boosting |
| 🔬 **Disease Detection** | Upload leaf images for disease diagnosis | Image Analysis |
| 🧪 **Fertilizer Advisor** | Optimal fertilizer based on soil composition | Decision Tree |
| 🌦️ **Weather Forecast** | 7-day forecast with farm alerts | Open-Meteo API |
| 💰 **Market Prices** | Price trends and best sell timing | Time Series |
| 💧 **Irrigation Advisor** | Smart water need calculation | Random Forest |
| 📊 **Analytics Dashboard** | KPIs, heatmaps, regional analytics | Plotly |

---

## 📂 Project Structure

```
AgroIntel Projet/
├── app.py                    ← Main Streamlit dashboard (home page)
├── requirements.txt
├── backend/
│   └── flask_api.py          ← REST API (4 prediction endpoints)
├── models/
│   ├── train_models.py       ← Training script (run first!)
│   ├── crop_model.pkl
│   ├── yield_model.pkl
│   ├── fertilizer_model.pkl
│   └── irrigation_model.pkl
├── data/
│   ├── crop_data.csv         ← 2,200+ crop samples (23 crops)
│   ├── yield_data.csv        ← 3,000 yield records
│   └── market_data.csv       ← 3-year weekly price data (8 crops)
├── pages/
│   ├── 1_Crop_Recommendation.py
│   ├── 2_Yield_Prediction.py
│   ├── 3_Disease_Detection.py
│   ├── 4_Fertilizer_Advisor.py
│   ├── 5_Weather_Forecast.py
│   ├── 6_Market_Prices.py
│   ├── 7_Irrigation_Advisor.py
│   └── 8_Analytics_Dashboard.py
└── utils/
    └── helpers.py            ← Shared utilities & chart helpers
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict/crop` | Crop recommendation |
| POST | `/predict/yield` | Yield prediction |
| POST | `/predict/fertilizer` | Fertilizer recommendation |
| POST | `/predict/irrigation` | Water need prediction |

### Example API Call
```python
import requests
response = requests.post("http://localhost:5000/predict/crop", json={
    "N": 90, "P": 42, "K": 43,
    "temperature": 20.9, "humidity": 82.0,
    "ph": 6.5, "rainfall": 202.9
})
print(response.json())
# {"recommended_crop": "Rice", "top_3": [...], "status": "success"}
```

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit + Plotly
- **Backend**: Flask REST API
- **ML Models**: Scikit-learn (Random Forest, Gradient Boosting, Decision Tree)
- **Data**: Pandas, NumPy (synthetic agricultural datasets)
- **Weather**: Open-Meteo free API (no API key needed)
- **Visualization**: Plotly Express + Graph Objects

---

*Built with ❤️ for smart, sustainable agriculture.*
