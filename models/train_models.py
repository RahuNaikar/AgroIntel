"""
AgroIntel - Model Training Script
Generates synthetic agricultural datasets and trains all ML models.
Run this once before launching the Streamlit app.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Create directories ──────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("=" * 60)
print("  AgroIntel - Model Training Pipeline")
print("=" * 60)

np.random.seed(42)
N_SAMPLES = 2200

# ═══════════════════════════════════════════════════════════════
# 1. CROP RECOMMENDATION DATASET & MODEL
# ═══════════════════════════════════════════════════════════════
print("\n[1/4] Generating Crop Recommendation data & training model...")

CROPS = [
    "Rice", "Wheat", "Maize", "Chickpea", "KidneyBeans",
    "PigeonPeas", "MothBeans", "MungBean", "Blackgram", "Lentil",
    "Pomegranate", "Banana", "Mango", "Grapes", "Watermelon",
    "Muskmelon", "Apple", "Orange", "Papaya", "Coconut",
    "Cotton", "Jute", "Coffee"
]

CROP_PROFILES = {
    "Rice":        dict(N=(60,100), P=(40,60),  K=(30,50),  T=(20,35), H=(70,90), ph=(5.5,7.0), R=(150,300)),
    "Wheat":       dict(N=(80,120), P=(30,60),  K=(20,50),  T=(10,25), H=(40,60), ph=(6.0,7.5), R=(50,120)),
    "Maize":       dict(N=(80,120), P=(40,70),  K=(20,50),  T=(18,35), H=(50,70), ph=(5.8,7.0), R=(50,100)),
    "Chickpea":    dict(N=(30,60),  P=(50,80),  K=(20,40),  T=(15,30), H=(40,60), ph=(5.5,7.0), R=(40,90)),
    "KidneyBeans": dict(N=(10,40),  P=(50,80),  K=(20,50),  T=(18,30), H=(50,70), ph=(6.0,7.5), R=(50,100)),
    "PigeonPeas":  dict(N=(10,40),  P=(50,80),  K=(20,50),  T=(20,35), H=(50,70), ph=(6.0,7.5), R=(60,120)),
    "MothBeans":   dict(N=(10,30),  P=(30,60),  K=(20,40),  T=(25,40), H=(20,40), ph=(6.0,7.5), R=(20,60)),
    "MungBean":    dict(N=(20,40),  P=(40,60),  K=(20,40),  T=(25,35), H=(60,80), ph=(6.0,7.5), R=(50,100)),
    "Blackgram":   dict(N=(20,40),  P=(40,60),  K=(20,40),  T=(25,35), H=(60,80), ph=(6.0,7.5), R=(60,110)),
    "Lentil":      dict(N=(10,30),  P=(50,80),  K=(20,40),  T=(15,30), H=(40,60), ph=(6.0,7.5), R=(40,80)),
    "Pomegranate": dict(N=(10,30),  P=(10,30),  K=(30,60),  T=(18,38), H=(40,65), ph=(5.5,7.0), R=(50,100)),
    "Banana":      dict(N=(80,120), P=(60,90),  K=(80,120), T=(25,40), H=(70,90), ph=(5.0,6.5), R=(100,200)),
    "Mango":       dict(N=(10,30),  P=(10,30),  K=(20,50),  T=(24,42), H=(50,70), ph=(5.5,7.5), R=(60,120)),
    "Grapes":      dict(N=(10,30),  P=(10,30),  K=(20,50),  T=(15,35), H=(60,80), ph=(5.5,7.5), R=(50,100)),
    "Watermelon":  dict(N=(80,120), P=(60,90),  K=(50,80),  T=(24,40), H=(65,80), ph=(6.0,7.0), R=(40,100)),
    "Muskmelon":   dict(N=(80,120), P=(60,90),  K=(50,80),  T=(28,42), H=(60,75), ph=(6.0,7.0), R=(20,60)),
    "Apple":       dict(N=(10,20),  P=(100,150),K=(100,150),T=(5,20),  H=(50,70), ph=(5.5,7.0), R=(100,200)),
    "Orange":      dict(N=(10,20),  P=(10,30),  K=(10,30),  T=(15,35), H=(60,80), ph=(5.5,7.5), R=(90,150)),
    "Papaya":      dict(N=(40,70),  P=(30,60),  K=(40,70),  T=(25,40), H=(70,90), ph=(5.5,7.0), R=(100,200)),
    "Coconut":     dict(N=(10,30),  P=(10,30),  K=(30,60),  T=(25,40), H=(70,90), ph=(5.0,7.0), R=(100,200)),
    "Cotton":      dict(N=(80,120), P=(30,60),  K=(20,40),  T=(22,38), H=(50,70), ph=(5.8,7.0), R=(60,120)),
    "Jute":        dict(N=(60,100), P=(30,60),  K=(20,40),  T=(24,38), H=(70,90), ph=(5.5,7.0), R=(150,250)),
    "Coffee":      dict(N=(80,120), P=(40,70),  K=(30,60),  T=(15,28), H=(70,90), ph=(5.0,6.5), R=(100,200)),
}

rows = []
for crop, p in CROP_PROFILES.items():
    n = N_SAMPLES // len(CROPS)
    rows.append(pd.DataFrame({
        "N":           np.random.randint(*p["N"], n),
        "P":           np.random.randint(*p["P"], n),
        "K":           np.random.randint(*p["K"], n),
        "temperature": np.round(np.random.uniform(*p["T"], n), 1),
        "humidity":    np.round(np.random.uniform(*p["H"], n), 1),
        "ph":          np.round(np.random.uniform(*p["ph"], n), 2),
        "rainfall":    np.round(np.random.uniform(*p["R"], n), 1),
        "label":       crop,
    }))

crop_df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
crop_df.to_csv("data/crop_data.csv", index=False)

X = crop_df.drop("label", axis=1)
y = crop_df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
acc = accuracy_score(y_te, rf.predict(X_te))
joblib.dump(rf, "models/crop_model.pkl")
print(f"   [OK] Crop model saved  |  Accuracy: {acc*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# 2. YIELD PREDICTION DATASET & MODEL
# ═══════════════════════════════════════════════════════════════
print("\n[2/4] Generating Yield Prediction data & training model...")

yield_crops  = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean", "Sunflower", "Groundnut"]
states       = ["Punjab", "Haryana", "UP", "MP", "Maharashtra", "Karnataka", "AP", "Rajasthan"]
seasons      = ["Kharif", "Rabi", "Zaid"]

ny = 3000
yield_df = pd.DataFrame({
    "crop":        np.random.choice(yield_crops, ny),
    "state":       np.random.choice(states, ny),
    "season":      np.random.choice(seasons, ny),
    "area_ha":     np.random.uniform(0.5, 50, ny).round(2),
    "rainfall_mm": np.random.uniform(300, 1500, ny).round(1),
    "temperature": np.random.uniform(18, 42, ny).round(1),
    "fertilizer_kg": np.random.uniform(50, 400, ny).round(1),
    "pesticide_kg":  np.random.uniform(0.5, 10, ny).round(2),
})
# Synthetic yield (tons/ha) with some realism
base = {"Rice":4.5,"Wheat":3.8,"Maize":5.0,"Cotton":2.2,"Sugarcane":70,"Soybean":2.0,"Sunflower":1.8,"Groundnut":2.5}
yield_df["yield_tons_ha"] = yield_df.apply(
    lambda r: round(
        base[r["crop"]]
        * (1 + 0.001*(r["rainfall_mm"]-600))
        * (1 + 0.002*r["fertilizer_kg"])
        + np.random.normal(0, 0.4),
        2
    ), axis=1
).clip(0.5)

le_crop   = LabelEncoder(); yield_df["crop_enc"]   = le_crop.fit_transform(yield_df["crop"])
le_state  = LabelEncoder(); yield_df["state_enc"]  = le_state.fit_transform(yield_df["state"])
le_season = LabelEncoder(); yield_df["season_enc"] = le_season.fit_transform(yield_df["season"])
yield_df.to_csv("data/yield_data.csv", index=False)

feats = ["crop_enc","state_enc","season_enc","area_ha","rainfall_mm","temperature","fertilizer_kg","pesticide_kg"]
X2 = yield_df[feats]; y2 = yield_df["yield_tons_ha"]
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)

gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
gb.fit(X2_tr, y2_tr)
r2 = r2_score(y2_te, gb.predict(X2_te))
joblib.dump(gb, "models/yield_model.pkl")
joblib.dump({"crop": le_crop, "state": le_state, "season": le_season}, "models/yield_encoders.pkl")
print(f"   [OK] Yield model saved  |  R2 Score: {r2:.4f}")

# ═══════════════════════════════════════════════════════════════
# 3. FERTILIZER RECOMMENDATION MODEL
# ═══════════════════════════════════════════════════════════════
print("\n[3/4] Generating Fertilizer data & training model...")

FERT_CROPS = ["Wheat","Rice","Maize","Sugarcane","Cotton","Pulses","Groundnut","Soybean","Sunflower","Potato"]
FERT_TYPES = ["Urea","DAP","MOP","NPK 10-26-26","NPK 12-32-16","SSP","Ammonium Sulphate","Calcium Ammonium Nitrate"]
SOIL_TYPES = ["Sandy","Loamy","Black","Red","Clayey"]

nf = 2000
N_val = np.random.randint(0, 140, nf)
P_val = np.random.randint(0, 145, nf)
K_val = np.random.randint(0, 205, nf)
temp  = np.random.uniform(10, 45, nf)
humid = np.random.uniform(10, 100, nf)
moist = np.random.uniform(0, 100, nf)
soil  = np.random.choice(SOIL_TYPES, nf)
crop  = np.random.choice(FERT_CROPS, nf)

# Simple rule-based labelling
def fert_rule(n, p, k):
    if n < 40:
        return "Urea"
    elif p < 30:
        return "DAP"
    elif k < 40:
        return "MOP"
    elif n < 80 and p < 80:
        return "NPK 12-32-16"
    else:
        return np.random.choice(["NPK 10-26-26", "Ammonium Sulphate", "SSP"])

fert_label = [fert_rule(n, p, k) for n, p, k in zip(N_val, P_val, K_val)]

le_soil_f = LabelEncoder()
le_crop_f = LabelEncoder()
le_fert   = LabelEncoder()

fert_df = pd.DataFrame({
    "temperature": temp, "humidity": humid, "moisture": moist,
    "soil_type": soil, "crop_type": crop,
    "nitrogen": N_val, "potassium": K_val, "phosphorous": P_val,
    "fertilizer": fert_label
})
fert_df["soil_enc"] = le_soil_f.fit_transform(fert_df["soil_type"])
fert_df["crop_enc"] = le_crop_f.fit_transform(fert_df["crop_type"])
fert_df["fert_enc"] = le_fert.fit_transform(fert_df["fertilizer"])

Xf = fert_df[["temperature","humidity","moisture","soil_enc","crop_enc","nitrogen","potassium","phosphorous"]]
yf = fert_df["fert_enc"]
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xf, yf, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(Xf_tr, yf_tr)
acc_f = accuracy_score(yf_te, dt.predict(Xf_te))
joblib.dump(dt, "models/fertilizer_model.pkl")
joblib.dump({"soil": le_soil_f, "crop": le_crop_f, "fert": le_fert}, "models/fertilizer_encoders.pkl")
print(f"   [OK] Fertilizer model saved  |  Accuracy: {acc_f*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# 4. MARKET PRICE DATA (for visualization / trend)
# ═══════════════════════════════════════════════════════════════
print("\n[4/4] Generating Market Price time-series data...")

market_crops = ["Rice","Wheat","Maize","Cotton","Onion","Potato","Tomato","Soybean"]
dates = pd.date_range("2022-01-01", "2024-12-31", freq="W")
mkt_rows = []
base_prices = {"Rice":2200,"Wheat":1975,"Maize":1800,"Cotton":6200,"Onion":1500,"Potato":1200,"Tomato":2000,"Soybean":4100}
for crop in market_crops:
    bp = base_prices[crop]
    trend = np.cumsum(np.random.randn(len(dates)) * bp * 0.015)
    seasonal = bp * 0.12 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
    noise = np.random.randn(len(dates)) * bp * 0.04
    prices = bp + trend + seasonal + noise
    prices = np.clip(prices, bp*0.4, bp*2.5)
    mkt_rows.append(pd.DataFrame({"date": dates, "crop": crop, "price_per_quintal": prices.round(0)}))

mkt_df = pd.concat(mkt_rows, ignore_index=True)
mkt_df.to_csv("data/market_data.csv", index=False)
print("   [OK] Market data saved")

# ═══════════════════════════════════════════════════════════════
# IRRIGATION ADVISOR MODEL
# ═══════════════════════════════════════════════════════════════
print("\n[Bonus] Training Irrigation Advisor model...")

ni = 2000
irr_df = pd.DataFrame({
    "temperature":   np.random.uniform(15, 45, ni),
    "humidity":      np.random.uniform(10, 100, ni),
    "soil_moisture": np.random.uniform(0, 100, ni),
    "rainfall_mm":   np.random.uniform(0, 50, ni),
    "crop_stage":    np.random.choice(["Seedling","Vegetative","Flowering","Fruiting","Maturity"], ni),
})
# Rule-based label: water_liters_per_day
stage_mult = {"Seedling":0.6,"Vegetative":1.0,"Flowering":1.3,"Fruiting":1.2,"Maturity":0.7}
irr_df["water_need_L"] = irr_df.apply(
    lambda r: round(
        max(0, (35 - r["soil_moisture"]) * 0.8 * stage_mult[r["crop_stage"]]
            + max(0, r["temperature"] - 28) * 0.5
            - r["rainfall_mm"] * 0.3
            + np.random.normal(0, 2)), 1
    ), axis=1
)

le_stage = LabelEncoder()
irr_df["stage_enc"] = le_stage.fit_transform(irr_df["crop_stage"])
Xi = irr_df[["temperature","humidity","soil_moisture","rainfall_mm","stage_enc"]]
yi = irr_df["water_need_L"]
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.2, random_state=42)

rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr.fit(Xi_tr, yi_tr)
r2i = r2_score(yi_te, rfr.predict(Xi_te))
joblib.dump(rfr, "models/irrigation_model.pkl")
joblib.dump(le_stage, "models/irrigation_stage_enc.pkl")
print(f"   [OK] Irrigation model saved  |  R2 Score: {r2i:.4f}")

print("\n" + "=" * 60)
print("  [DONE] All models trained & data generated successfully!")
print("  >> Run:  streamlit run app.py")
print("=" * 60)
