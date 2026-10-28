from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

yield_crops = [
    "Rice", "Wheat", "Maize", "Cotton",
    "Sugarcane", "Soybean", "Sunflower", "Groundnut"
]
states = [
    "Punjab", "Haryana", "UP", "MP",
    "Maharashtra", "Karnataka", "AP", "Rajasthan"
]
seasons = ["Kharif", "Rabi", "Zaid"]

n = 3000

df = pd.DataFrame({
    "crop": np.random.choice(yield_crops, n),
    "state": np.random.choice(states, n),
    "season": np.random.choice(seasons, n),
    "area_ha": np.random.uniform(0.5, 50, n).round(2),
    "rainfall_mm": np.random.uniform(300, 1500, n).round(1),
    "temperature": np.random.uniform(18, 42, n).round(1),
    "fertilizer_kg": np.random.uniform(50, 400, n).round(1),
    "pesticide_kg": np.random.uniform(0.5, 10, n).round(2),
})

base = {
    "Rice": 4.5,
    "Wheat": 3.8,
    "Maize": 5.0,
    "Cotton": 2.2,
    "Sugarcane": 70,
    "Soybean": 2.0,
    "Sunflower": 1.8,
    "Groundnut": 2.5,
}

df["yield_tons_ha"] = df.apply(
    lambda r: round(
        base[r["crop"]]
        * (1 + 0.001 * (r["rainfall_mm"] - 600))
        * (1 + 0.002 * r["fertilizer_kg"])
        + np.random.normal(0, 0.4),
        2
    ),
    axis=1
).clip(0.5)

le_crop = LabelEncoder()
le_state = LabelEncoder()
le_season = LabelEncoder()

df["crop_enc"] = le_crop.fit_transform(df["crop"])
df["state_enc"] = le_state.fit_transform(df["state"])
df["season_enc"] = le_season.fit_transform(df["season"])

features = [
    "crop_enc", "state_enc", "season_enc", "area_ha",
    "rainfall_mm", "temperature", "fertilizer_kg", "pesticide_kg"
]

X = df[features]
y = df["yield_tons_ha"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, MODEL_DIR / "yield_model.pkl")
joblib.dump(
    {
        "crop": le_crop,
        "state": le_state,
        "season": le_season
    },
    MODEL_DIR / "yield_encoders.pkl"
)

df.to_csv(DATA_DIR / "yield_data.csv", index=False)

print(f"Yield model trained. R² score: {r2_score(y_test, model.predict(X_test)):.4f}")
print("Yield model and encoders saved successfully.")
