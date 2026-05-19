"""
AgroIntel – Flask REST API Backend
Exposes prediction endpoints for integration with other services.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(__file__)

def load_model(filename):
    path = os.path.join(BASE, "..", "models", filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "name": "AgroIntel API",
        "version": "1.0",
        "status": "running",
        "endpoints": [
            "/predict/crop",
            "/predict/yield",
            "/predict/fertilizer",
            "/predict/irrigation",
            "/health",
        ]
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "timestamp": str(__import__("datetime").datetime.now())})

@app.route("/predict/crop", methods=["POST"])
def predict_crop():
    try:
        model = load_model("crop_model.pkl")
        if not model:
            return jsonify({"error": "Model not loaded. Run train_models.py first."}), 500

        d = request.json
        X = np.array([[d["N"], d["P"], d["K"], d["temperature"], d["humidity"], d["ph"], d["rainfall"]]])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        top3_idx = proba.argsort()[-3:][::-1]
        top3 = [{"crop": model.classes_[i], "confidence": round(float(proba[i])*100, 2)} for i in top3_idx]

        return jsonify({
            "recommended_crop": prediction,
            "top_3": top3,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/yield", methods=["POST"])
def predict_yield():
    try:
        model    = load_model("yield_model.pkl")
        encoders = load_model("yield_encoders.pkl")
        if not model:
            return jsonify({"error": "Model not loaded."}), 500

        d = request.json
        crops   = ["Rice","Wheat","Maize","Cotton","Sugarcane","Soybean","Sunflower","Groundnut"]
        states  = ["Punjab","Haryana","UP","MP","Maharashtra","Karnataka","AP","Rajasthan"]
        seasons = ["Kharif","Rabi","Zaid"]

        ce = encoders["crop"].transform([d["crop"]])[0]   if encoders else crops.index(d["crop"])
        se = encoders["state"].transform([d["state"]])[0] if encoders else states.index(d["state"])
        sn = encoders["season"].transform([d["season"]])[0] if encoders else seasons.index(d["season"])

        X = np.array([[ce, se, sn, d["area_ha"], d["rainfall_mm"], d["temperature"],
                       d["fertilizer_kg"], d["pesticide_kg"]]])
        pred = float(model.predict(X)[0])
        return jsonify({"yield_tons_per_ha": round(pred, 3), "total_yield_tons": round(pred * d["area_ha"], 2), "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/fertilizer", methods=["POST"])
def predict_fertilizer():
    try:
        model    = load_model("fertilizer_model.pkl")
        encoders = load_model("fertilizer_encoders.pkl")
        if not model:
            return jsonify({"error": "Model not loaded."}), 500

        d = request.json
        soil_types = ["Sandy","Loamy","Black","Red","Clayey"]
        crop_types = ["Wheat","Rice","Maize","Sugarcane","Cotton","Pulses","Groundnut","Soybean","Sunflower","Potato"]

        soil_e = encoders["soil"].transform([d["soil_type"]])[0] if encoders else soil_types.index(d["soil_type"])
        crop_e = encoders["crop"].transform([d["crop_type"]])[0] if encoders else crop_types.index(d["crop_type"])

        X = np.array([[d["temperature"], d["humidity"], d["moisture"], soil_e, crop_e,
                       d["nitrogen"], d["potassium"], d["phosphorous"]]])
        pred_enc = model.predict(X)[0]
        fert = encoders["fert"].inverse_transform([pred_enc])[0] if encoders else str(pred_enc)

        return jsonify({"recommended_fertilizer": fert, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/irrigation", methods=["POST"])
def predict_irrigation():
    try:
        model = load_model("irrigation_model.pkl")
        enc   = load_model("irrigation_stage_enc.pkl")
        if not model:
            return jsonify({"error": "Model not loaded."}), 500

        d = request.json
        stages = ["Seedling","Vegetative","Flowering","Fruiting","Maturity"]
        stage_e = enc.transform([d["crop_stage"]])[0] if enc else stages.index(d["crop_stage"])
        X = np.array([[d["temperature"], d["humidity"], d["soil_moisture"], d["rainfall_mm"], stage_e]])
        water = max(0, float(model.predict(X)[0]))

        level = "Low" if water < 5 else "Moderate" if water < 15 else "High" if water < 30 else "Critical"
        return jsonify({"water_need_L_per_m2": round(water, 2), "level": level, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("🌿 AgroIntel Flask API running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
