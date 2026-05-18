import os
PORT = int(os.environ.get("PORT", 8000))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model  = joblib.load('models/final_model.pkl')
scaler = joblib.load('models/scaler.pkl')

app = FastAPI(title="Food Waste Emission Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input schema
class WasteInput(BaseModel):
    household_kg_per_capita   : float
    household_tonnes          : float
    retail_kg_per_capita      : float
    retail_tonnes             : float
    foodservice_kg_per_capita : float
    foodservice_tonnes        : float
    confidence_encoded        : int    # 0=High, 1=Low, 2=Medium, 3=Very Low
    region_encoded            : int    # 0-16 for different regions

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(data: WasteInput):
    features = np.array([[
        data.household_kg_per_capita,
        data.household_tonnes,
        data.retail_kg_per_capita,
        data.retail_tonnes,
        data.foodservice_kg_per_capita,
        data.foodservice_tonnes,
        data.confidence_encoded,
        data.region_encoded
    ]])

    scaled   = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    # Calculate CO2 equivalent
    co2_estimate = (
        data.household_kg_per_capita   * 2.5 +
        data.retail_kg_per_capita      * 1.8 +
        data.foodservice_kg_per_capita * 2.1
    )

    return {
        "predicted_waste_kg_per_capita" : round(float(prediction), 2),
        "co2_estimate_kg_per_capita"    : round(float(co2_estimate), 2),
        "status" : "success"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost R²=0.9457"}