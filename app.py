import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model  = joblib.load('models/final_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Region mapping
REGIONS = {
    "Australia & New Zealand" : 0,
    "Caribbean"               : 1,
    "Central America"         : 2,
    "Central Asia"            : 3,
    "Eastern Asia"            : 4,
    "Eastern Europe"          : 5,
    "Melanesia"               : 6,
    "Northern Africa"         : 7,
    "Northern America"        : 8,
    "Northern Europe"         : 9,
    "Polynesia"               : 10,
    "South America"           : 11,
    "Southern Asia"           : 12,
    "Southern Europe"         : 13,
    "Sub-Saharan Africa"      : 14,
    "Western Asia"            : 15,
    "Western Europe"          : 16
}

CONFIDENCE = {
    "High Confidence"     : 0,
    "Low Confidence"      : 1,
    "Medium Confidence"   : 2,
    "Very Low Confidence" : 3
}

def predict(household_kg, household_tonnes, retail_kg, retail_tonnes,
            foodservice_kg, foodservice_tonnes, confidence, region):

    features = pd.DataFrame([[
        household_kg, household_tonnes,
        retail_kg, retail_tonnes,
        foodservice_kg, foodservice_tonnes,
        CONFIDENCE[confidence],
        REGIONS[region]
    ]], columns=[
        'household_kg_per_capita', 'household_tonnes',
        'retail_kg_per_capita', 'retail_tonnes',
        'foodservice_kg_per_capita', 'foodservice_tonnes',
        'confidence_encoded', 'region_encoded'
    ])

    scaled     = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    co2_estimate = (
        household_kg   * 2.5 +
        retail_kg      * 1.8 +
        foodservice_kg * 2.1
    )

    if co2_estimate < 200:
        level = "🟢 Low Emissions"
    elif co2_estimate < 400:
        level = "🟡 Medium Emissions"
    else:
        level = "🔴 High Emissions"

    return (
        f"{prediction:.2f} kg/capita/year",
        f"{co2_estimate:.2f} kg CO₂e/capita",
        level
    )

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Food Waste Predictor") as app:

    gr.Markdown("""
    # 🌱 Food Waste Emission Predictor
    **XGBoost Model · R² = 0.9457 · 214 Countries · UNEP Dataset**
    
    Enter food waste data by sector to predict total waste and CO₂ emissions.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏠 Household Waste")
            household_kg     = gr.Number(label="kg per capita / year", value=82)
            household_tonnes = gr.Number(label="Total tonnes / year",  value=3109153)

        with gr.Column():
            gr.Markdown("### 🛒 Retail Waste")
            retail_kg     = gr.Number(label="kg per capita / year", value=16)
            retail_tonnes = gr.Number(label="Total tonnes / year",  value=594982)

        with gr.Column():
            gr.Markdown("### 🍽️ Food Service Waste")
            foodservice_kg     = gr.Number(label="kg per capita / year", value=28)
            foodservice_tonnes = gr.Number(label="Total tonnes / year",  value=1051783)

    with gr.Row():
        confidence = gr.Dropdown(
            choices=list(CONFIDENCE.keys()),
            value="Very Low Confidence",
            label="📋 Data Confidence"
        )
        region = gr.Dropdown(
            choices=list(REGIONS.keys()),
            value="Southern Asia",
            label="🌍 Region"
        )

    predict_btn = gr.Button("⚡ Generate Prediction", variant="primary", size="lg")

    gr.Markdown("### 📊 Results")
    with gr.Row():
        waste_output = gr.Text(label="🗑️ Predicted Food Waste")
        co2_output   = gr.Text(label="🌿 CO₂ Estimate")
        level_output = gr.Text(label="📈 Emission Level")

    predict_btn.click(
        fn=predict,
        inputs=[
            household_kg, household_tonnes,
            retail_kg, retail_tonnes,
            foodservice_kg, foodservice_tonnes,
            confidence, region
        ],
        outputs=[waste_output, co2_output, level_output]
    )

    gr.Markdown("""
    ---
    **Dataset:** UNEP Food Waste Index Report · 
    **Model:** XGBoost Regressor · 
    **Built by:** Jayaranjini
    """)

app.launch()