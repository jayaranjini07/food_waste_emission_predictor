# 🌱 Food Waste Emission Predictor

> A machine learning web app that predicts food waste patterns and 
> estimates CO₂ equivalent emissions across 214 countries using XGBoost.

![Python](https://img.shields.io/badge/Python-3.11-green)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-brightgreen)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![R2](https://img.shields.io/badge/R²_Score-0.9457-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

🔗 **Live Demo:** 

---

## 📌 Project Overview

This project builds an end-to-end ML pipeline to analyze food waste 
patterns and predict total food waste per capita (kg/year) using 
real-world data from the UNEP Food Waste Index Report covering 214 countries.

The model also estimates CO₂ equivalent emissions by sector 
(household, retail, food service) to support environmental analysis.

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| R² Score | 0.9457 |
| MAE | 2.55 kg/capita |
| RMSE | 4.85 kg/capita |
| Training Samples | 171 |
| Test Samples | 43 |
| Features Used | 8 |

---

## 🔍 Key Findings

- **Household waste** is the dominant predictor (16x more important 
  than any other feature per SHAP analysis)
- **Nigeria & Malaysia** have the highest CO₂ emissions per capita
- **Sub-Saharan Africa** shows highest regional food waste levels
- Model correctly classifies emission levels (Low / Medium / High) 
  across all tested countries

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Model | XGBoost Regressor |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Explainability | SHAP |
| API Backend | FastAPI + Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Dataset | UNEP Food Waste Index (214 countries) |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/jayaranjini07/food-waste-emission-predictor.git
cd food-waste-emission-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
uvicorn app:app --reload

# 5. Open browser
http://localhost:8000
```

---

## 📊 Model Development Process

1. **EDA** — Explored 214-country dataset, identified waste patterns by 
   region and sector
2. **Feature Engineering** — Encoded categorical variables (region, 
   confidence level), scaled features with StandardScaler
3. **Model Comparison** — Evaluated Random Forest vs XGBoost; 
   XGBoost won with R²=0.9457
4. **Hyperparameter Tuning** — GridSearchCV with 5-fold cross validation 
   improved RMSE from 5.24 → 4.85
5. **Explainability** — SHAP values confirmed household waste as 
   dominant feature
6. **Deployment** — FastAPI backend + HTML/CSS/JS frontend

---

## 💡 Sample Predictions

| Country | Actual (kg/cap) | Predicted (kg/cap) | CO₂ Estimate |
|---|---|---|---|
| Afghanistan | 126 | 126.62 | 292.6 kg CO₂e |
| Nigeria | 260 | 241.17 | 610.5 kg CO₂e |

---

## 📈 SHAP Feature Importance

Household waste per capita dominates prediction with 16x more 
impact than any other feature, confirming that domestic consumption 
behavior is the primary driver of total food waste.

---

## 🙋 Author

**Jayaranjini** 
- GitHub: [@jayaranjini07](https://github.com/jayaranjini07)

---

## 📄 License
MIT License — feel free to use and build on this project.