
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from xgb_optuna_pipeline import reduce_memory_usage, predict_test_set

# Initialisation de l'app FastAPI
app = FastAPI(title="API de Prédiction CM", version="1.0")

# Chargement des modèles (à adapter avec les bons chemins)
model_cm, encoder_cm = joblib.load("best_model_encoder_cm_xgb.pkl")
model_freq = joblib.load("best_model_freq.pkl")  # À créer pour la prédiction de FREQ

# === Schéma d'entrée ===
class InputData(BaseModel):
    data: list  # list of dicts (one per observation)

# === Route de santé ===
@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API opérationnelle"}

# === Prédiction du montant CM ===
@app.post("/predict_montant")
def predict_cm(input_data: InputData):
    df = pd.DataFrame(input_data.data)
    df = reduce_memory_usage(df, verbose=False)
    y_pred = predict_test_set(df, df.select_dtypes(include='number').columns, df.select_dtypes(include='object').columns, encoder_cm, model_cm)
    return {"prediction_cm": y_pred.tolist()}

# === Prédiction de la fréquence ===
@app.post("/predict_freq")
def predict_freq(input_data: InputData):
    df = pd.DataFrame(input_data.data)
    df = reduce_memory_usage(df, verbose=False)
    # À adapter avec les bons encodeurs + traitement
    y_pred = model_freq.predict(df)
    return {"prediction_freq": y_pred.tolist()}

# === Prédiction combinée CM x FREQ ===
@app.post("/predict_global")
def predict_combined(input_data: InputData):
    df = pd.DataFrame(input_data.data)
    df = reduce_memory_usage(df, verbose=False)
    y_cm = predict_test_set(df, df.select_dtypes(include='number').columns, df.select_dtypes(include='object').columns, encoder_cm, model_cm)
    y_freq = model_freq.predict(df)
    y_global = (y_cm * y_freq).tolist()
    return {
        "prediction_cm": y_cm.tolist(),
        "prediction_freq": y_freq.tolist(),
        "prediction_total": y_global
    }
