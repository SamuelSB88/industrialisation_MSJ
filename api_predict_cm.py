from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from cm_predictor import CMPredictor

# Initialisation de l'app FastAPI
app = FastAPI(title="API de Prédiction CM", version="1.0")

# Chargement des modèles (à adapter si chemins différents)
model_cm, encoder_cm = joblib.load("best_model_encoder_cm_xgb.pkl")
model_freq = joblib.load("best_model_freq.pkl")  # Facultatif

predictor = CMPredictor()
predictor.model = model_cm
predictor.encoder = encoder_cm

class InputData(BaseModel):
    data: list  # list of dicts (one per observation)

@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API opérationnelle"}

@app.post("/predict_montant")
def predict_cm(input_data: InputData):
    df = pd.DataFrame(input_data.data)
    df = CMPredictor.reduce_memory_usage(df, verbose=False)
    y_pred = predictor.predict(df, df.select_dtypes(include='number').columns,
                                   df.select_dtypes(include='object').columns)
    return {"prediction_cm": y_pred.tolist()}

@app.post("/predict_freq")
def predict_freq(input_data: InputData):
    df = pd.DataFrame(input_data.data)
    df = CMPredictor.reduce_memory_usage(df, verbose=False)
    y_pred = model_freq.predict(df)  # À adapter selon preprocessing
    return {"prediction_freq": y_pred.tolist()}

@app.post("/predict_global")
def predict_combined(input_data: InputData):
    df = pd.DataFrame(input_data.data)
    df = CMPredictor.reduce_memory_usage(df, verbose=False)
    y_cm = predictor.predict(df, df.select_dtypes(include='number').columns,
                                 df.select_dtypes(include='object').columns)
    y_freq = model_freq.predict(df)
    y_global = (y_cm * y_freq).tolist()
    return {
        "prediction_cm": y_cm.tolist(),
        "prediction_freq": y_freq.tolist(),
        "prediction_total": y_global
    }
