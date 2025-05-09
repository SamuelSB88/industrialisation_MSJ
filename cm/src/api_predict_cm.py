from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from cm_predictor import CMPredictor
from schemas import CMPredictionInput  # le schéma généré avec toutes les variables en int

# Initialisation de l'app FastAPI
app = FastAPI(title="API de Prédiction CM", version="1.0")

# Chargement du modèle de coût moyen
model_cm, encoder_cm = joblib.load("best_model_encoder_cm_xgb.pkl")

predictor = CMPredictor()
predictor.model = model_cm
predictor.encoder = encoder_cm


@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API opérationnelle (CM uniquement)"}


@app.post("/predict_montant")
def predict_cm(input_data: CMPredictionInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        df = CMPredictor.reduce_memory_usage(df, verbose=False)

        y_pred = predictor.predict(
            df,
            df.select_dtypes(include='number').columns,
            df.select_dtypes(include='object').columns
        )

        return {"prediction_cm": float(y_pred[0])}  

    except Exception as e:
        return {"error": str(e)}
