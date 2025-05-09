from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from cm_predictor import CMPredictor

# Initialisation de l'app FastAPI
app = FastAPI(title="API de Prédiction CM", version="1.0")

# Chargement du modèle de coût moyen
model_cm, encoder_cm = joblib.load("best_model_encoder_cm_xgb.pkl")

predictor = CMPredictor()
predictor.model = model_cm
predictor.encoder = encoder_cm

class InputData(BaseModel):
    data: list  # list of dicts (one per observation)

@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API opérationnelle (CM uniquement)"}

@app.post("/predict_montant")
def predict_cm(input_data: InputData):
    try:
        df = pd.DataFrame(input_data.data)

        # ✅ Supprimer les colonnes si elles existent
        df.drop(columns=["ID", "ANNEE_ASSURANCE"], inplace=True, errors="ignore")

        df = CMPredictor.reduce_memory_usage(df, verbose=False)

        y_pred = predictor.predict(
            df,
            df.select_dtypes(include='number').columns,
            df.select_dtypes(include='object').columns
        )
        return {"prediction_cm": y_pred.tolist()}
    except Exception as e:
        return {"error": str(e)}

    
