# 📊 API de Prédiction du Coût Moyen (CM)

Cette API permet de prédire un **coût moyen (CM)** à partir de variables d'entrée (structurelles, géographiques, météorologiques, etc.). Elle repose sur un modèle entraîné (XGBoost) encapsulé dans une classe `CMPredictor` avec encodage automatique des variables catégorielles.

---

## 🚀 Fonctionnalités

- ✅ Prédiction du coût moyen à partir de données JSON
- ✅ Chargement automatique du modèle et de l’encodeur
- ✅ Réduction automatique de l’usage mémoire
- ✅ Encodage (Target/Count) sur les variables catégorielles
- ✅ API REST avec FastAPI + Swagger intégré

---

## 🗂 Arborescence du projet

```
.
├── api_predict_cm.py             # Déclaration de l’API FastAPI
├── best_model_encoder_cm_xgb.pkl # Modèle + encodeur (dump joblib)
├── cm_predictor.py               # Classe CMPredictor (modèle, preprocessing, prédiction)
├── main.py                       # Point d’entrée (Uvicorn)
├── preprocessing.py              # Fonctions de nettoyage (optionnel)
├── test_pipeline_unit.py         # Tests unitaires
├── requirements.txt              # Dépendances Python
└── README.md                     # 📄 Ce fichier
```

---

## ⚙️ Installation

1. **Créer un environnement virtuel (optionnel mais recommandé)**  
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

---

## ▶️ Lancer l’API

```bash
python3 main.py
```

L’API tourne par défaut sur [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📌 Endpoints

### `GET /health`

Permet de vérifier que l’API est bien active.

**Réponse :**
```json
{
  "status": "OK",
  "message": "API opérationnelle (CM uniquement)"
}
```

---

### `POST /predict_montant`

Fournit une prédiction de coût moyen pour un ou plusieurs jeux de données.

#### Exemple de requête (via Swagger ou `curl`) :

```json
{
  "data": [
    {
      "ACTIVIT2": 0,
      "VOCATION": 0,
      "TYPERS": 0,
      ...
      "ZONE": 0
    }
  ]
}
```

📎 Les colonnes `ID` et `ANNEE_ASSURANCE` ne sont **pas obligatoires** dans la requête, et seront **automatiquement exclues** si présentes.

**Réponse :**
```json
{
  "prediction_cm": [0.001029136241413653]
}
```

---

## ✅ Format attendu

- `data` est une **liste de dictionnaires**
- Tous les noms de colonnes doivent correspondre **exactement** à ceux attendus par le modèle (voir première ligne du fichier CSV d’entraînement)

---

## 🧠 Développement

Le cœur du système est dans `cm_predictor.py` :

- `predict()` applique automatiquement :
  - le **remplissage des valeurs manquantes**
  - l’**encodage** des variables catégorielles
  - la suppression des colonnes inutiles (`ID`, `ANNEE_ASSURANCE`)
  - la **prédiction log puis exponentiation** pour retrouver une valeur exploitable

---

## 📦 Dépendances principales

- `fastapi`
- `uvicorn`
- `pydantic`
- `pandas`
- `numpy`
- `xgboost`
- `category_encoders`
- `joblib`

---

## 🧪 Test rapide

Une fois l’API lancée :

👉 Accède à la documentation interactive :  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
