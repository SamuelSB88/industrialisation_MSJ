# -*- coding: utf-8 -*-
import sys
import io

# Encodage pour Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Imports
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd
import numpy as np
from category_encoders import TargetEncoder, CountEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import joblib
import gc

# Réduction de la mémoire
def reduce_memory_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    else:
                        df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"🧠 Mémoire utilisée : {start_mem:.2f} MB ➜ {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% gain)")
    return df

# Chargement des données
print("📂 Chargement des fichiers CSV...")
X_train = pd.read_csv("train_input_cleaned.csv")
y_train = pd.read_csv("train_output.csv")
X_test = pd.read_csv("test_input_cleaned.csv")
print("✅ Fichiers chargés.")

print("⚙️ Réduction de mémoire...")
X_train = reduce_memory_usage(X_train)
y_train = reduce_memory_usage(y_train)
X_test = reduce_memory_usage(X_test)

pd.set_option('display.max_columns', 375)

# Sauvegarde des identifiants
annee_assurance_train = X_train['ANNEE_ASSURANCE'].copy()
id_train = X_train['ID'].copy()
annee_assurance_test = X_test['ANNEE_ASSURANCE'].copy()
id_test = X_test['ID'].copy()

# === Benchmark : stratification sur CHARGE ===
Xy_full = X_train.copy()
Xy_full['CHARGE'] = y_train['CHARGE']
Xy_full['CHARGE_BIN'] = pd.qcut(Xy_full['CHARGE'], q=10, duplicates='drop')

train_benchmark, valid_benchmark = train_test_split(
    Xy_full,
    test_size=0.2,
    stratify=Xy_full['CHARGE_BIN'],
    random_state=42
)

X_train_benchmark = train_benchmark.drop(['CHARGE', 'CHARGE_BIN'], axis=1)
y_train_benchmark = train_benchmark['CHARGE']
annee_assurance_train_benchmark = X_train_benchmark['ANNEE_ASSURANCE'].copy()

X_valid_benchmark = valid_benchmark.drop(['CHARGE', 'CHARGE_BIN'], axis=1)
y_valid_benchmark = valid_benchmark['CHARGE']
annee_assurance_valid_benchmark = X_valid_benchmark['ANNEE_ASSURANCE'].copy()

print(f"✅ Benchmark : {X_train_benchmark.shape[0]} train / {X_valid_benchmark.shape[0]} validation.")

# Traitement des valeurs manquantes
numeric_columns = X_train.select_dtypes(include=['number']).columns
non_numeric_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"🔠 Colonnes catégorielles encodées (CountEncoder) : {len(non_numeric_cols)}")

X_train_benchmark[numeric_columns] = X_train_benchmark[numeric_columns].fillna(0)
X_train_benchmark[non_numeric_cols] = X_train_benchmark[non_numeric_cols].fillna(-999)
X_valid_benchmark[numeric_columns] = X_valid_benchmark[numeric_columns].fillna(0)
X_valid_benchmark[non_numeric_cols] = X_valid_benchmark[non_numeric_cols].fillna(-999)

X_train_benchmark = X_train_benchmark.drop(['ID', 'ANNEE_ASSURANCE'], axis=1)
X_valid_benchmark = X_valid_benchmark.drop(['ID', 'ANNEE_ASSURANCE'], axis=1)

# Appel au garbage collector pour libérer la RAM
gc.collect()

# Cible : log(CM)
y_train_cm_raw = y_train['CM'].loc[X_train_benchmark.index].copy()
y_train_cm_raw[y_train_cm_raw <= 0] = 1e-3
y_train_cm_log = np.log(y_train_cm_raw)

# 🔍 OPTUNA avec encodeur dynamique
best_rmse = np.inf

def objective(trial):
    global best_rmse

    encoder_type = trial.suggest_categorical("encoder", ["target", "count"])
    encoder = TargetEncoder(cols=non_numeric_cols) if encoder_type == "target" else CountEncoder(cols=non_numeric_cols)

    encoder.fit(X_train_benchmark, y_train_cm_log)
    X_train_enc = encoder.transform(X_train_benchmark)
    X_valid_enc = encoder.transform(X_valid_benchmark)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 50),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "verbosity": 0,
        "objective": "reg:squarederror",
        "tree_method": "hist"
    }

    model = XGBRegressor(**params)
    model.fit(X_train_enc, y_train_cm_log)

    y_pred_log = model.predict(X_valid_enc)
    y_pred = np.exp(y_pred_log)

    freq_valid = y_train['FREQ'].loc[X_valid_benchmark.index]
    charge_valid = y_valid_benchmark
    annee_valid = annee_assurance_valid_benchmark

    mask = freq_valid > 0
    cm_reel = charge_valid[mask] / (freq_valid[mask] * annee_valid[mask])
    cm_pred = y_pred[mask]

    rmse = np.sqrt(mean_squared_error(cm_reel, cm_pred))

    if rmse < best_rmse:
        best_rmse = rmse
        joblib.dump((model, encoder), "best_model_encoder_cm_xgb.pkl")

        rrmse = rmse / cm_reel.mean()

        print(f"\n💾 Nouveau meilleur modèle sauvegardé !")
        print(f"   ↪️ Encodeur     : {encoder_type}")
        print(f"   ↪️ RMSE absolu : {rmse:,.2f} €")
        print(f"   ↪️ rRMSE        : {rrmse:.2%}")

    return rmse

print("🔧 Optuna : optimisation modèle + encodeur...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, show_progress_bar=True)

print(f"\n✅ Meilleurs paramètres : {study.best_params}")
print(f"✅ Meilleur RMSE Optuna : {study.best_value:.4f}")

# ✅ Chargement du meilleur modèle + encodeur
best_model, best_encoder = joblib.load("best_model_encoder_cm_xgb.pkl")

X_valid_enc = best_encoder.transform(X_valid_benchmark)
y_pred_cm_log_valid = best_model.predict(X_valid_enc)
y_pred_cm_valid = np.exp(y_pred_cm_log_valid)

freq_valid = y_train['FREQ'].loc[X_valid_benchmark.index]
charge_valid = y_valid_benchmark
annee_valid = annee_assurance_valid_benchmark

mask = freq_valid > 0
cm_reel = charge_valid[mask] / (freq_valid[mask] * annee_valid[mask])
cm_pred = y_pred_cm_valid[mask]

# Analyse des prédictions
print("\n📊 Analyse des prédictions CM :")
print(f"   ↪️ CM réel    : Min = {cm_reel.min():.2f}, Max = {cm_reel.max():.2f}, Moyenne = {cm_reel.mean():.2f}")
print(f"   ↪️ CM prédit  : Min = {cm_pred.min():.2f}, Max = {cm_pred.max():.2f}, Moyenne = {cm_pred.mean():.2f}")

rmse_cm = np.sqrt(mean_squared_error(cm_reel, cm_pred))
rrmse = rmse_cm / cm_reel.mean()
mape = np.mean(np.abs((cm_reel - cm_pred) / cm_reel)) * 100

print(f"\n📉 RMSE absolu (FREQ > 0) : {rmse_cm:,.2f} €")
print(f"📏 CM moyen réel (FREQ > 0) : {cm_reel.mean():,.2f} €")
print(f"📊 RMSE relatif (rRMSE) : {rrmse:.2%}")
print(f"📊 MAPE (erreur moyenne en %) : {mape:.2f}%")

# 📊 Distribution CM réel
plt.figure(figsize=(7, 3))
sns.histplot(cm_reel, bins=50, kde=True)
plt.title("Distribution du CM réel (benchmark, FREQ > 0)")
plt.xlabel("CM réel")
plt.tight_layout()
plt.show()

# 📈 Scatter log-log préd vs réel
plt.figure(figsize=(6, 6))
plt.scatter(cm_reel, cm_pred, alpha=0.4)
plt.xscale("log")
plt.yscale("log")
plt.plot([cm_reel.min(), cm_reel.max()],
         [cm_reel.min(), cm_reel.max()],
         'r--', lw=2)
plt.xlabel("CM réel (log)")
plt.ylabel("CM prédit (log)")
plt.title("📊 CM prédit vs réel (log-log, FREQ > 0)")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# 🔮 Prédiction sur test & export
X_test[numeric_columns] = X_test[numeric_columns].fillna(0)
X_test[non_numeric_cols] = X_test[non_numeric_cols].fillna(-999)
X_test_model = X_test.drop(['ID', 'ANNEE_ASSURANCE'], axis=1)
X_test_enc = best_encoder.transform(X_test_model)

y_pred_cm_log_test = best_model.predict(X_test_enc)
y_pred_cm_test = np.exp(y_pred_cm_log_test)

submission = pd.DataFrame({
    'ID': id_test,
    'ANNEE_ASSURANCE': annee_assurance_test,
    'CM': y_pred_cm_test
})

submission.to_csv("submission_cm_xgb_encoder_optuna.csv", index=False)
print("📦 Fichier 'submission_cm_xgb_encoder_optuna.csv' généré avec le meilleur encodeur.")