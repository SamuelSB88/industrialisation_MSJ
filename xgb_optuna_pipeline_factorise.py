
import pandas as pd
import numpy as np
import joblib
import gc
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from category_encoders import TargetEncoder, CountEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

def prepare_benchmark(X_train, y_train):
    Xy = X_train.copy()
    Xy['CHARGE'] = y_train['CHARGE']
    Xy['CHARGE_BIN'] = pd.qcut(Xy['CHARGE'], q=10, duplicates='drop')
    train, valid = train_test_split(Xy, test_size=0.2, stratify=Xy['CHARGE_BIN'], random_state=42)
    return train, valid

def fill_missing_values(X_train, X_valid, numeric_cols, cat_cols):
    X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
    X_train[cat_cols] = X_train[cat_cols].fillna(-999)
    X_valid[numeric_cols] = X_valid[numeric_cols].fillna(0)
    X_valid[cat_cols] = X_valid[cat_cols].fillna(-999)
    return X_train, X_valid

def compute_cm_metrics(cm_true, cm_pred):
    rmse = np.sqrt(mean_squared_error(cm_true, cm_pred))
    rrmse = rmse / cm_true.mean()
    mape = np.mean(np.abs((cm_true - cm_pred) / cm_true)) * 100
    return rmse, rrmse, mape

def train_optuna(X_train, y_train_log, X_valid, y_train_full, y_valid_full, annee_valid, cat_cols, n_trials=10):
    best_rmse = np.inf

    def objective(trial):
        nonlocal best_rmse
        encoder_type = trial.suggest_categorical("encoder", ["target", "count"])
        encoder = TargetEncoder(cols=cat_cols) if encoder_type == "target" else CountEncoder(cols=cat_cols)

        encoder.fit(X_train, y_train_log)
        X_train_enc = encoder.transform(X_train)
        X_valid_enc = encoder.transform(X_valid)

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
        model.fit(X_train_enc, y_train_log)

        y_pred_log = model.predict(X_valid_enc)
        y_pred = np.exp(y_pred_log)

        freq_valid = y_train_full['FREQ'].loc[X_valid.index]
        charge_valid = y_valid_full
        mask = freq_valid > 0
        cm_reel = charge_valid[mask] / (freq_valid[mask] * annee_valid[mask])
        cm_pred = y_pred[mask]

        rmse = np.sqrt(mean_squared_error(cm_reel, cm_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            joblib.dump((model, encoder), "best_model_encoder_cm_xgb.pkl")
            print(f"\n💾 Nouveau meilleur modèle sauvegardé !\n   ↪️ Encodeur : {encoder_type} | RMSE : {rmse:,.2f} €")

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def plot_cm_distribution(cm_true, cm_pred):
    plt.figure(figsize=(7, 3))
    sns.histplot(cm_true, bins=50, kde=True)
    plt.title("Distribution du CM réel (FREQ > 0)")
    plt.xlabel("CM réel")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(cm_true, cm_pred, alpha=0.4)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot([cm_true.min(), cm_true.max()], [cm_true.min(), cm_true.max()], 'r--', lw=2)
    plt.xlabel("CM réel (log)")
    plt.ylabel("CM prédit (log)")
    plt.title("📊 CM prédit vs réel (log-log, FREQ > 0)")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

def predict_test_set(X_test, numeric_columns, cat_columns, encoder, model):
    X_test[numeric_columns] = X_test[numeric_columns].fillna(0)
    X_test[cat_columns] = X_test[cat_columns].fillna(-999)
    X_test_model = X_test.drop(['ID', 'ANNEE_ASSURANCE'], axis=1)
    X_test_enc = encoder.transform(X_test_model)
    y_pred_log = model.predict(X_test_enc)
    return np.exp(y_pred_log)
