import sys
import os

# Ajoute le dossier 'src' au path pour que l'import fonctionne
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from cm_predictor import CMPredictor

def test_reduce_memory_usage():
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4],
        'float_col': [1.0, 2.0, 3.0, 4.0]
    })

    reduced_df = CMPredictor.reduce_memory_usage(df.copy(), verbose=False)
    assert reduced_df.dtypes['int_col'] in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]
    assert reduced_df.dtypes['float_col'] == np.float32

def test_compute_metrics():
    cm_true = np.array([100.0, 200.0, 300.0])
    cm_pred = np.array([110.0, 190.0, 310.0])
    rmse, rrmse, mape = CMPredictor.compute_metrics(cm_true, cm_pred)

    assert rmse > 0
    assert 0 <= mape <= 100
    assert 0 <= rrmse <= 1

def test_fill_missing():
    df_train = pd.DataFrame({
        'num': [1, 2, np.nan],
        'cat': ['A', 'A', None]
    })

    df_valid = pd.DataFrame({
        'num': [np.nan, 4, 5],
        'cat': ['C', 'D', None]
    })

    numeric_cols = ['num']
    cat_cols = ['cat']
    filled_train, filled_valid = CMPredictor.fill_missing(df_train.copy(), df_valid.copy(), numeric_cols, cat_cols)

    assert filled_train['num'].isna().sum() == 0
    assert filled_valid['cat'].isna().sum() == 0
