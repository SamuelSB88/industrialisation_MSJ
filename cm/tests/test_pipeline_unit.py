import sys
import os
import unittest
import pandas as pd
import numpy as np

# Ajoute le dossier 'src' au path pour que l'import fonctionne
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cm_predictor import CMPredictor

class TestCMPredictor(unittest.TestCase):

    def test_reduce_memory_usage(self):
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.0, 2.0, 3.0, 4.0]
        })

        reduced_df = CMPredictor.reduce_memory_usage(df.copy(), verbose=False)
        self.assertIn(reduced_df.dtypes['int_col'], [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32])
        self.assertEqual(reduced_df.dtypes['float_col'], np.float32)

    def test_compute_metrics(self):
        cm_true = np.array([100.0, 200.0, 300.0])
        cm_pred = np.array([110.0, 190.0, 310.0])
        rmse, rrmse, mape = CMPredictor.compute_metrics(cm_true, cm_pred)

        self.assertGreater(rmse, 0)
        self.assertGreaterEqual(mape, 0)
        self.assertLessEqual(mape, 100)
        self.assertGreaterEqual(rrmse, 0)
        self.assertLessEqual(rrmse, 1)

    def test_fill_missing(self):
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

        self.assertEqual(filled_train['num'].isna().sum(), 0)
        self.assertEqual(filled_valid['cat'].isna().sum(), 0)

if __name__ == "__main__":
    unittest.main()
