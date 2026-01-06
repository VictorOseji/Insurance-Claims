# ============================================================================
# FILE: src/evaluation/metrics.py
# ============================================================================
"""Model evaluation metrics"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


class ModelEvaluator:
    """Calculate and manage evaluation metrics"""
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
