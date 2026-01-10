# ============================================================================
# FILE: src/training/hyperparameter_tuner.py
# ============================================================================
"""Hyperparameter tuning utilities"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Handle hyperparameter tuning with GridSearchCV"""

    def __init__(self, config: dict):
        self.config = config['training']

        # Create the scorer functions
        self.smape_scorer = make_scorer(self._smape, greater_is_better=False)
        self.rmsle_scorer = make_scorer(self._rmsle, greater_is_better=False)

        self.scoring = {
            "rmsle": self.rmsle_scorer,
            "smape": self.smape_scorer
        }

    # Note: static or instance methods both work if called properly
    def _smape(self, y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error"""
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        if not np.any(mask):
            return 0.0  # no valid pairs
        return np.mean(
            2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]
        )

    def _rmsle(self, y_true, y_pred):
        """Root Mean Squared Logarithmic Error"""
        # mean_squared_log_error returns a single float
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    def tune(self,
             model,
             param_grid: Dict[str, list],
             X_train,
             y_train) -> tuple:
        """Perform grid search for hyperparameter tuning"""
        logger.info("Performing hyperparameter tuning...")

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.config['cv_folds'],
            scoring=self.scoring,
            refit="rmsle",                # refit using RMSLE
            n_jobs=self.config['n_jobs'],
            verbose=self.config['verbose'],
            return_train_score=True       # optional, but useful
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV RMSLE: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

