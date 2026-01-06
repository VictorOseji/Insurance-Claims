# ============================================================================
# FILE: src/training/hyperparameter_tuner.py
# ============================================================================
"""Hyperparameter tuning utilities"""
from sklearn.model_selection import GridSearchCV
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Handle hyperparameter tuning with GridSearchCV"""
    
    def __init__(self, config: dict):
        self.config = config['training']
    
    def tune(self, model, param_grid: Dict[str, list], 
             X_train, y_train) -> tuple:
        """Perform grid search for hyperparameter tuning"""
        logger.info("Performing hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.config['cv_folds'],
            scoring=self.config['scoring'],
            n_jobs=self.config['n_jobs'],
            verbose=self.config['verbose']
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
