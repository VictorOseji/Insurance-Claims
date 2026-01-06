# ============================================================================
# FILE: src/models/gradient_boosting.py
# ============================================================================
"""Gradient Boosting model implementation"""
from sklearn.ensemble import GradientBoostingRegressor
from .base_model import BaseModel
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model wrapper"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = "Gradient Boosting"
        self.best_params = None
    
    def build_model(config: dict) -> GradientBoostingRegressor:
        model = GradientBoostingRegressor(
            n_estimators=config.get("n_estimator", 100),
            learning_rate=config.get("learning_rate", 0.01),
            max_depth=config.get("max_depth", 3),
            min_samples_split=config.get("min_samples_split", 2),
            random_state=config.get("random_state",42)
        )
        return model
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if self.best_params:
            return self.best_params
        return self.model.get_params()
    
    def set_params(self, params: Dict[str, Any]):
        """Set model parameters"""
        self.model.set_params(**params)
        self.best_params = params
