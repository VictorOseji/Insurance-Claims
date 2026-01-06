# ============================================================================
# FILE: src/models/xgboost_model.py
# ============================================================================
"""XGBoost model implementation"""
from xgboost import XGBRegressor
from .base_model import BaseModel
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost model wrapper"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = "XGBoost"
        self.best_params = None
    
    def build_model(config: dict) -> XGBRegressor:
        model = XGBRegressor(
            tree_method=config.get("tree_method", 100),
            n_estimator=config.get("n_estimator", 500),
            learning_rate=config.get("learning_rate", 0.01),
            max_bin=config.get("max_bin", 3),
            max_depth=config.get("max_depth", 2),
            random_state=config.get("random_state",42),
            objective='reg:squarederror'
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

