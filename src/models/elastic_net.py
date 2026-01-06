# ============================================================================
# FILE: src/models/elastic_net.py
# ============================================================================
"""ElasticNet Regression model implementation"""
from sklearn.linear_model import ElasticNet
from .base_model import BaseModel
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ElasticNetModel(BaseModel):
    """ElasticNet Regression model wrapper"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = "Elastic Net"
        
        self.alpha = config.get("alpha", 1.0)
        self.l1_ratio = config.get("l1_ratio", 0.5)
        self.max_iter = config.get("max_iter", 1000)
        self.tol = config.get("tol", 0.0001)
        self.random_state = config.get("random_state",42)

    def build_model(config: dict) -> ElasticNet:
        model = ElasticNet(
            alpha=config.get("alpha", 1.0),
            l1_ratio=config.get("l1_ratio", 0.5),
            max_iter=config.get("max_iter", 1000),
            tol=config.get("tol", 0.0001),
            random_state=config.get("random_state",42)
        )
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Elastic Net model"""
        logger.info(f"Training {self.model_name} (alpha={self.alpha}, l1_ratio={self.l1_ratio})...")
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters used in training"""
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "tol": self.tol
        }
