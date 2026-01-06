# ============================================================================
# FILE: src/models/base_model.py
# ============================================================================
"""Base model class"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any #, Tuple


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.model_name = None
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        pass
