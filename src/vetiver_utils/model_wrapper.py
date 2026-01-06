# ============================================================================
# FILE: src/vetiver_utils/model_wrapper.py
# ============================================================================
"""Vetiver model wrapping utilities"""
from vetiver import VetiverModel
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VetiverModelWrapper:
    """Wrap models in Vetiver format"""
    
    @staticmethod
    def create_vetiver_model(model, model_name: str, X_train, feature_names=None) -> VetiverModel:
        """Create a Vetiver model with metadata
        
        Args:
            model: Trained sklearn/xgboost model
            model_name: Name for the model
            X_train: Training data (can be numpy array or DataFrame)
            feature_names: List of feature names (required if X_train is numpy array)
            
        Returns:
            VetiverModel instance
        """
        logger.info(f"Creating Vetiver model: {model_name}")
        
        # Convert to DataFrame if it's a numpy array
        if isinstance(X_train, np.ndarray):
            if feature_names is None:
                # Generate generic feature names if none provided
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                logger.warning(f"No feature names provided, using generic names: {feature_names[:5]}...")
            
            prototype_df = pd.DataFrame(X_train[:5], columns=feature_names)
        else:
                # Already a DataFrame
            prototype_df = X_train[:5].copy()
        
        logger.info(f"Prototype shape: {prototype_df.shape}")
        logger.info(f"Prototype columns: {list(prototype_df.columns)[:5]}...")
        
        # Create vetiver model with prototype data
        v_model = VetiverModel(
            model=model,
            model_name=model_name,
            prototype_data=prototype_df
        )
        
        return v_model

