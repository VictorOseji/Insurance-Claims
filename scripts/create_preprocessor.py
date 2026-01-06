# ============================================================================
# FILE: scripts/create_preprocessor.py
# ============================================================================

"""
Create preprocessor and save train/test splits
This runs once before parallel model training
"""

import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data.data_loader import InsuranceDataLoader
from src.data.preprocessing import DataPreprocessor

def main():
    logger = setup_logger("create_preprocessor", "logs/create_preprocessor.log")
    
    logger.info("="*60)
    logger.info("CREATING PREPROCESSOR AND SPLITTING DATA")
    logger.info("="*60)
    
    # Load configuration
    logger.info("Loading configuration...")
    config_loader = ConfigLoader()
    config = config_loader.get_main_config()
    feature_config = config_loader.load_config("feature_config.yaml")
    
    # Load processed data
    logger.info("Loading processed data...")
    data_loader = InsuranceDataLoader(config)
    processed_data = data_loader.load_processed_data()
    logger.info(f"Data loaded: {processed_data.shape}")
    
    # Prepare features
    logger.info("Initializing preprocessor...")
    preprocessor = DataPreprocessor(config, feature_config)
    
    target_col = config['models']['target_variable']
    numeric_features = feature_config['numeric_features']
    categorical_features = feature_config['categorical_features']
    
    # Filter to available features
    available_numeric = [f for f in numeric_features if f in processed_data.columns]
    available_categorical = [f for f in categorical_features if f in processed_data.columns]
    
    logger.info(f"Numeric features: {len(available_numeric)}")
    logger.info(f"Categorical features: {len(available_categorical)}")
    
    # Prepare modeling data
    all_features = available_numeric + available_categorical + [target_col]
    modeling_data = processed_data[all_features].copy()
    modeling_data = modeling_data.dropna(subset=[target_col])
    
    logger.info(f"Modeling data shape: {modeling_data.shape}")
    
    # Prepare X and y
    logger.info("Separating features and target...")
    X, y = preprocessor.prepare_features(modeling_data, target_col)
    
    # Split data
    logger.info("Splitting into train and test sets...")
    split = preprocessor.split_data(X, y)
    
    logger.info(f"Train set: {split["X_train"].shape[0]} samples")
    logger.info(f"Test set: {split["X_test"].shape[0]} samples")

    # Save preprocessor
    logger.info("Saving data split...")
    preprocessor.save_preprocessor(split,available_numeric,available_categorical)
    
    # Save train/test splits
    logger.info("Data splits saved...")
    
    logger.info("="*60)
    logger.info("✓ PREPROCESSOR CREATION COMPLETE!")
    logger.info("="*60)
    logger.info("Outputs:")
    logger.info("  - Data Split saved to: data/interim/train_test_data_split")
    logger.info("  - A dictionary object containing X_train, Y_train, X_test, Y_test")
    logger.info("="*60)

if __name__ == "__main__":
    main()
