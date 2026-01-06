# ============================================================================
# FILE: scripts/train_insurance_models.py
# ============================================================================
"""Main training script for insurance claims prediction models"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data.preprocessing import DataPreprocessor
from src.training.trainer import ModelTrainer
from src.evaluation.model_comparison import ModelComparator
from src.pins_utils.board_manager import PinsBoardManager


def main():
    """Main training pipeline for insurance claims"""
    # Setup logger
    logger = setup_logger("insurance_training", "logs/training.log")
    
    logger.info("="*80)
    logger.info("Starting Insurance Claims Prediction Pipeline")
    logger.info("="*80)
    
    # Load configuration
    logger.info("\n1. Loading configuration...")
    config_loader = ConfigLoader()
    config = config_loader.get_main_config()
    model_params = config_loader.get_model_params()
    feature_config = config_loader.load_config("feature_config.yaml")
    
    # Load processed data
    logger.info("\n2. Loading processed data...")
    data_loader = DataPreprocessor(config,feature_config)
    
    try:
        processed_data = data_loader.load_preprocessor()
        meta_data = data_loader.load_metadata()
    except FileNotFoundError:
        logger.error("Processed data not found. Please run create_preprocessor.py first.")
        return
        
    # Load Split data
    X_train = processed_data['X_train']
    y_train = processed_data['Y_train']

    X_test = processed_data['X_test']
    y_test = processed_data['Y_test']

    # dataset features
    numeric_variable = meta_data.user['split_params']['num_vars']
    categorical_variable = meta_data.user['split_params']['cat_vars']
    
    # build preprocessor
    preprocessor = data_loader.build_preprocessor(
        numeric_features=numeric_variable,                # feature_config["numeric"]
        categorical_features=categorical_variable         # feature_config["categorical"]
    )
    
    # Get feature names after preprocessing
    feature_names = data_loader._get_feature_names_out(
        numeric_variable,
        categorical_variable,
        X_train)
    logger.info(f"Feature names after preprocessing: {len(feature_names)} features")
    
    # Initialize trainer
    logger.info("\n4. Initializing model trainer...")
    trainer = ModelTrainer(config, model_params)
    
    # Store feature names in trainer for Vetiver
    trainer.feature_names = feature_names
    
    # Train all models
    logger.info("\n5. Training models with hyperparameter tuning...")
    logger.info("-"*80)
    
    results = {}
    
    if model_params['elastic_net']['enabled']:
        metrics, run_id = trainer.train_elastic_net(
            preprocessor,
            X_train, X_test, y_train, y_test,
            feature_names
        )
        results['elastic_net'] = {'metrics': metrics, 'run_id': run_id}
    
    if model_params['random_forest']['enabled']:
        metrics, run_id = trainer.train_random_forest(
            preprocessor,
            X_train, X_test, y_train, y_test,
            feature_names
        )
        results['random_forest'] = {'metrics': metrics, 'run_id': run_id}
    
    if model_params['gradient_boosting']['enabled']:
        metrics, run_id = trainer.train_gradient_boosting(
            preprocessor,
            X_train, X_test, y_train, y_test,
            feature_names
        )
        results['gradient_boosting'] = {'metrics': metrics, 'run_id': run_id}
    
    if model_params['xgboost']['enabled']:
        metrics, run_id = trainer.train_xgboost(
            preprocessor,
            X_train, X_test, y_train, y_test,
            feature_names
        )
        results['xgboost'] = {'metrics': metrics, 'run_id': run_id}
    
    # List pinned models
    logger.info("\n6. Reviewing pinned models...")
    pins_manager = PinsBoardManager(config)
    
    # Compare models
    logger.info("\n7. Comparing all models...")
    comparator = ModelComparator(config['mlflow']['experiment_name'], config['pins'])
    best_pin_name = comparator.compare_models()
    
    # Demonstrate loading best model
    logger.info("\n8. Loading best model from pins...")
    if best_pin_name and best_pin_name != 'N/A':
        loaded_model = pins_manager.load_model(best_pin_name)
        if loaded_model:
            # Make a sample prediction
            sample_pred = loaded_model.predict(X_test[:3])
            logger.info(f"Sample predictions: {sample_pred}")
            logger.info(f"Actual values: {y_test.iloc[:3].values}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info("\n📊 View MLflow UI: mlflow ui")
    logger.info("Then open http://localhost:5000")
    logger.info(f"\n📌 Pins board location: {config['pins']['board_path']}")
    logger.info("Models are versioned and ready for deployment")


if __name__ == "__main__":
    main()