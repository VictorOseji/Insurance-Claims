# ============================================================================
# FILE: scripts/prepare_data.py
# ============================================================================
"""Script to prepare and process insurance claims data"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data.data_loader import InsuranceDataLoader
from src.data.data_validator import DataValidator
from src.data.feature_engineering import FeatureEngineer


def main():
    """Main data preparation pipeline"""
    # Setup logger
    logger = setup_logger("data_preparation", "logs/data_preparation.log")
    
    logger.info("="*80)
    logger.info("Starting Data Preparation Pipeline")
    logger.info("="*80)
    
    # Load configuration
    logger.info("\n1. Loading configuration...")
    config_loader = ConfigLoader()
    config = config_loader.get_main_config()
    
    # Load feature configuration
    feature_config = config_loader.load_config("feature_config.yaml")
    
    # Load raw data
    logger.info("\n2. Loading raw data...")
    data_loader = InsuranceDataLoader(config)
    claims_data = data_loader.load_raw_data()
    
    # Validate data
    logger.info("\n3. Validating data quality...")
    validator = DataValidator(config)
    validation_results = validator.validate(claims_data)
    
    if not validation_results['validation_passed']:
        logger.error("Data validation failed. Please check data quality.")
        return
    
    logger.info("Data validation passed ✓")
    
    # Feature engineering
    logger.info("\n4. Performing feature engineering...")
    feature_engineer = FeatureEngineer(config, feature_config)
    processed_data = feature_engineer.transform(claims_data)
    
    # Save processed data
    logger.info("\n5. Saving processed data...")
    data_loader.save_processed_data(processed_data)
    
    logger.info("\n" + "="*80)
    logger.info("✓ Data Preparation Complete!")
    logger.info("="*80)
    logger.info(f"Final dataset shape: {processed_data.shape}")
    logger.info("Processed data saved to: data/processed/")


if __name__ == "__main__":
    main()
