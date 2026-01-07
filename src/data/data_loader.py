# ============================================================================
# FILE: src/data/data_loader.py
# ============================================================================
"""Data loading utilities for insurance claims"""
import pandas as pd
import logging
import sys
from pathlib import Path
from vetiver import VetiverModel
import pins

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import generate_dataset_metadata
from src.data.preprocessing import _schema_fingerprint

logger = logging.getLogger(__name__)


class InsuranceDataLoader:
    """Load and merge insurance claims data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_config = config['data']
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load and merge claims and policy data"""
        logger.info("Loading raw data files...")
        
        # Load datasets
        claims_path = self.data_config['claims_file']
        policy_path = self.data_config['policy_file']
        
        logger.info(f"Loading claims data from: {claims_path}")
        claims = pd.read_csv(claims_path)
        
        logger.info(f"Loading policy data from: {policy_path}")
        policy = pd.read_csv(policy_path)
        
        # Merge datasets
        logger.info("Merging claims and policy data...")
        claims_data = pd.merge(
            claims, 
            policy, 
            on=['Policy_ID', 'Customer_ID']
        )
        
        logger.info(f"Merged data shape: {claims_data.shape}")
        logger.info(f"Columns: {list(claims_data.columns)}")
        
        return claims_data
    
    def save_processed_data(self, df: pd.DataFrame, pin_name: str = "processed_claims"):
        """Version processed data using pins"""
        output_path = Path(self.data_config['processed_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = generate_dataset_metadata(
            artifact_stage="processed",
            target="Ultimate_Claim_Amount",
            schema_version="v3",
            data=df,
            preprocessing_steps = [
                "median_imputation_numeric",
                "standard_scaling",
                "one_hot_encoding for character & Factor Variables"
            ]
        )

        board = pins.board_folder( output_path, versioned=True )
        board.pin_write( df, name=pin_name, type = "parquet",
            title = "Processed Insurance Claims Dataset", 
            description = "This dataset contains Insurance Claims dataset with added features",
            metadata = metadata )

        logger.info(
            f"Processed data pinned as '{pin_name}' "
            f"with versioning enabled")
      
    def load_processed_data(self, filename: str = "processed_claims") -> pd.DataFrame:
        """Load previously processed data"""
        input_path = Path(self.data_config['processed_path'])
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found at: {input_path}")

        board = pins.board_folder( input_path, versioned=True )
        df_processed = board.pin_read( "processed_claims")

        logger.info(f"Loading processed data from: {input_path}")

        return df_processed