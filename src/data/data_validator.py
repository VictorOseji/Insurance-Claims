# ============================================================================
# FILE: src/data/data_validator.py
# ============================================================================
"""Data validation utilities"""
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and completeness"""
    
    def __init__(self, config: dict):
        self.config = config
        self.missing_threshold = config['data'].get('missing_threshold', 0.3)
    
    def validate(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive data validation"""
        logger.info("Starting data validation...")
        
        validation_results = {
            'shape': df.shape,
            'missing_values': self._check_missing_values(df),
            'duplicate_rows': self._check_duplicates(df),
            'data_types': self._check_data_types(df),
            'validation_passed': True
        }
        
        # Check if validation passed
        if validation_results['missing_values']['critical_missing']:
            validation_results['validation_passed'] = False
            logger.warning("Validation failed: Critical missing values detected")
        
        return validation_results
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        critical_missing = []
        for col in missing[missing > 0].index:
            pct = missing_pct[col]
            logger.info(f"Column '{col}': {missing[col]} missing ({pct:.2f}%)")
            
            if pct > self.missing_threshold * 100:
                critical_missing.append(col)
        
        return {
            'total_missing': missing.sum(),
            'columns_with_missing': missing[missing > 0].to_dict(),
            'critical_missing': critical_missing
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows"""
        duplicates = df.duplicated().sum()
        logger.info(f"Duplicate rows found: {duplicates}")
        
        return {
            'count': int(duplicates),
            'percentage': (duplicates / len(df)) * 100
        }
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict:
        """Check data types"""
        dtypes = df.dtypes.astype(str).to_dict()
        logger.info(f"Data types: {dtypes}")
        return dtypes
