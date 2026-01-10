# ============================================================================
# FILE: src/data/feature_engineering.py
# ============================================================================
"""Feature engineering for insurance claims"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create derived features for insurance claims prediction"""
    
    def __init__(self, config: dict, feature_config: dict):
        self.config = config
        self.feature_config = feature_config
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations"""
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        
        # Convert date columns
        df = self._convert_date_columns(df)
        
        # Handle missing values
        df = self._impute_missing_values(df)
        
        # Create derived features
        df = self._create_derived_features(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        logger.info(f"New features: {list(df.columns)}")
        
        return df
    
    def _convert_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert string columns to datetime"""
        logger.info("Converting date columns...")
        
        date_columns = self.feature_config.get('date_columns', [])
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"Converted {col} to datetime")
        
        return df
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values based on configuration"""
        logger.info("Imputing missing values...")
        
        imputation_config = self.feature_config.get('imputation', {})
        
        # Numeric imputation
        numeric_config = imputation_config.get('numeric', {})
        for col in numeric_config.get('columns', []):
            if col in df.columns:
                strategy = numeric_config.get('strategy', 'median')
                if strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                logger.info(f"Imputed {col} using {strategy}")
        
        # Categorical imputation
        categorical_config = imputation_config.get('categorical', {})
        for col in categorical_config.get('columns', []):
            if col in df.columns:
                strategy = categorical_config.get('strategy', 'most_frequent')
                if strategy == 'most_frequent':
                    df[col] = df[col].fillna(df[col].mode()[0])
                logger.info(f"Imputed {col} using {strategy}")
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all derived features"""
        logger.info("Creating derived features...")
        
        # FNOL_Delay: Hours between accident and FNOL reporting
        if 'FNOL_Date' in df.columns and 'Accident_Date' in df.columns:
            df['FNOL_Delay'] = (df['FNOL_Date'] - df['Accident_Date']) / pd.Timedelta(hours=1)
            logger.info("Created FNOL_Delay feature")
        
        # Driver_Age: Driver age at time of accident (in years)
        if 'Accident_Date' in df.columns and 'Date_of_Birth' in df.columns:
            df['Driver_Age'] = (df['Accident_Date'] - df['Date_of_Birth']).dt.days // 365
            logger.info("Created Driver_Age feature")
        
        # License_Age: Years since license issue
        if 'Accident_Date' in df.columns and 'Full_License_Issue_Date' in df.columns:
            df['License_Age'] = (df['Accident_Date'] - df['Full_License_Issue_Date']).dt.days // 365
            df['License_Age'] = df['License_Age'].abs()
            logger.info("Created License_Age feature")
        
        # Vehicle_Age: Vehicle age at time of accident
        if 'Accident_Date' in df.columns and 'Vehicle_Year' in df.columns:
            df['Vehicle_Age'] = (df['Accident_Date'].dt.year - df['Vehicle_Year']).abs()
            logger.info("Created Vehicle_Age feature")
        
        # High_Risk_Driver: Flag for high risk age groups (< 25 or > 70)
        if 'Driver_Age' in df.columns:
            df['High_Risk_Driver'] = np.where(
                (df['Driver_Age'] < 25) | (df['Driver_Age'] > 65),
                'Yes',
                'No'
            )
            logger.info("Created High_Risk_Driver feature")
        
        # Inexperienced_Driver: Flag for drivers with < 2 years license
        if 'License_Age' in df.columns:
            df['Inexperienced_Driver'] = np.where(
                df['License_Age'] < 2,
                'Yes',
                'No'
            )
            logger.info("Created Inexperienced_Driver feature")
        
        # Old_Vehicle: Flag for vehicles > 10 years old
        if 'Vehicle_Age' in df.columns:
            df['Old_Vehicle'] = np.where(
                df['Vehicle_Age'] > 10,
                'Yes',
                'No'
            )
            logger.info("Created Old_Vehicle feature")
        
        # Early_FNOL: Flag for FNOL reported within 1 day
        if 'FNOL_Delay' in df.columns:
            df['Early_FNOL'] = np.where(
                df['FNOL_Delay'] <= 24,
                'Yes',
                'No'
            )
            logger.info("Created Early_FNOL feature")
        
        # Estimated_to_Age_Ratio: Derive Estimated Amount Ratios
        if 'Estimated_Claim_Amount' in df.columns:
            df['Estimate_to_Age_Ratio'] = df['Estimated_Claim_Amount'] / (df['Vehicle_Age'] + 1)
            logger.info("Created Estimate_to_Age_Ratio feature")

        # Estimate_Bucket: Categorize the Estimate
        if 'Estimated_Claim_Amount' in df.columns:
            df['Estimate_Bucket'] = pd.cut(
                df['Estimated_Claim_Amount'],
                bins=[0,1000,5000,10000,20000,50000,np.inf], 
                labels=["0-1k","1k-5k","5k-10k","10k-20k","20k-50k","50+"],
                right = True
            )
            logger.info("Created Estimate_Bucket feature")

        # Estimate_Relative_to_Type: Group-based Normalization
        if 'Estimated_Claim_Amount' in df.columns:
            df['Type_Avg_Estimate'] = df.groupby('Claim_Type')['Estimated_Claim_Amount'].transform('mean')
            df['Estimate_Relative_to_Type'] = df['Estimated_Claim_Amount'] / df['Type_Avg_Estimate']
            logger.info("Created Estimate_Relative_to_Type feature")

        # Severity_Flag: Interaction Feature
        if 'Estimated_Claim_Amount' in df.columns:
            df['Complexity_Score'] = np.where(
                df['Traffic_Condition'].isin(['High','Severe']) & 
                (df['Weather_Condition'] == 'Stormy'), "Yes", "No")
            logger.info("Created Estimate_Relative_to_Type feature")
        
        return df
    
    def get_feature_names(self, include_original: bool = False) -> List[str]:
        """Get list of engineered feature names"""
        derived_features = [
            'FNOL_Delay',
            'Driver_Age',
            'License_Age',
            'Vehicle_Age',
            'High_Risk_Driver',
            'Inexperienced_Driver',
            'Old_Vehicle',
            'Early_FNOL',
            'Estimate_to_Age_Ratio',
            'Estimate_Bucket',
            'Estimate_Relative_to_Type',
            'Complexity_Score'
        ]
        
        if include_original:
            derived_features.extend(
                self.feature_config.get('categorical_features', []) +
                self.feature_config.get('numeric_features', [])
            )
        
        return derived_features
