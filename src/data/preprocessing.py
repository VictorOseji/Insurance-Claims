# ============================================================================
# FILE: src/data/preprocessing.py
# ============================================================================
"""Data preprocessing pipeline"""
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import pins

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Build preprocessing components for sklearn pipelines"""
    
    def __init__(self, config: dict, feature_config: dict):
        self.config = config
        self.feature_config = feature_config
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target"""
        logger.info(f"Preparing features with target: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        X = df.drop(columns=[target_col])
        y = np.log1p(df[target_col])
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str,object]:
        """Split data into train and test sets"""
        logger.info("Splitting data into train and test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        
        """ 
        It the dataset is large, divide the dictionary into two seperate list contains training and test dataset
        """
        dict_data = {
            "X_train": X_train,
            "Y_train": y_train,
            "X_test": X_test,
            "Y_test": y_test
        }

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return dict_data  # X_train, X_test, y_train, y_test

    def _get_feature_names_out(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        X: pd.DataFrame
    ) -> List[str]:
        """Get feature names after one-hot encoding
        
        Args:
            numeric_features: Original numeric feature names
            categorical_features: Original categorical feature names
            X: Original DataFrame (before transformation)
            
        Returns:
            List of feature names after transformation
        """
        feature_names = []
        
        # Add numeric feature names (they stay the same)
        feature_names.extend(numeric_features)
        
        # Add one-hot encoded feature names
        for cat_feat in categorical_features:
            if cat_feat in X.columns:
                # Get unique values for this categorical feature
                unique_values = sorted(X[cat_feat].dropna().unique())
                for val in unique_values:
                    # Create feature name: feature_value
                    feature_names.append(f"{cat_feat}_{val}")
        
        logger.info(f"Generated {len(feature_names)} feature names")
        
        return feature_names
    
    def build_preprocessor(
        self,
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline"""
        logger.info("Building Preprocessing ColumnTransformer...")
        
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='drop'
        )
        
        logger.info(f"Pipeline created with {len(numeric_features)} numeric and "
                   f"{len(categorical_features)} categorical features")
        
        return preprocessor

    ##########################################################################################
    def save_preprocessor(self, df: dict, num_vars: list, cat_vars: list, pin_name: str = "train_test_data_split"):
        """Save Train-Test Dataset"""
        output_path = Path(self.config['data']['interim_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = generate_dataset_metadata(
            artifact_stage="split",
            target="Ultimate_Claim_Amount",
            schema_version="v3",
            train_df= df["X_train"],
            test_df= df["X_test"],
            split_method="random-based sampling",
            split_params={"test_size": 0.2,
                          "random_state": 42,
                          "num_vars": num_vars,
                          "cat_vars": cat_vars}
        )

        board = pins.board_folder( output_path, versioned=True )
        board.pin_write( df, name=pin_name, type = "joblib",
            title = "train-test Insurance Claims Dataset", 
            description = "This dictionary contains train and test dataset for modeling",
            metadata = metadata )

        logger.info(
            f"Train/Test Insurance Claims dataset pinned as '{pin_name}' "
            f"with versioning enabled")

    def load_preprocessor(self, pin_name: str = "train_test_data_split"):
        """Load fitted preprocessor"""
        input_path = Path(self.config['data']['interim_path'])
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found at: {input_path}")

        board = pins.board_folder( input_path, versioned=True, allow_pickle_read=True )
        df_processed = board.pin_read( "train_test_data_split",)

        logger.info(f"Data Split loaded from: {input_path / pin_name}")
        return df_processed

    def load_metadata(self, pin_name: str = "train_test_data_split"):
        """Load metadata from preprocessor"""
        input_path = Path(self.config['data']['interim_path'])
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found at: {input_path}")

        board = pins.board_folder( input_path, versioned=True )
        df_metadata = board.pin_meta( "train_test_data_split")

        logger.info(f"Metadata loaded from: {input_path / pin_name}")
        return df_metadata

##################################################################################
def build_training_pipeline(
        preprocessor: ColumnTransformer,
        model
    ) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model)
            ]
        )

def _schema_fingerprint(df: pd.DataFrame) -> Dict[str, str]:
    schema = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    payload = json.dumps(schema, sort_keys=True)
    return {
        "schema_hash": hashlib.md5(payload.encode()).hexdigest(),
        "n_columns": len(schema["columns"])
    }

def generate_dataset_metadata(
    *,
    artifact_stage: str,
    target: str,
    schema_version: str,
    data: Optional[pd.DataFrame] = None,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    split_method: Optional[str] = None,
    split_params: Optional[dict] = None,
    preprocessing_steps: Optional[list] = None
) -> dict:
    """
    Generate deterministic metadata for dataset artifacts.
        
    artifact_stage:
        - "split"       → raw → train/test
        - "processed"   → feature-engineered dataset
    """

    metadata = {
        # Identity
        "artifact_type": "dataset",
        "artifact_stage": artifact_stage,
        "created_at_utc": datetime.utcnow().isoformat(),

        # Modeling contract
        "target": target,
        "schema_version": schema_version,
    }

    # ---------------------------
    # SPLIT STAGE METADATA
    # ---------------------------
    if artifact_stage == "split":
        if train_df is None or test_df is None:
            raise ValueError("train_df and test_df required for split metadata")

        train_schema = _schema_fingerprint(train_df)
        test_schema = _schema_fingerprint(test_df)

        metadata.update({
            # Split logic
            "split_method": split_method,
            "split_params": split_params,

            # Row counts
            "train_rows": len(train_df),
            "test_rows": len(test_df),

            # Schema fingerprints
            "train_schema_hash": train_schema["schema_hash"],
            "test_schema_hash": test_schema["schema_hash"],

            # Sanity checks
            "schema_consistent": train_schema["schema_hash"] == test_schema["schema_hash"]
        })

    # ---------------------------
    # PROCESSED STAGE METADATA
    # ---------------------------
    elif artifact_stage == "processed":
        if data is None:
            raise ValueError("data required for processed metadata")

        schema = _schema_fingerprint(data)

        metadata.update({
            # Shape
            "rows": len(data),
            "columns": schema["n_columns"],
            "schema_hash": schema["schema_hash"],

            # Feature diagnostics
            "n_numeric_features": data.select_dtypes("number").shape[1],
            "n_categorical_features": data.select_dtypes("object").shape[1],

            # Preprocessing lineage
            "preprocessing_steps": preprocessing_steps,
        })

    else:
        raise ValueError(f"Unknown artifact_stage: {artifact_stage}")

    return metadata
