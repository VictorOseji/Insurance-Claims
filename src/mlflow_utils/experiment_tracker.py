# ============================================================================
# FILE: src/mlflow_utils/experiment_tracker.py
# ============================================================================
"""MLflow experiment tracking utilities"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
from typing import Dict, Any
from xgboost import XGBRegressor, XGBClassifier

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Manage MLflow experiment tracking"""
    
    def __init__(self, config: dict):
        self.config = config['mlflow']
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(self.config['tracking_uri'])
        mlflow.set_experiment(self.config['experiment_name'])

        # log_models=False ensures it doesn't conflict with your custom log_model method
        mlflow.sklearn.autolog(log_models=False)
        mlflow.xgboost.autolog(log_models=False)

        logger.info(f"MLflow tracking and autologging initialized: {self.config['experiment_name']}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow"""
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log model to MLflow with automatic type detection
        
        Args:
            model: Trained model (sklearn or xgboost)
            artifact_path: Path within MLflow run to save model
        """
        try:
            # Check if it's an XGBoost model
            if isinstance(model, (XGBRegressor, XGBClassifier)):
                # For XGBoost models from GridSearchCV, use sklearn logging
                # to avoid _estimator_type error
                if not hasattr(model, '_estimator_type'):
                    logger.warning(
                        "XGBoost model missing _estimator_type, using sklearn logging"
                    )
                    mlflow.sklearn.log_model(model, artifact_path)
                else:
                    mlflow.xgboost.log_model(model, artifact_path)
            else:
                # For all other sklearn models
                mlflow.sklearn.log_model(model, artifact_path)
                
            logger.info(f"Model logged to MLflow at: {artifact_path}")
            
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {e}")
            # Fallback to sklearn logging
            logger.info("Attempting fallback to sklearn logging...")
            mlflow.sklearn.log_model(model, artifact_path)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the run"""
        for tag_name, tag_value in tags.items():
            mlflow.set_tag(tag_name, tag_value)

