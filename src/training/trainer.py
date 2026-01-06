# ============================================================================
# FILE: src/training/trainer.py
# ============================================================================
"""Main training orchestration"""
import mlflow
# import numpy as np
# from typing import Dict, Any
import logging
from ..models.elastic_net import ElasticNetModel
from ..models.random_forest import RandomForestModel
from ..models.gradient_boosting import GradientBoostingModel
from ..models.xgboost import XGBoostModel
from ..evaluation.metrics import ModelEvaluator
from ..mlflow_utils.experiment_tracker import ExperimentTracker
from ..vetiver_utils.model_wrapper import VetiverModelWrapper
from ..pins_utils.board_manager import PinsBoardManager
from .hyperparameter_tuner import HyperparameterTuner
from ..data.preprocessing import build_training_pipeline

from sklearn.compose import ColumnTransformer


logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrate model training pipeline"""
    
    def __init__(self, config: dict, model_params: dict):
        self.config = config
        self.model_params = model_params
        self.experiment_tracker = ExperimentTracker(config)
        self.pins_manager = PinsBoardManager(config)
        self.tuner = HyperparameterTuner(config)
        self.evaluator = ModelEvaluator()
        self.vetiver_wrapper = VetiverModelWrapper()
    
    def train_elastic_net(self, 
        preprocessor: ColumnTransformer, 
        X_train, X_test, y_train, y_test,   
        features_names: list[str]
    ):
        """Train Linear Regression model"""
        with mlflow.start_run(run_name="ElasticNet_Tuned") as run:

            # ---------------------------
            # 1. Build base model
            # ---------------------------
            base_model = ElasticNetModel.build_model(self.config)
            pipeline = build_training_pipeline( preprocessor=preprocessor, model=base_model )

            # ---------------------------
            # 2. Hyperparameter tuning
            # ---------------------------
            param_grid = self.model_params["elastic_net"]["param_grid"]

            best_pipeline, best_params = self.tuner.tune(
                pipeline,
                param_grid,
                X_train,
                y_train
            )

            # ---------------------------
            # 3. Evaluate
            # ---------------------------
            y_pred = best_pipeline.predict(X_test)
            metric_score = self.evaluator.evaluate_model(y_test, y_pred)

            # ---------------------------
            # 4. MLflow logging
            # ---------------------------
            self.experiment_tracker.set_tags({
                "model_type": "ElastiNet Model",
                "run_id": run.info.run_id
            })

            self.experiment_tracker.log_params(best_params)
            self.experiment_tracker.log_metrics(metric_score)
            self.experiment_tracker.log_model(best_pipeline)

            # ---------------------------
            # 5. Vetiver + Pins
            # ---------------------------
            #feature_names = getattr(preprocessor, "get_feature_names_out", None)
            feature_names = features_names if features_names else None

            v_models = self.vetiver_wrapper.create_vetiver_model(
                model=best_pipeline,
                model_name="elastic_net",
                X_train=X_train,
                feature_names=feature_names
            )

            pin_name = self.pins_manager.pin_model(
                vetiver_model=v_models,
                model_name="ElasticNet",
                metrics=metric_score,
                params=best_params
            )

            mlflow.set_tag("vetiver_pin_name", pin_name)

            logger.info(
                f"ElasticNet - R²: {metric_score['r2']:.4f}, "
                f"RMSE: {metric_score['rmse']:.4f}"
            )

            return metric_score, run.info.run_id
    
    def train_random_forest(self, 
        preprocessor: ColumnTransformer, 
        X_train, X_test, y_train, y_test,   
        features_names: list[str]
    ):
        """Train Random Forest with tuning"""
        with mlflow.start_run(run_name="Random_Forest_Tuned") as run:

            # ---------------------------
            # 1. Build base model
            # ---------------------------
            base_model = RandomForestModel.build_model(self.config)
            pipeline = build_training_pipeline( preprocessor=preprocessor, model=base_model )

            # ---------------------------
            # 2. Hyperparameter tuning
            # ---------------------------
            param_grid = self.model_params["random_forest"]["param_grid"]

            best_pipeline, best_params = self.tuner.tune(
                pipeline,
                param_grid,
                X_train,
                y_train
            )

            # ---------------------------
            # 3. Evaluate
            # ---------------------------
            y_pred = best_pipeline.predict(X_test)
            metric_score = self.evaluator.evaluate_model(y_test, y_pred)

            # ---------------------------
            # 4. MLflow logging
            # ---------------------------
            self.experiment_tracker.set_tags({
                "model_type": "RandomForest",
                "run_id": run.info.run_id
            })

            self.experiment_tracker.log_params(best_params)
            self.experiment_tracker.log_metrics(metric_score)
            self.experiment_tracker.log_model(best_pipeline)

            # ---------------------------
            # 5. Vetiver + Pins
            # ---------------------------
            # feature_names = getattr(preprocessor, "get_feature_names_out", None)
            feature_names = features_names if features_names else None

            v_models = self.vetiver_wrapper.create_vetiver_model(
                model=best_pipeline,
                model_name="random_forest",
                X_train=X_train,
                feature_names=feature_names
            )

            pin_name = self.pins_manager.pin_model(
                vetiver_model=v_models,
                model_name="random_forest",
                metrics=metric_score,
                params=best_params
            )

            mlflow.set_tag("vetiver_pin_name", pin_name)

            logger.info(
                f"Random Forest - R²: {metric_score['r2']:.4f}, "
                f"RMSE: {metric_score['rmse']:.4f}"
            )

            return metric_score, run.info.run_id
    
    def train_gradient_boosting(self, 
        preprocessor: ColumnTransformer, 
        X_train, X_test, y_train, y_test,   
        features_names: list[str]
    ):
        """Train Gradient Boosting using unified pipeline + tuning"""

        with mlflow.start_run(run_name="Gradient_Boosting_Tuned") as run:

            # ---------------------------
            # 1. Build base model
            # ---------------------------
            base_model = GradientBoostingModel.build_model(self.config)
            pipeline = build_training_pipeline( preprocessor=preprocessor, model=base_model )

            # ---------------------------
            # 2. Hyperparameter tuning
            # ---------------------------
            param_grid = self.model_params["gradient_boosting"]["param_grid"]

            best_pipeline, best_params = self.tuner.tune(
                pipeline,
                param_grid,
                X_train,
                y_train
            )

            # ---------------------------
            # 3. Evaluate
            # ---------------------------
            y_pred = best_pipeline.predict(X_test)
            metric_score = self.evaluator.evaluate_model(y_test, y_pred)

            # ---------------------------
            # 4. MLflow logging
            # ---------------------------
            self.experiment_tracker.set_tags({
                "model_type": "GradientBoosting",
                "run_id": run.info.run_id
            })

            self.experiment_tracker.log_params(best_params)
            self.experiment_tracker.log_metrics(metric_score)
            self.experiment_tracker.log_model(best_pipeline)

            # ---------------------------
            # 5. Vetiver + Pins
            # ---------------------------
            # feature_names = getattr(preprocessor, "get_feature_names_out", None)
            feature_names = features_names if features_names else None

            v_models = self.vetiver_wrapper.create_vetiver_model(
                model=best_pipeline,
                model_name="gradient_boosting",
                X_train=X_train,
                feature_names=feature_names
            )

            pin_name = self.pins_manager.pin_model(
                vetiver_model=v_models,
                model_name="GradientBoosting",
                metrics=metric_score,
                params=best_params
            )

            mlflow.set_tag("vetiver_pin_name", pin_name)

            logger.info(
                f"Gradient Boosting - R²: {metric_score['r2']:.4f}, "
                f"RMSE: {metric_score['rmse']:.4f}"
            )

            return metric_score, run.info.run_id

    
    def train_xgboost(self, 
        preprocessor: ColumnTransformer, 
        X_train, X_test, y_train, y_test,   
        features_names: list[str]
    ):
        """Train XGBoost with tuning"""
        with mlflow.start_run(run_name="Xgboost_Tuned") as run:

            # ---------------------------
            # 1. Build base model
            # ---------------------------
            base_model = XGBoostModel.build_model(self.config)
            pipeline = build_training_pipeline( preprocessor=preprocessor, model=base_model )

            # ---------------------------
            # 2. Hyperparameter tuning
            # ---------------------------
            param_grid = self.model_params["xgboost"]["param_grid"]

            best_pipeline, best_params = self.tuner.tune(
                pipeline,
                param_grid,
                X_train,
                y_train
            )

            # ---------------------------
            # 3. Evaluate
            # ---------------------------
            y_pred = best_pipeline.predict(X_test)
            metric_score = self.evaluator.evaluate_model(y_test, y_pred)

            # ---------------------------
            # 4. MLflow logging
            # ---------------------------
            self.experiment_tracker.set_tags({
                "model_type": "Xgboost",
                "run_id": run.info.run_id
            })

            self.experiment_tracker.log_params(best_params)
            self.experiment_tracker.log_metrics(metric_score)
            self.experiment_tracker.log_model(best_pipeline)

            # ---------------------------
            # 5. Vetiver + Pins
            # ---------------------------
            # feature_names = getattr(preprocessor, "get_feature_names_out", None)
            feature_names = features_names if features_names else None

            v_models = self.vetiver_wrapper.create_vetiver_model(
                model=best_pipeline,
                model_name="xgboost",
                X_train=X_train,
                feature_names=feature_names
            )

            pin_name = self.pins_manager.pin_model(
                vetiver_model=v_models,
                model_name="Xgboost",
                metrics=metric_score,
                params=best_params
            )

            mlflow.set_tag("vetiver_pin_name", pin_name)

            logger.info(
                f"Xgboost - R²: {metric_score['r2']:.4f}, "
                f"RMSE: {metric_score['rmse']:.4f}"
            )

            return metric_score, run.info.run_id

