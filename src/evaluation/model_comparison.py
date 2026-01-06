# ============================================================================
# FILE: src/evaluation/model_comparison.py
# ============================================================================
"""Model comparison utilities"""
import mlflow
import logging
from datetime import datetime, timezone
from typing import Optional
import pins 

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare and analyze model performance"""
    
    def __init__(self, experiment_name: str, config: dict):
        self.experiment_name = experiment_name
        board_path = config['board_path']
        self.board = pins.board_folder(board_path, allow_pickle_read=True)
    
    def compare_models(self) -> Optional[str]:
        """
        Compare all MLflow-tracked models, identify the best one by R²,
        and promote it as the Vetiver production model via a Pins alias.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if not experiment:
            logger.error(f"Experiment {self.experiment_name} not found")
            return None
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            logger.warning("No runs found in experiment")
            return None
        
        # Sort by R² score
        runs_sorted = runs.sort_values('metrics.r2', ascending=False)
        
        # ---- Display Comparison Summary ----
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Pin Name':<25}")
        print("-"*80)
        
        for _, run in runs_sorted.iterrows():
            model_name = run.get('tags.model_type') or 'Unknown'
            r2 = run.get('metrics.r2') or 0.0
            rmse = run.get('metrics.rmse') or 0.0
            mae = run.get('metrics.mae') or 0.0
            pin_name = run.get('tags.vetiver_pin_name') or 'N/A'
            print(f"{model_name:<25} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f} {pin_name:<20}")
        
        print("="*90)

        # ---- Identify best run ----
        best_run = runs_sorted.iloc[0]
        
        best_r2 = best_run.get("metrics.r2") or 0.0
        best_model_type = best_run.get("tags.model_type") or "Unknown"
        best_pin_name = best_run.get('tags.vetiver_pin_name')
        best_run_id = best_run.get("run_id")

        if not best_pin_name:
            logger.error("Best run does not have a vetiver_pin_name tag")
            return None

        # ---- Load best model from Pins ----
        logger.info(f"Loading best model from pin: {best_pin_name}")
        best_model = self.board.pin_read(best_pin_name)

        # ---- Promote best model via alias ----
        logger.info("Promoting best model to production alias: claims_model_best")

        self.board.pin_write(
            best_model,
            name="claims_model_best",
            versioned = False,
            description="The best model created after parameter tuning",
            type="joblib",
            metadata={
                "source_pin": best_pin_name,
                "model_type": best_model_type,
                "selection_metric": "r2",
                "selection_value": best_r2,
                "mlflow_run_id": best_run_id,
                "selected_at": datetime.now(timezone.utc).isoformat(timespec='seconds'),
                "selected_by": "mlflow_r2_max"
            }
        )

        # ---- Final summary ----
        print("\n" + "=" * 90)
        print("🏆 BEST MODEL PROMOTED TO PRODUCTION")
        print("=" * 90)
        print(f"Model Type      : {best_model_type}")
        print(f"R² Score        : {best_r2:.4f}")
        print(f"MLflow Run ID   : {best_run_id}")
        print(f"Source Pin      : {best_pin_name}")
        print("Production Pin  : claims_model_best")
        print("=" * 90 + "\n")

        # Always return the stable Vetiver alias
        return "claims_model_best"