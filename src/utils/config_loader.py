# ============================================================================
# FILE: src/utils/config_loader.py
# ============================================================================
"""Configuration loading utilities"""
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        config_path = Path(config_dir)

        if not config_path.is_absolute():
            # Get project root: src/utils/config_loader.py -> src -> project_root
            project_root = Path(__file__).resolve().parent.parent.parent
            self.config_dir = project_root / config_dir
        else:
            self.config_dir = config_path
    
    def load_config(self, filename: str) -> Dict[Any, Any]:
        """Load a YAML configuration file"""
        config_path = self.config_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Config directory: {self.config_dir}"
            )

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_main_config(self) -> Dict[Any, Any]:
        """Load main configuration"""
        return self.load_config("config.yaml")
    
    def get_model_params(self) -> Dict[Any, Any]:
        """Load model parameters"""
        return self.load_config("model_params.yaml")

