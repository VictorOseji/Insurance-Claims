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
        self.config_dir = Path(config_dir)
    
    def load_config(self, filename: str) -> Dict[Any, Any]:
        """Load a YAML configuration file"""
        config_path = self.config_dir / filename
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_main_config(self) -> Dict[Any, Any]:
        """Load main configuration"""
        return self.load_config("config.yaml")
    
    def get_model_params(self) -> Dict[Any, Any]:
        """Load model parameters"""
        return self.load_config("model_params.yaml")

