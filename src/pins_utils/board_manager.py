# ============================================================================
# FILE: src/pins_utils/board_manager.py
# ============================================================================
"""Pins board management utilities"""
from pins import board_folder
from vetiver import vetiver_pin_write, VetiverModel
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class PinsBoardManager:
    """Manage pins board operations"""
    
    def __init__(self, config: dict):
        self.config = config['pins']
        self.board = self._initialize_board()
    
    def _initialize_board(self):
        """Initialize pins board"""
        board_path = self.config['board_path']
        os.makedirs(board_path, exist_ok=True)
        board = board_folder(board_path, allow_pickle_read=True)
        logger.info(f"Pins board initialized: {board_path}")
        return board
    
    def pin_model(self, vetiver_model, model_name: str, 
                  metrics: Dict[str, float], params: Optional[Dict[str, Any]] = None) -> str:
        """Pin vetiver model to board with metadata"""
        try:
            # Create pin name
            pin_name = model_name.lower().replace(" ", "_")
            
            # Pin the vetiver model
            vetiver_pin_write(self.board, vetiver_model, pin_name)
            
            # Create and save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "parameters": params if params else {},
                "model_type": model_name
            }
            
            metadata_pin_name = f"{pin_name}_metadata"
            self.board.pin_write(metadata, metadata_pin_name, type="json")
            
            logger.info(f"Model pinned successfully: {pin_name}")
            return pin_name
            
        except Exception as e:
            logger.error(f"Error pinning model {model_name}: {e}")
            return None
    
    def load_model(self, pin_name: str):
        """Load model from pins board"""
        try:
            v_models = VetiverModel.from_pin(self.board, pin_name)
            logger.info(f"Model loaded from pin: {pin_name}")
            return v_models
        except Exception as e:
            logger.error(f"Error loading model {pin_name}: {e}")
            return None
    
    def list_models(self):
        """List all pinned models"""
        pins_list = self.board.pin_list()
        model_pins = [p for p in pins_list if not p.endswith('_metadata')]
        return model_pins
