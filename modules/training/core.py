"""
Core Training Module for Microspore Phenotyping
Contains main training functions and model management.
Implements DRY principle - uses centralized config and utils modules.
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import centralized configuration
from ..config import (
    YOLO_MODELS,
    get_all_models,
    get_data_yaml_path,
    get_trained_models_path,
)

# Import consolidated utilities
from ..utils import generate_experiment_name, check_gpu, ensure_dir

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics or torch not installed. Run setup_conda_training.sh first.")


class YOLOTrainer:
    """
    YOLO Training Manager for Microspore Phenotyping.
    Handles model training, validation, and export.
    Uses centralized configuration and utilities.
    """
    
    def __init__(self, 
                 model_path: str = "yolov8s.pt",
                 data_yaml: Optional[str] = None,
                 project_dir: Optional[str] = None):
        """
        Initialize YOLO Trainer.
        
        Args:
            model_path: Path to YOLO model weights
            data_yaml: Path to dataset configuration (defaults to config)
            project_dir: Output directory for trained models (defaults to config)
        """
        self.model_path = model_path
        self.data_yaml = data_yaml or str(get_data_yaml_path())
        self.project_dir = Path(project_dir) if project_dir else get_trained_models_path()
        ensure_dir(self.project_dir)
        
        self.model = None
        self.results = None
        self.training_config = {}
        
    def load_model(self) -> 'YOLO':
        """Load YOLO model"""
        self.model = YOLO(self.model_path)
        return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        if self.model is None:
            self.load_model()
        
        info = {
            "model_type": self.model_path,
            "task": getattr(self.model, 'task', 'detect'),
            "device": str(next(self.model.model.parameters()).device) if hasattr(self.model, 'model') else 'unknown',
        }
        
        # Get parameter count
        if hasattr(self.model, 'model'):
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            info["total_parameters"] = total_params
            info["trainable_parameters"] = trainable_params
            info["parameters_millions"] = round(total_params / 1e6, 2)
        
        return info
    
    def train(self, **kwargs) -> Any:
        """
        Train the model with given parameters.
        
        Args:
            **kwargs: Training parameters (epochs, batch, imgsz, etc.)
            
        Returns:
            Training results
        """
        if self.model is None:
            self.load_model()
        
        # Store training config
        self.training_config = {
            "data": self.data_yaml,
            "project": str(self.project_dir),
            **kwargs
        }
        
        # Generate experiment name if not provided - use centralized function
        if "name" not in kwargs:
            kwargs["name"] = generate_experiment_name(
                model_name=self.model_path,
                epochs=kwargs.get("epochs", 100),
                batch_size=kwargs.get("batch", 16),
                img_size=kwargs.get("imgsz", 640),
            )
        
        # Train
        self.results = self.model.train(
            data=self.data_yaml,
            project=str(self.project_dir),
            **kwargs
        )
        
        return self.results
    
    def validate(self, model_path: Optional[str] = None) -> Any:
        """
        Validate the model
        
        Args:
            model_path: Path to model weights (uses best.pt from training if None)
            
        Returns:
            Validation results
        """
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available. Train or provide model_path.")
        
        return model.val(data=self.data_yaml)
    
    def export(self, 
               model_path: Optional[str] = None,
               format: str = "onnx",
               **kwargs) -> str:
        """
        Export model to different formats
        
        Args:
            model_path: Path to model weights
            format: Export format (onnx, torchscript, tflite, etc.)
            
        Returns:
            Path to exported model
        """
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available.")
        
        return model.export(format=format, **kwargs)


class DatasetManager:
    """
    Dataset Management for YOLO Training
    Handles data.yaml configuration and dataset statistics
    """
    
    def __init__(self, dataset_path: str = "Dataset"):
        self.dataset_path = Path(dataset_path)
        self.data_yaml_path = self.dataset_path / "data.yaml"
        self.classes_path = self.dataset_path / "classes.txt"
        
    def load_classes(self) -> List[str]:
        """Load class names from classes.txt"""
        if not self.classes_path.exists():
            raise FileNotFoundError(f"classes.txt not found at {self.classes_path}")
        
        with open(self.classes_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            "train": {"images": 0, "labels": 0},
            "test": {"images": 0, "labels": 0},
            "classes": [],
            "class_distribution": {}
        }
        
        # Load classes
        try:
            stats["classes"] = self.load_classes()
        except FileNotFoundError:
            pass
        
        # Count files
        for split in ["Train", "Test"]:
            split_path = self.dataset_path / split
            if split_path.exists():
                images = list(split_path.glob("*.jpg")) + list(split_path.glob("*.png"))
                labels = list(split_path.glob("*.txt"))
                stats[split.lower()]["images"] = len(images)
                stats[split.lower()]["labels"] = len(labels)
        
        return stats
    
    def create_data_yaml(self, 
                         train_path: str = "Train",
                         val_path: str = "Test") -> str:
        """
        Create or update data.yaml configuration
        
        Returns:
            Path to created data.yaml
        """
        classes = self.load_classes()
        
        config = {
            "path": ".",
            "train": train_path,
            "val": val_path,
            "names": {i: name for i, name in enumerate(classes)},
            "nc": len(classes)
        }
        
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return str(self.data_yaml_path)


def get_available_models() -> Dict[str, List[str]]:
    """
    Get list of available YOLO models.
    Wrapper for centralized YOLO_MODELS config (DRY principle).
    
    Returns:
        Dictionary of model variants by version
    """
    return YOLO_MODELS.copy()


# Re-export check_gpu from utils for backwards compatibility
# The actual implementation is in utils.py (single source of truth)
__all__ = [
    'YOLOTrainer',
    'DatasetManager', 
    'check_gpu',  # Re-exported from utils
    'get_available_models',
]
