"""
Configuration Module for Microspore Phenotyping
Centralized configuration constants and default values.
Implements DRY principle for shared settings across all modules.
"""

from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

# Get the base directory (C_TRAINING - grandparent of config folder)
# Path: modules/config/config.py -> modules/config -> modules -> C_TRAINING
BASE_DIR = Path(__file__).parent.parent.parent
MODULES_DIR = Path(__file__).parent.parent

# Default directory paths (relative to BASE_DIR)
DEFAULT_DATASETS_DIR = "TRAINING_WD"  # Parent folder containing all datasets
DEFAULT_DATASET_NAME = "Dataset_1"  # Default dataset name
DEFAULT_DATASET_DIR = f"{DEFAULT_DATASETS_DIR}/{DEFAULT_DATASET_NAME}"
DEFAULT_WEIGHTS_DIR = "modules/yolo_models_weights"
DEFAULT_TRAINED_MODELS_DIR = "trained_models_output"
DEFAULT_DATA_YAML = f"{DEFAULT_DATASETS_DIR}/{DEFAULT_DATASET_NAME}/data.yaml"
DEFAULT_CLASSES_FILE = f"{DEFAULT_DATASETS_DIR}/{DEFAULT_DATASET_NAME}/classes.txt"

def get_dataset_path() -> Path:
    """Get absolute path to dataset directory."""
    return BASE_DIR / DEFAULT_DATASET_DIR


def get_weights_path() -> Path:
    """Get absolute path to weights directory."""
    return BASE_DIR / DEFAULT_WEIGHTS_DIR


def get_trained_models_path() -> Path:
    """Get absolute path to trained models directory."""
    return BASE_DIR / DEFAULT_TRAINED_MODELS_DIR


def get_data_yaml_path() -> Path:
    """Get absolute path to data.yaml file."""
    return BASE_DIR / DEFAULT_DATA_YAML


# =============================================================================
# YOLO MODEL CONFIGURATION
# =============================================================================

# Centralized model definitions - Single Source of Truth
YOLO_MODELS: Dict[str, List[str]] = {
    "yolov5": ["yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt"],
    "yolov8": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    "yolov9": ["yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt"],
    "yolov10": ["yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt"],
    "yolo11": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
}

# Model size variants in order from smallest to largest
MODEL_SIZES = ["n", "s", "m", "l", "x"]  # nano, small, medium, large, xlarge

# Special cases for size naming (yolov9 uses different names)
YOLOV9_SIZE_MAP = {"n": "t", "s": "s", "m": "m", "l": "c", "x": "e"}  # t=tiny, c=compact, e=extended


def get_all_models() -> List[str]:
    """Get flat list of all available YOLO models."""
    return [model for models in YOLO_MODELS.values() for model in models]


def get_models_by_version(version: str) -> List[str]:
    """
    Get models for a specific YOLO version.
    
    Args:
        version: YOLO version string (e.g., 'yolov8', 'yolo11')
        
    Returns:
        List of model names for that version
    """
    return YOLO_MODELS.get(version.lower(), [])


def get_model_version(model_name: str) -> str:
    """
    Extract YOLO version from model name.
    
    Args:
        model_name: Model filename (e.g., 'yolov8s.pt')
        
    Returns:
        Version string (e.g., 'yolov8')
    """
    name = model_name.lower().replace(".pt", "")
    for version in YOLO_MODELS.keys():
        if name.startswith(version):
            return version
    return "unknown"


# =============================================================================
# TRAINING DEFAULT PARAMETERS
# =============================================================================

@dataclass
class TrainingDefaults:
    """Default training hyperparameters (aligned with microspores.cfg)."""
    # Core parameters
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640  # microspores.cfg uses 608
    patience: int = 50
    workers: int = 4
    
    # Learning rate & optimizer (from microspores.cfg)
    lr0: float = 0.001  # from microspores.cfg: learning_rate=0.001
    lrf: float = 0.01
    momentum: float = 0.949  # from microspores.cfg: momentum=0.949
    weight_decay: float = 0.0005  # from microspores.cfg: decay=0.0005
    optimizer: str = "auto"
    
    # Augmentation (from microspores.cfg)
    hsv_h: float = 0.1  # from microspores.cfg: hue=0.1
    hsv_s: float = 0.7  # microspores.cfg: saturation=1.5 (uses multiplier)
    hsv_v: float = 0.4  # microspores.cfg: exposure=1.5 (uses multiplier)
    degrees: float = 0.0  # from microspores.cfg: angle=0
    translate: float = 0.1
    scale: float = 0.5  # similar to microspores.cfg: jitter=0.3
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0  # from microspores.cfg: mosaic=1
    mixup: float = 0.0
    copy_paste: float = 0.0
    
    # Warmup parameters (equivalent to microspores.cfg: burn_in=1000)
    warmup_epochs: float = 3.0  # burn_in=1000 batches ≈ 3 epochs
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Loss weights (from microspores.cfg: iou_normalizer, cls_normalizer)
    box_loss: float = 7.5  # box loss gain
    cls_loss: float = 0.5  # classification loss gain
    dfl_loss: float = 1.5  # distribution focal loss gain
    
    # IoU and NMS (from microspores.cfg: iou_thresh, beta_nms)
    iou_threshold: float = 0.7  # microspores.cfg: iou_thresh=0.213
    label_smoothing: float = 0.0
    
    # Mosaic and training modes
    close_mosaic: int = 10  # disable mosaic for final epochs
    multi_scale: bool = False
    rect: bool = False
    
    # Model settings
    pretrained: bool = True
    resume: bool = False
    cache: str = "disk"
    amp: bool = True
    freeze: int = 0
    
    # Device
    device: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YOLO training."""
        return {
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.img_size,
            "patience": self.patience,
            "workers": self.workers,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            # Warmup parameters
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            # Loss weights
            "box": self.box_loss,
            "cls": self.cls_loss,
            "dfl": self.dfl_loss,
            # IoU and training modes
            "iou": self.iou_threshold,
            "label_smoothing": self.label_smoothing,
            "close_mosaic": self.close_mosaic,
            "multi_scale": self.multi_scale,
            "rect": self.rect,
            # Model settings
            "pretrained": self.pretrained,
            "resume": self.resume,
            "cache": self.cache,
            "amp": self.amp,
            "freeze": self.freeze,
            "device": self.device,
        }


# Singleton default instance
TRAINING_DEFAULTS = TrainingDefaults()


# =============================================================================
# GPU CONFIGURATION
# =============================================================================

# VRAM-based batch size recommendations
VRAM_BATCH_SIZE_MAP = {
    4: 4,    # 4GB VRAM -> batch 4
    6: 8,    # 6GB VRAM -> batch 8
    8: 8,    # 8GB VRAM -> batch 8
    12: 16,  # 12GB VRAM -> batch 16
    16: 24,  # 16GB VRAM -> batch 24
    24: 32,  # 24GB VRAM -> batch 32
}


def get_recommended_batch_size(vram_gb: int) -> int:
    """
    Get recommended batch size based on VRAM.
    
    Args:
        vram_gb: GPU VRAM in gigabytes
        
    Returns:
        Recommended batch size
    """
    for threshold, batch in sorted(VRAM_BATCH_SIZE_MAP.items()):
        if vram_gb <= threshold:
            return batch
    return 32  # Default for large GPUs


# =============================================================================
# IMAGE EXTENSIONS
# =============================================================================

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
LABEL_EXTENSION = ".txt"


# =============================================================================
# EXPORT FORMATS
# =============================================================================

EXPORT_FORMATS = {
    "onnx": "ONNX (Open Neural Network Exchange)",
    "torchscript": "TorchScript",
    "openvino": "OpenVINO",
    "engine": "TensorRT",
    "coreml": "CoreML (iOS)",
    "tflite": "TFLite (Android/Edge)",
    "pb": "TensorFlow SavedModel",
    "paddle": "PaddlePaddle",
}


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
LOG_FORMAT_DETAILED = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# Export public API
__all__ = [
    # Directories
    "BASE_DIR",
    "MODULES_DIR",
    "DEFAULT_DATASETS_DIR",
    "DEFAULT_DATASET_NAME",
    "DEFAULT_DATASET_DIR",
    "DEFAULT_WEIGHTS_DIR",
    "DEFAULT_TRAINED_MODELS_DIR",
    "DEFAULT_DATA_YAML",
    "DEFAULT_CLASSES_FILE",
    "get_dataset_path",
    "get_weights_path",
    "get_trained_models_path",
    "get_data_yaml_path",
    # Models
    "YOLO_MODELS",
    "MODEL_SIZES",
    "get_all_models",
    "get_models_by_version",
    "get_model_version",
    # Training
    "TrainingDefaults",
    "TRAINING_DEFAULTS",
    # GPU
    "VRAM_BATCH_SIZE_MAP",
    "get_recommended_batch_size",
    # Extensions
    "IMAGE_EXTENSIONS",
    "LABEL_EXTENSION",
    "EXPORT_FORMATS",
    # Logging
    "LOG_FORMAT",
    "LOG_FORMAT_DETAILED",
    "LOG_DATE_FORMAT",
]
