"""Training Module for Microspore Phenotyping
Core training components, trainer classes, and statistics.

Files:
    core.py     - YOLOTrainer class for model training
    train.py    - Training runner and CLI interface
    stats.py    - Statistics, metrics, and reporting
"""

from .core import YOLOTrainer, YOLO_AVAILABLE
from .stats import TrainingStats, generate_training_report
from .train import run_training, generate_stats, load_or_download_model, export_to_onnx

__all__ = [
    # Core training
    'YOLOTrainer',
    'YOLO_AVAILABLE',
    # Statistics
    'TrainingStats',
    'generate_training_report',
    # Training runner
    'run_training',
    'generate_stats',
    'load_or_download_model',
    'export_to_onnx',
]
