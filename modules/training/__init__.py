"""Training Module for Microspore Phenotyping
Core training components, trainer classes, and statistics.

Files:
    core.py     - YOLOTrainer class for model training
    train.py    - Training runner and CLI interface
    stats.py    - Statistics, metrics, and reporting
"""

from .core import YOLOTrainer, YOLO_AVAILABLE
from .stats import TrainingStats, generate_training_report

# Lazy import for train.py to avoid circular import warning when running as __main__
def __getattr__(name):
    """Lazy import for train.py functions to avoid RuntimeWarning."""
    if name in ('run_training', 'generate_stats', 'load_or_download_model', 'export_to_onnx'):
        from . import train as train_module
        return getattr(train_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core training
    'YOLOTrainer',
    'YOLO_AVAILABLE',
    # Statistics
    'TrainingStats',
    'generate_training_report',
    # Training runner (lazy loaded)
    'run_training',
    'generate_stats',
    'load_or_download_model',
    'export_to_onnx',
]
