"""Training Module for Microspore Phenotyping
Core training components, trainer classes, and statistics.

Files:
    core.py           - YOLOTrainer class for model training
    train.py          - Training runner and CLI interface
    stats.py          - Statistics, metrics, and reporting
    class_balancer.py - Class imbalance handling via oversampling

Note: All imports are lazy-loaded via __getattr__ to avoid RuntimeWarning
when running as: python -m modules.training.train
"""

# Lazy import everything to avoid RuntimeWarning when running train.py as __main__
# This warning occurs because Python imports the package before executing the script
def __getattr__(name):
    """Lazy import for all module attributes to avoid RuntimeWarning."""
    # Core training
    if name == 'YOLOTrainer':
        from .core import YOLOTrainer
        return YOLOTrainer
    if name == 'YOLO_AVAILABLE':
        from .core import YOLO_AVAILABLE
        return YOLO_AVAILABLE
    # Statistics
    if name == 'TrainingStats':
        from .stats import TrainingStats
        return TrainingStats
    if name == 'generate_training_report':
        from .stats import generate_training_report
        return generate_training_report
    # Class balancing
    if name == 'ClassBalancer':
        from .class_balancer import ClassBalancer
        return ClassBalancer
    if name == 'create_balanced_training_data':
        from .class_balancer import create_balanced_training_data
        return create_balanced_training_data
    if name == 'cleanup_balanced_dataset':
        from .class_balancer import cleanup_balanced_dataset
        return cleanup_balanced_dataset
    if name == 'generate_balancing_report':
        from .class_balancer import generate_balancing_report
        return generate_balancing_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Wrapper functions keep `from modules.training import *` import-light. The real
# train.py module imports Ultralytics at module import time, so loading it here
# would break utility-only workflows on machines that have not activated the
# training environment yet.
def run_training(*args, **kwargs):
    from .train import run_training as _run_training
    return _run_training(*args, **kwargs)


def generate_stats(*args, **kwargs):
    from .train import generate_stats as _generate_stats
    return _generate_stats(*args, **kwargs)


def load_or_download_model(*args, **kwargs):
    from .train import load_or_download_model as _load_or_download_model
    return _load_or_download_model(*args, **kwargs)


def export_to_onnx(*args, **kwargs):
    from .train import export_to_onnx as _export_to_onnx
    return _export_to_onnx(*args, **kwargs)

__all__ = [
    # Core training
    'YOLOTrainer',
    'YOLO_AVAILABLE',
    # Statistics
    'TrainingStats',
    'generate_training_report',
    # Class balancing
    'ClassBalancer',
    'create_balanced_training_data',
    'cleanup_balanced_dataset',
    'generate_balancing_report',
    # Training runner (lazy loaded)
    'run_training',
    'generate_stats',
    'load_or_download_model',
    'export_to_onnx',
]
