"""Training Module for Microspore Phenotyping
Core training components, trainer classes, and statistics.

Files:
    core.py           - YOLOTrainer class for model training
    train.py          - Training runner and CLI interface
    stats.py          - Statistics, metrics, and reporting
    class_balancer.py - Class imbalance handling via oversampling
"""

import sys

# Skip imports when running train.py as __main__ to avoid RuntimeWarning
# This occurs when using: python -m modules.training.train
_running_as_main = 'modules.training.train' in sys.modules

if not _running_as_main:
    from .core import YOLOTrainer, YOLO_AVAILABLE
    from .stats import TrainingStats, generate_training_report
    from .class_balancer import ClassBalancer, create_balanced_training_data, cleanup_balanced_dataset, generate_balancing_report

# Lazy import for train.py functions to avoid circular import warning when running as __main__
def __getattr__(name):
    """Lazy import for train.py functions and conditionally loaded modules."""
    # Handle train.py functions
    if name in ('run_training', 'generate_stats', 'load_or_download_model', 'export_to_onnx'):
        from . import train as train_module
        return getattr(train_module, name)
    # Handle core/stats/balancer imports when running as __main__
    if name == 'YOLOTrainer':
        from .core import YOLOTrainer
        return YOLOTrainer
    if name == 'YOLO_AVAILABLE':
        from .core import YOLO_AVAILABLE
        return YOLO_AVAILABLE
    if name == 'TrainingStats':
        from .stats import TrainingStats
        return TrainingStats
    if name == 'generate_training_report':
        from .stats import generate_training_report
        return generate_training_report
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
