"""Logging Module for Microspore Phenotyping
Comprehensive logging for ML training: metrics, GPU, system, errors.

Files:
    logging.py              - Python training logger, metrics tracking
    logging_utils.sh        - Shell logging utilities (GPU/system monitoring)
    optimization_metrics.py - Detailed metrics for hyperparameter optimization
"""

from .logging import (
    TrainingLogger,
    TrainingMetrics,
    CheckpointInfo,
    LOGS_DIR,
)

# Import YOLOTrainingLogger if available
try:
    from .logging import YOLOTrainingLogger, create_logger
except ImportError:
    pass

# Import optimization metrics logger
try:
    from .optimization_metrics import (
        OptimizationMetricsLogger,
        PerClassMetrics,
        GradientStats,
        LossBreakdown,
        ConvergenceIndicators,
        ThroughputMetrics,
        OptimizationRecommendation,
        extract_yolo_class_metrics,
    )
except ImportError:
    pass

__all__ = [
    'TrainingLogger',
    'TrainingMetrics', 
    'CheckpointInfo',
    'LOGS_DIR',
    'YOLOTrainingLogger',
    'create_logger',
    # Optimization metrics
    'OptimizationMetricsLogger',
    'PerClassMetrics',
    'GradientStats',
    'LossBreakdown',
    'ConvergenceIndicators',
    'ThroughputMetrics',
    'OptimizationRecommendation',
    'extract_yolo_class_metrics',
]
