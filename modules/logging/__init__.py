"""Logging Module for Microspore Phenotyping
Comprehensive logging for ML training: metrics, GPU, system, errors.

Files:
    logging.py         - Python training logger, metrics tracking
    logging_utils.sh   - Shell logging utilities (GPU/system monitoring)
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

__all__ = [
    'TrainingLogger',
    'TrainingMetrics', 
    'CheckpointInfo',
    'LOGS_DIR',
    'YOLOTrainingLogger',
    'create_logger',
]
