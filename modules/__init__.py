"""Microspore Phenotyping Training Modules
Core and utility functions for YOLO training pipeline.
Implements DRY principle - centralized configuration and utilities.

Directory Structure (Organized by Function):
    modules/
    ├── __init__.py              # This file - main module entry point
    ├── config/                  # Configuration (Python + Shell)
    │   ├── config.py            # Python configuration constants
    │   └── common_functions.sh  # Shell shared config & functions
    ├── training/                # Training core logic
    │   ├── core.py              # YOLOTrainer class
    │   ├── train.py             # Training runner & CLI
    │   └── stats.py             # Statistics and reporting
    ├── logging/                 # Logging utilities
    │   ├── logging.py           # Python training logger
    │   └── logging_utils.sh     # Shell logging (GPU/system)
    ├── setup/                   # Environment/GPU setup
    │   ├── gpu_setup_core.sh    # GPU detection & CUDA setup
    │   └── setup_conda_core.sh  # Conda environment setup
    ├── utils/                   # Shared utilities
    │   └── utils.py             # Helper functions
    └── yolo_models_weights/     # Pre-trained YOLO weights
"""

# Import centralized configuration first
from .config import *

# Import utilities (single source of truth for common functions)
from .utils import *

# Import training components
from .training import *

# Import logging utilities (optional - may not be needed in all contexts)
try:
    from .logging import (
        TrainingLogger,
        YOLOTrainingLogger,
        TrainingMetrics,
        CheckpointInfo,
        create_logger,
        LOGS_DIR,
    )
except ImportError:
    pass

__version__ = "1.0.0"
__author__ = "Microspore Phenotyping Team"
