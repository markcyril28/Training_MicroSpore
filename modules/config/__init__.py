"""Configuration Module for Microspore Phenotyping
Centralized configuration constants and defaults for both Python and Shell.

Files:
    config.py              - Python configuration (paths, models, defaults)
    common_functions.sh    - Shell shared configuration and common functions
"""

from .config import *

__all__ = [
    # Directory configuration
    'BASE_DIR',
    'MODULES_DIR',
    'DEFAULT_DATASETS_DIR',
    'DEFAULT_DATASET_NAME',
    'DEFAULT_DATASET_DIR',
    'DEFAULT_WEIGHTS_DIR',
    'DEFAULT_TRAINED_MODELS_DIR',
    'DEFAULT_DATA_YAML',
    'DEFAULT_CLASSES_FILE',
    'get_dataset_path',
    'get_weights_path',
    'get_trained_models_path',
    'get_data_yaml_path',
    # YOLO model configuration
    'YOLO_MODELS',
    'MODEL_SIZES',
    'get_all_models',
    'get_models_by_version',
    'get_model_version',
    # Training defaults
    'TrainingDefaults',
    'TRAINING_DEFAULTS',
    'get_recommended_batch_size',
    'VRAM_BATCH_SIZE_MAP',
    # Logging configuration
    'LOG_FORMAT',
    'LOG_FORMAT_DETAILED',
    'LOG_DATE_FORMAT',
    # File extensions
    'IMAGE_EXTENSIONS',
    'LABEL_EXTENSION',
    'EXPORT_FORMATS',
]
