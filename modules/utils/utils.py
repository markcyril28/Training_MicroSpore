"""
Utilities Module for Microspore Phenotyping
Helper functions for file management, logging, and system operations.
This module serves as the single source of truth for common utilities.
Implements DRY principle - all shared functions should be defined here.
"""

import os
import sys
import json
import shutil
import logging
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

# Import centralized configuration
from ..config import (
    LOG_FORMAT,
    LOG_FORMAT_DETAILED,
    LOG_DATE_FORMAT,
    IMAGE_EXTENSIONS,
    VRAM_BATCH_SIZE_MAP,
    get_recommended_batch_size,
)


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logger(
    name: str = "microspore_training",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.
    Uses centralized log format configuration.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Console handler - uses simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler - uses detailed format
    if log_file:
        ensure_dir(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(LOG_FORMAT_DETAILED, datefmt=LOG_DATE_FORMAT)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# FILE UTILITIES
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_count(directory: Union[str, Path]) -> int:
    """
    Count image files in directory using standard image extensions.
    
    Args:
        directory: Directory path
        
    Returns:
        Image file count
    """
    return get_file_count(directory, IMAGE_EXTENSIONS)


def get_file_count(directory: Union[str, Path], extensions: List[str] = None) -> int:
    """
    Count files in directory with optional extension filter.
    
    Args:
        directory: Directory path
        extensions: List of extensions to count (e.g., ['.jpg', '.png'])
        
    Returns:
        File count
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    if extensions:
        count = 0
        for ext in extensions:
            count += len(list(directory.glob(f"*{ext}")))
        return count
    else:
        return len([f for f in directory.iterdir() if f.is_file()])


def copy_files(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    extensions: List[str] = None,
    overwrite: bool = False
) -> int:
    """
    Copy files from source to destination
    
    Args:
        src_dir: Source directory
        dst_dir: Destination directory
        extensions: File extensions to copy
        overwrite: Whether to overwrite existing files
        
    Returns:
        Number of files copied
    """
    src_dir = Path(src_dir)
    dst_dir = ensure_dir(dst_dir)
    
    copied = 0
    for file in src_dir.iterdir():
        if file.is_file():
            if extensions and file.suffix.lower() not in extensions:
                continue
            
            dst_file = dst_dir / file.name
            if dst_file.exists() and not overwrite:
                continue
            
            shutil.copy2(file, dst_file)
            copied += 1
    
    return copied


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_readable_size(size_bytes: int) -> str:
    """
    Convert bytes to human readable size
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human readable string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def get_dir_size(directory: Union[str, Path]) -> int:
    """
    Get total size of directory in bytes
    
    Args:
        directory: Directory path
        
    Returns:
        Size in bytes
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    total = 0
    for file in directory.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


# =============================================================================
# JSON UTILITIES
# =============================================================================

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        path: File path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load data from JSON file
    
    Args:
        path: File path
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# SYSTEM UTILITIES
# =============================================================================

def run_command(
    command: str,
    cwd: Optional[str] = None,
    capture_output: bool = True
) -> Tuple[int, str, str]:
    """
    Run shell command
    
    Args:
        command: Command to run
        cwd: Working directory
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=capture_output,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def check_conda_env(env_name: str = "training") -> bool:
    """
    Check if conda environment exists
    
    Args:
        env_name: Environment name
        
    Returns:
        True if environment exists
    """
    returncode, stdout, _ = run_command("conda env list")
    return env_name in stdout


def get_python_version() -> str:
    """Get current Python version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_gpu() -> Dict[str, Any]:
    """
    Check GPU availability and specifications.
    This is the single source of truth for GPU information.
    
    Returns:
        Dictionary with GPU information including:
        - cuda_available: bool
        - gpu_count: int
        - gpus: list of GPU info dicts
        - recommended_batch_size: int
    """
    info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "recommended_batch_size": 16
    }
    
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count()
        info["torch_version"] = torch.__version__
        
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            
            for i in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory // (1024**3)
                
                gpu_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_bytes": props.total_memory,
                    "memory_total_gb": round(props.total_memory / (1024**3), 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
                info["gpus"].append(gpu_info)
                
                # Use centralized batch size recommendation
                info["recommended_batch_size"] = get_recommended_batch_size(vram_gb)
                
    except ImportError:
        info["torch_installed"] = False
    except Exception as e:
        info["error"] = str(e)
    
    return info


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information including GPU.
    Uses check_gpu() for GPU information (DRY principle).
    
    Returns:
        Dictionary with system info
    """
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": get_python_version(),
        "is_wsl": "microsoft" in platform.release().lower()
    }
    
    # Add GPU info from centralized function
    gpu_info = check_gpu()
    info.update({
        "torch_version": gpu_info.get("torch_version", "Not installed"),
        "cuda_available": gpu_info.get("cuda_available", False),
        "cuda_version": gpu_info.get("cuda_version"),
        "gpu_name": gpu_info["gpus"][0]["name"] if gpu_info["gpus"] else None,
        "gpu_count": gpu_info.get("gpu_count", 0),
    })
    
    return info


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def estimate_training_time(
    num_images: int,
    epochs: int,
    batch_size: int,
    gpu_speed: float = 1.0
) -> str:
    """
    Estimate training time
    
    Args:
        num_images: Number of training images
        epochs: Number of epochs
        batch_size: Batch size
        gpu_speed: Relative GPU speed factor (1.0 = RTX 3060)
        
    Returns:
        Estimated time string
    """
    # Rough estimate: ~0.01 seconds per image per epoch on RTX 3060
    seconds_per_image = 0.01 / gpu_speed
    total_seconds = num_images * epochs * seconds_per_image
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def generate_experiment_name(
    model_name: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    optimizer: str = "auto"
) -> str:
    """
    Generate experiment name with parameters and timestamp
    
    Args:
        model_name: Model name (e.g., yolov8s)
        epochs: Number of epochs
        batch_size: Batch size
        img_size: Image size
        optimizer: Optimizer type (e.g., SGD, Adam, AdamW, auto)
        
    Returns:
        Experiment name string
    """
    timestamp = get_timestamp()
    # Remove .pt extension if present for cleaner name
    model_stem = Path(model_name).stem if "." in model_name else model_name
    # Normalize optimizer name (lowercase for consistency)
    opt_name = optimizer.lower() if optimizer else "auto"
    # Order: model_color_imgsize_optimizer_balance_epochs_batch_lr_timestamp
    return f"{model_stem}_img{img_size}_{opt_name}_e{epochs}_b{batch_size}_{timestamp}"


# =============================================================================
# DATA.YAML PATH UTILITIES (Portability Support)
# =============================================================================

def update_data_yaml_path(data_yaml_path: Union[str, Path], dataset_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Update the 'path' field in data.yaml to use the absolute path of the dataset directory.
    This ensures the project is portable across different machines and directories.
    
    Args:
        data_yaml_path: Path to the data.yaml file
        dataset_dir: Optional explicit dataset directory path. If not provided,
                     uses the parent directory of data.yaml
        
    Returns:
        Path to the updated data.yaml file
        
    Example:
        >>> update_data_yaml_path('/new/location/Dataset_1/data.yaml')
        # Updates data.yaml 'path' field to '/new/location/Dataset_1'
    """
    import yaml
    
    data_yaml_path = Path(data_yaml_path).resolve()
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    
    # Determine the dataset directory
    if dataset_dir:
        dataset_dir = Path(dataset_dir).resolve()
    else:
        # Use parent directory of data.yaml as dataset root
        dataset_dir = data_yaml_path.parent.resolve()
    
    # Read current data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get the old path for comparison
    old_path = data_config.get('path', '')
    new_path = str(dataset_dir)
    
    # Only update if the path has changed
    if old_path != new_path:
        data_config['path'] = new_path
        
        # Write back to data.yaml
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"[Portability] Updated data.yaml path:")
        print(f"  Old: {old_path}")
        print(f"  New: {new_path}")
    
    return data_yaml_path


def ensure_portable_data_yaml(data_yaml_path: Union[str, Path]) -> Path:
    """
    Ensure data.yaml uses a relative or current absolute path for portability.
    This is a convenience wrapper that auto-detects the dataset directory.
    
    Args:
        data_yaml_path: Path to the data.yaml file
        
    Returns:
        Path to the (potentially updated) data.yaml file
    """
    return update_data_yaml_path(data_yaml_path)


# Export public API
__all__ = [
    # Logging
    'setup_logger',
    # File utilities
    'ensure_dir',
    'get_file_count',
    'get_image_count',
    'copy_files',
    'get_timestamp',
    'get_readable_size',
    'get_dir_size',
    # JSON utilities
    'save_json',
    'load_json',
    # System utilities
    'run_command',
    'check_conda_env',
    'get_python_version',
    'get_system_info',
    # GPU utilities (consolidated)
    'check_gpu',
    # Training utilities
    'estimate_training_time',
    'generate_experiment_name',
    # Portability utilities
    'update_data_yaml_path',
    'ensure_portable_data_yaml',
]
