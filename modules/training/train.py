"""Training Script Module for Microspore Phenotyping
Command-line interface for YOLO model training.
Implements DRY principle - uses centralized config and utilities.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from ultralytics import YOLO, settings
except ImportError:
    raise ImportError("ultralytics not installed. Run setup_conda_training.sh first.")

# Import from centralized modules (DRY principle)
from ..config import (
    TRAINING_DEFAULTS,
    get_weights_path,
    get_trained_models_path,
    get_data_yaml_path,
)
from ..utils import check_gpu, get_system_info, ensure_dir
from .stats import TrainingStats, generate_training_report

# Import logging module (optional - gracefully handle if not available)
try:
    from ..logging import YOLOTrainingLogger, TrainingMetrics
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def export_to_onnx(
    model_path: Path,
    output_dir: Path,
    exp_name: str,
    img_size: int = 640,
) -> Optional[Path]:
    """
    Export trained model to ONNX format.
    
    Args:
        model_path: Path to the best.pt weights
        output_dir: Directory to save ONNX file
        exp_name: Experiment folder name (used for ONNX filename)
        img_size: Image size used during training
        
    Returns:
        Path to exported ONNX file, or None if export failed
    """
    try:
        print("\n[Export] Exporting model to ONNX format...")
        model = YOLO(str(model_path))
        
        # Export to ONNX
        onnx_path = model.export(format='onnx', imgsz=img_size, simplify=True)
        
        # Move to output directory with naming based on experiment folder name
        if onnx_path and Path(onnx_path).exists():
            # Use experiment folder name for ONNX file: e.g., Dataset_1_yolov5xu_e10_b8_img320.onnx
            output_onnx = output_dir / f"{exp_name}.onnx"
            shutil.copy(onnx_path, output_onnx)
            print(f"[Export] ONNX saved to: {output_onnx}")
            return output_onnx
    except Exception as e:
        print(f"[Export] ONNX export failed: {e}")
    return None


def copy_general_guide(
    output_dir: Path,
    source_guide: Optional[Path] = None,
    results_csv: Optional[Path] = None,
) -> Optional[Path]:
    """
    Copy the general interpretation guide to the training output directory.
    Adds accuracy summary at start and end of the file.
    
    Args:
        output_dir: Directory to save the guide
        source_guide: Path to source guide (defaults to TRAINING_WD/OUTPUT_INTERPRETATION_GUIDE.md)
        results_csv: Path to results.csv for extracting accuracy metrics
        
    Returns:
        Path to copied guide, or None if copy failed
    """
    try:
        output_path = output_dir / "GENERAL_INTERPRETATION_GUIDE_and_ANALYSIS.md"
        
        # Try to find the source guide
        if source_guide is None:
            # Look for the guide in TRAINING_WD
            possible_paths = [
                output_dir.parent.parent / "TRAINING_WD" / "OUTPUT_INTERPRETATION_GUIDE.md",
                output_dir.parent / "TRAINING_WD" / "OUTPUT_INTERPRETATION_GUIDE.md",
                Path(__file__).parent.parent.parent / "TRAINING_WD" / "OUTPUT_INTERPRETATION_GUIDE.md",
            ]
            for path in possible_paths:
                if path.exists():
                    source_guide = path
                    break
        
        # Extract accuracy metrics from results.csv if available
        accuracy_summary = _extract_accuracy_summary(results_csv)
        
        if source_guide and source_guide.exists():
            with open(source_guide, 'r') as f:
                guide_content = f.read()
            
            # Add accuracy summary at start
            header = f"""# Training Accuracy Summary

{accuracy_summary}

---

"""
            # Add accuracy summary at end
            footer = f"""

---

# Final Accuracy Summary

{accuracy_summary}
"""
            # Write with accuracy at start and end
            with open(output_path, 'w') as f:
                f.write(header + guide_content + footer)
            
            print(f"[Export] General guide with accuracy saved to: {output_path}")
            return output_path
        else:
            print("[Export] Warning: OUTPUT_INTERPRETATION_GUIDE.md not found")
            return None
    except Exception as e:
        print(f"[Export] General guide copy failed: {e}")
    return None


def _extract_accuracy_summary(results_csv: Optional[Path]) -> str:
    """
    Extract accuracy metrics from results.csv for summary.
    
    Args:
        results_csv: Path to results.csv file
        
    Returns:
        Formatted accuracy summary string
    """
    if results_csv is None or not results_csv.exists():
        return "*(Accuracy metrics not available - results.csv not found)*"
    
    try:
        import csv
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return "*(No training data available)*"
        
        # Clean column names (strip whitespace)
        first_row = {k.strip(): v for k, v in rows[0].items()}
        last_row = {k.strip(): v for k, v in rows[-1].items()}
        
        # Find mAP columns
        map50_col = None
        map5095_col = None
        for col in first_row.keys():
            if 'mAP50' in col and '95' not in col:
                map50_col = col
            elif 'mAP50-95' in col:
                map5095_col = col
        
        # Extract metrics
        epochs = len(rows)
        final_map50 = float(last_row.get(map50_col, 0)) if map50_col else 0
        final_map5095 = float(last_row.get(map5095_col, 0)) if map5095_col else 0
        
        # Find best metrics
        best_map50 = max(float(r.get(map50_col.strip() if map50_col else '', 0) or 0) for r in [{k.strip(): v for k, v in row.items()} for row in rows]) if map50_col else 0
        best_map5095 = max(float(r.get(map5095_col.strip() if map5095_col else '', 0) or 0) for r in [{k.strip(): v for k, v in row.items()} for row in rows]) if map5095_col else 0
        
        summary = f"""## Model Performance Metrics

| Metric | Final Value | Best Value |
|--------|-------------|------------|
| **mAP@50** | {final_map50:.4f} ({final_map50*100:.1f}%) | {best_map50:.4f} ({best_map50*100:.1f}%) |
| **mAP@50-95** | {final_map5095:.4f} ({final_map5095*100:.1f}%) | {best_map5095:.4f} ({best_map5095*100:.1f}%) |
| **Epochs Trained** | {epochs} | - |

**Interpretation:**
- mAP@50 > 0.70 (70%): Good detection accuracy
- mAP@50-95 > 0.50 (50%): Good localization precision
"""
        return summary
        
    except Exception as e:
        return f"*(Error extracting accuracy: {e})*"


def export_raw_data_csvs(
    experiment_path: Path,
    stats_dir: Path,
) -> Dict[str, Optional[Path]]:
    """
    Export raw training data and statistics to CSV files for later analysis.
    
    Args:
        experiment_path: Path to training experiment folder
        stats_dir: Directory to save CSV files
        
    Returns:
        Dictionary of exported CSV file paths
    """
    csv_files = {}
    
    try:
        # 1. Copy results.csv to stats folder (already done, but ensure it's there)
        results_src = experiment_path / "results.csv"
        if results_src.exists():
            results_dst = stats_dir / "training_metrics.csv"
            shutil.copy(str(results_src), str(results_dst))
            csv_files['training_metrics'] = results_dst
        
        # 2. Generate epoch summary CSV
        if results_src.exists():
            epoch_summary_path = stats_dir / "epoch_summary.csv"
            _generate_epoch_summary_csv(results_src, epoch_summary_path)
            csv_files['epoch_summary'] = epoch_summary_path
        
        # 3. Generate class performance CSV if confusion matrix data available
        # (This would need actual training callback data in a real implementation)
        
        print(f"[Export] Generated {len(csv_files)} raw data CSVs")
        
    except Exception as e:
        print(f"[Export] Raw data CSV export failed: {e}")
    
    return csv_files


def _generate_epoch_summary_csv(results_csv: Path, output_path: Path) -> None:
    """
    Generate a simplified epoch summary CSV with key metrics.
    """
    try:
        import csv
        
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return
        
        # Define key columns to extract
        key_metrics = ['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                       'metrics/precision(B)', 'metrics/recall(B)', 
                       'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                       'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'lr/pg0']
        
        with open(output_path, 'w', newline='') as f:
            # Clean headers
            available_cols = [k.strip() for k in rows[0].keys()]
            output_cols = [col for col in key_metrics if col in available_cols]
            
            writer = csv.writer(f)
            writer.writerow(output_cols)
            
            for row in rows:
                clean_row = {k.strip(): v for k, v in row.items()}
                writer.writerow([clean_row.get(col, '') for col in output_cols])
                
    except Exception as e:
        print(f"[Export] Epoch summary generation failed: {e}")


def generate_output_analysis_files(
    experiment_path: Path,
    config: Dict[str, Any],
    classes: List[str],
) -> Dict[str, Optional[Path]]:
    """
    Generate accompanying analysis text files for important output files.
    
    Args:
        experiment_path: Path to training experiment folder
        config: Training configuration dictionary
        classes: List of class names
        
    Returns:
        Dictionary of generated analysis file paths
    """
    analysis_files = {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # 1. Generate weights analysis
        weights_analysis_path = experiment_path / "weights_ANALYSIS.txt"
        weights_content = f"""# Weights Directory Analysis
# Generated: {timestamp}

## Overview
This directory contains the trained model weights.

## Files
- best.pt: Best performing model checkpoint (highest mAP50-95)
- last.pt: Final epoch model checkpoint

## Usage
Load best.pt for inference:
    from ultralytics import YOLO
    model = YOLO('weights/best.pt')
    results = model('image.jpg')

## Deployment
For production deployment, use the ONNX export in the parent directory.
"""
        with open(weights_analysis_path, 'w') as f:
            f.write(weights_content)
        analysis_files['weights_analysis'] = weights_analysis_path
        
        # 2. Generate results.csv analysis
        results_analysis_path = experiment_path / "results_csv_ANALYSIS.txt"
        results_content = f"""# results.csv Analysis Guide
# Generated: {timestamp}

## Overview
This file contains epoch-by-epoch training metrics.

## Key Columns
- epoch: Training epoch number
- train/box_loss: Bounding box regression loss (should decrease)
- train/cls_loss: Classification loss (should decrease)
- train/dfl_loss: Distribution focal loss (should decrease)
- metrics/precision(B): Detection precision
- metrics/recall(B): Detection recall
- metrics/mAP50(B): Mean Average Precision at IoU=0.50
- metrics/mAP50-95(B): Mean AP at IoU=0.50-0.95 (primary metric)

## Interpretation
- All losses should decrease over epochs
- mAP values should increase over epochs
- Large gap between train/val metrics indicates overfitting
- Smooth curves indicate stable training

## Quick Analysis (Python)
```python
import pandas as pd
df = pd.read_csv('results.csv')
print(f"Best mAP50-95: {{df['metrics/mAP50-95(B)'].max():.4f}}")
```
"""
        with open(results_analysis_path, 'w') as f:
            f.write(results_content)
        analysis_files['results_analysis'] = results_analysis_path
        
        # 3. Generate args.yaml analysis
        args_analysis_path = experiment_path / "args_yaml_ANALYSIS.txt"
        args_content = f"""# args.yaml Analysis Guide
# Generated: {timestamp}

## Overview
This file contains all YOLO training arguments used for this experiment.

## Key Parameters
- model: {config.get('model', 'Unknown')}
- epochs: {config.get('epochs', 'Unknown')}
- batch: {config.get('batch_size', 'Unknown')}
- imgsz: {config.get('img_size', 'Unknown')}
- lr0: {config.get('lr0', 'Unknown')}
- optimizer: {config.get('optimizer', 'Unknown')}

## Augmentation Settings
- hsv_h: {config.get('hsv_h', 0.015)} (hue augmentation)
- hsv_s: {config.get('hsv_s', 0.7)} (saturation augmentation)
- hsv_v: {config.get('hsv_v', 0.4)} (value augmentation)
- mosaic: {config.get('mosaic', 1.0)}
- flipud: {config.get('flipud', 0.5)} (vertical flip probability)
- fliplr: {config.get('fliplr', 0.5)} (horizontal flip probability)

## Classes Trained
{chr(10).join(f'  {i}: {name}' for i, name in enumerate(classes)) if classes else '  (classes not available)'}

## Reproducibility
Use these settings to reproduce the training run.
"""
        with open(args_analysis_path, 'w') as f:
            f.write(args_content)
        analysis_files['args_analysis'] = args_analysis_path
        
        # 4. Generate ONNX usage guide
        onnx_analysis_path = experiment_path / "onnx_ANALYSIS.txt"
        model_name = config.get('model', 'model').replace('.pt', '').replace('.', '')
        onnx_content = f"""# ONNX Export Analysis Guide
# Generated: {timestamp}

## Overview
The ONNX file ({model_name}_microspore.onnx) is for cross-platform deployment.

## Supported Runtimes
- ONNX Runtime (CPU/GPU)
- TensorRT (NVIDIA GPU optimization)
- OpenVINO (Intel optimization)
- CoreML (Apple devices)

## Usage with ONNX Runtime
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('{model_name}_microspore.onnx')
input_name = session.get_inputs()[0].name
output = session.run(None, {{input_name: image_array}})
```

## Input Format
- Shape: (1, 3, {config.get('img_size', 640)}, {config.get('img_size', 640)})
- Type: float32
- Normalization: 0-1 range

## Output Format
- Detection boxes with class probabilities
- Format: (batch, num_detections, 5 + num_classes)
"""
        with open(onnx_analysis_path, 'w') as f:
            f.write(onnx_content)
        analysis_files['onnx_analysis'] = onnx_analysis_path
        
        print(f"[Export] Generated {len(analysis_files)} analysis files")
        
    except Exception as e:
        print(f"[Export] Analysis file generation failed: {e}")
    
    return analysis_files


def create_obj_names(
    classes_file: Path,
    output_dir: Path,
) -> Optional[Path]:
    """
    Create obj.names file (class names for inference).
    
    Args:
        classes_file: Path to classes.txt from dataset
        output_dir: Directory to save obj.names
        
    Returns:
        Path to obj.names file, or None if creation failed
    """
    try:
        output_path = output_dir / "obj.names"
        
        if classes_file.exists():
            shutil.copy(classes_file, output_path)
        else:
            # Try to find classes.txt in dataset directory
            print(f"[Export] Warning: classes.txt not found at {classes_file}")
            return None
            
        print(f"[Export] obj.names saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"[Export] obj.names creation failed: {e}")
    return None


def create_training_config(
    output_dir: Path,
    model_name: str,
    config: Dict[str, Any],
    num_classes: int,
) -> Optional[Path]:
    """
    Create a training configuration summary file (.cfg).
    
    Args:
        output_dir: Directory to save config file
        model_name: YOLO model name
        config: Training configuration dictionary
        num_classes: Number of classes in the dataset
        
    Returns:
        Path to config file, or None if creation failed
    """
    try:
        clean_name = model_name.replace('.pt', '').replace('.', '')
        output_path = output_dir / f"{clean_name}_microspore.cfg"
        
        # Build config content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cfg_content = f"""# Microspore Phenotyping Training Configuration
# Generated: {timestamp}
# Model: {model_name}

[training]
model={model_name}
epochs={config.get('epochs', 100)}
batch_size={config.get('batch_size', 16)}
img_size={config.get('img_size', 640)}
patience={config.get('patience', 50)}
optimizer={config.get('optimizer', 'auto')}
learning_rate={config.get('lr0', 0.01)}
amp={config.get('amp', True)}
pretrained={config.get('pretrained', True)}

[dataset]
num_classes={num_classes}
classes={config.get('classes', [])}

[augmentation]
hsv_h={config.get('hsv_h', 0.015)}
hsv_s={config.get('hsv_s', 0.7)}
hsv_v={config.get('hsv_v', 0.4)}
degrees={config.get('degrees', 0.0)}
translate={config.get('translate', 0.1)}
scale={config.get('scale', 0.5)}
flipud={config.get('flipud', 0.5)}
fliplr={config.get('fliplr', 0.5)}
mosaic={config.get('mosaic', 1.0)}
mixup={config.get('mixup', 0.0)}

[output]
weights_file=best.pt
onnx_file={clean_name}_microspore.onnx
names_file=obj.names
"""
        
        with open(output_path, 'w') as f:
            f.write(cfg_content)
            
        print(f"[Export] Config saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"[Export] Config creation failed: {e}")
    return None


def organize_output_folders(experiment_path: Path) -> Dict[str, Path]:
    """
    Create organized folder structure for training outputs.
    
    Args:
        experiment_path: Path to training experiment folder
        
    Returns:
        Dictionary of folder paths
    """
    folders = {
        'weights': experiment_path / "weights",           # Model weights (already exists from YOLO)
        'configs': experiment_path / "configs",           # .cfg and .yaml config files
        'guides': experiment_path / "guides",             # Interpretation guides and analysis files
        'exports': experiment_path / "exports",           # ONNX, obj.names for deployment
        'stats': experiment_path / "stats",               # JSON statistics and reports
        'visualizations': experiment_path / "visualizations",  # Plots and images
    }
    
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    return folders


def move_yolo_outputs_to_folders(experiment_path: Path, folders: Dict[str, Path]) -> None:
    """
    Move YOLO-generated outputs to organized folders.
    
    Args:
        experiment_path: Path to training experiment folder
        folders: Dictionary of organized folder paths
    """
    import shutil as sh
    
    # Move visualization files (plots, images)
    viz_patterns = ['*.png', '*.jpg', '*.jpeg']
    for pattern in viz_patterns:
        for f in experiment_path.glob(pattern):
            if f.is_file():
                dest = folders['visualizations'] / f.name
                sh.move(str(f), str(dest))
    
    # Move args.yaml to configs folder
    args_file = experiment_path / "args.yaml"
    if args_file.exists():
        sh.copy(str(args_file), str(folders['configs'] / "args.yaml"))
    
    # Move results.csv to stats folder (keep copy in root for compatibility)
    results_file = experiment_path / "results.csv"
    if results_file.exists():
        sh.copy(str(results_file), str(folders['stats'] / "results.csv"))


def export_training_outputs(
    experiment_path: Path,
    model_name: str,
    config: Dict[str, Any],
    classes_file: Path,
    img_size: int = 640,
) -> Dict[str, Optional[Path]]:
    """
    Export all additional training outputs (ONNX, obj.names, config).
    Organizes outputs into logical folder groups.
    
    Args:
        experiment_path: Path to training experiment folder
        model_name: YOLO model name
        config: Training configuration dictionary
        classes_file: Path to classes.txt file
        img_size: Image size used during training
        
    Returns:
        Dictionary of exported file paths
    """
    print("\n" + "="*60)
    print("  Exporting & Organizing Output Files")
    print("="*60)
    
    # Create organized folder structure
    folders = organize_output_folders(experiment_path)
    
    exports = {}
    weights_dir = experiment_path / "weights"
    best_weights = weights_dir / "best.pt"
    exp_name = experiment_path.name  # Use folder name for ONNX naming
    
    # Read class names for config
    classes = []
    num_classes = 0
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
            num_classes = len(classes)
    config['classes'] = classes
    
    # Export ONNX to exports folder (named based on output folder name)
    if best_weights.exists():
        exports['onnx'] = export_to_onnx(best_weights, folders['exports'], exp_name, img_size)
    else:
        print(f"[Export] Warning: best.pt not found at {best_weights}")
        exports['onnx'] = None
    
    # Create obj.names in exports folder
    exports['obj_names'] = create_obj_names(classes_file, folders['exports'])
    
    # Create config file in configs folder
    exports['config'] = create_training_config(folders['configs'], model_name, config, num_classes)
    
    # Copy general interpretation guide to guides folder (with accuracy summary)
    results_csv = experiment_path / "results.csv"
    exports['general_guide'] = copy_general_guide(folders['guides'], results_csv=results_csv)
    
    # Generate accompanying analysis text files in guides folder
    analysis_files = generate_output_analysis_files(folders['guides'], config, classes)
    exports.update(analysis_files)
    
    # Export raw data CSVs to stats folder
    csv_files = export_raw_data_csvs(experiment_path, folders['stats'])
    exports.update(csv_files)
    
    # Move YOLO-generated outputs to organized folders
    move_yolo_outputs_to_folders(experiment_path, folders)
    
    print("\n" + "-"*60)
    print("  Folder Structure:")
    for name, path in folders.items():
        file_count = len(list(path.glob('*'))) if path.exists() else 0
        print(f"    📁 {name}/ ({file_count} files)")
    print("-"*60)
    print("  Export Summary:")
    for key, path in exports.items():
        status = "✓" if path else "✗"
        print(f"    [{status}] {key}: {path.name if path else 'Failed'}")
    print("-"*60 + "\n")
    
    return exports


def load_or_download_model(model_name: str, weights_dir: Optional[Path] = None) -> YOLO:
    """
    Load YOLO model from local weights directory or download if not available.
    Uses centralized default weights path (DRY principle).
    
    Args:
        model_name: Name of the YOLO model (e.g., 'yolov8s.pt')
        weights_dir: Directory to store/load model weights (defaults to config)
        
    Returns:
        Loaded YOLO model
    """
    # Use default from config if not provided
    weights_dir = Path(weights_dir) if weights_dir else get_weights_path()
    ensure_dir(weights_dir)
    
    local_model = weights_dir / model_name
    
    if local_model.exists():
        print(f'Using local model: {local_model}')
        model = YOLO(str(local_model))
    else:
        print(f'Downloading model: {model_name}')
        model = YOLO(model_name)
        # Copy downloaded model to weights directory
        downloaded = Path(settings['weights_dir']) / model_name
        if downloaded.exists():
            shutil.copy(downloaded, local_model)
            print(f'Model saved to: {local_model}')
        
        # Clean up model file if downloaded to current working directory
        # (Ultralytics sometimes downloads to cwd in addition to settings dir)
        cwd_model = Path(model_name)
        if cwd_model.exists() and cwd_model.absolute() != local_model.absolute():
            cwd_model.unlink()
            print(f'Cleaned up: {cwd_model}')
    
    return model


def run_training(
    data_yaml: str,
    model_name: str,
    weights_dir: str,
    project_dir: str,
    exp_name: str,
    # Core parameters
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    patience: int = 50,
    workers: int = 4,
    # Learning rate & optimizer
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    optimizer: str = "auto",
    # Augmentation
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.5,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
    grayscale: bool = False,
    # Model settings
    pretrained: bool = True,
    resume: bool = False,
    cache: str = "disk",
    amp: bool = True,
    freeze: int = 0,
    # Device
    device: int = 0,
    # Logging
    log_dir: Optional[str] = None,
    # Advanced parameters (from microspores.cfg)
    warmup_epochs: float = 3.0,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.1,
    box_loss: float = 7.5,
    cls_loss: float = 0.5,
    dfl_loss: float = 1.5,
    iou_threshold: float = 0.7,
    label_smoothing: float = 0.0,
    close_mosaic: int = 10,
    multi_scale: bool = False,
    rect: bool = False,
) -> Any:
    """
    Run YOLO model training with specified parameters.
    
    Args:
        data_yaml: Path to dataset configuration YAML
        model_name: YOLO model name (e.g., 'yolov8s.pt')
        weights_dir: Directory for model weights
        project_dir: Output directory for training results
        exp_name: Experiment name
        ... (training parameters)
        
    Returns:
        Training results
    """
    # Initialize Python logging if available and log_dir provided
    logger = None
    if LOGGING_AVAILABLE and log_dir:
        logger = YOLOTrainingLogger(model_name, exp_name)
        # Update log_dir and all subdirectories
        logger.log_dir = Path(log_dir)
        logger.metrics_dir = logger.log_dir / "training_metrics"
        logger.vis_dir = logger.log_dir / "visualization_logs"
        logger.errors_dir = logger.log_dir / "errors_logs"
        # Recreate directories at the new location
        logger.metrics_dir.mkdir(parents=True, exist_ok=True)
        logger.vis_dir.mkdir(parents=True, exist_ok=True)
        logger.errors_dir.mkdir(parents=True, exist_ok=True)
        # Reinitialize metrics CSV at new location
        logger.metrics_csv = logger.metrics_dir / "metrics.csv"
        logger._init_metrics_csv()
        print(f"[TrainingLogger] Logging to: {log_dir}")
    
    # Load or download model
    model = load_or_download_model(model_name, Path(weights_dir))
    
    # Add YOLO callbacks for logging if available
    if logger:
        callbacks = logger.get_callbacks()
        for event, callback in callbacks.items():
            model.add_callback(event, callback)
    
    # If grayscale mode, adjust HSV saturation to 0 to train on desaturated images
    # This effectively trains on grayscale-like images while keeping 3 channels
    effective_hsv_s = 0.0 if grayscale else hsv_s
    
    # Log grayscale mode
    if grayscale:
        print("[Training] Grayscale mode enabled - HSV saturation set to 0")
    
    # Train the model
    results = model.train(
        # Data
        data=data_yaml,
        
        # Core parameters
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        workers=workers,
        
        # Learning rate & optimizer
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer=optimizer,
        
        # Augmentation
        hsv_h=hsv_h,
        hsv_s=effective_hsv_s,  # 0 if grayscale mode
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        
        # Model settings
        pretrained=pretrained,
        resume=resume,
        cache=cache,
        amp=amp,
        freeze=freeze,
        
        # Device
        device=device,
        
        # Advanced parameters (from microspores.cfg)
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        box=box_loss,
        cls=cls_loss,
        dfl=dfl_loss,
        iou=iou_threshold,
        label_smoothing=label_smoothing,
        close_mosaic=close_mosaic,
        multi_scale=multi_scale,
        rect=rect,
        
        # Output
        project=project_dir,
        name=exp_name,
        exist_ok=False,
        save=True,
        save_period=-1,
        
        # Logging
        plots=True,
        verbose=True,
    )
    
    print()
    print('============================================')
    print('  Training Complete!')
    print('============================================')
    print()
    print(f'Results saved to: {project_dir}/{exp_name}')
    print()
    
    # Print logger summary if available
    if logger:
        logger.print_summary()
    
    return results


def generate_stats(
    project_dir: str,
    exp_name: str,
    dataset_dir: str,
    model_name: str,
    config: Dict[str, Any],
    img_size: int = 640,
) -> None:
    """
    Generate training statistics, reports, and export additional file formats.
    Organizes outputs into logical folder groups.
    
    Args:
        project_dir: Output directory for training results
        exp_name: Experiment name
        dataset_dir: Path to dataset directory
        model_name: YOLO model name
        config: Training configuration dictionary
        img_size: Image size used during training (for ONNX export)
    """
    try:
        experiment_path = Path(f'{project_dir}/{exp_name}')
        stats_dir = experiment_path / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training stats (save to stats folder)
        stats = TrainingStats(str(experiment_path))
        
        # Save configuration
        stats.set_config(config)
        
        # Save system info
        stats.set_system_info(get_system_info())
        
        # Save GPU info
        gpu_info = check_gpu()
        stats.stats['gpu_info'] = gpu_info
        stats.save()
        
        # Move stats files to stats folder
        stats_file = experiment_path / "training_stats.json"
        if stats_file.exists():
            shutil.copy(str(stats_file), str(stats_dir / "training_stats.json"))
        
        # Generate full report (save to stats folder)
        report = generate_training_report(
            str(experiment_path),
            dataset_dir,
            str(stats_dir / "training_report.json")
        )
        
        print('Training statistics saved!')
        print(f'  📁 stats/training_stats.json')
        print(f'  📁 stats/training_report.json')
        
    except Exception as e:
        print(f'Could not generate training statistics: {e}')
    
    # Export additional file formats (ONNX, obj.names, config)
    try:
        experiment_path = Path(f'{project_dir}/{exp_name}')
        dataset_path = Path(dataset_dir)
        classes_file = dataset_path / "classes.txt"
        
        # Fallback: check Train subdirectory if classes.txt not in root
        if not classes_file.exists():
            train_classes = dataset_path / "Train" / "classes.txt"
            if train_classes.exists():
                classes_file = train_classes
        
        # Export ONNX, obj.names, and config files
        export_training_outputs(
            experiment_path=experiment_path,
            model_name=model_name,
            config=config,
            classes_file=classes_file,
            img_size=img_size,
        )
    except Exception as e:
        print(f'Could not export additional files: {e}')


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='YOLO Training for Microspore Phenotyping')
    
    # Required arguments
    parser.add_argument('--data-yaml', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, required=True, help='YOLO model name')
    parser.add_argument('--weights-dir', type=str, required=True, help='Weights directory')
    parser.add_argument('--project-dir', type=str, required=True, help='Project output directory')
    parser.add_argument('--exp-name', type=str, required=True, help='Experiment name')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Dataset directory')
    
    # Core parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    
    # Learning rate & optimizer
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--optimizer', type=str, default='auto')
    
    # Augmentation
    parser.add_argument('--hsv-h', type=float, default=0.015)
    parser.add_argument('--hsv-s', type=float, default=0.7)
    parser.add_argument('--hsv-v', type=float, default=0.4)
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--shear', type=float, default=0.0)
    parser.add_argument('--perspective', type=float, default=0.0)
    parser.add_argument('--flipud', type=float, default=0.5)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--copy-paste', type=float, default=0.0)
    parser.add_argument('--grayscale', type=lambda x: x.lower() == 'true', default=False,
                        help='Convert images to grayscale for training')
    
    # Model settings
    parser.add_argument('--pretrained', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--resume', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--cache', type=str, default='disk')
    parser.add_argument('--amp', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--freeze', type=int, default=0)
    
    # Advanced parameters (from microspores.cfg)
    parser.add_argument('--warmup-epochs', type=float, default=3.0,
                        help='Warmup epochs (from cfg: burn_in equivalent)')
    parser.add_argument('--warmup-momentum', type=float, default=0.8,
                        help='Warmup initial momentum')
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1,
                        help='Warmup initial bias LR')
    parser.add_argument('--box-loss', type=float, default=7.5,
                        help='Box loss gain (from cfg: iou_normalizer)')
    parser.add_argument('--cls-loss', type=float, default=0.5,
                        help='Classification loss gain (from cfg: cls_normalizer)')
    parser.add_argument('--dfl-loss', type=float, default=1.5,
                        help='Distribution focal loss gain')
    parser.add_argument('--iou-threshold', type=float, default=0.7,
                        help='IoU training threshold (from cfg: iou_thresh)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon')
    parser.add_argument('--close-mosaic', type=int, default=10,
                        help='Disable mosaic for final epochs')
    parser.add_argument('--multi-scale', type=lambda x: x.lower() == 'true', default=False,
                        help='Multi-scale training (+/- 50%% img size)')
    parser.add_argument('--rect', type=lambda x: x.lower() == 'true', default=False,
                        help='Rectangular training (non-square images)')
    
    # Device
    parser.add_argument('--device', type=int, default=0)
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='', help='Logging directory')
    
    args = parser.parse_args()
    
    # Build config for stats and export
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'patience': args.patience,
        'lr0': args.lr0,
        'optimizer': args.optimizer,
        'amp': args.amp,
        'pretrained': args.pretrained,
        # Augmentation params for config file
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'grayscale': args.grayscale,
        # Advanced parameters from microspores.cfg
        'warmup_epochs': args.warmup_epochs,
        'warmup_momentum': args.warmup_momentum,
        'warmup_bias_lr': args.warmup_bias_lr,
        'box': args.box_loss,
        'cls': args.cls_loss,
        'dfl': args.dfl_loss,
        'iou': args.iou_threshold,
        'label_smoothing': args.label_smoothing,
        'close_mosaic': args.close_mosaic,
        'multi_scale': args.multi_scale,
        'rect': args.rect,
    }
    
    # Run training
    run_training(
        data_yaml=args.data_yaml,
        model_name=args.model,
        weights_dir=args.weights_dir,
        project_dir=args.project_dir,
        exp_name=args.exp_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        patience=args.patience,
        workers=args.workers,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        grayscale=args.grayscale,
        pretrained=args.pretrained,
        resume=args.resume,
        cache=args.cache,
        amp=args.amp,
        freeze=args.freeze,
        device=args.device,
        # Advanced parameters from microspores.cfg
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        box_loss=args.box_loss,
        cls_loss=args.cls_loss,
        dfl_loss=args.dfl_loss,
        iou_threshold=args.iou_threshold,
        label_smoothing=args.label_smoothing,
        close_mosaic=args.close_mosaic,
        multi_scale=args.multi_scale,
        rect=args.rect,
        log_dir=args.log_dir if args.log_dir else None,
    )
    
    # Generate stats and export additional files
    generate_stats(
        project_dir=args.project_dir,
        exp_name=args.exp_name,
        dataset_dir=args.dataset_dir,
        model_name=args.model,
        config=config,
        img_size=args.img_size,
    )


if __name__ == '__main__':
    main()
