"""Training Script Module for Microspore Phenotyping
Command-line interface for YOLO model training.
Implements DRY principle - uses centralized config and utilities.
"""

import argparse
import csv
import json
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
from ..utils import check_gpu, get_system_info, ensure_dir, update_data_yaml_path
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
        model_path: Path to the best.pt weights (already renamed to {exp_name}_best.pt)
        output_dir: Directory to save ONNX file (weights folder)
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
        
        # Move to output directory with naming: {exp_name}_best.onnx
        if onnx_path and Path(onnx_path).exists():
            output_onnx = output_dir / f"{exp_name}_best.onnx"
            shutil.copy(onnx_path, output_onnx)
            # Clean up temp ONNX file
            Path(onnx_path).unlink()
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
        # Add timestamp to filename as per spec
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"GENERAL_GUIDE_and_ANALYSIS_{timestamp}.md"
        
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


def generate_general_results(
    output_dir: Path,
    results_csv: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    exp_name: str = "",
) -> Optional[Path]:
    """
    Generate GENERAL_RESULTS_{timestamp}.md with concise results only.
    
    Args:
        output_dir: Directory to save the results file
        results_csv: Path to results.csv for extracting metrics
        config: Training configuration dictionary
        exp_name: Experiment folder name
        
    Returns:
        Path to generated file, or None if generation failed
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"GENERAL_RESULTS_{timestamp}.md"
        
        # Extract metrics
        metrics = _extract_concise_results(results_csv, config, exp_name)
        
        with open(output_path, 'w') as f:
            f.write(metrics)
        
        print(f"[Export] GENERAL_RESULTS saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"[Export] GENERAL_RESULTS generation failed: {e}")
    return None


def _extract_concise_results(
    results_csv: Optional[Path],
    config: Optional[Dict[str, Any]],
    exp_name: str,
) -> str:
    """
    Extract concise training results for GENERAL_RESULTS.md.
    
    Returns:
        Formatted markdown string with concise results
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Default values
    total_epochs = 0
    best_epoch = 0
    best_map50 = 0.0
    best_map5095 = 0.0
    final_precision = 0.0
    final_recall = 0.0
    final_f1 = 0.0
    
    if results_csv and results_csv.exists():
        try:
            import csv
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if rows:
                total_epochs = len(rows)
                cleaned_rows = [{k.strip(): v for k, v in row.items()} for row in rows]
                sample_row = cleaned_rows[0]
                
                # Find columns
                map50_col = next((c for c in sample_row.keys() if 'mAP50' in c and '95' not in c), None)
                map5095_col = next((c for c in sample_row.keys() if 'mAP50-95' in c), None)
                prec_col = next((c for c in sample_row.keys() if 'precision' in c.lower()), None)
                rec_col = next((c for c in sample_row.keys() if 'recall' in c.lower()), None)
                
                # Find best mAP50-95 and corresponding epoch
                if map5095_col:
                    best_val = 0.0
                    for i, row in enumerate(cleaned_rows):
                        val = float(row.get(map5095_col, 0) or 0)
                        if val > best_val:
                            best_val = val
                            best_epoch = i + 1
                    best_map5095 = best_val
                
                if map50_col:
                    best_map50 = max(float(r.get(map50_col, 0) or 0) for r in cleaned_rows)
                
                # Final values
                last_row = cleaned_rows[-1]
                if prec_col:
                    final_precision = float(last_row.get(prec_col, 0) or 0)
                if rec_col:
                    final_recall = float(last_row.get(rec_col, 0) or 0)
                
                # Calculate F1
                if final_precision + final_recall > 0:
                    final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall)
        except Exception:
            pass
    
    # Config info
    model = config.get('model', 'N/A') if config else 'N/A'
    epochs_cfg = config.get('epochs', 'N/A') if config else 'N/A'
    batch = config.get('batch_size', 'N/A') if config else 'N/A'
    img_size = config.get('img_size', 'N/A') if config else 'N/A'
    lr = config.get('lr', 'N/A') if config else 'N/A'
    optimizer = config.get('optimizer', 'N/A') if config else 'N/A'
    
    # Build concise results
    results = f"""# Training Results Summary
**Experiment:** {exp_name}  
**Generated:** {timestamp}

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Best mAP@50** | {best_map50:.4f} ({best_map50*100:.1f}%) |
| **Best mAP@50-95** | {best_map5095:.4f} ({best_map5095*100:.1f}%) |
| **Best Epoch** | {best_epoch} / {total_epochs} |
| **Final Precision** | {final_precision:.4f} ({final_precision*100:.1f}%) |
| **Final Recall** | {final_recall:.4f} ({final_recall*100:.1f}%) |
| **Final F1 Score** | {final_f1:.4f} ({final_f1*100:.1f}%) |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | {model} |
| Epochs | {epochs_cfg} |
| Batch Size | {batch} |
| Image Size | {img_size} |
| Learning Rate | {lr} |
| Optimizer | {optimizer} |

---

## Quick Assessment

"""
    # Add quick assessment based on metrics
    if best_map5095 >= 0.5:
        results += "✅ **GOOD** - mAP@50-95 ≥ 50% indicates strong localization\n"
    elif best_map5095 >= 0.3:
        results += "⚠️ **MODERATE** - mAP@50-95 between 30-50%\n"
    else:
        results += "❌ **NEEDS IMPROVEMENT** - mAP@50-95 < 30%\n"
    
    if best_map50 >= 0.7:
        results += "✅ **GOOD** - mAP@50 ≥ 70% indicates reliable detection\n"
    elif best_map50 >= 0.5:
        results += "⚠️ **MODERATE** - mAP@50 between 50-70%\n"
    else:
        results += "❌ **NEEDS IMPROVEMENT** - mAP@50 < 50%\n"
    
    return results


def create_training_config(
    output_dir: Path,
    model_name: str,
    config: Dict[str, Any],
    num_classes: int,
    exp_name: str = "",
) -> Optional[Path]:
    """
    Create a training configuration summary file (.cfg).
    
    Args:
        output_dir: Directory to save config file
        model_name: YOLO model name
        config: Training configuration dictionary
        num_classes: Number of classes in the dataset
        exp_name: Experiment folder name for file naming
        
    Returns:
        Path to config file, or None if creation failed
    """
    try:
        # Use exp_name if provided, otherwise fall back to model name
        file_prefix = exp_name if exp_name else model_name.replace('.pt', '').replace('.', '')
        output_path = output_dir / f"{file_prefix}_args.cfg"
        
        # Build config content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cfg_content = f"""# Microspore Phenotyping Training Configuration
# Generated: {timestamp}
# Experiment: {exp_name}
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
weights_file={exp_name}_best.pt
onnx_file={exp_name}_best.onnx
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
    Follows specification:
        weights/
            configs/
        stats/
        visualizations/
            curves/
            matrices/
            samples/
            overviews/
    
    Args:
        experiment_path: Path to training experiment folder
        
    Returns:
        Dictionary of folder paths
    """
    folders = {
        # Main folders
        'weights': experiment_path / "weights",
        'weights_configs': experiment_path / "weights" / "configs",
        'stats': experiment_path / "stats",
        'visualizations': experiment_path / "visualizations",
        # Visualization subfolders
        'viz_curves': experiment_path / "visualizations" / "curves",
        'viz_matrices': experiment_path / "visualizations" / "matrices",
        'viz_samples': experiment_path / "visualizations" / "samples",
        'viz_overviews': experiment_path / "visualizations" / "overviews",
    }
    
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    return folders


def move_yolo_outputs_to_folders(experiment_path: Path, folders: Dict[str, Path], exp_name: str) -> None:
    """
    Move YOLO-generated outputs to organized folders with proper naming.
    
    Visualization organization:
        curves/ - loss_curves, precision_recall, PR_curve, P_curve, R_curve, F1_curve
        matrices/ - confusion_matrix, labels_correlogram
        samples/ - val_batch_predictions, train_batch_samples
        overviews/ - results, Box_curve, labels
    
    Args:
        experiment_path: Path to training experiment folder
        folders: Dictionary of organized folder paths
        exp_name: Experiment folder name for file naming
    """
    import shutil as sh
    
    # Visualization file mappings: original YOLO name -> (new name, destination folder)
    viz_mappings = {
        # Curves
        'results.png': ('results.png', folders['viz_overviews']),
        'F1_curve.png': ('F1_curve.png', folders['viz_curves']),
        'P_curve.png': ('P_curve.png', folders['viz_curves']),
        'R_curve.png': ('R_curve.png', folders['viz_curves']),
        'PR_curve.png': ('PR_curve.png', folders['viz_curves']),
        'Box_curve.png': ('Box_curve.png', folders['viz_overviews']),
        # Matrices
        'confusion_matrix.png': ('confusion_matrix.png', folders['viz_matrices']),
        'confusion_matrix_normalized.png': ('confusion_matrix_normalized.png', folders['viz_matrices']),
        'labels_correlogram.jpg': ('labels_correlogram.png', folders['viz_matrices']),
        'labels.jpg': ('labels.png', folders['viz_overviews']),
        # Samples
        'train_batch0.jpg': ('train_batch_samples.png', folders['viz_samples']),
        'train_batch1.jpg': ('train_batch_samples_1.png', folders['viz_samples']),
        'train_batch2.jpg': ('train_batch_samples_2.png', folders['viz_samples']),
        'val_batch0_labels.jpg': ('val_batch_labels.png', folders['viz_samples']),
        'val_batch0_pred.jpg': ('val_batch_predictions.png', folders['viz_samples']),
        'val_batch1_pred.jpg': ('val_batch_predictions_1.png', folders['viz_samples']),
        'val_batch2_pred.jpg': ('val_batch_predictions_2.png', folders['viz_samples']),
    }
    
    # Move visualization files with proper naming
    for original_name, (new_name, dest_folder) in viz_mappings.items():
        src_file = experiment_path / original_name
        if src_file.exists():
            dest = dest_folder / new_name
            sh.copy(str(src_file), str(dest))
            src_file.unlink()  # Remove original
    
    # Move remaining images to overviews
    viz_patterns = ['*.png', '*.jpg', '*.jpeg']
    for pattern in viz_patterns:
        for f in experiment_path.glob(pattern):
            if f.is_file():
                dest = folders['viz_overviews'] / f.name
                sh.move(str(f), str(dest))
    
    # Keep args.yaml at root level (per spec)
    # No movement needed as YOLO creates it there
    
    # Copy args.yaml to weights/configs folder as well with proper naming
    args_file = experiment_path / "args.yaml"
    if args_file.exists():
        sh.copy(str(args_file), str(folders['weights_configs'] / f"{exp_name}_args.yaml"))


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
    weights_dir = folders['weights']
    best_weights = weights_dir / "best.pt"
    exp_name = experiment_path.name  # Use folder name for file naming
    
    # Read class names for config
    classes = []
    num_classes = 0
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
            num_classes = len(classes)
    config['classes'] = classes
    
    # Rename weight files with dynamic naming: {exp_name}_best.pt, {exp_name}_last.pt
    rename_weight_files(weights_dir, exp_name)
    best_weights = weights_dir / f"{exp_name}_best.pt"
    
    # Export ONNX to weights folder (named based on experiment folder name)
    if best_weights.exists():
        exports['onnx'] = export_to_onnx(best_weights, weights_dir, exp_name, img_size)
    else:
        print(f"[Export] Warning: {exp_name}_best.pt not found at {weights_dir}")
        exports['onnx'] = None
    
    # Create config files in weights/configs folder
    exports['config_cfg'] = create_training_config(folders['weights_configs'], model_name, config, num_classes, exp_name)
    exports['config_yaml'] = copy_args_yaml_with_naming(experiment_path, folders['weights_configs'], exp_name)
    
    # Copy general interpretation guide to root folder (with accuracy summary)
    results_csv = experiment_path / "results.csv"
    exports['general_guide'] = copy_general_guide(experiment_path, results_csv=results_csv)
    
    # Generate concise results file
    exports['general_results'] = generate_general_results(experiment_path, results_csv=results_csv, config=config, exp_name=exp_name)
    
    # Generate stats files
    exports.update(generate_stats_files(experiment_path, folders['stats'], config, exp_name))
    
    # Generate custom visualizations (loss_curves.png, precision_recall.png)
    generate_custom_visualizations(results_csv, folders['viz_curves'])
    
    # Generate visualization interpretation guides
    generate_visualization_guides(folders)
    
    # Move YOLO-generated outputs to organized folders
    move_yolo_outputs_to_folders(experiment_path, folders, exp_name)
    
    # Clean up results.csv from root (moved to stats)
    if results_csv.exists():
        results_csv.unlink()
    
    print("\n" + "-"*60)
    print("  Folder Structure:")
    for name, path in folders.items():
        if not name.startswith('viz_') and name != 'weights_configs':
            file_count = len(list(path.glob('*'))) if path.exists() else 0
            print(f"    📁 {name}/ ({file_count} files)")
    print("-"*60)
    print("  Export Summary:")
    for key, path in exports.items():
        status = "✓" if path else "✗"
        print(f"    [{status}] {key}: {path.name if path else 'Failed'}")
    print("-"*60 + "\n")
    
    return exports


def rename_weight_files(weights_dir: Path, exp_name: str) -> None:
    """
    Rename weight files with dynamic naming convention.
    best.pt -> {exp_name}_best.pt
    last.pt -> {exp_name}_last.pt
    
    Args:
        weights_dir: Path to weights directory
        exp_name: Experiment folder name
    """
    mappings = [
        ('best.pt', f'{exp_name}_best.pt'),
        ('last.pt', f'{exp_name}_last.pt'),
    ]
    
    for old_name, new_name in mappings:
        old_path = weights_dir / old_name
        new_path = weights_dir / new_name
        if old_path.exists() and not new_path.exists():
            shutil.move(str(old_path), str(new_path))
            print(f"[Export] Renamed: {old_name} -> {new_name}")


def copy_args_yaml_with_naming(experiment_path: Path, configs_dir: Path, exp_name: str) -> Optional[Path]:
    """
    Copy args.yaml to configs folder with proper naming.
    
    Args:
        experiment_path: Path to experiment folder
        configs_dir: Destination configs directory
        exp_name: Experiment folder name
        
    Returns:
        Path to copied file or None
    """
    try:
        args_file = experiment_path / "args.yaml"
        if args_file.exists():
            dest = configs_dir / f"{exp_name}_args.yaml"
            shutil.copy(str(args_file), str(dest))
            return dest
    except Exception as e:
        print(f"[Export] Failed to copy args.yaml: {e}")
    return None


def generate_stats_files(
    experiment_path: Path,
    stats_dir: Path,
    config: Dict[str, Any],
    exp_name: str,
) -> Dict[str, Optional[Path]]:
    """
    Generate stats files as per specification:
    - epoch_metrics.csv
    - training_summary.json
    - model_comparison.csv
    - hardware_stats.csv
    
    Args:
        experiment_path: Path to experiment folder
        stats_dir: Stats directory path
        config: Training configuration
        exp_name: Experiment name
        
    Returns:
        Dictionary of generated file paths
    """
    exports = {}
    results_csv = experiment_path / "results.csv"
    
    try:
        # 1. epoch_metrics.csv - Per-epoch: loss, mAP, precision, recall, lr
        if results_csv.exists():
            epoch_metrics_path = stats_dir / "epoch_metrics.csv"
            shutil.copy(str(results_csv), str(epoch_metrics_path))
            exports['epoch_metrics'] = epoch_metrics_path
            print(f"[Export] epoch_metrics.csv saved")
        
        # 2. training_summary.json - Final stats: best_epoch, total_time, best_mAP, config
        summary_path = stats_dir / "training_summary.json"
        summary = generate_training_summary(results_csv, config, exp_name)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        exports['training_summary'] = summary_path
        print(f"[Export] training_summary.json saved")
        
        # 3. model_comparison.csv - Columns for comparing multiple runs
        comparison_path = stats_dir / "model_comparison.csv"
        generate_model_comparison_csv(results_csv, config, comparison_path, exp_name)
        exports['model_comparison'] = comparison_path
        print(f"[Export] model_comparison.csv saved")
        
        # 4. hardware_stats.csv - GPU/CPU usage (if available from logs)
        hardware_path = stats_dir / "hardware_stats.csv"
        generate_hardware_stats_csv(experiment_path, hardware_path)
        exports['hardware_stats'] = hardware_path
        print(f"[Export] hardware_stats.csv saved")
        
    except Exception as e:
        print(f"[Export] Stats generation error: {e}")
    
    return exports


def generate_training_summary(results_csv: Path, config: Dict[str, Any], exp_name: str) -> Dict[str, Any]:
    """Generate training_summary.json content."""
    import csv
    
    summary = {
        "experiment_name": exp_name,
        "generated_at": datetime.now().isoformat(),
        "config": config,
        "best_epoch": 0,
        "total_epochs": 0,
        "best_mAP50": 0.0,
        "best_mAP50_95": 0.0,
        "final_precision": 0.0,
        "final_recall": 0.0,
    }
    
    if not results_csv.exists():
        return summary
    
    try:
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if rows:
            summary["total_epochs"] = len(rows)
            
            # Clean column names
            cleaned_rows = [{k.strip(): v for k, v in row.items()} for row in rows]
            
            # Find mAP columns
            sample_row = cleaned_rows[0]
            map50_col = next((c for c in sample_row.keys() if 'mAP50' in c and '95' not in c), None)
            map5095_col = next((c for c in sample_row.keys() if 'mAP50-95' in c), None)
            prec_col = next((c for c in sample_row.keys() if 'precision' in c.lower()), None)
            rec_col = next((c for c in sample_row.keys() if 'recall' in c.lower()), None)
            
            # Find best epoch
            if map5095_col:
                best_idx = 0
                best_val = 0.0
                for i, row in enumerate(cleaned_rows):
                    val = float(row.get(map5095_col, 0) or 0)
                    if val > best_val:
                        best_val = val
                        best_idx = i
                summary["best_epoch"] = best_idx + 1
                summary["best_mAP50_95"] = best_val
            
            if map50_col:
                summary["best_mAP50"] = max(float(r.get(map50_col, 0) or 0) for r in cleaned_rows)
            
            # Final values
            last_row = cleaned_rows[-1]
            if prec_col:
                summary["final_precision"] = float(last_row.get(prec_col, 0) or 0)
            if rec_col:
                summary["final_recall"] = float(last_row.get(rec_col, 0) or 0)
                
    except Exception as e:
        summary["error"] = str(e)
    
    return summary


def generate_model_comparison_csv(results_csv: Path, config: Dict[str, Any], output_path: Path, exp_name: str) -> None:
    """Generate model_comparison.csv with key metrics for comparing runs."""
    import csv
    
    # Calculate metrics
    metrics = {
        "experiment": exp_name,
        "model": config.get('model', ''),
        "epochs": config.get('epochs', 0),
        "img_size": config.get('img_size', 0),
        "optimizer": config.get('optimizer', ''),
        "batch": config.get('batch_size', 0),
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "train_time": "N/A",
    }
    
    if results_csv.exists():
        try:
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if rows:
                cleaned_rows = [{k.strip(): v for k, v in row.items()} for row in rows]
                sample_row = cleaned_rows[0]
                
                map50_col = next((c for c in sample_row.keys() if 'mAP50' in c and '95' not in c), None)
                map5095_col = next((c for c in sample_row.keys() if 'mAP50-95' in c), None)
                prec_col = next((c for c in sample_row.keys() if 'precision' in c.lower()), None)
                rec_col = next((c for c in sample_row.keys() if 'recall' in c.lower()), None)
                
                last_row = cleaned_rows[-1]
                if map50_col:
                    metrics["mAP50"] = float(last_row.get(map50_col, 0) or 0)
                if map5095_col:
                    metrics["mAP50_95"] = float(last_row.get(map5095_col, 0) or 0)
                if prec_col:
                    metrics["precision"] = float(last_row.get(prec_col, 0) or 0)
                if rec_col:
                    metrics["recall"] = float(last_row.get(rec_col, 0) or 0)
                
                # Calculate F1
                p, r = metrics["precision"], metrics["recall"]
                metrics["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        except Exception:
            pass
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)


def generate_hardware_stats_csv(experiment_path: Path, output_path: Path) -> None:
    """Generate hardware_stats.csv placeholder."""
    import csv
    
    # Check for GPU logs in parent logs folder
    headers = ["epoch", "gpu_util", "vram_peak", "cpu_util", "ram_peak", "throughput_img_sec"]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        # Placeholder row - actual data would come from monitoring
        writer.writerow(["1", "N/A", "N/A", "N/A", "N/A", "N/A"])


def generate_custom_visualizations(results_csv: Path, curves_folder: Path) -> None:
    """
    Generate custom visualization plots from results.csv data.
    Creates loss_curves.png and precision_recall.png as per specification.
    
    Args:
        results_csv: Path to results.csv file
        curves_folder: Path to visualizations/curves folder
    """
    if not results_csv.exists():
        print("[Viz] Warning: results.csv not found, skipping custom visualizations")
        return
    
    try:
        import matplotlib.pyplot as plt
        import csv
        
        # Read CSV data
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return
        
        # Clean column names
        cleaned_rows = [{k.strip(): v for k, v in row.items()} for row in rows]
        epochs = list(range(1, len(cleaned_rows) + 1))
        
        # Find columns dynamically
        sample_row = cleaned_rows[0]
        
        # 1. Generate loss_curves.png
        try:
            loss_cols = {
                'train_box': next((c for c in sample_row.keys() if 'train/box_loss' in c.lower()), None),
                'train_cls': next((c for c in sample_row.keys() if 'train/cls_loss' in c.lower()), None),
                'train_dfl': next((c for c in sample_row.keys() if 'train/dfl_loss' in c.lower()), None),
                'val_box': next((c for c in sample_row.keys() if 'val/box_loss' in c.lower()), None),
                'val_cls': next((c for c in sample_row.keys() if 'val/cls_loss' in c.lower()), None),
                'val_dfl': next((c for c in sample_row.keys() if 'val/dfl_loss' in c.lower()), None),
            }
            
            if any(loss_cols.values()):
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                fig.suptitle('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
                
                # Box Loss
                if loss_cols['train_box'] and loss_cols['val_box']:
                    train_box = [float(r.get(loss_cols['train_box'], 0) or 0) for r in cleaned_rows]
                    val_box = [float(r.get(loss_cols['val_box'], 0) or 0) for r in cleaned_rows]
                    axes[0].plot(epochs, train_box, label='Train', color='blue', linewidth=2)
                    axes[0].plot(epochs, val_box, label='Validation', color='orange', linewidth=2)
                    axes[0].set_title('Box Loss')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                
                # Classification Loss
                if loss_cols['train_cls'] and loss_cols['val_cls']:
                    train_cls = [float(r.get(loss_cols['train_cls'], 0) or 0) for r in cleaned_rows]
                    val_cls = [float(r.get(loss_cols['val_cls'], 0) or 0) for r in cleaned_rows]
                    axes[1].plot(epochs, train_cls, label='Train', color='blue', linewidth=2)
                    axes[1].plot(epochs, val_cls, label='Validation', color='orange', linewidth=2)
                    axes[1].set_title('Classification Loss')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Loss')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                
                # DFL Loss
                if loss_cols['train_dfl'] and loss_cols['val_dfl']:
                    train_dfl = [float(r.get(loss_cols['train_dfl'], 0) or 0) for r in cleaned_rows]
                    val_dfl = [float(r.get(loss_cols['val_dfl'], 0) or 0) for r in cleaned_rows]
                    axes[2].plot(epochs, train_dfl, label='Train', color='blue', linewidth=2)
                    axes[2].plot(epochs, val_dfl, label='Validation', color='orange', linewidth=2)
                    axes[2].set_title('DFL Loss')
                    axes[2].set_xlabel('Epoch')
                    axes[2].set_ylabel('Loss')
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                loss_curves_path = curves_folder / 'loss_curves.png'
                plt.savefig(loss_curves_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[Viz] loss_curves.png generated")
        except Exception as e:
            print(f"[Viz] Error generating loss_curves.png: {e}")
        
        # 2. Generate precision_recall.png
        try:
            metric_cols = {
                'precision': next((c for c in sample_row.keys() if 'precision' in c.lower() and 'all' in c.lower()), None),
                'recall': next((c for c in sample_row.keys() if 'recall' in c.lower() and 'all' in c.lower()), None),
                'map50': next((c for c in sample_row.keys() if 'map50' in c.lower() and '95' not in c.lower()), None),
                'map5095': next((c for c in sample_row.keys() if 'map50-95' in c.lower()), None),
            }
            
            if any(metric_cols.values()):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.suptitle('Precision, Recall, and mAP over Epochs', fontsize=14, fontweight='bold')
                
                # Precision and Recall
                if metric_cols['precision'] and metric_cols['recall']:
                    precision = [float(r.get(metric_cols['precision'], 0) or 0) for r in cleaned_rows]
                    recall = [float(r.get(metric_cols['recall'], 0) or 0) for r in cleaned_rows]
                    axes[0].plot(epochs, precision, label='Precision', color='green', linewidth=2, marker='o', markersize=4)
                    axes[0].plot(epochs, recall, label='Recall', color='red', linewidth=2, marker='s', markersize=4)
                    axes[0].set_title('Precision & Recall')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Score')
                    axes[0].set_ylim([0, 1.05])
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                
                # mAP
                if metric_cols['map50'] and metric_cols['map5095']:
                    map50 = [float(r.get(metric_cols['map50'], 0) or 0) for r in cleaned_rows]
                    map5095 = [float(r.get(metric_cols['map5095'], 0) or 0) for r in cleaned_rows]
                    axes[1].plot(epochs, map50, label='mAP@50', color='blue', linewidth=2, marker='o', markersize=4)
                    axes[1].plot(epochs, map5095, label='mAP@50-95', color='purple', linewidth=2, marker='s', markersize=4)
                    axes[1].set_title('Mean Average Precision')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('mAP')
                    axes[1].set_ylim([0, 1.05])
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                precision_recall_path = curves_folder / 'precision_recall.png'
                plt.savefig(precision_recall_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[Viz] precision_recall.png generated")
        except Exception as e:
            print(f"[Viz] Error generating precision_recall.png: {e}")
            
    except ImportError:
        print("[Viz] matplotlib not available, skipping custom visualizations")
    except Exception as e:
        print(f"[Viz] Error generating custom visualizations: {e}")


def generate_visualization_guides(folders: Dict[str, Path]) -> None:
    """
    Generate interpretation guide txt files for each visualization.
    
    Creates *_guide_and_interpretation.txt files for:
    - curves/: loss_curves, precision_recall, PR_curve, P_curve, R_curve, F1_curve
    - matrices/: confusion_matrix, labels_correlogram
    - samples/: val_batch_predictions, train_batch_samples
    - overviews/: results, Box_curve, labels
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Curves guides
    curves_guides = {
        'loss_curves': """# Loss Curves Interpretation Guide
Generated: {timestamp}

## What This Shows
Training and validation loss curves over epochs.

## Key Metrics
- Box Loss: Bounding box regression accuracy
- Classification Loss: Class prediction accuracy
- DFL Loss: Distribution focal loss

## Interpretation
- GOOD: Both losses decrease smoothly and converge
- WARNING: Val loss increases while train loss decreases = Overfitting
- GOOD: Small gap between train/val = Good generalization
""",
        'precision_recall': """# Precision-Recall Interpretation Guide
Generated: {timestamp}

## What This Shows
Trade-off between precision and recall across confidence thresholds.

## Key Metrics
- Precision: TP / (TP + FP) - How many detections are correct
- Recall: TP / (TP + FN) - How many objects were found

## Interpretation
- GOOD: Curve stays high (close to top-right corner)
- Area under curve (AUC) = Average Precision (AP)
- Higher AUC = Better model performance
""",
        'PR_curve': """# PR Curve (Precision-Recall Curve) Guide
Generated: {timestamp}

## What This Shows
Precision vs Recall at different confidence thresholds.

## Interpretation
- Ideal curve hugs top-right corner
- Steep drop indicates confidence threshold sensitivity
- Use to choose optimal confidence threshold for deployment
""",
        'P_curve': """# P Curve (Precision-Confidence) Guide
Generated: {timestamp}

## What This Shows
Precision at different confidence thresholds.

## Interpretation
- Higher confidence = Higher precision (fewer false positives)
- Find the knee point for optimal confidence threshold
- Steep curves indicate clear decision boundaries
""",
        'R_curve': """# R Curve (Recall-Confidence) Guide
Generated: {timestamp}

## What This Shows
Recall at different confidence thresholds.

## Interpretation
- Lower confidence = Higher recall (more detections)
- Trade-off: More detections = more false positives
- Use for applications where missing objects is costly
""",
        'F1_curve': """# F1 Curve Guide
Generated: {timestamp}

## What This Shows
F1 score (harmonic mean of precision and recall) vs confidence.

## Key Formula
F1 = 2 * (Precision * Recall) / (Precision + Recall)

## Interpretation
- Peak indicates optimal confidence threshold
- GOOD: High peak (>0.8) with broad plateau
- Use peak confidence for balanced precision/recall
""",
    }
    
    # Matrices guides
    matrices_guides = {
        'confusion_matrix': """# Confusion Matrix Interpretation Guide
Generated: {timestamp}

## What This Shows
Predicted vs actual class distribution for all detections.

## Reading the Matrix
- Diagonal: Correct predictions (True Positives)
- Off-diagonal: Misclassifications
- Row: Actual class, Column: Predicted class

## Interpretation
- GOOD: Strong diagonal, weak off-diagonal
- Common confusions appear as bright off-diagonal cells
- Use to identify problematic class pairs
""",
        'labels_correlogram': """# Labels Correlogram Interpretation Guide
Generated: {timestamp}

## What This Shows
Correlation between bounding box dimensions and positions.

## Subplots
- x, y: Center position distributions
- width, height: Box size distributions
- Scatter plots: Correlations between dimensions

## Interpretation
- Clustered distributions = Consistent object sizes/positions
- Wide spreads = High variability in dataset
- Use to understand dataset characteristics
""",
    }
    
    # Samples guides
    samples_guides = {
        'val_batch_predictions': """# Validation Batch Predictions Guide
Generated: {timestamp}

## What This Shows
Model predictions on validation images with bounding boxes.

## Elements
- Colored boxes: Predicted bounding boxes
- Labels: Class name and confidence score
- Ground truth may be shown for comparison

## Interpretation
- Check box alignment with objects
- Verify class predictions
- Note confidence scores for typical detections
""",
        'train_batch_samples': """# Training Batch Samples Guide
Generated: {timestamp}

## What This Shows
Augmented training images as seen by the model.

## Elements
- Applied augmentations (mosaic, flip, color jitter)
- Ground truth bounding boxes
- Class labels

## Interpretation
- Verify augmentations are appropriate
- Check label correctness
- Ensure objects remain recognizable after augmentation
""",
    }
    
    # Overviews guides
    overviews_guides = {
        'results': """# Results Overview Interpretation Guide
Generated: {timestamp}

## What This Shows
Combined training metrics dashboard.

## Panels
- Loss curves (train/val for box, cls, dfl)
- Precision, Recall over epochs
- mAP50, mAP50-95 over epochs

## Interpretation
- All metrics should improve over epochs
- Convergence indicates training completion
- Divergence between train/val indicates overfitting
""",
        'Box_curve': """# Box Curve Interpretation Guide
Generated: {timestamp}

## What This Shows
Bounding box IoU (Intersection over Union) distribution.

## Interpretation
- Higher IoU = Better localization
- IoU > 0.5: Acceptable detection
- IoU > 0.75: Good localization
- IoU > 0.9: Excellent localization
""",
        'labels': """# Labels Distribution Guide
Generated: {timestamp}

## What This Shows
Dataset label statistics and distributions.

## Elements
- Class frequency histogram
- Bounding box center heatmap
- Box dimension distribution

## Interpretation
- Check for class imbalance
- Identify position/size biases
- Use to understand dataset characteristics
""",
    }
    
    # Write all guides
    all_guides = [
        (folders['viz_curves'], curves_guides),
        (folders['viz_matrices'], matrices_guides),
        (folders['viz_samples'], samples_guides),
        (folders['viz_overviews'], overviews_guides),
    ]
    
    for folder, guides in all_guides:
        for name, content in guides.items():
            guide_path = folder / f"{name}_guide_and_interpretation.txt"
            with open(guide_path, 'w') as f:
                f.write(content.format(timestamp=timestamp))


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
    
    # Always clean up model file if it exists in current working directory
    # (Ultralytics sometimes downloads/creates .pt file in cwd as a side effect)
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
    # Class focus parameters
    class_focus_mode: str = "none",
    class_weights: str = "{}",
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
    # ===========================================================================
    # PORTABILITY: Update data.yaml path to current location
    # This ensures the project works on any machine/directory
    # ===========================================================================
    try:
        data_yaml = str(update_data_yaml_path(data_yaml))
    except Exception as e:
        print(f"[Warning] Could not update data.yaml path: {e}")
        # Continue with original path
    
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
    
    # Log class focus mode
    if class_focus_mode != "none":
        print(f"[Training] Class focus mode: {class_focus_mode}")
        try:
            weights_dict = json.loads(class_weights) if class_weights else {}
            if weights_dict:
                print("[Training] Class weights for balancing:")
                for cls_name, weight in sorted(weights_dict.items(), key=lambda x: -x[1]):
                    boost_status = "(boosted)" if weight > 1.0 else ""
                    print(f"    {cls_name}: {weight:.2f}x {boost_status}")
                print("[Training] Note: Class weights are logged for reference.")
                print("           Future versions may implement oversampling based on these weights.")
        except (json.JSONDecodeError, TypeError):
            print(f"[Training] Class weights: {class_weights}")
    
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
    
    # Class focus / class imbalance parameters
    parser.add_argument('--class-focus-mode', type=str, default='none',
                        choices=['none', 'manual', 'auto', 'sqrt'],
                        help='Class focus mode: none, manual, auto, sqrt')
    parser.add_argument('--class-weights', type=str, default='{}',
                        help='JSON string of class weights for oversampling')
    
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
        # Class focus parameters for addressing class imbalance
        'class_focus_mode': args.class_focus_mode,
        'class_weights': args.class_weights,
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
        # Class focus parameters
        class_focus_mode=args.class_focus_mode,
        class_weights=args.class_weights,
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
