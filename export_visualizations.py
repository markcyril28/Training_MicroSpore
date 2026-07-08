#!/usr/bin/env python3
"""
Standalone script to run export_training_outputs() on existing training folders.

This script fixes training folders where visualizations were not generated 
(e.g., if training was interrupted before post-processing completed).

Usage:
    python export_visualizations.py <experiment_path> [--dataset-dir <path>]

Example:
    python export_visualizations.py trained_models_output/server/04_class_balancing_combination_continue_2/Dataset_2_OPTIMIZATION_best_gray_img1280_bal-manual_auto_e300_b8_lr0_00001_20260201_101533_cont1
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.training.train import (
    export_training_outputs,
    generate_custom_visualizations,
    move_yolo_outputs_to_folders,
    generate_visualization_guides,
    organize_output_folders,
)


def parse_config_from_args_yaml(args_yaml_path: Path) -> dict:
    """Parse configuration from args.yaml file."""
    config = {}
    if args_yaml_path.exists():
        import yaml
        with open(args_yaml_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    return config


def find_classes_file(experiment_path: Path, dataset_dir: str = None) -> Path:
    """Find classes.txt file from various possible locations."""
    # Try to find from args.yaml
    args_yaml = experiment_path / "args.yaml"
    if args_yaml.exists():
        config = parse_config_from_args_yaml(args_yaml)
        if 'data' in config:
            # Read data.yaml to find dataset path
            data_yaml_path = Path(config['data'])
            if data_yaml_path.exists():
                import yaml
                with open(data_yaml_path, 'r') as f:
                    data_config = yaml.safe_load(f) or {}
                if 'path' in data_config:
                    dataset_path = Path(data_config['path'])
                    classes_file = dataset_path / "classes.txt"
                    if classes_file.exists():
                        return classes_file
                    # Try Train subfolder
                    train_classes = dataset_path / "Train" / "classes.txt"
                    if train_classes.exists():
                        return train_classes
    
    # Use provided dataset_dir
    if dataset_dir:
        dataset_path = Path(dataset_dir)
        classes_file = dataset_path / "classes.txt"
        if classes_file.exists():
            return classes_file
        train_classes = dataset_path / "Train" / "classes.txt"
        if train_classes.exists():
            return train_classes
    
    # Fallback: create empty classes file
    return Path("/dev/null")


def export_existing_folder(
    experiment_path: Path,
    dataset_dir: str = None,
    img_size: int = 640,
) -> None:
    """
    Run export_training_outputs on an existing training folder.
    
    Args:
        experiment_path: Path to the training experiment folder
        dataset_dir: Optional path to dataset directory (for classes.txt)
        img_size: Image size (for ONNX export if weights exist)
    """
    print("\n" + "="*70)
    print(f"  Exporting Visualizations for: {experiment_path.name}")
    print("="*70 + "\n")
    
    if not experiment_path.exists():
        print(f"ERROR: Experiment path does not exist: {experiment_path}")
        return
    
    # Read config from args.yaml
    args_yaml = experiment_path / "args.yaml"
    config = parse_config_from_args_yaml(args_yaml)
    
    # Extract model name from folder name or config
    model_name = config.get('model', 'yolov8x')
    if not model_name:
        # Try to extract from folder name
        folder_name = experiment_path.name.lower()
        for model in ['yolo11x', 'yolo11l', 'yolo11m', 'yolov10x', 'yolov9e', 'yolov8x', 'yolov8l', 'yolov5xu']:
            if model in folder_name:
                model_name = model
                break
    
    # Extract img_size from config
    if 'imgsz' in config:
        img_size = config['imgsz']
    elif 'img_size' in config:
        img_size = config['img_size']
    
    # Find classes file
    classes_file = find_classes_file(experiment_path, dataset_dir)
    
    print(f"  Model: {model_name}")
    print(f"  Image Size: {img_size}")
    print(f"  Classes File: {classes_file}")
    print()
    
    # Create folder structure
    folders = organize_output_folders(experiment_path)
    
    # Check if results.csv exists
    results_csv = experiment_path / "results.csv"
    stats_results_csv = experiment_path / "stats" / "epoch_metrics.csv"
    
    if not results_csv.exists() and stats_results_csv.exists():
        # Copy from stats if original was already moved
        import shutil
        shutil.copy(str(stats_results_csv), str(results_csv))
        print("[Info] Restored results.csv from stats/epoch_metrics.csv")
    
    if results_csv.exists():
        # Generate custom visualizations
        print("[1/5] Generating custom visualizations...")
        generate_custom_visualizations(results_csv, folders['viz_curves'])
    else:
        print("[1/5] SKIP: results.csv not found")
    
    # Move YOLO outputs to organized folders
    print("[2/5] Organizing YOLO output files...")
    exp_name = experiment_path.name
    move_yolo_outputs_to_folders(experiment_path, folders, exp_name)

    # Generate visualization guides after files are organized
    print("[3/5] Generating visualization guides...")
    generate_visualization_guides(folders)
    
    # Run full export if weights exist
    weights_dir = experiment_path / "weights"
    best_weights = weights_dir / "best.pt"
    if not best_weights.exists():
        # Check for renamed weights
        for f in weights_dir.glob("*_best.pt"):
            best_weights = f
            break
    
    # Find data_yaml for confusion matrix stats
    data_yaml_path = None
    args_yaml = experiment_path / "args.yaml"
    if args_yaml.exists():
        try:
            import yaml
            with open(args_yaml, 'r') as f:
                args_config = yaml.safe_load(f) or {}
            if 'data' in args_config:
                data_yaml_path = args_config['data']
        except Exception:
            pass
    
    if best_weights.exists() and classes_file.exists() and str(classes_file) != "/dev/null":
        print("[4/5] Running full export (ONNX, configs)...")
        try:
            export_training_outputs(
                experiment_path=experiment_path,
                model_name=model_name,
                config=config,
                classes_file=classes_file,
                img_size=img_size,
            )
        except Exception as e:
            print(f"[4/5] Warning: Full export failed: {e}")
    else:
        print("[4/5] SKIP: Weights or classes file not found")
    
    # Generate confusion matrix statistics if data_yaml available
    if best_weights.exists() and data_yaml_path:
        print("[5/5] Generating confusion matrix statistics...")
        try:
            from modules.training.train import generate_confusion_matrix_stats
            cm_exports = generate_confusion_matrix_stats(
                weights_path=best_weights,
                data_yaml=data_yaml_path,
                stats_dir=folders['stats'],
                matrices_dir=folders['viz_matrices'],
                img_size=img_size,
            )
            if cm_exports:
                print(f"[5/5] Generated {len(cm_exports)} confusion matrix files")
        except ImportError:
            print("[5/5] SKIP: confusion matrix statistics helper is not available")
        except Exception as e:
            print(f"[5/5] Warning: Confusion matrix stats failed: {e}")
    else:
        print("[5/5] SKIP: Confusion matrix stats (no weights or data.yaml)")
    
    # Summary
    print("\n" + "-"*70)
    print("  Export Complete! Folder structure:")
    for name, path in folders.items():
        if path.exists():
            file_count = len(list(path.glob('*')))
            if file_count > 0:
                print(f"    📁 {name}/ ({file_count} files)")
    print("-"*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Export visualizations for existing training folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single folder
  python export_visualizations.py path/to/experiment_folder

  # Export with dataset directory (for classes.txt)
  python export_visualizations.py path/to/experiment --dataset-dir /path/to/dataset
  
  # Export all folders in a directory
  for d in trained_models_output/server/*/; do python export_visualizations.py "$d"; done
"""
    )
    
    parser.add_argument('experiment_path', type=str, 
                        help='Path to training experiment folder')
    parser.add_argument('--dataset-dir', type=str, default=None,
                        help='Path to dataset directory (for classes.txt)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for ONNX export (default: 640)')
    
    args = parser.parse_args()
    
    experiment_path = Path(args.experiment_path).resolve()
    
    # Handle if user passes a parent directory
    if not (experiment_path / "weights").exists():
        # Check if it's a parent with subdirectories
        subdirs = [d for d in experiment_path.iterdir() if d.is_dir() and (d / "weights").exists()]
        if subdirs:
            print(f"Found {len(subdirs)} experiment folder(s) in {experiment_path}")
            for subdir in subdirs:
                export_existing_folder(subdir, args.dataset_dir, args.img_size)
            return
    
    export_existing_folder(experiment_path, args.dataset_dir, args.img_size)


if __name__ == '__main__':
    main()
