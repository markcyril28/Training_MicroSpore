"""
Statistics Module for Microspore Phenotyping
Generates training statistics, metrics, and reports
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
# Import utility functions (avoid circular imports by importing at module level)
from ..utils import load_json, save_json, ensure_dir
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TrainingStats:
    """Training Statistics Manager.
    Collects, saves, and visualizes training metrics.
    Uses centralized JSON utilities (DRY principle).
    """
    
    def __init__(self, experiment_dir: Union[str, Path]):
        """
        Initialize Training Stats.
        
        Args:
            experiment_dir: Path to experiment output directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.stats_file = self.experiment_dir / "training_stats.json"
        self.stats = self._load_or_init_stats()
        
    def _load_or_init_stats(self) -> Dict[str, Any]:
        """Load existing stats or initialize new ones."""
        if self.stats_file.exists():
            return load_json(self.stats_file)
        
        return {
            "experiment_name": self.experiment_dir.name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config": {},
            "training": {},
            "validation": {},
            "model_info": {},
            "dataset_info": {},
            "system_info": {}
        }
    
    def save(self) -> None:
        """Save stats to JSON file using centralized utility."""
        self.stats["updated_at"] = datetime.now().isoformat()
        ensure_dir(self.experiment_dir)
        save_json(self.stats, self.stats_file)
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set training configuration"""
        self.stats["config"] = config
        self.save()
    
    def set_model_info(self, info: Dict[str, Any]) -> None:
        """Set model information"""
        self.stats["model_info"] = info
        self.save()
    
    def set_dataset_info(self, info: Dict[str, Any]) -> None:
        """Set dataset information"""
        self.stats["dataset_info"] = info
        self.save()
    
    def set_system_info(self, info: Dict[str, Any]) -> None:
        """Set system information"""
        self.stats["system_info"] = info
        self.save()
    
    def update_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update training metrics"""
        self.stats["training"].update(metrics)
        self.save()
    
    def update_validation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update validation metrics"""
        self.stats["validation"].update(metrics)
        self.save()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get stats summary"""
        return {
            "experiment": self.stats["experiment_name"],
            "model": self.stats.get("config", {}).get("model", "Unknown"),
            "epochs": self.stats.get("config", {}).get("epochs", 0),
            "best_mAP50": self.stats.get("validation", {}).get("mAP50", 0),
            "best_mAP50_95": self.stats.get("validation", {}).get("mAP50-95", 0),
            "training_time": self.stats.get("training", {}).get("total_time", "Unknown"),
        }


class ModelStatsCollector:
    """
    Collects and organizes statistics from trained YOLO models
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        """
        Initialize Model Stats Collector
        
        Args:
            model_dir: Path to models directory
        """
        self.model_dir = Path(model_dir)
        
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Get list of all experiment directories with their stats
        
        Returns:
            List of experiment info dictionaries
        """
        experiments = []
        
        for exp_dir in self.model_dir.iterdir():
            if exp_dir.is_dir():
                exp_info = self._get_experiment_info(exp_dir)
                if exp_info:
                    experiments.append(exp_info)
        
        # Sort by creation date (newest first)
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return experiments
    
    def _get_experiment_info(self, exp_dir: Path) -> Optional[Dict[str, Any]]:
        """Get info from single experiment directory"""
        info = {
            "name": exp_dir.name,
            "path": str(exp_dir),
            "has_weights": False,
            "has_results": False,
        }
        
        # Check for weights
        weights_dir = exp_dir / "weights"
        if weights_dir.exists():
            info["has_weights"] = True
            info["best_weights"] = str(weights_dir / "best.pt") if (weights_dir / "best.pt").exists() else None
            info["last_weights"] = str(weights_dir / "last.pt") if (weights_dir / "last.pt").exists() else None
        
        # Check for results
        results_csv = exp_dir / "results.csv"
        if results_csv.exists():
            info["has_results"] = True
            info["metrics"] = self._parse_results_csv(results_csv)
        
        # Check for training stats - use centralized load_json
        stats_file = exp_dir / "training_stats.json"
        if stats_file.exists():
            stats = load_json(stats_file)
            info["created_at"] = stats.get("created_at", "")
            info["config"] = stats.get("config", {})
        
        # Check for args.yaml (YOLO default config file)
        args_file = exp_dir / "args.yaml"
        if args_file.exists():
            try:
                import yaml
                with open(args_file, 'r') as f:
                    info["args"] = yaml.safe_load(f)
            except:
                pass
        
        return info
    
    def _parse_results_csv(self, csv_path: Path) -> Dict[str, Any]:
        """Parse YOLO results.csv file"""
        metrics = {
            "epochs_completed": 0,
            "best_mAP50": 0,
            "best_mAP50_95": 0,
            "final_train_loss": 0,
        }
        
        try:
            if PANDAS_AVAILABLE:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                
                metrics["epochs_completed"] = len(df)
                
                # Find mAP columns
                map50_col = [c for c in df.columns if 'mAP50' in c and '95' not in c]
                map5095_col = [c for c in df.columns if 'mAP50-95' in c]
                
                if map50_col:
                    metrics["best_mAP50"] = float(df[map50_col[0]].max())
                if map5095_col:
                    metrics["best_mAP50_95"] = float(df[map5095_col[0]].max())
                
                # Get final losses
                loss_cols = [c for c in df.columns if 'loss' in c.lower()]
                if loss_cols:
                    metrics["final_train_loss"] = float(df[loss_cols[0]].iloc[-1])
            else:
                # Fallback without pandas
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    metrics["epochs_completed"] = len(rows)
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def generate_comparison_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comparison report of all experiments
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Comparison report dictionary
        """
        experiments = self.get_all_experiments()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_experiments": len(experiments),
            "experiments": [],
            "best_model": None
        }
        
        best_map = 0
        for exp in experiments:
            exp_summary = {
                "name": exp["name"],
                "has_weights": exp["has_weights"],
                "epochs": exp.get("metrics", {}).get("epochs_completed", 0),
                "mAP50": exp.get("metrics", {}).get("best_mAP50", 0),
                "mAP50_95": exp.get("metrics", {}).get("best_mAP50_95", 0),
                "best_weights": exp.get("best_weights"),
            }
            report["experiments"].append(exp_summary)
            
            if exp_summary["mAP50_95"] > best_map:
                best_map = exp_summary["mAP50_95"]
                report["best_model"] = exp_summary
        
        # Use centralized save_json utility
        if output_path:
            save_json(report, output_path)
        
        return report


class DatasetStats:
    """
    Dataset Statistics Calculator
    """
    
    def __init__(self, dataset_path: Union[str, Path]):
        """
        Initialize Dataset Stats
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        
    def get_class_distribution(self, split: str = "Train") -> Dict[str, int]:
        """
        Get class distribution from label files
        
        Args:
            split: Dataset split (Train or Test)
            
        Returns:
            Dictionary of class_id: count
        """
        split_path = self.dataset_path / split
        if not split_path.exists():
            return {}
        
        class_counts = Counter()
        
        for label_file in split_path.glob("*.txt"):
            # Skip classes.txt if in same directory
            if label_file.name == "classes.txt":
                continue
            
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
            except:
                continue
        
        return dict(class_counts)
    
    def get_full_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics
        
        Returns:
            Dictionary with full statistics
        """
        stats = {
            "dataset_path": str(self.dataset_path),
            "splits": {},
            "total_images": 0,
            "total_annotations": 0,
            "classes": [],
            "class_distribution": {}
        }
        
        # Load class names
        classes_file = self.dataset_path / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                stats["classes"] = [line.strip() for line in f if line.strip()]
        
        # Get stats for each split
        for split in ["Train", "Test"]:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
            
            images = list(split_path.glob("*.jpg")) + list(split_path.glob("*.png"))
            labels = [f for f in split_path.glob("*.txt") if f.name != "classes.txt"]
            
            split_stats = {
                "images": len(images),
                "labels": len(labels),
                "missing_labels": len(images) - len(labels),
            }
            
            # Get class distribution for this split
            class_dist = self.get_class_distribution(split)
            split_stats["class_distribution"] = class_dist
            split_stats["total_annotations"] = sum(class_dist.values())
            
            stats["splits"][split.lower()] = split_stats
            stats["total_images"] += split_stats["images"]
            stats["total_annotations"] += split_stats["total_annotations"]
        
        # Aggregate class distribution
        for split in ["train", "test"]:
            if split in stats["splits"]:
                for class_id, count in stats["splits"][split].get("class_distribution", {}).items():
                    if class_id not in stats["class_distribution"]:
                        stats["class_distribution"][class_id] = 0
                    stats["class_distribution"][class_id] += count
        
        return stats
    
    def plot_class_distribution(
        self, 
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Optional[str]:
        """
        Plot class distribution bar chart
        
        Args:
            output_path: Path to save plot
            figsize: Figure size
            
        Returns:
            Path to saved plot or None
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not installed. Cannot generate plots.")
            return None
        
        stats = self.get_full_stats()
        class_names = stats.get("classes", [])
        class_dist = stats.get("class_distribution", {})
        
        if not class_dist:
            return None
        
        # Prepare data
        classes = []
        counts = []
        for class_id in sorted(class_dist.keys()):
            if class_id < len(class_names):
                classes.append(class_names[class_id])
            else:
                classes.append(f"Class {class_id}")
            counts.append(class_dist[class_id])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(classes, counts, color='steelblue', edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Dataset Class Distribution', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None


def generate_training_report(
    experiment_dir: Union[str, Path],
    dataset_path: Union[str, Path],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive training report
    
    Args:
        experiment_dir: Path to experiment directory
        dataset_path: Path to dataset
        output_path: Path to save report
        
    Returns:
        Report dictionary
    """
    experiment_dir = Path(experiment_dir)
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "experiment_name": experiment_dir.name,
        "experiment_path": str(experiment_dir),
    }
    
    # Get training stats
    stats = TrainingStats(experiment_dir)
    report["training_config"] = stats.stats.get("config", {})
    report["model_info"] = stats.stats.get("model_info", {})
    report["training_metrics"] = stats.stats.get("training", {})
    report["validation_metrics"] = stats.stats.get("validation", {})
    
    # Get dataset stats
    dataset_stats = DatasetStats(dataset_path)
    report["dataset_stats"] = dataset_stats.get_full_stats()
    
    # Get model files
    weights_dir = experiment_dir / "weights"
    if weights_dir.exists():
        report["model_files"] = {
            "best_weights": str(weights_dir / "best.pt") if (weights_dir / "best.pt").exists() else None,
            "last_weights": str(weights_dir / "last.pt") if (weights_dir / "last.pt").exists() else None,
        }
    
    # Save report using centralized utility
    if output_path:
        save_json(report, output_path)
    
    return report


# Export public API
__all__ = [
    'TrainingStats',
    'ModelStatsCollector',
    'DatasetStats',
    'generate_training_report'
]
