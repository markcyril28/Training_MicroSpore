"""
Microspore Phenotyping - Python Logging Utilities
Comprehensive logging for ML training metrics and visualization
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# Get base directory (C_TRAINING - grandparent of logging folder)
# Path: modules/logging/logging.py -> modules/logging -> modules -> C_TRAINING
BASE_DIR = Path(__file__).parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"


@dataclass
class TrainingMetrics:
    """Container for training metrics at a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    learning_rate: float = 0.0
    images_per_sec: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class CheckpointInfo:
    """Information about a saved checkpoint."""
    epoch: int
    metric_name: str
    metric_value: float
    checkpoint_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TrainingLogger:
    """
    Comprehensive training logger for ML experiments.
    Handles metrics, checkpoints, and visualization logging.
    """
    
    def __init__(self, experiment_name: str, log_dir: Optional[Path] = None):
        """
        Initialize training logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Optional custom log directory
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up directories
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = LOGS_DIR / f"{experiment_name}_{self.timestamp}"
        
        self.metrics_dir = self.log_dir / "training_metrics"
        self.vis_dir = self.log_dir / "visualization_logs"
        self.errors_dir = self.log_dir / "errors_logs"
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.best_checkpoints: List[CheckpointInfo] = []
        self.best_metrics: Dict[str, float] = {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "val_loss": float('inf')
        }
        
        # Initialize CSV file
        self.metrics_csv = self.metrics_dir / "metrics.csv"
        self._init_metrics_csv()
        
        print(f"[TrainingLogger] Initialized for: {experiment_name}")
        print(f"[TrainingLogger] Log directory: {self.log_dir}")
    
    def _init_metrics_csv(self):
        """Initialize metrics CSV file with headers."""
        headers = [
            "timestamp", "epoch", "train_loss", "val_loss", 
            "mAP50", "mAP50_95", "precision", "recall",
            "learning_rate", "images_per_sec"
        ]
        with open(self.metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_epoch(self, metrics: TrainingMetrics) -> Optional[str]:
        """
        Log metrics for an epoch.
        
        Args:
            metrics: TrainingMetrics object with epoch data
            
        Returns:
            Improvement message if new best, None otherwise
        """
        self.metrics_history.append(metrics)
        
        # Write to CSV
        with open(self.metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.timestamp, metrics.epoch, metrics.train_loss,
                metrics.val_loss, metrics.mAP50, metrics.mAP50_95,
                metrics.precision, metrics.recall, metrics.learning_rate,
                metrics.images_per_sec
            ])
        
        # Check for improvements
        improvements = []
        
        if metrics.mAP50 > self.best_metrics["mAP50"]:
            self.best_metrics["mAP50"] = metrics.mAP50
            improvements.append(f"mAP50: {metrics.mAP50:.4f}")
        
        if metrics.mAP50_95 > self.best_metrics["mAP50_95"]:
            self.best_metrics["mAP50_95"] = metrics.mAP50_95
            improvements.append(f"mAP50-95: {metrics.mAP50_95:.4f}")
        
        if metrics.val_loss < self.best_metrics["val_loss"]:
            self.best_metrics["val_loss"] = metrics.val_loss
            improvements.append(f"val_loss: {metrics.val_loss:.4f}")
        
        if improvements:
            return f"New best at epoch {metrics.epoch}: " + ", ".join(improvements)
        return None
    
    def log_checkpoint(self, checkpoint: CheckpointInfo):
        """Log a saved checkpoint."""
        self.best_checkpoints.append(checkpoint)
        
        # Append to checkpoints log
        checkpoints_file = self.metrics_dir / "best_checkpoints.json"
        checkpoints_data = []
        
        if checkpoints_file.exists():
            with open(checkpoints_file, 'r') as f:
                checkpoints_data = json.load(f)
        
        checkpoints_data.append(asdict(checkpoint))
        
        with open(checkpoints_file, 'w') as f:
            json.dump(checkpoints_data, f, indent=2)
        
        print(f"[Checkpoint] Saved best {checkpoint.metric_name}: {checkpoint.metric_value:.4f}")
    
    def log_error(self, error_type: str, message: str, details: Optional[Dict] = None):
        """
        Log an error event.
        
        Args:
            error_type: Type of error (OOM, INTERRUPTION, WARNING, etc.)
            message: Error message
            details: Optional additional details
        """
        error_log = self.errors_dir / "errors.json"
        errors = []
        
        if error_log.exists():
            with open(error_log, 'r') as f:
                errors = json.load(f)
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": message,
            "details": details or {}
        }
        errors.append(error_entry)
        
        with open(error_log, 'w') as f:
            json.dump(errors, f, indent=2)
        
        print(f"[{error_type}] {message}")
    
    def log_oom(self, gpu_memory_state: Optional[Dict] = None):
        """Log an Out of Memory event."""
        self.log_error("OOM", "Out of memory event detected", gpu_memory_state)
    
    def log_visualization(self, vis_type: str, epoch: int, 
                          source_path: str, output_path: str,
                          metadata: Optional[Dict] = None):
        """
        Log a visualization output.
        
        Args:
            vis_type: Type of visualization (prediction, confusion_matrix, etc.)
            epoch: Training epoch
            source_path: Source image path
            output_path: Output visualization path
            metadata: Optional additional metadata
        """
        vis_log = self.vis_dir / f"{vis_type}_log.json"
        entries = []
        
        if vis_log.exists():
            with open(vis_log, 'r') as f:
                entries = json.load(f)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "source": source_path,
            "output": output_path,
            "metadata": metadata or {}
        }
        entries.append(entry)
        
        with open(vis_log, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def log_class_performance(self, epoch: int, class_metrics: Dict[str, Dict[str, float]]):
        """
        Log per-class performance metrics.
        
        Args:
            epoch: Training epoch
            class_metrics: Dict mapping class names to their metrics
        """
        class_log = self.vis_dir / "class_performance.json"
        entries = []
        
        if class_log.exists():
            with open(class_log, 'r') as f:
                entries = json.load(f)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "classes": class_metrics
        }
        entries.append(entry)
        
        with open(class_log, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def get_loss_curves(self) -> Dict[str, List[float]]:
        """Get training and validation loss curves."""
        return {
            "epochs": [m.epoch for m in self.metrics_history],
            "train_loss": [m.train_loss for m in self.metrics_history],
            "val_loss": [m.val_loss for m in self.metrics_history]
        }
    
    def get_map_curves(self) -> Dict[str, List[float]]:
        """Get mAP curves over training."""
        return {
            "epochs": [m.epoch for m in self.metrics_history],
            "mAP50": [m.mAP50 for m in self.metrics_history],
            "mAP50_95": [m.mAP50_95 for m in self.metrics_history]
        }
    
    def get_lr_schedule(self) -> Dict[str, List[float]]:
        """Get learning rate schedule over training."""
        return {
            "epochs": [m.epoch for m in self.metrics_history],
            "learning_rate": [m.learning_rate for m in self.metrics_history]
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate training summary."""
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "log_directory": str(self.log_dir),
            "total_epochs": len(self.metrics_history),
            "best_metrics": self.best_metrics,
            "checkpoints_saved": len(self.best_checkpoints),
            "final_metrics": asdict(self.metrics_history[-1]) if self.metrics_history else None
        }
        
        # Save summary
        summary_file = self.log_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_summary(self):
        """Print training summary to console."""
        summary = self.generate_summary()
        
        print("\n" + "=" * 50)
        print("  Training Summary")
        print("=" * 50)
        print(f"  Experiment: {summary['experiment_name']}")
        print(f"  Total Epochs: {summary['total_epochs']}")
        print(f"\n  Best Metrics:")
        for metric, value in summary['best_metrics'].items():
            print(f"    {metric}: {value:.4f}")
        print(f"\n  Checkpoints Saved: {summary['checkpoints_saved']}")
        print(f"  Log Directory: {summary['log_directory']}")
        print("=" * 50 + "\n")


class YOLOTrainingLogger(TrainingLogger):
    """
    Specialized logger for YOLO training.
    Integrates with Ultralytics training callbacks.
    """
    
    def __init__(self, model_name: str, experiment_name: Optional[str] = None):
        """
        Initialize YOLO training logger.
        
        Args:
            model_name: YOLO model name (e.g., 'yolo11n')
            experiment_name: Optional experiment name (defaults to model name)
        """
        exp_name = experiment_name or model_name
        super().__init__(exp_name)
        self.model_name = model_name
    
    def parse_yolo_results(self, results) -> TrainingMetrics:
        """
        Parse YOLO training results into TrainingMetrics.
        
        Args:
            results: YOLO training results object
            
        Returns:
            TrainingMetrics object
        """
        # Extract metrics from YOLO results
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        
        return TrainingMetrics(
            epoch=getattr(results, 'epoch', 0),
            train_loss=metrics.get('train/box_loss', 0) + metrics.get('train/cls_loss', 0),
            val_loss=metrics.get('val/box_loss', 0) + metrics.get('val/cls_loss', 0),
            mAP50=metrics.get('metrics/mAP50(B)', 0),
            mAP50_95=metrics.get('metrics/mAP50-95(B)', 0),
            precision=metrics.get('metrics/precision(B)', 0),
            recall=metrics.get('metrics/recall(B)', 0),
            learning_rate=metrics.get('lr/pg0', 0)
        )
    
    def get_callbacks(self) -> Dict[str, callable]:
        """
        Get YOLO training callbacks for automatic logging.
        
        Returns:
            Dict of callback functions
        """
        def on_train_epoch_end(trainer):
            """Called at the end of each training epoch."""
            epoch = trainer.epoch
            metrics = trainer.metrics
            
            # Get train loss, using detach() to avoid gradient tracking issues
            train_loss = 0
            if hasattr(trainer, 'loss') and trainer.loss is not None:
                loss_tensor = trainer.loss.mean()
                train_loss = float(loss_tensor.detach()) if loss_tensor.requires_grad else float(loss_tensor)
            
            training_metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=metrics.get('val/box_loss', 0) + metrics.get('val/cls_loss', 0),
                mAP50=metrics.get('metrics/mAP50(B)', 0),
                mAP50_95=metrics.get('metrics/mAP50-95(B)', 0),
                precision=metrics.get('metrics/precision(B)', 0),
                recall=metrics.get('metrics/recall(B)', 0),
                learning_rate=trainer.optimizer.param_groups[0]['lr'] if hasattr(trainer, 'optimizer') else 0
            )
            
            improvement = self.log_epoch(training_metrics)
            if improvement:
                print(f"[Logger] {improvement}")
        
        def on_train_end(trainer):
            """Called at the end of training."""
            self.print_summary()
        
        return {
            'on_train_epoch_end': on_train_epoch_end,
            'on_train_end': on_train_end
        }


def create_logger(experiment_name: str, model_name: Optional[str] = None) -> TrainingLogger:
    """
    Factory function to create appropriate logger.
    
    Args:
        experiment_name: Name of the experiment
        model_name: Optional YOLO model name for specialized logging
        
    Returns:
        TrainingLogger instance
    """
    if model_name and 'yolo' in model_name.lower():
        return YOLOTrainingLogger(model_name, experiment_name)
    return TrainingLogger(experiment_name)


# Convenience exports
__all__ = [
    'TrainingMetrics',
    'CheckpointInfo', 
    'TrainingLogger',
    'YOLOTrainingLogger',
    'create_logger',
    'LOGS_DIR'
]
