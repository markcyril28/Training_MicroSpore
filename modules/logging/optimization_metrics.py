"""
Optimization Metrics Logger for YOLO Training
Captures detailed metrics useful for hyperparameter tuning and training optimization.

Key metrics tracked:
1. Per-class performance (precision, recall, mAP per class)
2. Gradient statistics (norm, magnitude trends)
3. Learning rate analysis (effective LR, schedule visualization)
4. Loss component breakdown (box, cls, dfl contributions)
5. Convergence indicators (plateau detection, overfitting signals)
6. Throughput metrics (images/sec, GPU utilization efficiency)
7. Memory efficiency (batch size optimization hints)
8. Early stopping signals
9. Augmentation effectiveness indicators
"""

import csv
import json
import math
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import deque
import statistics

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PerClassMetrics:
    """Per-class performance metrics for a single epoch."""
    epoch: int
    class_id: int
    class_name: str
    precision: float = 0.0
    recall: float = 0.0
    ap50: float = 0.0
    ap50_95: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    support: int = 0  # Number of ground truth instances


@dataclass
class GradientStats:
    """Gradient statistics for an epoch."""
    epoch: int
    grad_norm_mean: float = 0.0
    grad_norm_max: float = 0.0
    grad_norm_min: float = 0.0
    grad_norm_std: float = 0.0
    num_zero_grads: int = 0
    num_exploding_grads: int = 0  # > threshold
    grad_clip_fraction: float = 0.0  # Fraction of updates that were clipped
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LossBreakdown:
    """Detailed loss component breakdown."""
    epoch: int
    # Training losses
    train_box_loss: float = 0.0
    train_cls_loss: float = 0.0
    train_dfl_loss: float = 0.0
    train_total_loss: float = 0.0
    # Validation losses
    val_box_loss: float = 0.0
    val_cls_loss: float = 0.0
    val_dfl_loss: float = 0.0
    val_total_loss: float = 0.0
    # Loss ratios (for balance analysis)
    box_cls_ratio: float = 0.0  # box_loss / cls_loss
    train_val_ratio: float = 0.0  # train_loss / val_loss (overfitting indicator)
    # Loss component contributions (as percentages)
    box_contribution_pct: float = 0.0
    cls_contribution_pct: float = 0.0
    dfl_contribution_pct: float = 0.0


@dataclass
class ConvergenceIndicators:
    """Convergence and training health indicators."""
    epoch: int
    # Plateau detection
    is_plateau: bool = False
    plateau_epochs: int = 0  # Consecutive epochs without improvement
    # Overfitting detection
    overfitting_score: float = 0.0  # train_loss << val_loss indicates overfitting
    val_loss_increasing: bool = False
    val_loss_increase_epochs: int = 0
    # Learning signals
    improvement_rate: float = 0.0  # Rate of mAP improvement
    loss_variance: float = 0.0  # High variance may indicate LR issues
    # Recommendations
    suggested_action: str = ""  # "continue", "reduce_lr", "early_stop", "increase_augmentation"


@dataclass
class ThroughputMetrics:
    """Training throughput and efficiency metrics."""
    epoch: int
    images_per_second: float = 0.0
    batches_per_second: float = 0.0
    epoch_time_seconds: float = 0.0
    avg_batch_time_ms: float = 0.0
    gpu_utilization_avg: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_memory_avg_mb: float = 0.0
    cpu_utilization_avg: float = 0.0
    data_loading_time_pct: float = 0.0  # Time spent loading data vs training


@dataclass
class OptimizationRecommendation:
    """Hyperparameter optimization recommendation."""
    parameter: str
    current_value: Any
    recommended_value: Any
    confidence: float  # 0.0 to 1.0
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)


class OptimizationMetricsLogger:
    """
    Comprehensive optimization metrics logger for YOLO training.
    Tracks detailed metrics useful for hyperparameter tuning.
    """
    
    def __init__(
        self,
        experiment_dir: Union[str, Path],
        class_names: Optional[List[str]] = None,
        window_size: int = 10,  # Rolling window for trend analysis
    ):
        """
        Initialize optimization metrics logger.
        
        Args:
            experiment_dir: Path to experiment output directory
            class_names: List of class names
            window_size: Window size for rolling statistics
        """
        self.experiment_dir = Path(experiment_dir)
        self.class_names = class_names or []
        self.window_size = window_size
        
        # Create directories
        self.optimization_dir = self.experiment_dir / "stats" / "optimization"
        self.optimization_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Rolling windows for trend analysis
        self.loss_history = deque(maxlen=window_size)
        self.val_loss_history = deque(maxlen=window_size)
        self.map_history = deque(maxlen=window_size)
        self.grad_norm_history = deque(maxlen=window_size)
        
        # Tracking state
        self.best_map50_95 = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.val_loss_increasing_count = 0
        self.last_val_loss = float('inf')
        
        # Per-class history for trend analysis
        self.per_class_history: Dict[int, List[PerClassMetrics]] = {}
        
        print(f"[OptimizationLogger] Initialized at: {self.optimization_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Per-class metrics CSV
        self.per_class_csv = self.optimization_dir / "per_class_metrics.csv"
        with open(self.per_class_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "class_id", "class_name", "precision", "recall",
                "ap50", "ap50_95", "f1", "true_positives", "false_positives",
                "false_negatives", "support"
            ])
        
        # Gradient stats CSV
        self.gradient_csv = self.optimization_dir / "gradient_stats.csv"
        with open(self.gradient_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "grad_norm_mean", "grad_norm_max", "grad_norm_min",
                "grad_norm_std", "num_zero_grads", "num_exploding_grads",
                "grad_clip_fraction", "timestamp"
            ])
        
        # Loss breakdown CSV
        self.loss_breakdown_csv = self.optimization_dir / "loss_breakdown.csv"
        with open(self.loss_breakdown_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_box_loss", "train_cls_loss", "train_dfl_loss",
                "train_total_loss", "val_box_loss", "val_cls_loss", "val_dfl_loss",
                "val_total_loss", "box_cls_ratio", "train_val_ratio",
                "box_contribution_pct", "cls_contribution_pct", "dfl_contribution_pct"
            ])
        
        # Convergence indicators CSV
        self.convergence_csv = self.optimization_dir / "convergence_indicators.csv"
        with open(self.convergence_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "is_plateau", "plateau_epochs", "overfitting_score",
                "val_loss_increasing", "val_loss_increase_epochs",
                "improvement_rate", "loss_variance", "suggested_action"
            ])
        
        # Throughput metrics CSV
        self.throughput_csv = self.optimization_dir / "throughput_metrics.csv"
        with open(self.throughput_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "images_per_second", "batches_per_second",
                "epoch_time_seconds", "avg_batch_time_ms", "gpu_utilization_avg",
                "gpu_memory_peak_mb", "gpu_memory_avg_mb", "cpu_utilization_avg",
                "data_loading_time_pct"
            ])
        
        # Learning rate analysis CSV
        self.lr_analysis_csv = self.optimization_dir / "lr_analysis.csv"
        with open(self.lr_analysis_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr_pg0", "lr_pg1", "lr_pg2", "effective_lr",
                "lr_loss_correlation", "optimal_lr_estimate"
            ])
    
    def log_per_class_metrics(
        self,
        epoch: int,
        class_metrics: Dict[int, Dict[str, float]],
    ) -> None:
        """
        Log per-class performance metrics.
        
        Args:
            epoch: Current epoch
            class_metrics: Dict mapping class_id to metrics dict with keys:
                           precision, recall, ap50, ap50_95, etc.
        """
        rows = []
        for class_id, metrics in class_metrics.items():
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            pcm = PerClassMetrics(
                epoch=epoch,
                class_id=class_id,
                class_name=class_name,
                precision=precision,
                recall=recall,
                ap50=metrics.get('ap50', metrics.get('mAP50', 0.0)),
                ap50_95=metrics.get('ap50_95', metrics.get('mAP50-95', 0.0)),
                f1=f1,
                true_positives=int(metrics.get('tp', 0)),
                false_positives=int(metrics.get('fp', 0)),
                false_negatives=int(metrics.get('fn', 0)),
                support=int(metrics.get('support', metrics.get('gt', 0))),
            )
            
            rows.append([
                pcm.epoch, pcm.class_id, pcm.class_name, pcm.precision,
                pcm.recall, pcm.ap50, pcm.ap50_95, pcm.f1,
                pcm.true_positives, pcm.false_positives, pcm.false_negatives,
                pcm.support
            ])
            
            # Store in history
            if class_id not in self.per_class_history:
                self.per_class_history[class_id] = []
            self.per_class_history[class_id].append(pcm)
        
        # Append to CSV
        with open(self.per_class_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def log_gradient_stats(
        self,
        epoch: int,
        model: Any = None,
        grad_norms: Optional[List[float]] = None,
        grad_clip_threshold: float = 1.0,
    ) -> GradientStats:
        """
        Log gradient statistics for the epoch.
        
        Args:
            epoch: Current epoch
            model: PyTorch model (optional, for extracting gradient info)
            grad_norms: Pre-computed list of gradient norms
            grad_clip_threshold: Threshold for gradient clipping
            
        Returns:
            GradientStats object
        """
        if grad_norms is None and model is not None and TORCH_AVAILABLE:
            grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.data.norm(2).item())
        
        if not grad_norms:
            grad_norms = [0.0]
        
        stats = GradientStats(
            epoch=epoch,
            grad_norm_mean=statistics.mean(grad_norms),
            grad_norm_max=max(grad_norms),
            grad_norm_min=min(grad_norms),
            grad_norm_std=statistics.stdev(grad_norms) if len(grad_norms) > 1 else 0.0,
            num_zero_grads=sum(1 for g in grad_norms if g == 0.0),
            num_exploding_grads=sum(1 for g in grad_norms if g > grad_clip_threshold * 10),
            grad_clip_fraction=sum(1 for g in grad_norms if g > grad_clip_threshold) / len(grad_norms),
        )
        
        # Add to history
        self.grad_norm_history.append(stats.grad_norm_mean)
        
        # Write to CSV
        with open(self.gradient_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                stats.epoch, stats.grad_norm_mean, stats.grad_norm_max,
                stats.grad_norm_min, stats.grad_norm_std, stats.num_zero_grads,
                stats.num_exploding_grads, stats.grad_clip_fraction, stats.timestamp
            ])
        
        return stats
    
    def log_loss_breakdown(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
    ) -> LossBreakdown:
        """
        Log detailed loss component breakdown.
        
        Args:
            epoch: Current epoch
            train_losses: Dict with 'box', 'cls', 'dfl' keys
            val_losses: Dict with 'box', 'cls', 'dfl' keys
            
        Returns:
            LossBreakdown object
        """
        train_box = train_losses.get('box', train_losses.get('box_loss', 0.0))
        train_cls = train_losses.get('cls', train_losses.get('cls_loss', 0.0))
        train_dfl = train_losses.get('dfl', train_losses.get('dfl_loss', 0.0))
        train_total = train_box + train_cls + train_dfl
        
        val_box = val_losses.get('box', val_losses.get('box_loss', 0.0))
        val_cls = val_losses.get('cls', val_losses.get('cls_loss', 0.0))
        val_dfl = val_losses.get('dfl', val_losses.get('dfl_loss', 0.0))
        val_total = val_box + val_cls + val_dfl
        
        breakdown = LossBreakdown(
            epoch=epoch,
            train_box_loss=train_box,
            train_cls_loss=train_cls,
            train_dfl_loss=train_dfl,
            train_total_loss=train_total,
            val_box_loss=val_box,
            val_cls_loss=val_cls,
            val_dfl_loss=val_dfl,
            val_total_loss=val_total,
            box_cls_ratio=train_box / train_cls if train_cls > 0 else 0.0,
            train_val_ratio=train_total / val_total if val_total > 0 else 0.0,
            box_contribution_pct=100 * train_box / train_total if train_total > 0 else 0.0,
            cls_contribution_pct=100 * train_cls / train_total if train_total > 0 else 0.0,
            dfl_contribution_pct=100 * train_dfl / train_total if train_total > 0 else 0.0,
        )
        
        # Add to history
        self.loss_history.append(train_total)
        self.val_loss_history.append(val_total)
        
        # Write to CSV
        with open(self.loss_breakdown_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                breakdown.epoch, breakdown.train_box_loss, breakdown.train_cls_loss,
                breakdown.train_dfl_loss, breakdown.train_total_loss,
                breakdown.val_box_loss, breakdown.val_cls_loss, breakdown.val_dfl_loss,
                breakdown.val_total_loss, breakdown.box_cls_ratio, breakdown.train_val_ratio,
                breakdown.box_contribution_pct, breakdown.cls_contribution_pct,
                breakdown.dfl_contribution_pct
            ])
        
        return breakdown
    
    def log_convergence_indicators(
        self,
        epoch: int,
        current_map50_95: float,
        current_val_loss: float,
        patience: int = 50,
    ) -> ConvergenceIndicators:
        """
        Calculate and log convergence indicators.
        
        Args:
            epoch: Current epoch
            current_map50_95: Current mAP50-95 value
            current_val_loss: Current validation loss
            patience: Early stopping patience
            
        Returns:
            ConvergenceIndicators object
        """
        # Update improvement tracking
        if current_map50_95 > self.best_map50_95:
            improvement = current_map50_95 - self.best_map50_95
            self.best_map50_95 = current_map50_95
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            improvement = 0.0
            self.epochs_without_improvement += 1
        
        # Track validation loss trend
        if current_val_loss > self.last_val_loss:
            self.val_loss_increasing_count += 1
        else:
            self.val_loss_increasing_count = 0
        self.last_val_loss = current_val_loss
        
        # Add to mAP history
        self.map_history.append(current_map50_95)
        
        # Calculate improvement rate (slope of mAP over window)
        improvement_rate = 0.0
        if len(self.map_history) >= 3:
            recent = list(self.map_history)[-5:]
            if len(recent) >= 2:
                improvement_rate = (recent[-1] - recent[0]) / len(recent)
        
        # Calculate loss variance
        loss_variance = 0.0
        if len(self.val_loss_history) >= 3:
            loss_variance = statistics.variance(list(self.val_loss_history))
        
        # Calculate overfitting score
        overfitting_score = 0.0
        if len(self.loss_history) > 0 and len(self.val_loss_history) > 0:
            train_avg = statistics.mean(list(self.loss_history)[-5:])
            val_avg = statistics.mean(list(self.val_loss_history)[-5:])
            if train_avg > 0:
                overfitting_score = max(0, (val_avg - train_avg) / train_avg)
        
        # Determine suggested action
        suggested_action = "continue"
        if self.epochs_without_improvement >= patience:
            suggested_action = "early_stop"
        elif self.epochs_without_improvement >= patience // 2:
            suggested_action = "reduce_lr"
        elif overfitting_score > 0.3:
            suggested_action = "increase_regularization"
        elif loss_variance > 0.1:
            suggested_action = "reduce_lr"
        
        indicators = ConvergenceIndicators(
            epoch=epoch,
            is_plateau=self.epochs_without_improvement >= 10,
            plateau_epochs=self.epochs_without_improvement,
            overfitting_score=overfitting_score,
            val_loss_increasing=self.val_loss_increasing_count >= 3,
            val_loss_increase_epochs=self.val_loss_increasing_count,
            improvement_rate=improvement_rate,
            loss_variance=loss_variance,
            suggested_action=suggested_action,
        )
        
        # Write to CSV
        with open(self.convergence_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                indicators.epoch, indicators.is_plateau, indicators.plateau_epochs,
                indicators.overfitting_score, indicators.val_loss_increasing,
                indicators.val_loss_increase_epochs, indicators.improvement_rate,
                indicators.loss_variance, indicators.suggested_action
            ])
        
        return indicators
    
    def log_throughput_metrics(
        self,
        epoch: int,
        epoch_time_seconds: float,
        num_images: int,
        batch_size: int,
        gpu_stats: Optional[Dict[str, float]] = None,
    ) -> ThroughputMetrics:
        """
        Log training throughput metrics.
        
        Args:
            epoch: Current epoch
            epoch_time_seconds: Time taken for the epoch
            num_images: Number of images processed
            batch_size: Batch size
            gpu_stats: Optional GPU statistics
            
        Returns:
            ThroughputMetrics object
        """
        gpu_stats = gpu_stats or {}
        
        num_batches = math.ceil(num_images / batch_size)
        
        metrics = ThroughputMetrics(
            epoch=epoch,
            images_per_second=num_images / epoch_time_seconds if epoch_time_seconds > 0 else 0.0,
            batches_per_second=num_batches / epoch_time_seconds if epoch_time_seconds > 0 else 0.0,
            epoch_time_seconds=epoch_time_seconds,
            avg_batch_time_ms=1000 * epoch_time_seconds / num_batches if num_batches > 0 else 0.0,
            gpu_utilization_avg=gpu_stats.get('gpu_util_avg', 0.0),
            gpu_memory_peak_mb=gpu_stats.get('memory_peak_mb', 0.0),
            gpu_memory_avg_mb=gpu_stats.get('memory_avg_mb', 0.0),
            cpu_utilization_avg=gpu_stats.get('cpu_util_avg', 0.0),
            data_loading_time_pct=gpu_stats.get('data_loading_pct', 0.0),
        )
        
        # Write to CSV
        with open(self.throughput_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.epoch, metrics.images_per_second, metrics.batches_per_second,
                metrics.epoch_time_seconds, metrics.avg_batch_time_ms,
                metrics.gpu_utilization_avg, metrics.gpu_memory_peak_mb,
                metrics.gpu_memory_avg_mb, metrics.cpu_utilization_avg,
                metrics.data_loading_time_pct
            ])
        
        return metrics
    
    def log_lr_analysis(
        self,
        epoch: int,
        lr_pg0: float,
        lr_pg1: float,
        lr_pg2: float,
        current_loss: float,
    ) -> None:
        """
        Log learning rate analysis metrics.
        
        Args:
            epoch: Current epoch
            lr_pg0: Learning rate for parameter group 0
            lr_pg1: Learning rate for parameter group 1
            lr_pg2: Learning rate for parameter group 2
            current_loss: Current training loss
        """
        effective_lr = (lr_pg0 + lr_pg1 + lr_pg2) / 3
        
        # Calculate LR-loss correlation over window
        lr_loss_correlation = 0.0
        optimal_lr_estimate = effective_lr
        
        # Simple correlation: if loss is decreasing, LR is probably good
        if len(self.loss_history) >= 5:
            recent_losses = list(self.loss_history)[-5:]
            if recent_losses[0] > 0:
                loss_trend = (recent_losses[-1] - recent_losses[0]) / recent_losses[0]
                lr_loss_correlation = -loss_trend  # Positive if loss is decreasing
                
                # Estimate optimal LR based on convergence
                if loss_trend > 0.1:  # Loss increasing
                    optimal_lr_estimate = effective_lr * 0.5
                elif abs(loss_trend) < 0.01:  # Plateau
                    optimal_lr_estimate = effective_lr * 1.5
        
        # Write to CSV
        with open(self.lr_analysis_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, lr_pg0, lr_pg1, lr_pg2, effective_lr,
                lr_loss_correlation, optimal_lr_estimate
            ])
    
    def get_class_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of per-class performance for optimization.
        
        Returns:
            Dict with underperforming classes, class imbalance indicators, etc.
        """
        summary = {
            "underperforming_classes": [],
            "well_performing_classes": [],
            "high_fp_classes": [],  # Classes with high false positive rates
            "high_fn_classes": [],  # Classes with high false negative rates
            "class_balance_recommendation": "",
        }
        
        if not self.per_class_history:
            return summary
        
        # Analyze latest epoch's metrics
        avg_map = 0.0
        class_maps = {}
        
        for class_id, history in self.per_class_history.items():
            if history:
                latest = history[-1]
                class_maps[class_id] = latest.ap50_95
                
                if latest.ap50_95 < 0.3:
                    summary["underperforming_classes"].append({
                        "class_id": class_id,
                        "class_name": latest.class_name,
                        "ap50_95": latest.ap50_95,
                        "precision": latest.precision,
                        "recall": latest.recall,
                    })
                elif latest.ap50_95 > 0.6:
                    summary["well_performing_classes"].append({
                        "class_id": class_id,
                        "class_name": latest.class_name,
                        "ap50_95": latest.ap50_95,
                    })
                
                # Check for FP/FN imbalance
                total = latest.true_positives + latest.false_positives + latest.false_negatives
                if total > 0:
                    fp_rate = latest.false_positives / total
                    fn_rate = latest.false_negatives / total
                    
                    if fp_rate > 0.3:
                        summary["high_fp_classes"].append({
                            "class_id": class_id,
                            "class_name": latest.class_name,
                            "fp_rate": fp_rate,
                        })
                    if fn_rate > 0.3:
                        summary["high_fn_classes"].append({
                            "class_id": class_id,
                            "class_name": latest.class_name,
                            "fn_rate": fn_rate,
                        })
        
        # Class balance recommendation
        if class_maps:
            avg_map = statistics.mean(class_maps.values())
            map_std = statistics.stdev(class_maps.values()) if len(class_maps) > 1 else 0
            
            if map_std > 0.15:
                summary["class_balance_recommendation"] = (
                    f"High variance in class performance (std={map_std:.3f}). "
                    f"Consider class-balanced sampling or class-specific augmentation."
                )
            elif len(summary["underperforming_classes"]) > len(class_maps) // 3:
                summary["class_balance_recommendation"] = (
                    f"Multiple underperforming classes. Consider increasing training data "
                    f"or using focal loss for hard examples."
                )
        
        return summary
    
    def generate_optimization_recommendations(
        self,
        config: Dict[str, Any],
    ) -> List[OptimizationRecommendation]:
        """
        Generate hyperparameter optimization recommendations based on collected metrics.
        
        Args:
            config: Current training configuration
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze convergence
        if self.epochs_without_improvement >= 20:
            # Check if LR should be reduced
            if len(self.val_loss_history) > 10:
                recent_variance = statistics.variance(list(self.val_loss_history)[-10:])
                if recent_variance > 0.05:
                    recommendations.append(OptimizationRecommendation(
                        parameter="lr0",
                        current_value=config.get('lr0', 0.01),
                        recommended_value=config.get('lr0', 0.01) * 0.5,
                        confidence=0.7,
                        reason="High validation loss variance suggests learning rate is too high",
                        evidence={"loss_variance": recent_variance, "plateau_epochs": self.epochs_without_improvement}
                    ))
        
        # Analyze class performance
        class_summary = self.get_class_performance_summary()
        if len(class_summary["underperforming_classes"]) > 0:
            recommendations.append(OptimizationRecommendation(
                parameter="class_weights",
                current_value=config.get('class_weights', {}),
                recommended_value="Use class-balanced sampling",
                confidence=0.6,
                reason=f"{len(class_summary['underperforming_classes'])} classes have mAP < 30%",
                evidence={"underperforming_classes": class_summary["underperforming_classes"]}
            ))
        
        # Check for overfitting
        if len(self.loss_history) > 10 and len(self.val_loss_history) > 10:
            train_avg = statistics.mean(list(self.loss_history)[-10:])
            val_avg = statistics.mean(list(self.val_loss_history)[-10:])
            
            if val_avg > train_avg * 1.5:
                recommendations.append(OptimizationRecommendation(
                    parameter="augmentation",
                    current_value={
                        "hsv_h": config.get('hsv_h', 0.015),
                        "mosaic": config.get('mosaic', 1.0),
                        "mixup": config.get('mixup', 0.0),
                    },
                    recommended_value={
                        "hsv_h": min(0.1, config.get('hsv_h', 0.015) * 2),
                        "mosaic": 1.0,
                        "mixup": 0.1,
                    },
                    confidence=0.65,
                    reason="Validation loss significantly higher than training loss indicates overfitting",
                    evidence={"train_loss_avg": train_avg, "val_loss_avg": val_avg}
                ))
        
        # Batch size recommendation based on throughput
        # (Would need actual GPU memory data for this)
        
        return recommendations
    
    def save_optimization_summary(self) -> Path:
        """
        Save comprehensive optimization summary.
        
        Returns:
            Path to saved summary file
        """
        summary = {
            "generated_at": datetime.now().isoformat(),
            "best_epoch": self.best_epoch,
            "best_map50_95": self.best_map50_95,
            "epochs_without_improvement": self.epochs_without_improvement,
            "class_performance_summary": self.get_class_performance_summary(),
            "convergence_status": {
                "is_converged": self.epochs_without_improvement >= 50,
                "plateau_detected": self.epochs_without_improvement >= 10,
                "overfitting_detected": False,  # Would need more data
            },
        }
        
        # Add loss statistics
        if self.loss_history:
            summary["loss_statistics"] = {
                "train_loss_final": list(self.loss_history)[-1],
                "train_loss_mean": statistics.mean(self.loss_history),
                "train_loss_min": min(self.loss_history),
            }
        
        if self.val_loss_history:
            summary["loss_statistics"]["val_loss_final"] = list(self.val_loss_history)[-1]
            summary["loss_statistics"]["val_loss_mean"] = statistics.mean(self.val_loss_history)
            summary["loss_statistics"]["val_loss_min"] = min(self.val_loss_history)
        
        output_path = self.optimization_dir / "optimization_summary.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return output_path


def extract_yolo_class_metrics(results) -> Dict[int, Dict[str, float]]:
    """
    Extract per-class metrics from YOLO validation results.
    
    Args:
        results: YOLO validation results object
        
    Returns:
        Dict mapping class_id to metrics dict
    """
    class_metrics = {}
    
    try:
        if hasattr(results, 'box'):
            box = results.box
            
            # Get per-class data
            if hasattr(box, 'ap_class_index'):
                class_indices = box.ap_class_index
                
                for i, class_id in enumerate(class_indices):
                    class_id = int(class_id)
                    class_metrics[class_id] = {
                        'precision': float(box.p[i]) if hasattr(box, 'p') and i < len(box.p) else 0.0,
                        'recall': float(box.r[i]) if hasattr(box, 'r') and i < len(box.r) else 0.0,
                        'ap50': float(box.ap50[i]) if hasattr(box, 'ap50') and i < len(box.ap50) else 0.0,
                        'ap50_95': float(box.ap[i]) if hasattr(box, 'ap') and i < len(box.ap) else 0.0,
                    }
    except Exception as e:
        print(f"[OptimizationLogger] Warning: Could not extract class metrics: {e}")
    
    return class_metrics


# Export public API
__all__ = [
    'OptimizationMetricsLogger',
    'PerClassMetrics',
    'GradientStats',
    'LossBreakdown',
    'ConvergenceIndicators',
    'ThroughputMetrics',
    'OptimizationRecommendation',
    'extract_yolo_class_metrics',
]
