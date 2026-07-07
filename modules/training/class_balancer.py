"""Class Balancer Module for YOLO Training

Implements class-weighted oversampling for addressing class imbalance in object detection.
Creates symbolic links or copies of underrepresented class images to balance the dataset.

This module provides two approaches:
1. Image-level oversampling: Duplicate images containing minority classes
2. Weighted sampling info: Generate weights for custom training loops

Author: GitHub Copilot
Date: 2026-01-09
"""

import csv
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random


class ClassBalancer:
    """
    Handles class imbalance by oversampling images containing minority classes.
    
    Modes:
        - none: No balancing, use original dataset
        - manual: Apply specified fold to selected classes
        - auto: Automatically balance all classes towards target (max/median/mean)
        - sqrt: Square root balancing (softer than full equalization)
    """
    
    def __init__(
        self,
        mode: str = "none",
        class_weights: Optional[Dict[str, float]] = None,
        max_fold: float = 3.0,
        seed: int = 42,
    ):
        """
        Initialize the ClassBalancer.
        
        Args:
            mode: Balancing mode ('none', 'manual', 'auto', 'sqrt')
            class_weights: Pre-calculated class weights {class_name: weight}
            max_fold: Maximum oversampling factor to prevent extreme duplication
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.class_weights = class_weights or {}
        self.max_fold = max_fold
        self.seed = seed
        random.seed(seed)
        
    @classmethod
    def from_distribution_csv(
        cls,
        csv_path: str,
        mode: str = "auto",
        target: str = "median",
        max_fold: float = 3.0,
        focus_classes: Optional[List[str]] = None,
    ) -> "ClassBalancer":
        """
        Create a ClassBalancer from distribution CSV file.
        
        Args:
            csv_path: Path to distribution_sorted.csv
            mode: Balancing mode
            target: Target for auto mode ('max', 'median', 'mean')
            max_fold: Maximum oversampling factor
            focus_classes: Classes to focus on (for manual mode)
            
        Returns:
            ClassBalancer instance with calculated weights
        """
        class_counts = cls._parse_distribution_csv(csv_path)
        
        if not class_counts:
            print(f"[ClassBalancer] Warning: Could not parse {csv_path}")
            return cls(mode="none")
        
        class_weights = cls._calculate_weights(
            class_counts=class_counts,
            mode=mode,
            target=target,
            max_fold=max_fold,
            focus_classes=focus_classes,
        )
        
        return cls(mode=mode, class_weights=class_weights, max_fold=max_fold)
    
    @classmethod
    def from_json_weights(cls, weights_json: str, mode: str = "auto") -> "ClassBalancer":
        """
        Create a ClassBalancer from JSON weights string.
        
        Args:
            weights_json: JSON string of class weights
            mode: Balancing mode
            
        Returns:
            ClassBalancer instance
        """
        try:
            weights = json.loads(weights_json) if weights_json else {}
            return cls(mode=mode, class_weights=weights)
        except json.JSONDecodeError:
            print(f"[ClassBalancer] Warning: Could not parse weights JSON")
            return cls(mode="none")
    
    @staticmethod
    def _parse_distribution_csv(csv_path: str) -> Dict[str, int]:
        """Parse distribution_sorted.csv to get class counts."""
        class_counts = {}
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class_name = row.get('Class', '').strip()
                    train_count = int(row.get('Train', 0))
                    if class_name and train_count > 0:
                        class_counts[class_name] = train_count
        except Exception as e:
            print(f"[ClassBalancer] Error reading {csv_path}: {e}")
        return class_counts
    
    @staticmethod
    def _calculate_weights(
        class_counts: Dict[str, int],
        mode: str,
        target: str,
        max_fold: float,
        focus_classes: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Calculate class weights based on mode and target."""
        if not class_counts:
            return {}
        
        counts = list(class_counts.values())
        
        # Calculate target count
        if target == "max":
            target_count = max(counts)
        elif target == "median":
            sorted_counts = sorted(counts)
            mid = len(sorted_counts) // 2
            target_count = sorted_counts[mid] if len(sorted_counts) % 2 else \
                          (sorted_counts[mid-1] + sorted_counts[mid]) // 2
        else:  # mean
            target_count = sum(counts) // len(counts)
        
        weights = {}
        for class_name, count in class_counts.items():
            if mode == "manual":
                # Apply weight only to focus classes
                if focus_classes and class_name in focus_classes:
                    fold = target_count / count if count > 0 else 1.0
                    weights[class_name] = min(max_fold, max(1.0, fold))
                else:
                    weights[class_name] = 1.0
            elif mode == "auto":
                # Auto-balance all classes
                if count > 0:
                    fold = target_count / count
                    weights[class_name] = min(max_fold, max(1.0, fold))
                else:
                    weights[class_name] = 1.0
            elif mode == "sqrt":
                # Square root balancing (softer)
                if count > 0:
                    fold = (target_count / count) ** 0.5
                    weights[class_name] = min(max_fold, max(1.0, fold))
                else:
                    weights[class_name] = 1.0
            else:
                weights[class_name] = 1.0
        
        return weights
    
    def get_image_weight(self, label_path: Path) -> float:
        """
        Calculate weight for an image based on its annotations.
        Uses the maximum weight of all classes present in the image.
        
        Args:
            label_path: Path to YOLO format label file (.txt)
            
        Returns:
            Weight for the image (1.0 = no oversampling)
        """
        if self.mode == "none" or not self.class_weights:
            return 1.0
        
        # Read label file to find which classes are present
        classes_in_image = set()
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        classes_in_image.add(class_id)
        except Exception:
            return 1.0
        
        # Can't apply weights without class name mapping
        # Return 1.0 - actual weighting happens in create_balanced_dataset
        return 1.0
    
    def create_balanced_dataset(
        self,
        train_images_dir: str,
        train_labels_dir: str,
        class_names: List[str],
        output_dir: Optional[str] = None,
        use_symlinks: bool = True,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Create a balanced dataset by oversampling images with minority classes.
        
        Args:
            train_images_dir: Path to training images directory
            train_labels_dir: Path to training labels directory
            class_names: List of class names in order (index = class_id)
            output_dir: Output directory for balanced dataset (None = temp dir)
            use_symlinks: Use symbolic links instead of copying files
            
        Returns:
            Tuple of (balanced_images_dir, balanced_labels_dir, stats_dict)
        """
        if self.mode == "none" or not self.class_weights:
            # No balancing needed
            return train_images_dir, train_labels_dir, {"mode": "none", "original": True}
        
        # Create output directories
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="balanced_dataset_")
        
        output_path = Path(output_dir)
        balanced_images = output_path / "images" / "train"
        balanced_labels = output_path / "labels" / "train"
        balanced_images.mkdir(parents=True, exist_ok=True)
        balanced_labels.mkdir(parents=True, exist_ok=True)
        
        train_images_path = Path(train_images_dir)
        train_labels_path = Path(train_labels_dir)
        
        # Build class_id to weight mapping
        class_id_weights = {}
        for i, class_name in enumerate(class_names):
            class_id_weights[i] = self.class_weights.get(class_name, 1.0)
        
        # Analyze each image and calculate its weight
        image_weights = {}
        image_classes = {}
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        for img_file in train_images_path.iterdir():
            if img_file.suffix.lower() not in image_extensions:
                continue
            
            label_file = train_labels_path / (img_file.stem + ".txt")
            if not label_file.exists():
                continue
            
            # Find classes in this image
            classes_in_image = set()
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id < len(class_names):
                                classes_in_image.add(class_id)
            except Exception:
                continue
            
            if not classes_in_image:
                # No valid annotations, include once
                image_weights[img_file] = 1.0
                image_classes[img_file] = set()
            else:
                # Weight = max weight of classes in image
                max_weight = max(class_id_weights.get(cid, 1.0) for cid in classes_in_image)
                image_weights[img_file] = max_weight
                image_classes[img_file] = classes_in_image
        
        # Calculate original class counts (before balancing)
        original_class_counts = defaultdict(int)
        for img_file, classes in image_classes.items():
            for class_id in classes:
                if class_id < len(class_names):
                    original_class_counts[class_names[class_id]] += 1
        
        # Create balanced dataset
        stats = {
            "mode": self.mode,
            "original_count": len(image_weights),
            "balanced_count": 0,
            "class_weights": self.class_weights,
            "original_class_counts": dict(original_class_counts),
            "duplications": defaultdict(int),
        }
        
        file_counter = 0
        for img_file, weight in image_weights.items():
            label_file = train_labels_path / (img_file.stem + ".txt")
            
            # Calculate number of copies (weight rounded, minimum 1)
            num_copies = max(1, int(round(weight)))
            
            for copy_idx in range(num_copies):
                file_counter += 1
                
                if copy_idx == 0:
                    # Original - use original name
                    new_img_name = img_file.name
                    new_label_name = label_file.name
                else:
                    # Duplicate - add suffix
                    new_img_name = f"{img_file.stem}_dup{copy_idx}{img_file.suffix}"
                    new_label_name = f"{label_file.stem}_dup{copy_idx}.txt"
                
                new_img_path = balanced_images / new_img_name
                new_label_path = balanced_labels / new_label_name
                
                # Create link or copy
                if use_symlinks:
                    try:
                        if not new_img_path.exists():
                            new_img_path.symlink_to(img_file.absolute())
                        if not new_label_path.exists():
                            new_label_path.symlink_to(label_file.absolute())
                    except OSError:
                        # Symlinks may not work on Windows without admin
                        shutil.copy2(img_file, new_img_path)
                        shutil.copy2(label_file, new_label_path)
                else:
                    shutil.copy2(img_file, new_img_path)
                    shutil.copy2(label_file, new_label_path)
                
                # Track duplications by class
                for class_id in image_classes.get(img_file, set()):
                    if class_id < len(class_names):
                        stats["duplications"][class_names[class_id]] += 1
        
        stats["balanced_count"] = file_counter
        stats["duplications"] = dict(stats["duplications"])
        
        print(f"[ClassBalancer] Created balanced dataset:")
        print(f"    Original images: {stats['original_count']}")
        print(f"    Balanced images: {stats['balanced_count']}")
        print(f"    Increase: {stats['balanced_count'] - stats['original_count']} images")
        
        return str(balanced_images), str(balanced_labels), stats
    
    def create_balanced_data_yaml(
        self,
        original_yaml_path: str,
        balanced_images_dir: str,
        balanced_labels_dir: str,
        output_yaml_path: Optional[str] = None,
    ) -> str:
        """
        Create a new data.yaml pointing to the balanced dataset.
        
        Args:
            original_yaml_path: Path to original data.yaml
            balanced_images_dir: Path to balanced images directory
            balanced_labels_dir: Path to balanced labels directory
            output_yaml_path: Path for new data.yaml (None = auto-generate)
            
        Returns:
            Path to new data.yaml
        """
        import yaml
        
        # Load original yaml
        with open(original_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Update train path to balanced dataset
        balanced_images_path = Path(balanced_images_dir)
        data['train'] = str(balanced_images_path)
        
        # Keep test/val paths from original
        # (we only balance training data)
        
        # Write new yaml
        if output_yaml_path is None:
            output_yaml_path = str(balanced_images_path.parent.parent / "data_balanced.yaml")
        
        with open(output_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"[ClassBalancer] Created balanced data.yaml: {output_yaml_path}")
        
        return output_yaml_path
    
    def get_class_names_from_yaml(self, yaml_path: str) -> List[str]:
        """Extract class names from data.yaml."""
        import yaml
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        names = data.get('names', [])
        
        # Handle both list and dict formats
        if isinstance(names, dict):
            # Sort by key (class id) and return values
            return [names[k] for k in sorted(names.keys())]
        return names
    
    def print_summary(self):
        """Print a summary of class weights."""
        if self.mode == "none":
            print("[ClassBalancer] Mode: none (no balancing)")
            return
        
        print(f"[ClassBalancer] Mode: {self.mode}")
        print(f"[ClassBalancer] Max fold: {self.max_fold}")
        print("[ClassBalancer] Class weights:")
        
        for class_name, weight in sorted(self.class_weights.items(), key=lambda x: -x[1]):
            boost_marker = " (boosted)" if weight > 1.0 else ""
            print(f"    {class_name}: {weight:.2f}x{boost_marker}")


def create_balanced_training_data(
    data_yaml: str,
    class_weights_json: str,
    mode: str = "auto",
    output_dir: Optional[str] = None,
    use_symlinks: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to create balanced training data.
    
    Args:
        data_yaml: Path to original data.yaml
        class_weights_json: JSON string of class weights
        mode: Balancing mode
        output_dir: Output directory (None = temp dir)
        use_symlinks: Use symbolic links instead of copying
        
    Returns:
        Tuple of (new_data_yaml_path, stats_dict)
    """
    import yaml
    
    if mode == "none":
        return data_yaml, {"mode": "none", "original": True}
    
    # Parse weights
    try:
        class_weights = json.loads(class_weights_json) if class_weights_json else {}
    except json.JSONDecodeError:
        print("[ClassBalancer] Warning: Invalid weights JSON, using original dataset")
        return data_yaml, {"mode": "none", "original": True, "error": "Invalid JSON"}
    
    if not class_weights or all(w <= 1.0 for w in class_weights.values()):
        print("[ClassBalancer] No class balancing needed (all weights <= 1.0)")
        return data_yaml, {"mode": "none", "original": True, "reason": "No boosting needed"}
    
    # Load data.yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get paths
    yaml_dir = Path(data_yaml).parent
    train_path = data.get('train', '')
    
    # Resolve relative paths
    if not Path(train_path).is_absolute():
        train_path = str(yaml_dir / train_path)
    
    train_images_dir = Path(train_path)
    
    # Infer labels directory (YOLO convention: images -> labels)
    train_labels_dir = Path(str(train_images_dir).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
    
    if not train_images_dir.exists():
        print(f"[ClassBalancer] Warning: Train images dir not found: {train_images_dir}")
        return data_yaml, {"mode": "none", "error": "Train dir not found"}
    
    if not train_labels_dir.exists():
        print(f"[ClassBalancer] Warning: Train labels dir not found: {train_labels_dir}")
        return data_yaml, {"mode": "none", "error": "Labels dir not found"}
    
    # Get class names
    names = data.get('names', [])
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys())]
    else:
        class_names = names
    
    # Create balancer
    balancer = ClassBalancer(mode=mode, class_weights=class_weights)
    balancer.print_summary()
    
    # Create balanced dataset
    balanced_images, balanced_labels, stats = balancer.create_balanced_dataset(
        train_images_dir=str(train_images_dir),
        train_labels_dir=str(train_labels_dir),
        class_names=class_names,
        output_dir=output_dir,
        use_symlinks=use_symlinks,
    )
    
    # Create new data.yaml
    new_yaml = balancer.create_balanced_data_yaml(
        original_yaml_path=data_yaml,
        balanced_images_dir=balanced_images,
        balanced_labels_dir=balanced_labels,
    )
    
    return new_yaml, stats


def cleanup_balanced_dataset(balanced_dir: str) -> None:
    """
    Clean up temporary balanced dataset directory.
    
    Args:
        balanced_dir: Path to balanced dataset directory to remove
    """
    try:
        balanced_path = Path(balanced_dir)
        if balanced_path.exists() and "balanced_dataset" in str(balanced_path):
            shutil.rmtree(balanced_path)
            print(f"[ClassBalancer] Cleaned up: {balanced_path}")
    except Exception as e:
        print(f"[ClassBalancer] Warning: Could not clean up {balanced_dir}: {e}")


def generate_balancing_report(
    stats: Dict[str, Any],
    output_dir: str,
    class_names: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Generate a detailed class balancing report and save to the output directory.
    
    Outputs to a 'distribution' subfolder:
        - class_distribution_comparison.csv: Before/after class counts
        - class_distribution_report.txt: Human-readable text report
        - class_distribution_comparison.jpg: Bar chart visualization
    
    Args:
        stats: Balancing statistics from create_balanced_training_data
        output_dir: Directory to save the report
        class_names: Optional list of class names for ordering
        
    Returns:
        Path to the distribution folder, or None if not applicable
    """
    if stats.get("original", True) or stats.get("mode") == "none":
        return None
    
    # Create distribution subfolder
    output_path = Path(output_dir) / "distribution"
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / "class_distribution_report.txt"
    csv_path = output_path / "class_distribution_comparison.csv"
    graph_path = output_path / "class_distribution_comparison.jpg"
    
    mode = stats.get("mode", "unknown")
    original_count = stats.get("original_count", 0)
    balanced_count = stats.get("balanced_count", 0)
    class_weights = stats.get("class_weights", {})
    original_class_counts = stats.get("original_class_counts", {})
    duplications = stats.get("duplications", {})
    
    # Calculate per-class statistics with actual before/after counts
    class_stats = []
    for class_name, weight in class_weights.items():
        original_instances = original_class_counts.get(class_name, 0)
        balanced_instances = duplications.get(class_name, 0)
        
        # If no original counts available, estimate from weight
        if original_instances == 0 and balanced_instances > 0:
            original_instances = int(balanced_instances / weight) if weight > 0 else balanced_instances
        
        added = balanced_instances - original_instances
        increase_pct = (added / original_instances * 100) if original_instances > 0 else 0
        
        class_stats.append({
            "class": class_name,
            "weight": weight,
            "before": original_instances,
            "after": balanced_instances,
            "added": added,
            "increase_pct": increase_pct,
        })
    
    # Sort by original count (lowest first to show minority classes at top)
    class_stats.sort(key=lambda x: x["before"])
    
    # Generate text report
    lines = [
        "=" * 90,
        "CLASS DISTRIBUTION REPORT - BEFORE AND AFTER BALANCING",
        "=" * 90,
        "",
        f"Balancing Mode:          {mode}",
        f"Original total images:   {original_count}",
        f"Balanced total images:   {balanced_count}",
        f"Images added:            {balanced_count - original_count}",
        f"Overall increase:        {((balanced_count - original_count) / original_count * 100):.1f}%" if original_count > 0 else "N/A",
        "",
        "-" * 90,
        "CLASS-WISE DISTRIBUTION COMPARISON",
        "-" * 90,
        "",
        f"{'Class':<25} {'Before':>10} {'After':>10} {'Weight':>8} {'Added':>8} {'Increase':>10}",
        "-" * 90,
    ]
    
    for cs in class_stats:
        lines.append(
            f"{cs['class']:<25} {cs['before']:>10} {cs['after']:>10} {cs['weight']:>8.2f}x {cs['added']:>8} {cs['increase_pct']:>9.1f}%"
        )
    
    # Add summary statistics
    before_counts = [cs['before'] for cs in class_stats if cs['before'] > 0]
    after_counts = [cs['after'] for cs in class_stats if cs['after'] > 0]
    
    if before_counts and after_counts:
        before_std = (sum((x - sum(before_counts)/len(before_counts))**2 for x in before_counts) / len(before_counts)) ** 0.5
        after_std = (sum((x - sum(after_counts)/len(after_counts))**2 for x in after_counts) / len(after_counts)) ** 0.5
        before_imbalance = max(before_counts) / min(before_counts) if min(before_counts) > 0 else float('inf')
        after_imbalance = max(after_counts) / min(after_counts) if min(after_counts) > 0 else float('inf')
        
        lines.extend([
            "-" * 90,
            "",
            "DISTRIBUTION STATISTICS:",
            f"  Before balancing:",
            f"    - Min instances:     {min(before_counts)}",
            f"    - Max instances:     {max(before_counts)}",
            f"    - Mean instances:    {sum(before_counts) / len(before_counts):.1f}",
            f"    - Std deviation:     {before_std:.1f}",
            f"    - Imbalance ratio:   {before_imbalance:.2f}x",
            f"  After balancing:",
            f"    - Min instances:     {min(after_counts)}",
            f"    - Max instances:     {max(after_counts)}",
            f"    - Mean instances:    {sum(after_counts) / len(after_counts):.1f}",
            f"    - Std deviation:     {after_std:.1f}",
            f"    - Imbalance ratio:   {after_imbalance:.2f}x",
            "",
        ])
    
    lines.extend([
        "-" * 90,
        "",
        "NOTES:",
        "  - 'Before' = Original class instance count in training images",
        "  - 'After' = Class instance count after oversampling",
        "  - Weight > 1.0 means the class was oversampled",
        "  - Oversampling is done via symbolic links (minimal disk space used)",
        "  - The balanced dataset is temporary and cleaned up after training",
        "  - Lower imbalance ratio after balancing indicates better class balance",
        "",
        "=" * 90,
    ])
    
    # Write text report
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    # Write CSV for easy analysis
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["class", "before", "after", "weight", "added", "increase_pct"])
        writer.writeheader()
        for cs in class_stats:
            writer.writerow(cs)
    
    # Generate comparison bar chart
    _generate_distribution_graph(class_stats, graph_path, mode)
    
    print(f"[ClassBalancer] Saved distribution report: {report_path}")
    print(f"[ClassBalancer] Saved distribution CSV: {csv_path}")
    print(f"[ClassBalancer] Saved distribution graph: {graph_path}")
    
    return str(output_path)


def _generate_distribution_graph(
    class_stats: List[Dict[str, Any]],
    output_path: Path,
    mode: str,
) -> None:
    """
    Generate a bar chart comparing before/after class distribution.
    
    Args:
        class_stats: List of class statistics dictionaries
        output_path: Path to save the JPEG image
        mode: Balancing mode for title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data
        classes = [cs['class'] for cs in class_stats]
        before_counts = [cs['before'] for cs in class_stats]
        after_counts = [cs['after'] for cs in class_stats]
        
        # Create figure with appropriate size
        fig_width = max(10, len(classes) * 0.8)
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Bar positions
        x = np.arange(len(classes))
        bar_width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - bar_width/2, before_counts, bar_width, 
                       label='Before Balancing', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + bar_width/2, after_counts, bar_width, 
                       label='After Balancing', color='#2ecc71', alpha=0.8)
        
        # Customize chart
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Instance Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Class Distribution: Before vs After Balancing\n(Mode: {mode})', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add value labels on bars
        def add_value_labels(bars, values):
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax.annotate(f'{int(val)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8, rotation=90)
        
        add_value_labels(bars1, before_counts)
        add_value_labels(bars2, after_counts)
        
        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save as JPEG
        plt.savefig(output_path, format='jpeg', dpi=150, 
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
    except ImportError as e:
        print(f"[ClassBalancer] Warning: Could not generate graph (matplotlib not available): {e}")
    except Exception as e:
        print(f"[ClassBalancer] Warning: Could not generate distribution graph: {e}")
