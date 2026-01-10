#!/usr/bin/env python3
"""
Tally script for counting class distribution in Train and Test datasets.
Outputs distribution as text file and bar graph visualization.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import matplotlib.pyplot as plt
import numpy as np

# ============== CONFIGURATION ==============
SCRIPT_DIR = Path(__file__).parent.resolve()
CLASSES_FILE = SCRIPT_DIR / "classes.txt"
TRAIN_DIR = SCRIPT_DIR / "Train"
TEST_DIR = SCRIPT_DIR / "Test"
POOLED_DIR = SCRIPT_DIR / "Pooled"
POOLED_COPY_DIR = SCRIPT_DIR / "Pooled_Backup"
TEMP_DIR = SCRIPT_DIR / "Temp"
OUTPUT_DIR = SCRIPT_DIR / "Distribution"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MAX_WORKERS = 12  # Number of threads for parallel file operations
INCLUDE_POOLED_COPY = False  # Toggle to include/exclude Pooled_copy folder
# ===========================================


def load_classes(classes_file: Path) -> list[str]:
    """Load class names from classes.txt."""
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def process_label_file(label_file: Path) -> tuple[dict[int, int], bool]:
    """Process a single label file and return class counts and whether it's blank."""
    counts = defaultdict(int)
    is_blank = True
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    counts[class_id] += 1
                    is_blank = False
    except (ValueError, IndexError):
        pass
    return counts, is_blank


def count_annotations_in_file(label_file: Path) -> int:
    """Count total annotations in a single label file."""
    count = 0
    try:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
    except:
        pass
    return count


def count_annotations_per_class_in_file(label_file: Path) -> dict[int, int]:
    """Count annotations per class in a single label file.
    Returns dict mapping class_id -> count."""
    counts = defaultdict(int)
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    counts[class_id] += 1
    except (ValueError, IndexError):
        pass
    return counts


def get_annotations_per_image_by_class(folder: Path, all_class_ids: list[int]) -> dict[int, dict[int, int]]:
    """Get distribution of annotations per image for each class.
    Returns dict mapping class_id -> {annotation_count -> number_of_images}."""
    # Initialize distributions for each class
    distributions = {cid: defaultdict(int) for cid in all_class_ids}
    
    if not folder.exists():
        return distributions
    
    # Find all image files
    image_files = [f for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    
    for img_file in image_files:
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            class_counts = count_annotations_per_class_in_file(label_file)
        else:
            class_counts = {}
        
        # For each class, record how many annotations this image has
        for cid in all_class_ids:
            ann_count = class_counts.get(cid, 0)
            distributions[cid][ann_count] += 1
    
    return distributions


def get_annotations_per_image_distribution(folder: Path) -> dict[int, int]:
    """Get distribution of annotations per image.
    Returns dict mapping annotation_count -> number_of_images."""
    distribution = defaultdict(int)
    
    if not folder.exists():
        return distribution
    
    # Find all image files
    image_files = [f for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    
    for img_file in image_files:
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            ann_count = count_annotations_in_file(label_file)
        else:
            ann_count = 0
        distribution[ann_count] += 1
    
    return distribution


def count_labels(folder: Path) -> tuple[dict[int, int], int]:
    """Count instances per class from YOLO label files (.txt) using multithreading.
    Also returns count of blank (empty) label files."""
    counts = defaultdict(int)
    blank_count = 0
    
    if not folder.exists():
        return counts, blank_count
    
    # Collect all label files first
    label_files = [f for f in folder.rglob("*.txt") if f.name != "classes.txt"]
    
    if not label_files:
        return counts, blank_count
    
    # Process files in parallel
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_label_file, lf): lf for lf in label_files}
        for future in as_completed(futures):
            file_counts, is_blank = future.result()
            with lock:
                for class_id, count in file_counts.items():
                    counts[class_id] += count
                if is_blank:
                    blank_count += 1
    
    return counts, blank_count


def check_unannotated(img_file: Path) -> bool:
    """Check if an image file has no corresponding annotation."""
    label_file = img_file.with_suffix('.txt')
    return not label_file.exists()


def count_unannotated_images(folder: Path) -> int:
    """Count images that don't have a corresponding annotation .txt file using multithreading."""
    if not folder.exists():
        return 0
    
    # Collect all image files first
    image_files = [f for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    
    if not image_files:
        return 0
    
    # Check files in parallel
    unannotated = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(check_unannotated, image_files)
        unannotated = sum(1 for is_unannotated in results if is_unannotated)
    
    return unannotated


def count_images(folder: Path) -> int:
    """Count total number of images in a folder."""
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)


def save_distribution_csv(train_counts: dict, test_counts: dict, pooled_counts: dict,
                          pooled_copy_counts: dict, temp_counts: dict,
                          class_names: list[str], all_class_ids: list[int],
                          unannotated: dict[str, int], output_dir: Path, 
                          image_counts: dict[str, int], include_pooled_copy: bool = True) -> None:
    """Save distribution data to separate CSV files."""
    
    # 0. Image count CSV
    image_count_file = output_dir / "image_counts.csv"
    with open(image_count_file, 'w') as f:
        f.write("Folder,Images\n")
        f.write(f"Train,{image_counts.get('Train', 0)}\n")
        f.write(f"Test,{image_counts.get('Test', 0)}\n")
        f.write(f"Pooled,{image_counts.get('Pooled', 0)}\n")
        f.write(f"Temp,{image_counts.get('Temp', 0)}\n")
        if include_pooled_copy:
            f.write(f"Pooled_copy,{image_counts.get('Pooled_copy', 0)}\n")
        f.write(f"TOTAL,{sum(image_counts.values())}\n")
    
    # 1. Summary CSV
    summary_file = output_dir / "distribution_summary.csv"
    with open(summary_file, 'w') as f:
        if include_pooled_copy:
            f.write("Class,Pooled,Temp,Train,Test,Pooled_copy,Ratio\n")
        else:
            f.write("Class,Pooled,Temp,Train,Test,Ratio\n")
        
        total_train = 0
        total_test = 0
        total_pooled = 0
        total_pooled_copy = 0
        total_temp = 0
        
        for class_id in all_class_ids:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            train_count = train_counts.get(class_id, 0)
            test_count = test_counts.get(class_id, 0)
            pooled_count = pooled_counts.get(class_id, 0)
            pooled_copy_count = pooled_copy_counts.get(class_id, 0) if include_pooled_copy else 0
            temp_count = temp_counts.get(class_id, 0)
            
            train_test_total = train_count + test_count
            if train_test_total > 0:
                train_pct = round(train_count / train_test_total * 100)
                test_pct = 100 - train_pct
                ratio = f"{train_pct}:{test_pct}"
            else:
                ratio = "N/A"
            
            if include_pooled_copy:
                f.write(f"{class_name},{pooled_count},{temp_count},{train_count},{test_count},{pooled_copy_count},{ratio}\n")
            else:
                f.write(f"{class_name},{pooled_count},{temp_count},{train_count},{test_count},{ratio}\n")
            
            total_train += train_count
            total_test += test_count
            total_pooled += pooled_count
            total_pooled_copy += pooled_copy_count
            total_temp += temp_count
        
        # Add Blank row
        blank_train = unannotated.get('Train', 0)
        blank_test = unannotated.get('Test', 0)
        blank_pooled = unannotated.get('Pooled', 0)
        blank_temp = unannotated.get('Temp', 0)
        blank_pooled_copy = unannotated.get('Pooled_copy', 0) if include_pooled_copy else 0
        
        blank_train_test_total = blank_train + blank_test
        if blank_train_test_total > 0:
            blank_train_pct = round(blank_train / blank_train_test_total * 100)
            blank_test_pct = 100 - blank_train_pct
            blank_ratio = f"{blank_train_pct}:{blank_test_pct}"
        else:
            blank_ratio = "N/A"
        
        if include_pooled_copy:
            f.write(f"Blank,{blank_pooled},{blank_temp},{blank_train},{blank_test},{blank_pooled_copy},{blank_ratio}\n")
            f.write(f"TOTAL,{total_pooled},{total_temp},{total_train},{total_test},{total_pooled_copy},\n")
        else:
            f.write(f"Blank,{blank_pooled},{blank_temp},{blank_train},{blank_test},{blank_ratio}\n")
            f.write(f"TOTAL,{total_pooled},{total_temp},{total_train},{total_test},\n")
    
    # 2. Percentage CSV
    percentage_file = output_dir / "distribution_percentage.csv"
    with open(percentage_file, 'w') as f:
        if include_pooled_copy:
            f.write("Class,Pooled %,Temp %,Train %,Test %,Pooled_copy %\n")
        else:
            f.write("Class,Pooled %,Temp %,Train %,Test %\n")
        
        total_train = sum(train_counts.values())
        total_test = sum(test_counts.values())
        total_pooled = sum(pooled_counts.values())
        total_temp = sum(temp_counts.values())
        total_pooled_copy = sum(pooled_copy_counts.values()) if include_pooled_copy else 0
        
        for class_id in all_class_ids:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            train_pct = (train_counts.get(class_id, 0) / total_train * 100) if total_train > 0 else 0
            test_pct = (test_counts.get(class_id, 0) / total_test * 100) if total_test > 0 else 0
            pooled_pct = (pooled_counts.get(class_id, 0) / total_pooled * 100) if total_pooled > 0 else 0
            temp_pct = (temp_counts.get(class_id, 0) / total_temp * 100) if total_temp > 0 else 0
            
            if include_pooled_copy:
                pooled_copy_pct = (pooled_copy_counts.get(class_id, 0) / total_pooled_copy * 100) if total_pooled_copy > 0 else 0
                f.write(f"{class_name},{pooled_pct:.2f}%,{temp_pct:.2f}%,{train_pct:.2f}%,{test_pct:.2f}%,{pooled_copy_pct:.2f}%\n")
            else:
                f.write(f"{class_name},{pooled_pct:.2f}%,{temp_pct:.2f}%,{train_pct:.2f}%,{test_pct:.2f}%\n")
    
    # 3. Sorted CSV
    sorted_file = output_dir / "distribution_sorted.csv"
    with open(sorted_file, 'w') as f:
        if include_pooled_copy:
            f.write("Rank,Class,Train,Test,Total,Pooled,Temp,Pooled_copy\n")
        else:
            f.write("Rank,Class,Train,Test,Total,Pooled,Temp\n")
        
        sorted_data = []
        for class_id in all_class_ids:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            train_count = train_counts.get(class_id, 0)
            test_count = test_counts.get(class_id, 0)
            total_count = train_count + test_count
            pooled_count = pooled_counts.get(class_id, 0)
            temp_count = temp_counts.get(class_id, 0)
            pooled_copy_count = pooled_copy_counts.get(class_id, 0) if include_pooled_copy else 0
            sorted_data.append((class_name, train_count, test_count, total_count, pooled_count, temp_count, pooled_copy_count))
        
        # Add Blank
        blank_train = unannotated.get('Train', 0)
        blank_test = unannotated.get('Test', 0)
        blank_total = blank_train + blank_test
        blank_pooled = unannotated.get('Pooled', 0)
        blank_temp = unannotated.get('Temp', 0)
        blank_pooled_copy = unannotated.get('Pooled_copy', 0) if include_pooled_copy else 0
        sorted_data.append(('Blank', blank_train, blank_test, blank_total, blank_pooled, blank_temp, blank_pooled_copy))
        
        sorted_data.sort(key=lambda x: x[3], reverse=True)
        
        for rank, (class_name, train_count, test_count, total_count, pooled_count, temp_count, pooled_copy_count) in enumerate(sorted_data, 1):
            if include_pooled_copy:
                f.write(f"{rank},{class_name},{train_count},{test_count},{total_count},{pooled_count},{temp_count},{pooled_copy_count}\n")
            else:
                f.write(f"{rank},{class_name},{train_count},{test_count},{total_count},{pooled_count},{temp_count}\n")


def save_distribution_text(train_counts: dict, test_counts: dict, pooled_counts: dict,
                           pooled_copy_counts: dict, temp_counts: dict, blank_counts: dict[str, int],
                           class_names: list[str], all_class_ids: list[int], 
                           unannotated: dict[str, int], output_file: Path, 
                           image_counts: dict[str, int], include_pooled_copy: bool = True) -> None:
    """Save distribution summary to text file."""
    line_width = 120 if include_pooled_copy else 100
    
    with open(output_file, 'w') as f:
        f.write("=" * line_width + "\n")
        f.write("DATASET CLASS DISTRIBUTION SUMMARY\n")
        f.write("=" * line_width + "\n\n")
        
        # Image count summary
        f.write("IMAGE COUNT SUMMARY:\n")
        f.write("-" * line_width + "\n")
        if include_pooled_copy:
            f.write(f"{'Folder':<20} {'Images':>10}\n")
        else:
            f.write(f"{'Folder':<20} {'Images':>10}\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Train':<20} {image_counts.get('Train', 0):>10}\n")
        f.write(f"{'Test':<20} {image_counts.get('Test', 0):>10}\n")
        f.write(f"{'Pooled':<20} {image_counts.get('Pooled', 0):>10}\n")
        f.write(f"{'Temp':<20} {image_counts.get('Temp', 0):>10}\n")
        if include_pooled_copy:
            f.write(f"{'Pooled_copy':<20} {image_counts.get('Pooled_copy', 0):>10}\n")
        total_images = sum(image_counts.values())
        f.write("-" * 40 + "\n")
        f.write(f"{'TOTAL':<20} {total_images:>10}\n")
        f.write("\n")
        
        if include_pooled_copy:
            f.write(f"{'Class':<20} {'Pooled':>10} {'Temp':>10} {'Train':>10} {'Test':>10} {'Pooled_copy':>12} {'Ratio':>10}\n")
        else:
            f.write(f"{'Class':<20} {'Pooled':>10} {'Temp':>10} {'Train':>10} {'Test':>10} {'Ratio':>10}\n")
        f.write("-" * line_width + "\n")
        
        total_train = 0
        total_test = 0
        total_pooled = 0
        total_pooled_copy = 0
        total_temp = 0
        
        for class_id in all_class_ids:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            train_count = train_counts.get(class_id, 0)
            test_count = test_counts.get(class_id, 0)
            pooled_count = pooled_counts.get(class_id, 0)
            pooled_copy_count = pooled_copy_counts.get(class_id, 0) if include_pooled_copy else 0
            temp_count = temp_counts.get(class_id, 0)
            # Calculate ratio as percentage (sums to 100)
            train_test_total = train_count + test_count
            if train_test_total > 0:
                train_pct = round(train_count / train_test_total * 100)
                test_pct = 100 - train_pct
                ratio = f"{train_pct}:{test_pct}"
            else:
                ratio = "N/A"
            
            if include_pooled_copy:
                f.write(f"{class_name:<20} {pooled_count:>10} {temp_count:>10} {train_count:>10} {test_count:>10} {pooled_copy_count:>12} {ratio:>10}\n")
            else:
                f.write(f"{class_name:<20} {pooled_count:>10} {temp_count:>10} {train_count:>10} {test_count:>10} {ratio:>10}\n")
            total_train += train_count
            total_test += test_count
            total_pooled += pooled_count
            total_pooled_copy += pooled_copy_count
            total_temp += temp_count
        
        # Add Blank row (unannotated images - images without corresponding .txt file)
        blank_train = unannotated.get('Train', 0)
        blank_test = unannotated.get('Test', 0)
        blank_pooled = unannotated.get('Pooled', 0)
        blank_temp = unannotated.get('Temp', 0)
        blank_pooled_copy = unannotated.get('Pooled_copy', 0) if include_pooled_copy else 0
        # Calculate ratio as percentage (sums to 100)
        blank_train_test_total = blank_train + blank_test
        if blank_train_test_total > 0:
            blank_train_pct = round(blank_train / blank_train_test_total * 100)
            blank_test_pct = 100 - blank_train_pct
            blank_ratio = f"{blank_train_pct}:{blank_test_pct}"
        else:
            blank_ratio = "N/A"
        
        if include_pooled_copy:
            f.write(f"{'Blank':<20} {blank_pooled:>10} {blank_temp:>10} {blank_train:>10} {blank_test:>10} {blank_pooled_copy:>12} {blank_ratio:>10}\n")
        else:
            f.write(f"{'Blank':<20} {blank_pooled:>10} {blank_temp:>10} {blank_train:>10} {blank_test:>10} {blank_ratio:>10}\n")
        
        f.write("-" * line_width + "\n")
        if include_pooled_copy:
            f.write(f"{'TOTAL':<20} {total_pooled:>10} {total_temp:>10} {total_train:>10} {total_test:>10} {total_pooled_copy:>12}\n")
        else:
            f.write(f"{'TOTAL':<20} {total_pooled:>10} {total_temp:>10} {total_train:>10} {total_test:>10}\n")
        f.write("=" * line_width + "\n")
        
        f.write("\nPERCENTAGE DISTRIBUTION:\n")
        f.write("-" * line_width + "\n")
        if include_pooled_copy:
            f.write(f"{'Class':<20} {'Pooled %':>10} {'Temp %':>10} {'Train %':>10} {'Test %':>10} {'Pooled_copy %':>14}\n")
        else:
            f.write(f"{'Class':<20} {'Pooled %':>10} {'Temp %':>10} {'Train %':>10} {'Test %':>10}\n")
        f.write("-" * line_width + "\n")
        
        for class_id in all_class_ids:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            train_pct = (train_counts.get(class_id, 0) / total_train * 100) if total_train > 0 else 0
            test_pct = (test_counts.get(class_id, 0) / total_test * 100) if total_test > 0 else 0
            pooled_pct = (pooled_counts.get(class_id, 0) / total_pooled * 100) if total_pooled > 0 else 0
            temp_pct = (temp_counts.get(class_id, 0) / total_temp * 100) if total_temp > 0 else 0
            if include_pooled_copy:
                pooled_copy_pct = (pooled_copy_counts.get(class_id, 0) / total_pooled_copy * 100) if total_pooled_copy > 0 else 0
                f.write(f"{class_name:<20} {pooled_pct:>9.2f}% {temp_pct:>9.2f}% {train_pct:>9.2f}% {test_pct:>9.2f}% {pooled_copy_pct:>13.2f}%\n")
            else:
                f.write(f"{class_name:<20} {pooled_pct:>9.2f}% {temp_pct:>9.2f}% {train_pct:>9.2f}% {test_pct:>9.2f}%\n")
        
        # Sorted Class Distribution (by Train + Test total, descending)
        f.write("\n" + "=" * line_width + "\n")
        f.write("SORTED CLASS DISTRIBUTION (by Train + Test total, descending)\n")
        f.write("=" * line_width + "\n\n")
        
        if include_pooled_copy:
            f.write(f"{'Rank':<6} {'Class':<20} {'Train':>10} {'Test':>10} {'Total':>10} {'Pooled':>10} {'Temp':>10} {'Pooled_copy':>12}\n")
        else:
            f.write(f"{'Rank':<6} {'Class':<20} {'Train':>10} {'Test':>10} {'Total':>10} {'Pooled':>10} {'Temp':>10}\n")
        f.write("-" * line_width + "\n")
        
        # Build list of (class_name, train, test, total, pooled, temp, pooled_copy)
        sorted_data = []
        for class_id in all_class_ids:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            train_count = train_counts.get(class_id, 0)
            test_count = test_counts.get(class_id, 0)
            total_count = train_count + test_count
            pooled_count = pooled_counts.get(class_id, 0)
            temp_count = temp_counts.get(class_id, 0)
            pooled_copy_count = pooled_copy_counts.get(class_id, 0) if include_pooled_copy else 0
            sorted_data.append((class_name, train_count, test_count, total_count, pooled_count, temp_count, pooled_copy_count))
        
        # Add Blank to sorted data
        blank_total = blank_train + blank_test
        sorted_data.append(('Blank', blank_train, blank_test, blank_total, blank_pooled, blank_temp, blank_pooled_copy))
        
        # Sort by total (Train + Test) descending
        sorted_data.sort(key=lambda x: x[3], reverse=True)
        
        for rank, (class_name, train_count, test_count, total_count, pooled_count, temp_count, pooled_copy_count) in enumerate(sorted_data, 1):
            if include_pooled_copy:
                f.write(f"{rank:<6} {class_name:<20} {train_count:>10} {test_count:>10} {total_count:>10} {pooled_count:>10} {temp_count:>10} {pooled_copy_count:>12}\n")
            else:
                f.write(f"{rank:<6} {class_name:<20} {train_count:>10} {test_count:>10} {total_count:>10} {pooled_count:>10} {temp_count:>10}\n")
        
        f.write("-" * line_width + "\n")


def save_annotations_per_image_text(distributions: dict[str, dict[int, int]], 
                                     output_file: Path, include_pooled_copy: bool = True) -> None:
    """Save annotations per image distribution to text file."""
    line_width = 100
    
    # Get all unique annotation counts across all folders
    all_ann_counts = set()
    for dist in distributions.values():
        all_ann_counts.update(dist.keys())
    all_ann_counts = sorted(all_ann_counts)
    
    with open(output_file, 'w') as f:
        f.write("=" * line_width + "\n")
        f.write("ANNOTATIONS PER IMAGE DISTRIBUTION\n")
        f.write("=" * line_width + "\n\n")
        f.write("This shows how many images have N annotations (bounding boxes).\n\n")
        
        # Header
        folders = ['Train', 'Test', 'Pooled', 'Temp']
        if include_pooled_copy:
            folders.append('Pooled_copy')
        
        header = f"{'Annotations':<15}"
        for folder in folders:
            header += f"{folder:>12}"
        header += f"{'Total':>12}"
        f.write(header + "\n")
        f.write("-" * line_width + "\n")
        
        totals = {folder: 0 for folder in folders}
        totals['Total'] = 0
        
        for ann_count in all_ann_counts:
            row = f"{ann_count:<15}"
            row_total = 0
            for folder in folders:
                count = distributions.get(folder, {}).get(ann_count, 0)
                row += f"{count:>12}"
                totals[folder] += count
                row_total += count
            row += f"{row_total:>12}"
            totals['Total'] += row_total
            f.write(row + "\n")
        
        f.write("-" * line_width + "\n")
        total_row = f"{'TOTAL':<15}"
        for folder in folders:
            total_row += f"{totals[folder]:>12}"
        total_row += f"{totals['Total']:>12}"
        f.write(total_row + "\n")
        f.write("=" * line_width + "\n")
        
        # Statistics summary
        f.write("\nSTATISTICS SUMMARY:\n")
        f.write("-" * line_width + "\n")
        f.write(f"{'Folder':<15} {'Images':>10} {'Total Ann':>12} {'Avg Ann/Img':>14} {'Max Ann':>10} {'Min Ann':>10}\n")
        f.write("-" * line_width + "\n")
        
        for folder in folders:
            dist = distributions.get(folder, {})
            total_images = sum(dist.values())
            total_annotations = sum(k * v for k, v in dist.items())
            avg_ann = total_annotations / total_images if total_images > 0 else 0
            max_ann = max(dist.keys()) if dist else 0
            min_ann = min(dist.keys()) if dist else 0
            f.write(f"{folder:<15} {total_images:>10} {total_annotations:>12} {avg_ann:>14.2f} {max_ann:>10} {min_ann:>10}\n")
        
        f.write("=" * line_width + "\n")


def save_annotations_per_image_csv(distributions: dict[str, dict[int, int]], 
                                    output_dir: Path, include_pooled_copy: bool = True) -> None:
    """Save annotations per image distribution to CSV."""
    # Get all unique annotation counts
    all_ann_counts = set()
    for dist in distributions.values():
        all_ann_counts.update(dist.keys())
    all_ann_counts = sorted(all_ann_counts)
    
    folders = ['Train', 'Test', 'Pooled', 'Temp']
    if include_pooled_copy:
        folders.append('Pooled_copy')
    
    # Distribution CSV
    csv_file = output_dir / "annotations_per_image.csv"
    with open(csv_file, 'w') as f:
        header = "Annotations," + ",".join(folders) + ",Total\n"
        f.write(header)
        
        for ann_count in all_ann_counts:
            row_values = [str(distributions.get(folder, {}).get(ann_count, 0)) for folder in folders]
            row_total = sum(distributions.get(folder, {}).get(ann_count, 0) for folder in folders)
            f.write(f"{ann_count},{','.join(row_values)},{row_total}\n")
    
    # Statistics CSV
    stats_file = output_dir / "annotations_per_image_stats.csv"
    with open(stats_file, 'w') as f:
        f.write("Folder,Images,Total_Annotations,Avg_Ann_Per_Image,Max_Ann,Min_Ann\n")
        for folder in folders:
            dist = distributions.get(folder, {})
            total_images = sum(dist.values())
            total_annotations = sum(k * v for k, v in dist.items())
            avg_ann = total_annotations / total_images if total_images > 0 else 0
            max_ann = max(dist.keys()) if dist else 0
            min_ann = min(dist.keys()) if dist else 0
            f.write(f"{folder},{total_images},{total_annotations},{avg_ann:.2f},{max_ann},{min_ann}\n")


def save_annotations_per_image_graph(distributions: dict[str, dict[int, int]], 
                                      output_dir: Path, include_pooled_copy: bool = True) -> None:
    """Generate individual histogram visualization for each folder's annotations per image distribution.
    Creates two versions: one including 0 annotations, one excluding 0 annotations."""
    folder_colors = {
        'Train': '#2ecc71',
        'Test': '#3498db', 
        'Pooled': '#e74c3c',
        'Temp': '#f39c12',
        'Pooled_copy': '#9b59b6'
    }
    
    folders = ['Train', 'Test', 'Pooled', 'Temp']
    if include_pooled_copy:
        folders.append('Pooled_copy')
    
    for folder in folders:
        dist = distributions.get(folder, {})
        if not dist:
            continue
        
        # Generate two versions: with and without 0 annotations
        for include_zero in [True, False]:
            if include_zero:
                ann_counts = sorted(dist.keys())
                suffix = ""
                title_suffix = "(Including 0)"
            else:
                ann_counts = sorted([k for k in dist.keys() if k > 0])
                suffix = "_no_zero"
                title_suffix = "(Excluding 0)"
            
            if not ann_counts:
                continue
            
            values = [dist.get(ann, 0) for ann in ann_counts]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x = np.arange(len(ann_counts))
            bars = ax.bar(x, values, color=folder_colors.get(folder, '#888888'), edgecolor='black')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Number of Annotations per Image', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
            ax.set_title(f'{folder}: Distribution of Annotations per Image {title_suffix}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([str(a) for a in ann_counts], fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            # Add statistics text
            total_images = sum(values)
            total_annotations = sum(k * v for k, v in dist.items() if (include_zero or k > 0))
            avg_ann = total_annotations / total_images if total_images > 0 else 0
            max_ann = max(ann_counts) if ann_counts else 0
            stats_text = f'Images: {total_images} | Annotations: {total_annotations} | Avg: {avg_ann:.2f}/img | Max: {max_ann}'
            ax.text(0.5, 0.97, stats_text, transform=ax.transAxes, ha='center', va='top', 
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            output_file = output_dir / f"annotations_per_image_{folder}{suffix}.jpeg"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()


def save_annotations_per_class_text(distributions: dict[str, dict[int, dict[int, int]]], 
                                     class_names: list[str], all_class_ids: list[int],
                                     output_file: Path, include_pooled_copy: bool = True) -> None:
    """Save per-class annotations per image distribution to text file."""
    line_width = 120
    
    folders = ['Train', 'Test', 'Pooled', 'Temp']
    if include_pooled_copy:
        folders.append('Pooled_copy')
    
    with open(output_file, 'w') as f:
        f.write("=" * line_width + "\n")
        f.write("ANNOTATIONS PER IMAGE DISTRIBUTION BY CLASS\n")
        f.write("=" * line_width + "\n\n")
        f.write("For each class, shows how many images have N annotations of that class.\n\n")
        
        for class_id in all_class_ids:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            # Get all unique annotation counts for this class across all folders
            all_ann_counts = set()
            for folder in folders:
                folder_dist = distributions.get(folder, {})
                class_dist = folder_dist.get(class_id, {})
                all_ann_counts.update(class_dist.keys())
            all_ann_counts = sorted(all_ann_counts)
            
            if not all_ann_counts:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"CLASS: {class_name}\n")
            f.write(f"{'='*80}\n")
            
            # Header
            header = f"{'Annotations':<15}"
            for folder in folders:
                header += f"{folder:>12}"
            header += f"{'Total':>12}"
            f.write(header + "\n")
            f.write("-" * 80 + "\n")
            
            totals = {folder: 0 for folder in folders}
            totals['Total'] = 0
            
            for ann_count in all_ann_counts:
                row = f"{ann_count:<15}"
                row_total = 0
                for folder in folders:
                    count = distributions.get(folder, {}).get(class_id, {}).get(ann_count, 0)
                    row += f"{count:>12}"
                    totals[folder] += count
                    row_total += count
                row += f"{row_total:>12}"
                totals['Total'] += row_total
                f.write(row + "\n")
            
            f.write("-" * 80 + "\n")
            
            # Statistics for this class
            f.write(f"\nStatistics for {class_name}:\n")
            for folder in folders:
                class_dist = distributions.get(folder, {}).get(class_id, {})
                total_images = sum(class_dist.values())
                total_annotations = sum(k * v for k, v in class_dist.items())
                avg_ann = total_annotations / total_images if total_images > 0 else 0
                non_zero_imgs = sum(v for k, v in class_dist.items() if k > 0)
                f.write(f"  {folder}: {total_images} images, {total_annotations} annotations, "
                        f"avg={avg_ann:.2f}/img, {non_zero_imgs} images with ≥1 annotation\n")
        
        f.write("\n" + "=" * line_width + "\n")


def save_annotations_per_class_csv(distributions: dict[str, dict[int, dict[int, int]]], 
                                    class_names: list[str], all_class_ids: list[int],
                                    output_dir: Path, include_pooled_copy: bool = True) -> None:
    """Save per-class annotations per image distribution to CSV."""
    folders = ['Train', 'Test', 'Pooled', 'Temp']
    if include_pooled_copy:
        folders.append('Pooled_copy')
    
    # Summary CSV with stats per class
    summary_file = output_dir / "annotations_per_class_summary.csv"
    with open(summary_file, 'w') as f:
        header = "Class," + ",".join([f"{folder}_Images,{folder}_Annotations,{folder}_Avg" for folder in folders]) + "\n"
        f.write(header)
        
        for class_id in all_class_ids:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            row = [class_name]
            
            for folder in folders:
                class_dist = distributions.get(folder, {}).get(class_id, {})
                total_images = sum(class_dist.values())
                total_annotations = sum(k * v for k, v in class_dist.items())
                avg_ann = total_annotations / total_images if total_images > 0 else 0
                row.extend([str(total_images), str(total_annotations), f"{avg_ann:.2f}"])
            
            f.write(",".join(row) + "\n")
    
    # Detailed CSV per class
    detail_file = output_dir / "annotations_per_class_detail.csv"
    with open(detail_file, 'w') as f:
        header = "Class,Annotations," + ",".join(folders) + ",Total\n"
        f.write(header)
        
        for class_id in all_class_ids:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            # Get all unique annotation counts for this class
            all_ann_counts = set()
            for folder in folders:
                class_dist = distributions.get(folder, {}).get(class_id, {})
                all_ann_counts.update(class_dist.keys())
            all_ann_counts = sorted(all_ann_counts)
            
            for ann_count in all_ann_counts:
                row_values = []
                row_total = 0
                for folder in folders:
                    count = distributions.get(folder, {}).get(class_id, {}).get(ann_count, 0)
                    row_values.append(str(count))
                    row_total += count
                f.write(f"{class_name},{ann_count},{','.join(row_values)},{row_total}\n")


def save_annotations_per_class_graph(distributions: dict[str, dict[int, dict[int, int]]], 
                                      class_names: list[str], all_class_ids: list[int],
                                      output_dir: Path, include_pooled_copy: bool = True) -> None:
    """Generate per-class histogram visualizations for annotations per image."""
    folders = ['Train', 'Test', 'Pooled', 'Temp']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    if include_pooled_copy:
        folders.append('Pooled_copy')
        colors.append('#9b59b6')
    
    # Create a combined overview graph showing average annotations per image for each class
    overview_file = output_dir / "annotations_per_class_overview.jpeg"
    
    x = np.arange(len(all_class_ids))
    width = 0.15 if include_pooled_copy else 0.2
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    num_folders = len(folders)
    for i, (folder, color) in enumerate(zip(folders, colors)):
        offset = (i - (num_folders - 1) / 2) * width
        avg_values = []
        for class_id in all_class_ids:
            class_dist = distributions.get(folder, {}).get(class_id, {})
            total_images = sum(class_dist.values())
            total_annotations = sum(k * v for k, v in class_dist.items())
            avg = total_annotations / total_images if total_images > 0 else 0
            avg_values.append(avg)
        
        bars = ax.bar(x + offset, avg_values, width, label=folder, color=color, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=6)
    
    display_names = [class_names[cid] if cid < len(class_names) else f"class_{cid}" for cid in all_class_ids]
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Annotations per Image', fontsize=12, fontweight='bold')
    ax.set_title('Average Annotations per Image by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(overview_file, dpi=150, bbox_inches='tight')
    plt.close()


def save_distribution_graph(train_counts: dict, test_counts: dict, pooled_counts: dict,
                            pooled_copy_counts: dict, temp_counts: dict, class_names: list[str], all_class_ids: list[int], 
                            output_dir: Path, include_pooled_copy: bool = True) -> None:
    """Generate and save bar chart visualizations.
    Creates multiple versions with different folder combinations and color schemes."""
    # Build display names for each class
    display_names = [class_names[cid] if cid < len(class_names) else f"class_{cid}" for cid in all_class_ids]
    
    x = np.arange(len(all_class_ids))
    
    train_values = [train_counts.get(cid, 0) for cid in all_class_ids]
    test_values = [test_counts.get(cid, 0) for cid in all_class_ids]
    pooled_values = [pooled_counts.get(cid, 0) for cid in all_class_ids]
    temp_values = [temp_counts.get(cid, 0) for cid in all_class_ids]
    pooled_copy_values = [pooled_copy_counts.get(cid, 0) for cid in all_class_ids] if include_pooled_copy else []
    
    def add_labels(ax, bars, fontsize=7):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=fontsize)
    
    # Define color schemes
    color_schemes = {
        'default': {
            'Pooled': '#e74c3c', 'Temp': '#f39c12', 'Train': '#2ecc71', 
            'Test': '#3498db', 'Pooled_copy': '#9b59b6'
        },
        'pastel': {
            'Pooled': '#ffb3ba', 'Temp': '#ffdfba', 'Train': '#baffc9', 
            'Test': '#bae1ff', 'Pooled_copy': '#e0b3ff'
        },
        'dark': {
            'Pooled': '#c0392b', 'Temp': '#d35400', 'Train': '#27ae60', 
            'Test': '#2980b9', 'Pooled_copy': '#8e44ad'
        },
        'colorblind': {
            'Pooled': '#d55e00', 'Temp': '#cc79a7', 'Train': '#009e73', 
            'Test': '#0072b2', 'Pooled_copy': '#f0e442'
        }
    }
    
    def create_graph(folder_config, colors, title, filename, width_factor=1.0):
        """Helper to create a bar graph with specified folders and colors."""
        n_bars = len(folder_config)
        width = 0.8 / n_bars * width_factor
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        all_bars = []
        for i, (folder_name, values) in enumerate(folder_config):
            offset = (i - (n_bars - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, label=folder_name, 
                         color=colors[folder_name], edgecolor='black')
            all_bars.append(bars)
            add_labels(ax, bars)
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Folder configurations
    all_folders = [('Pooled', pooled_values), ('Temp', temp_values), 
                   ('Train', train_values), ('Test', test_values)]
    if include_pooled_copy:
        all_folders.append(('Pooled_copy', pooled_copy_values))
    
    train_test_only = [('Train', train_values), ('Test', test_values)]
    no_temp = [('Pooled', pooled_values), ('Train', train_values), ('Test', test_values)]
    
    # Generate graphs for each color scheme
    for scheme_name, colors in color_schemes.items():
        suffix = f"_{scheme_name}" if scheme_name != 'default' else ""
        
        # All folders
        title_all = 'Dataset Class Distribution: Pooled vs Temp vs Train vs Test'
        if include_pooled_copy:
            title_all += ' vs Pooled_copy'
        create_graph(all_folders, colors, title_all, f"distribution_graph{suffix}.jpeg")
        
        # Train and Test only
        create_graph(train_test_only, colors, 
                    'Dataset Class Distribution: Train vs Test',
                    f"distribution_graph_train_test{suffix}.jpeg")
        
        # No Temp
        create_graph(no_temp, colors,
                    'Dataset Class Distribution: Pooled vs Train vs Test',
                    f"distribution_graph_no_temp{suffix}.jpeg")


def main():
    """Main execution."""
    # Load class names if available
    class_names = []
    if CLASSES_FILE.exists():
        class_names = load_classes(CLASSES_FILE)
        print(f"Loaded {len(class_names)} class names from classes.txt")
    else:
        print(f"Warning: {CLASSES_FILE} not found, using class IDs as names")
    
    print(f"\nInclude Pooled_copy: {INCLUDE_POOLED_COPY}")
    print(f"Counting labels using {MAX_WORKERS} threads...")
    
    blank_counts = {}
    
    print(f"  Processing Train folder: {TRAIN_DIR}")
    train_counts, blank_counts['Train'] = count_labels(TRAIN_DIR)
    
    print(f"  Processing Test folder: {TEST_DIR}")
    test_counts, blank_counts['Test'] = count_labels(TEST_DIR)
    
    print(f"  Processing Pooled folder: {POOLED_DIR}")
    pooled_counts, blank_counts['Pooled'] = count_labels(POOLED_DIR)
    
    print(f"  Processing Temp folder: {TEMP_DIR}")
    temp_counts, blank_counts['Temp'] = count_labels(TEMP_DIR)
    
    pooled_copy_counts = {}
    if INCLUDE_POOLED_COPY:
        print(f"  Processing Pooled_copy folder: {POOLED_COPY_DIR}")
        pooled_copy_counts, blank_counts['Pooled_copy'] = count_labels(POOLED_COPY_DIR)
    
    # Count images in each folder
    print(f"\nCounting images...")
    image_counts = {
        'Train': count_images(TRAIN_DIR),
        'Test': count_images(TEST_DIR),
        'Pooled': count_images(POOLED_DIR),
        'Temp': count_images(TEMP_DIR),
    }
    if INCLUDE_POOLED_COPY:
        image_counts['Pooled_copy'] = count_images(POOLED_COPY_DIR)
    
    image_counts_str = ", ".join(f"{k}={v}" for k, v in image_counts.items())
    print(f"  {image_counts_str}")
    
    # Count unannotated images
    print(f"\nCounting unannotated images...")
    unannotated = {
        'Train': count_unannotated_images(TRAIN_DIR),
        'Test': count_unannotated_images(TEST_DIR),
        'Pooled': count_unannotated_images(POOLED_DIR),
        'Temp': count_unannotated_images(TEMP_DIR),
    }
    if INCLUDE_POOLED_COPY:
        unannotated['Pooled_copy'] = count_unannotated_images(POOLED_COPY_DIR)
    
    unannotated_str = ", ".join(f"{k}={v}" for k, v in unannotated.items())
    print(f"  {unannotated_str}")
    
    # Get annotations per image distribution
    print(f"\nCalculating annotations per image distribution...")
    ann_per_image_dist = {
        'Train': get_annotations_per_image_distribution(TRAIN_DIR),
        'Test': get_annotations_per_image_distribution(TEST_DIR),
        'Pooled': get_annotations_per_image_distribution(POOLED_DIR),
        'Temp': get_annotations_per_image_distribution(TEMP_DIR),
    }
    if INCLUDE_POOLED_COPY:
        ann_per_image_dist['Pooled_copy'] = get_annotations_per_image_distribution(POOLED_COPY_DIR)
    
    for folder, dist in ann_per_image_dist.items():
        total_imgs = sum(dist.values())
        total_anns = sum(k * v for k, v in dist.items())
        avg = total_anns / total_imgs if total_imgs > 0 else 0
        print(f"  {folder}: {total_imgs} images, {total_anns} annotations, avg={avg:.2f}/img")
    
    # Discover all unique class IDs across all folders
    all_class_ids = sorted(set(train_counts.keys()) | set(test_counts.keys()) | set(pooled_counts.keys()) | set(pooled_copy_counts.keys()) | set(temp_counts.keys()))
    print(f"Found {len(all_class_ids)} unique classes: {all_class_ids}")
    
    # Clear and recreate output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create organized subdirectories
    class_dist_dir = OUTPUT_DIR / "1_class_distribution"
    ann_per_image_dir = OUTPUT_DIR / "2_annotations_per_image"
    ann_per_class_dir = OUTPUT_DIR / "3_annotations_per_class"
    class_dist_dir.mkdir(exist_ok=True)
    ann_per_image_dir.mkdir(exist_ok=True)
    ann_per_class_dir.mkdir(exist_ok=True)
    
    # Save class distribution outputs
    text_file = class_dist_dir / "distribution.txt"
    
    save_distribution_text(train_counts, test_counts, pooled_counts, pooled_copy_counts, temp_counts, blank_counts, class_names, all_class_ids, unannotated, text_file, image_counts, INCLUDE_POOLED_COPY)
    print(f"Saved distribution text: {text_file}")
    
    save_distribution_csv(train_counts, test_counts, pooled_counts, pooled_copy_counts, temp_counts, class_names, all_class_ids, unannotated, class_dist_dir, image_counts, INCLUDE_POOLED_COPY)
    print(f"Saved distribution CSVs: {class_dist_dir}")
    
    save_distribution_graph(train_counts, test_counts, pooled_counts, pooled_copy_counts, temp_counts, class_names, all_class_ids, class_dist_dir, INCLUDE_POOLED_COPY)
    print(f"Saved distribution graphs: {class_dist_dir}")
    
    # Save annotations per image distribution
    ann_per_image_text = ann_per_image_dir / "annotations_per_image.txt"
    
    save_annotations_per_image_text(ann_per_image_dist, ann_per_image_text, INCLUDE_POOLED_COPY)
    print(f"Saved annotations per image text: {ann_per_image_text}")
    
    save_annotations_per_image_csv(ann_per_image_dist, ann_per_image_dir, INCLUDE_POOLED_COPY)
    print(f"Saved annotations per image CSVs: {ann_per_image_dir}")
    
    save_annotations_per_image_graph(ann_per_image_dist, ann_per_image_dir, INCLUDE_POOLED_COPY)
    print(f"Saved annotations per image graphs: {ann_per_image_dir}")
    
    # Calculate and save per-class annotations per image distribution
    print(f"\nCalculating per-class annotations per image distribution...")
    ann_per_class_dist = {
        'Train': get_annotations_per_image_by_class(TRAIN_DIR, all_class_ids),
        'Test': get_annotations_per_image_by_class(TEST_DIR, all_class_ids),
        'Pooled': get_annotations_per_image_by_class(POOLED_DIR, all_class_ids),
        'Temp': get_annotations_per_image_by_class(TEMP_DIR, all_class_ids),
    }
    if INCLUDE_POOLED_COPY:
        ann_per_class_dist['Pooled_copy'] = get_annotations_per_image_by_class(POOLED_COPY_DIR, all_class_ids)
    
    ann_per_class_text = ann_per_class_dir / "annotations_per_class.txt"
    save_annotations_per_class_text(ann_per_class_dist, class_names, all_class_ids, ann_per_class_text, INCLUDE_POOLED_COPY)
    print(f"Saved per-class annotations text: {ann_per_class_text}")
    
    save_annotations_per_class_csv(ann_per_class_dist, class_names, all_class_ids, ann_per_class_dir, INCLUDE_POOLED_COPY)
    print(f"Saved per-class annotations CSVs: {ann_per_class_dir}")
    
    save_annotations_per_class_graph(ann_per_class_dist, class_names, all_class_ids, ann_per_class_dir, INCLUDE_POOLED_COPY)
    print(f"Saved per-class annotations graph: {ann_per_class_dir}")
    
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    total_pooled = sum(pooled_counts.values())
    total_temp = sum(temp_counts.values())
    total_pooled_copy = sum(pooled_copy_counts.values()) if INCLUDE_POOLED_COPY else 0
    total_images = sum(image_counts.values())
    
    print(f"\n{'='*60}")
    print(f"IMAGE STATISTICS:")
    print(f"  Total Images: {total_images}")
    for folder, count in image_counts.items():
        print(f"    {folder}: {count}")
    
    print(f"\nANNOTATION STATISTICS:")
    if INCLUDE_POOLED_COPY:
        print(f"  Train={total_train}, Test={total_test}, Pooled={total_pooled}, Pooled_copy={total_pooled_copy}, Temp={total_temp}")
        print(f"  Total Annotations={total_train + total_test + total_pooled + total_pooled_copy + total_temp}")
    else:
        print(f"  Train={total_train}, Test={total_test}, Pooled={total_pooled}, Temp={total_temp}")
        print(f"  Total Annotations={total_train + total_test + total_pooled + total_temp}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
