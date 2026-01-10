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


def save_distribution_csv(train_counts: dict, test_counts: dict, pooled_counts: dict,
                          pooled_copy_counts: dict, temp_counts: dict,
                          class_names: list[str], all_class_ids: list[int],
                          unannotated: dict[str, int], output_dir: Path, include_pooled_copy: bool = True) -> None:
    """Save distribution data to separate CSV files."""
    
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
                           unannotated: dict[str, int], output_file: Path, include_pooled_copy: bool = True) -> None:
    """Save distribution summary to text file."""
    line_width = 120 if include_pooled_copy else 100
    
    with open(output_file, 'w') as f:
        f.write("=" * line_width + "\n")
        f.write("DATASET CLASS DISTRIBUTION SUMMARY\n")
        f.write("=" * line_width + "\n\n")
        
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


def save_distribution_graph(train_counts: dict, test_counts: dict, pooled_counts: dict,
                            pooled_copy_counts: dict, temp_counts: dict, class_names: list[str], all_class_ids: list[int], 
                            output_file: Path, include_pooled_copy: bool = True) -> None:
    """Generate and save bar chart visualization."""
    # Build display names for each class
    display_names = [class_names[cid] if cid < len(class_names) else f"class_{cid}" for cid in all_class_ids]
    
    x = np.arange(len(all_class_ids))
    
    train_values = [train_counts.get(cid, 0) for cid in all_class_ids]
    test_values = [test_counts.get(cid, 0) for cid in all_class_ids]
    pooled_values = [pooled_counts.get(cid, 0) for cid in all_class_ids]
    temp_values = [temp_counts.get(cid, 0) for cid in all_class_ids]
    
    if include_pooled_copy:
        width = 0.15
        pooled_copy_values = [pooled_copy_counts.get(cid, 0) for cid in all_class_ids]
        fig, ax = plt.subplots(figsize=(18, 8))
        
        bars1 = ax.bar(x - 2*width, train_values, width, label='Train', color='#2ecc71', edgecolor='black')
        bars2 = ax.bar(x - width, test_values, width, label='Test', color='#3498db', edgecolor='black')
        bars3 = ax.bar(x, pooled_values, width, label='Pooled', color='#e74c3c', edgecolor='black')
        bars4 = ax.bar(x + width, pooled_copy_values, width, label='Pooled_copy', color='#9b59b6', edgecolor='black')
        bars5 = ax.bar(x + 2*width, temp_values, width, label='Temp', color='#f39c12', edgecolor='black')
        ax.set_title('Dataset Class Distribution: Train vs Test vs Pooled vs Pooled_copy vs Temp', fontsize=14, fontweight='bold')
    else:
        width = 0.2
        fig, ax = plt.subplots(figsize=(16, 8))
        
        bars1 = ax.bar(x - 1.5*width, train_values, width, label='Train', color='#2ecc71', edgecolor='black')
        bars2 = ax.bar(x - 0.5*width, test_values, width, label='Test', color='#3498db', edgecolor='black')
        bars3 = ax.bar(x + 0.5*width, pooled_values, width, label='Pooled', color='#e74c3c', edgecolor='black')
        bars5 = ax.bar(x + 1.5*width, temp_values, width, label='Temp', color='#f39c12', edgecolor='black')
        ax.set_title('Dataset Class Distribution: Train vs Test vs Pooled vs Temp', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    if include_pooled_copy:
        add_labels(bars4)
    add_labels(bars5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


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
    
    # Discover all unique class IDs across all folders
    all_class_ids = sorted(set(train_counts.keys()) | set(test_counts.keys()) | set(pooled_counts.keys()) | set(pooled_copy_counts.keys()) | set(temp_counts.keys()))
    print(f"Found {len(all_class_ids)} unique classes: {all_class_ids}")
    
    # Clear and recreate output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    text_file = OUTPUT_DIR / "distribution.txt"
    graph_file = OUTPUT_DIR / "distribution_graph.jpeg"
    
    save_distribution_text(train_counts, test_counts, pooled_counts, pooled_copy_counts, temp_counts, blank_counts, class_names, all_class_ids, unannotated, text_file, INCLUDE_POOLED_COPY)
    print(f"Saved distribution text: {text_file}")
    
    save_distribution_csv(train_counts, test_counts, pooled_counts, pooled_copy_counts, temp_counts, class_names, all_class_ids, unannotated, OUTPUT_DIR, INCLUDE_POOLED_COPY)
    print(f"Saved distribution CSVs: {OUTPUT_DIR}")
    
    save_distribution_graph(train_counts, test_counts, pooled_counts, pooled_copy_counts, temp_counts, class_names, all_class_ids, graph_file, INCLUDE_POOLED_COPY)
    print(f"Saved distribution graph: {graph_file}")
    
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    total_pooled = sum(pooled_counts.values())
    total_temp = sum(temp_counts.values())
    total_pooled_copy = sum(pooled_copy_counts.values()) if INCLUDE_POOLED_COPY else 0
    
    if INCLUDE_POOLED_COPY:
        print(f"\nSummary: Train={total_train}, Test={total_test}, Pooled={total_pooled}, Pooled_copy={total_pooled_copy}, Temp={total_temp}, Total={total_train + total_test + total_pooled + total_pooled_copy + total_temp}")
    else:
        print(f"\nSummary: Train={total_train}, Test={total_test}, Pooled={total_pooled}, Temp={total_temp}, Total={total_train + total_test + total_pooled + total_temp}")


if __name__ == "__main__":
    main()
