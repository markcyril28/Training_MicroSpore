#!/usr/bin/env python3
"""
Dataset Split Script for YOLO Object Detection

This script manages the distribution of images and annotations from a pooled dataset
into training and testing sets with balanced class representation.

WORKFLOW:
    1. Randomly extract image and annotation pairs from the 'Pooled' directory into the 'Temp' folder.  
        with not more than 1400 datapoints or samples per classes.  

    2. From the 'Temp' folder, randomly transfer image and annotation pairs to the 'Train' and 'Test' directory,
        with this distribution or ratio for each classes:
            Default (configurable): 
                Train:  90%
                Test:   10%

    3. Make sure there is no overlap of 'Test' set with 'Train' set.
        Run a check to ensure no images in 'Test' are present in 'Train'.
            Report if satisfied or not. 
    4. Do the same thing for the Blank images (images without annotation text files). 
        Extract from 'Blanks' folder to the 'Train' and 'Test' folder. 
            Default (configurable): 
                    Train:  80%
                    Test:   20%

FEATURES:
    - Preserves original 'Pooled' folder (only moves, never deletes or modify what is inside the 'Pooled' folder)
    - Clears 'Train' and 'Test' folders every time this python script is run. 
    - Random selection for reproducibility (configurable seed)
    - Reports counts of images and annotations in each set after splitting using the 'tally.py' script.
    - MULTITHREADING

CONFIGURATION:
    - RANDOM_SEED: For reproducible random selection
    - Ratio of datapoints per classses for 'Test' and 'Train' set or folder.

file: classes.txt
CLASS INDEX and NAMES REFERENCE:
    0 tetrad
    1 young_microspore
    2 mid_microspore
    3 late_microspore
    4 young_pollen
    5 midlate_pollen
    6 mature_pollen
    7 others
"""

import os
import shutil
import random
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============== CONFIGURATION ==============
SCRIPT_DIR = Path(__file__).parent.resolve()
CLASSES_FILE = SCRIPT_DIR / "classes.txt"
POOLED_DIR = SCRIPT_DIR / "Pooled"
BLANKS_DIR = SCRIPT_DIR / "Blanks"  # Images without annotation files
TEMP_DIR = SCRIPT_DIR / "Temp"
TRAIN_DIR = SCRIPT_DIR / "Train"
TEST_DIR = SCRIPT_DIR / "Test"
TALLY_SCRIPT = SCRIPT_DIR / "tally.py"

# Configurable parameters
RANDOM_SEED = 21  # For reproducibility

# Class-specific sample limits (adjust per class as needed)
# Class indices: 0=tetrad, 1=young_microspore, 2=mid_microspore, 3=late_microspore,
#                4=young_pollen, 5=midlate_pollen, 6=mature_pollen, 7=others
CLASS_SAMPLE_LIMITS = {
    6: 2000,   # mature_pollen
    4: 2000,   # young_pollen
    0: 1400,   # tetrad
    7: 1400,   # others
    3: 1000,   # late_microspore
    2: 1200,   # mid_microspore
    5: 400,   # midlate_pollen (reduced - overrepresented)
    1: 450,    # young_microspore (reduced - overrepresented)
}

DEFAULT_MAX_SAMPLES = 1500  # Fallback for any unlisted class

TRAIN_RATIO = 0.80  
TEST_RATIO = 0.20
MAX_WORKERS = 12    # Number of threads for parallel file operations

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# ===========================================


def load_classes(classes_file: Path) -> list[str]:
    """Load class names from classes.txt."""
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_classes_in_label(label_file: Path) -> set[int]:
    """Get set of class IDs present in a label file."""
    classes = set()
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    classes.add(class_id)
    except (ValueError, IndexError, FileNotFoundError):
        pass
    return classes


def find_image_label_pairs(folder: Path) -> list[tuple[Path, Path]]:
    """Find all image-label pairs in a folder."""
    pairs = []
    for img_file in folder.rglob("*"):
        if img_file.suffix.lower() in IMAGE_EXTENSIONS:
            label_file = img_file.with_suffix('.txt')
            if label_file.exists() and label_file.name != "classes.txt":
                pairs.append((img_file, label_file))
    return pairs


def clear_directory(folder: Path) -> None:
    """Clear all contents of a directory."""
    if folder.exists():
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        folder.mkdir(parents=True, exist_ok=True)


def copy_file_pair(src_img: Path, src_label: Path, dest_dir: Path) -> tuple[Path, Path]:
    """Copy an image-label pair to destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_img = dest_dir / src_img.name
    dest_label = dest_dir / src_label.name
    shutil.copy2(src_img, dest_img)
    shutil.copy2(src_label, dest_label)
    return (dest_img, dest_label)


def step1_pool_to_temp(pairs: list[tuple[Path, Path]], num_classes: int) -> list[tuple[Path, Path]]:
    """
    Step 1: Copy image-label pairs from Pooled to Temp with class balancing.
    Uses CLASS_SAMPLE_LIMITS for per-class maximum samples.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Pooled -> Temp (Class-balanced sampling)")
    print("=" * 60)
    
    # Group pairs by their primary class (first class in label file)
    # Actually, we need to consider all classes in each image for balancing
    class_to_pairs: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    
    for img, label in pairs:
        classes = get_classes_in_label(label)
        for class_id in classes:
            class_to_pairs[class_id].append((img, label))
    
    # Track selected pairs and class counts
    selected_pairs: set[tuple[str, str]] = set()  # Use string paths for hashability
    class_counts: dict[int, int] = defaultdict(int)
    
    # Randomly sample pairs while respecting class limits
    random.seed(RANDOM_SEED)
    
    # Process each class in priority order (based on CLASS_SAMPLE_LIMITS dictionary order)
    for class_id in CLASS_SAMPLE_LIMITS.keys():
        available_pairs = class_to_pairs.get(class_id, [])
        random.shuffle(available_pairs)
        
        for img, label in available_pairs:
            pair_key = (str(img), str(label))
            if pair_key in selected_pairs:
                continue
            
            # Check if adding this pair would exceed any class limit
            pair_classes = get_classes_in_label(label)
            can_add = True
            for c in pair_classes:
                class_limit = CLASS_SAMPLE_LIMITS.get(c, DEFAULT_MAX_SAMPLES)
                if class_counts[c] >= class_limit:
                    can_add = False
                    break
            
            if can_add:
                selected_pairs.add(pair_key)
                for c in pair_classes:
                    class_counts[c] += 1
    
    # Convert back to Path objects
    final_pairs = [(Path(img), Path(label)) for img, label in selected_pairs]
    
    # Clear and prepare Temp directory
    print(f"Clearing Temp directory...")
    clear_directory(TEMP_DIR)
    
    # Copy files to Temp using multithreading
    print(f"Copying {len(final_pairs)} pairs to Temp...")
    copied_pairs = []
    lock = threading.Lock()
    
    def copy_to_temp(pair):
        img, label = pair
        return copy_file_pair(img, label, TEMP_DIR)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(copy_to_temp, pair): pair for pair in final_pairs}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                copied_pairs.append(result)
    
    # Report class distribution
    print(f"\nTemp folder class distribution:")
    for class_id in range(num_classes):
        print(f"  Class {class_id}: {class_counts.get(class_id, 0)} samples")
    print(f"Total pairs copied to Temp: {len(copied_pairs)}")
    
    return copied_pairs


def step2_temp_to_train_test(temp_pairs: list[tuple[Path, Path]], num_classes: int) -> tuple[list[Path], list[Path]]:
    """
    Step 2: Split Temp pairs into Train and Test with class-balanced ratio.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Temp -> Train/Test (Class-balanced split)")
    print("=" * 60)
    
    # Clear Train and Test directories
    print("Clearing Train and Test directories...")
    clear_directory(TRAIN_DIR)
    clear_directory(TEST_DIR)
    
    # Group pairs by primary class for balanced splitting
    class_to_pairs: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    pair_primary_class: dict[tuple[str, str], int] = {}
    
    for img, label in temp_pairs:
        classes = get_classes_in_label(label)
        if classes:
            # Use the first class as primary for splitting
            primary_class = min(classes)
            class_to_pairs[primary_class].append((img, label))
            pair_primary_class[(str(img), str(label))] = primary_class
    
    train_pairs = []
    test_pairs = []
    
    random.seed(RANDOM_SEED)
    
    # Split each class according to ratio
    for class_id in range(num_classes):
        pairs = class_to_pairs.get(class_id, [])
        if not pairs:
            continue
            
        random.shuffle(pairs)
        split_idx = int(len(pairs) * TRAIN_RATIO)
        
        train_pairs.extend(pairs[:split_idx])
        test_pairs.extend(pairs[split_idx:])
    
    # Copy to Train directory using multithreading
    print(f"Copying {len(train_pairs)} pairs to Train...")
    train_images = []
    lock = threading.Lock()
    
    def copy_to_train(pair):
        img, label = pair
        return copy_file_pair(img, label, TRAIN_DIR)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(copy_to_train, pair): pair for pair in train_pairs}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                train_images.append(result[0])
    
    # Copy to Test directory using multithreading
    print(f"Copying {len(test_pairs)} pairs to Test...")
    test_images = []
    
    def copy_to_test(pair):
        img, label = pair
        return copy_file_pair(img, label, TEST_DIR)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(copy_to_test, pair): pair for pair in test_pairs}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                test_images.append(result[0])
    
    print(f"\nTrain: {len(train_images)} pairs")
    print(f"Test: {len(test_images)} pairs")
    
    return train_images, test_images


def step3_verify_no_overlap(train_images: list[Path], test_images: list[Path]) -> bool:
    """
    Step 3: Verify no overlap between Train and Test sets.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Verifying no overlap between Train and Test")
    print("=" * 60)
    
    train_names = {img.name for img in train_images}
    test_names = {img.name for img in test_images}
    
    overlap = train_names.intersection(test_names)
    
    if overlap:
        print(f"❌ OVERLAP DETECTED! {len(overlap)} files found in both Train and Test:")
        for name in list(overlap)[:10]:  # Show first 10
            print(f"  - {name}")
        if len(overlap) > 10:
            print(f"  ... and {len(overlap) - 10} more")
        return False
    else:
        print("✓ No overlap detected. Train and Test sets are disjoint.")
        return True


def find_blank_images(folder: Path) -> list[Path]:
    """
    Find all images in the Blanks folder (images without annotation text files).
    Since the Blanks folder specifically contains images without annotations,
    we simply retrieve all image files from this folder.
    """
    blank_images = []
    for img_file in folder.rglob("*"):
        if img_file.suffix.lower() in IMAGE_EXTENSIONS:
            blank_images.append(img_file)
    return blank_images


def copy_image(src_img: Path, dest_dir: Path) -> Path:
    """Copy a single image to destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_img = dest_dir / src_img.name
    shutil.copy2(src_img, dest_img)
    return dest_img


def step4_split_blank_images() -> tuple[list[Path], list[Path]]:
    """
    Step 4: Split blank images (images without annotations) from Blanks folder
    into Train and Test with the same ratio.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Splitting Blank Images (no annotations)")
    print("=" * 60)
    
    # Check if Blanks directory exists
    if not BLANKS_DIR.exists():
        print(f"Warning: Blanks directory not found at {BLANKS_DIR}")
        print("Skipping blank images split.")
        return [], []
    
    # Find all blank images
    blank_images = find_blank_images(BLANKS_DIR)
    print(f"Found {len(blank_images)} blank images in Blanks folder")
    
    if not blank_images:
        print("No blank images found. Skipping.")
        return [], []
    
    # Shuffle and split according to ratio
    random.seed(RANDOM_SEED)
    random.shuffle(blank_images)
    
    split_idx = int(len(blank_images) * TRAIN_RATIO)
    train_blanks = blank_images[:split_idx]
    test_blanks = blank_images[split_idx:]
    
    # Copy blank images to Train directory using multithreading
    print(f"Copying {len(train_blanks)} blank images to Train...")
    train_copied = []
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(copy_image, img, TRAIN_DIR): img for img in train_blanks}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                train_copied.append(result)
    
    # Copy blank images to Test directory using multithreading
    print(f"Copying {len(test_blanks)} blank images to Test...")
    test_copied = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(copy_image, img, TEST_DIR): img for img in test_blanks}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                test_copied.append(result)
    
    print(f"\nBlank Images Split:")
    print(f"  Train: {len(train_copied)} blank images")
    print(f"  Test: {len(test_copied)} blank images")
    print(f"  Ratio: {len(train_copied)}/{len(blank_images)} = {len(train_copied)/len(blank_images)*100:.1f}% Train")
    
    return train_copied, test_copied


def run_tally():
    """Run the tally.py script to report distribution."""
    print("\n" + "=" * 60)
    print("STEP 5: Running tally.py for distribution report")
    print("=" * 60)
    
    if TALLY_SCRIPT.exists():
        try:
            subprocess.run(["python", str(TALLY_SCRIPT)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running tally.py: {e}")
    else:
        print(f"Warning: tally.py not found at {TALLY_SCRIPT}")


def main():
    """Main function to orchestrate the dataset splitting."""
    print("=" * 60)
    print("DATASET SPLIT SCRIPT")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  Default Max Samples: {DEFAULT_MAX_SAMPLES}")
    print(f"  Class-specific limits: {CLASS_SAMPLE_LIMITS}")
    print(f"  Train/Test Ratio: {TRAIN_RATIO:.0%}/{TEST_RATIO:.0%}")
    print(f"  Max Workers: {MAX_WORKERS}")
    
    # Load classes
    if not CLASSES_FILE.exists():
        print(f"Error: Classes file not found at {CLASSES_FILE}")
        return
    
    class_names = load_classes(CLASSES_FILE)
    num_classes = len(class_names)
    print(f"\nLoaded {num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Check Pooled directory
    if not POOLED_DIR.exists():
        print(f"Error: Pooled directory not found at {POOLED_DIR}")
        return
    
    # Find all image-label pairs in Pooled
    print(f"\nScanning Pooled directory for image-label pairs...")
    pooled_pairs = find_image_label_pairs(POOLED_DIR)
    print(f"Found {len(pooled_pairs)} image-label pairs in Pooled")
    
    if not pooled_pairs:
        print("No image-label pairs found. Exiting.")
        return
    
    # Step 1: Pool to Temp with class balancing
    temp_pairs = step1_pool_to_temp(pooled_pairs, num_classes)
    
    # Step 2: Temp to Train/Test split
    train_images, test_images = step2_temp_to_train_test(temp_pairs, num_classes)
    
    # Step 3: Verify no overlap
    no_overlap = step3_verify_no_overlap(train_images, test_images)
    
    # Step 4: Split blank images (images without annotations)
    train_blanks, test_blanks = step4_split_blank_images()
    
    # Step 5: Run tally for reporting
    run_tally()
    
    # Step 6: Clear Temp folder after processing
    print("\n" + "=" * 60)
    print("STEP 6: Clearing Temp folder")
    print("=" * 60)
    clear_directory(TEMP_DIR)
    print("✓ Temp folder cleared.")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pooled pairs: {len(pooled_pairs)}")
    print(f"Temp pairs: {len(temp_pairs)}")
    print(f"Train pairs: {len(train_images)}")
    print(f"Test pairs: {len(test_images)}")
    print(f"Train blank images: {len(train_blanks)}")
    print(f"Test blank images: {len(test_blanks)}")
    print(f"Overlap check: {'PASSED ✓' if no_overlap else 'FAILED ❌'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
