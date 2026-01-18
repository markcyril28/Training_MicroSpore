"""
Orphan File Scanner for YOLO Dataset

This script scans for images without annotation files and annotation files without images.
It reports the findings and prompts whether to delete these orphaned files.

"""

import os
import sys
import argparse
from pathlib import Path
from typing import Set, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
# Annotation extension
ANNOTATION_EXTENSION = '.txt'
# Files to exclude from orphan detection (e.g., YOLO class definition files)
EXCLUDED_FILES = {'classes.txt'}
# Number of threads for parallel processing
MAX_WORKERS = 12


def process_file(file: Path) -> Tuple[str, str, str]:
    """
    Process a single file and return its classification.
    
    Args:
        file: Path to the file to process
        
    Returns:
        Tuple of (basename, file_type, full_path) where file_type is 'image', 'annotation', or 'other'
    """
    if not file.is_file():
        return (None, 'other', None)
    
    # Skip excluded files
    if file.name.lower() in {f.lower() for f in EXCLUDED_FILES}:
        return (None, 'excluded', None)
    
    suffix_lower = file.suffix.lower()
    
    if suffix_lower in IMAGE_EXTENSIONS:
        return (file.stem, 'image', str(file))
    elif suffix_lower == ANNOTATION_EXTENSION:
        return (file.stem, 'annotation', str(file))
    
    return (None, 'other', None)


def get_basenames(folder: Path, extensions: Set[str]) -> Set[str]:
    """
    Get all file basenames (without extension) for files matching given extensions.
    
    Args:
        folder: Path to the folder to scan
        extensions: Set of file extensions to look for (lowercase, with dot)
    
    Returns:
        Set of basenames (stems) found
    """
    basenames = set()
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            # Skip excluded files
            if file.name.lower() in {f.lower() for f in EXCLUDED_FILES}:
                continue
            basenames.add(file.stem)
    return basenames


def find_orphan_files(folder: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find orphan files in the folder using multithreading.
    
    Args:
        folder: Path to the folder to scan
        
    Returns:
        Tuple of (images_without_annotations, annotations_without_images)
    """
    # Collect all files first
    all_files = list(folder.iterdir())
    
    # Process files in parallel
    image_files = {}  # basename -> path
    annotation_files = {}  # basename -> path
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        
        for future in as_completed(futures):
            basename, file_type, full_path = future.result()
            
            if file_type == 'image' and basename:
                image_files[basename] = Path(full_path)
            elif file_type == 'annotation' and basename:
                annotation_files[basename] = Path(full_path)
    
    # Find orphans
    image_basenames = set(image_files.keys())
    annotation_basenames = set(annotation_files.keys())
    
    images_without_annotations = image_basenames - annotation_basenames
    annotations_without_images = annotation_basenames - image_basenames
    
    # Get orphan file paths
    orphan_images = [image_files[b] for b in images_without_annotations]
    orphan_annotations = [annotation_files[b] for b in annotations_without_images]
    
    # Sort for consistent output
    orphan_images.sort()
    orphan_annotations.sort()
    
    return orphan_images, orphan_annotations


def print_report(orphan_images: List[Path], orphan_annotations: List[Path]) -> None:
    """Print a detailed report of orphan files."""
    print("\n" + "=" * 70)
    print("ORPHAN FILE SCAN REPORT")
    print("=" * 70)
    
    # Images without annotations
    print(f"\n📷 IMAGES WITHOUT ANNOTATION FILES ({len(orphan_images)} found):")
    print("-" * 50)
    if orphan_images:
        for img in orphan_images:
            print(f"  - {img.name}")
    else:
        print("  None found ✓")
    
    # Annotations without images
    print(f"\n📝 ANNOTATION FILES WITHOUT IMAGES ({len(orphan_annotations)} found):")
    print("-" * 50)
    if orphan_annotations:
        for ann in orphan_annotations:
            print(f"  - {ann.name}")
    else:
        print("  None found ✓")
    
    # Summary
    total_orphans = len(orphan_images) + len(orphan_annotations)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {total_orphans} orphan file(s) found")
    print(f"  - {len(orphan_images)} image(s) without annotations")
    print(f"  - {len(orphan_annotations)} annotation(s) without images")
    print("=" * 70 + "\n")


def delete_file_task(file: Path) -> Tuple[bool, str, str]:
    """
    Delete a single file (for parallel deletion).
    
    Args:
        file: Path to the file to delete
        
    Returns:
        Tuple of (success, filename, error_message)
    """
    try:
        file.unlink()
        return (True, file.name, None)
    except Exception as e:
        return (False, file.name, str(e))


def delete_files(files: List[Path], file_type: str) -> int:
    """
    Delete the specified files using multithreading.
    
    Args:
        files: List of file paths to delete
        file_type: Description of file type for logging
        
    Returns:
        Number of files successfully deleted
    """
    if not files:
        return 0
    
    deleted_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(delete_file_task, f): f for f in files}
        
        for future in as_completed(futures):
            success, filename, error = future.result()
            if success:
                print(f"  ✓ Deleted: {filename}")
                deleted_count += 1
            else:
                print(f"  ✗ Failed to delete {filename}: {error}")
    
    return deleted_count


def prompt_deletion(orphan_images: List[Path], orphan_annotations: List[Path]) -> None:
    """Prompt user for deletion options and handle the deletion."""
    total_orphans = len(orphan_images) + len(orphan_annotations)
    
    if total_orphans == 0:
        print("No orphan files to delete. Dataset is clean! ✓")
        return
    
    print("\nDELETION OPTIONS:")
    print("-" * 50)
    print("  1. Delete ALL orphan files")
    print("  2. Delete only orphan IMAGES (without annotations)")
    print("  3. Delete only orphan ANNOTATIONS (without images)")
    print("  4. Cancel (do not delete anything)")
    print("-" * 50)
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\n⚠️  WARNING: This will permanently delete ALL orphan files!")
            confirm = input("Type 'YES' to confirm: ").strip()
            if confirm == 'YES':
                print("\nDeleting orphan images...")
                img_deleted = delete_files(orphan_images, "images")
                print("\nDeleting orphan annotations...")
                ann_deleted = delete_files(orphan_annotations, "annotations")
                print(f"\n✓ Deletion complete: {img_deleted + ann_deleted} file(s) deleted")
            else:
                print("Deletion cancelled.")
            break
            
        elif choice == '2':
            if not orphan_images:
                print("No orphan images to delete.")
            else:
                print(f"\n⚠️  WARNING: This will permanently delete {len(orphan_images)} orphan image(s)!")
                confirm = input("Type 'YES' to confirm: ").strip()
                if confirm == 'YES':
                    print("\nDeleting orphan images...")
                    img_deleted = delete_files(orphan_images, "images")
                    print(f"\n✓ Deletion complete: {img_deleted} image(s) deleted")
                else:
                    print("Deletion cancelled.")
            break
            
        elif choice == '3':
            if not orphan_annotations:
                print("No orphan annotations to delete.")
            else:
                print(f"\n⚠️  WARNING: This will permanently delete {len(orphan_annotations)} orphan annotation(s)!")
                confirm = input("Type 'YES' to confirm: ").strip()
                if confirm == 'YES':
                    print("\nDeleting orphan annotations...")
                    ann_deleted = delete_files(orphan_annotations, "annotations")
                    print(f"\n✓ Deletion complete: {ann_deleted} annotation(s) deleted")
                else:
                    print("Deletion cancelled.")
            break
            
        elif choice == '4':
            print("Operation cancelled. No files were deleted.")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Scan for orphan image/annotation files in a YOLO dataset folder.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python orphan_file_scanner.py                   # Scans 'Pooled' folder
    python orphan_file_scanner.py Train             # Scans 'Train' folder
    python orphan_file_scanner.py /path/to/folder   # Scans specified absolute path
        """
    )
    parser.add_argument(
        'target_folder',
        nargs='?',
        default='Pooled',
        help='Target folder to scan (default: Pooled)'
    )
    
    args = parser.parse_args()
    
    # Determine the target folder path
    script_dir = Path(__file__).parent
    target_folder = Path(args.target_folder)
    
    # If relative path, make it relative to script directory
    if not target_folder.is_absolute():
        target_folder = script_dir / target_folder
    
    # Validate folder exists
    if not target_folder.exists():
        print(f"❌ Error: Folder not found: {target_folder}")
        sys.exit(1)
    
    if not target_folder.is_dir():
        print(f"❌ Error: Path is not a directory: {target_folder}")
        sys.exit(1)
    
    print(f"\n🔍 Scanning folder: {target_folder}")
    print(f"   Looking for image extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}")
    print(f"   Annotation extension: {ANNOTATION_EXTENSION}")
    
    # Find orphan files
    orphan_images, orphan_annotations = find_orphan_files(target_folder)
    
    # Print report
    print_report(orphan_images, orphan_annotations)
    
    # Prompt for deletion
    prompt_deletion(orphan_images, orphan_annotations)


if __name__ == '__main__':
    main()
