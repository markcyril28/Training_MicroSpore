#!/usr/bin/env python3
"""
Generate Dataset Statistics Script
CLI script to generate dataset statistics, called from bash setup scripts.
Implements DRY principle - extracted from embedded Python in bash scripts.

Usage:
    python -m modules.scripts.generate_dataset_stats --dataset-path /path/to/dataset
    python modules/scripts/generate_dataset_stats.py --dataset-path /path/to/dataset
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for module imports when run directly
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    modules_dir = script_dir.parent
    project_dir = modules_dir.parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))


def generate_dataset_stats(dataset_path: str, save_stats: bool = True) -> dict:
    """
    Generate and optionally save dataset statistics.
    
    Args:
        dataset_path: Path to the dataset directory
        save_stats: Whether to save stats to JSON file
        
    Returns:
        Dictionary containing dataset statistics
    """
    from modules.training.stats import DatasetStats
    from modules.utils.utils import save_json
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"  Dataset path does not exist: {dataset_path}")
        return {}
    
    try:
        dataset = DatasetStats(dataset_path)
        stats = dataset.get_full_stats()
        
        # Print summary
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Classes: {len(stats['classes'])}")
        
        for split, split_stats in stats.get('splits', {}).items():
            print(f"  {split.capitalize()}: {split_stats['images']} images, {split_stats['total_annotations']} annotations")
        
        # Save stats if requested
        if save_stats:
            output_file = dataset_path / "dataset_stats.json"
            save_json(stats, output_file)
            print(f"\n  Stats saved to: {output_file}")
        
        return stats
        
    except Exception as e:
        print(f"  Could not generate stats: {e}")
        return {}


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics for YOLO training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m modules.scripts.generate_dataset_stats --dataset-path ./TRAINING_WD/Dataset_1_TEST
    python -m modules.scripts.generate_dataset_stats -d ./TRAINING_WD/Dataset_2_OPTIMIZATION --no-save
        """
    )
    
    parser.add_argument(
        "-d", "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not save stats to JSON file (only print)"
    )
    
    args = parser.parse_args()
    
    # Generate stats
    stats = generate_dataset_stats(
        dataset_path=args.dataset_path,
        save_stats=not args.no_save
    )
    
    # Return exit code based on success
    sys.exit(0 if stats else 1)


if __name__ == "__main__":
    main()
