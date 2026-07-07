#!/usr/bin/env python3
"""
Extract Hard Examples from YOLO Validation Predictions

This script analyzes validation predictions and extracts "hard examples" -
images where the model made specific types of mistakes. These can be:
1. False Negatives: Objects the model missed (predicted as background)
2. Class Confusions: Objects classified as the wrong class

Usage:
    python extract_hard_examples.py --model-dir <path_to_trained_model> --output-dir <output_path>

Example:
    python extract_hard_examples.py \
        --model-dir trained_models_output/server/04_class_balancing_combination_continue_4_GPT5.2_phase_2/Dataset_2_..._cont1 \
        --output-dir hard_examples_gallery \
        --confusion-pairs "mature_pollen:midlate_pollen,young_pollen:late_microspore"

The script will create galleries of:
- False negatives per class (class → background)
- Confusion pairs (class A → class B)

These galleries can be reviewed, cleaned, and added back to the training set.

Author: GitHub Copilot
Date: 2026-02-02
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_confusion_pairs(pairs_str: str) -> List[Tuple[str, str]]:
    """Parse confusion pairs from command line argument."""
    if not pairs_str:
        return []
    pairs = []
    for pair in pairs_str.split(","):
        if ":" in pair:
            true_cls, pred_cls = pair.split(":")
            pairs.append((true_cls.strip(), pred_cls.strip()))
    return pairs


def load_class_names(model_dir: Path) -> List[str]:
    """Load class names from the model's args.yaml or data.yaml."""
    args_yaml = model_dir / "args.yaml"
    if args_yaml.exists():
        import yaml
        with open(args_yaml, "r") as f:
            args = yaml.safe_load(f)
            if "names" in args:
                if isinstance(args["names"], dict):
                    return [args["names"][i] for i in sorted(args["names"].keys())]
                return args["names"]
            # Try to find data.yaml path
            if "data" in args:
                data_yaml_path = Path(args["data"])
                if data_yaml_path.exists():
                    with open(data_yaml_path, "r") as df:
                        data = yaml.safe_load(df)
                        if "names" in data:
                            if isinstance(data["names"], dict):
                                return [data["names"][i] for i in sorted(data["names"].keys())]
                            return data["names"]
    
    # Fallback: common microspore classes
    return [
        "tetrad", "young_microspore", "mid_microspore", "late_microspore",
        "young_pollen", "midlate_pollen", "mature_pollen", "others", "Blank"
    ]


def get_validation_images(model_dir: Path) -> List[Path]:
    """Get list of validation images from the model's data.yaml."""
    args_yaml = model_dir / "args.yaml"
    if args_yaml.exists():
        import yaml
        with open(args_yaml, "r") as f:
            args = yaml.safe_load(f)
            if "data" in args:
                data_yaml_path = Path(args["data"])
                if data_yaml_path.exists():
                    with open(data_yaml_path, "r") as df:
                        data = yaml.safe_load(df)
                        val_path = data.get("val")
                        if val_path:
                            val_dir = Path(val_path)
                            if not val_dir.is_absolute():
                                val_dir = data_yaml_path.parent / val_path
                            if val_dir.exists():
                                return list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    return []


def load_ground_truth(image_path: Path, class_names: List[str]) -> List[Dict]:
    """Load ground truth annotations for an image."""
    # Try to find label file
    label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    if not label_path.exists():
        label_path = image_path.with_suffix(".txt")
    
    if not label_path.exists():
        return []
    
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                annotations.append({
                    "class_id": cls_id,
                    "class_name": class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}",
                    "bbox_norm": [x_center, y_center, width, height]
                })
    return annotations


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes in [x_center, y_center, w, h] format."""
    # Convert to [x1, y1, x2, y2]
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2
    
    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def find_hard_examples(
    model: YOLO,
    val_images: List[Path],
    class_names: List[str],
    confusion_pairs: List[Tuple[str, str]],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25
) -> Dict[str, List[Dict]]:
    """
    Run inference and find hard examples.
    
    Returns:
        Dictionary with keys:
        - "false_negatives_<class>": List of images with missed objects
        - "confusion_<true>_to_<pred>": List of images with wrong predictions
    """
    hard_examples = {}
    
    # Initialize categories
    for cls_name in class_names:
        hard_examples[f"false_negatives_{cls_name}"] = []
    for true_cls, pred_cls in confusion_pairs:
        hard_examples[f"confusion_{true_cls}_to_{pred_cls}"] = []
    
    print(f"Processing {len(val_images)} validation images...")
    
    for i, img_path in enumerate(val_images):
        if i % 100 == 0:
            print(f"  Processing image {i+1}/{len(val_images)}...")
        
        # Load ground truth
        gt_annotations = load_ground_truth(img_path, class_names)
        if not gt_annotations:
            continue
        
        # Run inference
        results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            # All ground truth objects are false negatives
            for gt in gt_annotations:
                hard_examples[f"false_negatives_{gt['class_name']}"].append({
                    "image_path": str(img_path),
                    "gt_class": gt["class_name"],
                    "gt_bbox": gt["bbox_norm"],
                    "pred_class": None,
                    "pred_conf": 0.0,
                    "type": "false_negative"
                })
            continue
        
        # Get predictions
        boxes = results[0].boxes
        pred_bboxes = boxes.xywhn.cpu().numpy() if len(boxes) > 0 else []
        pred_classes = boxes.cls.cpu().numpy().astype(int) if len(boxes) > 0 else []
        pred_confs = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
        
        # Match GT to predictions
        gt_matched = [False] * len(gt_annotations)
        
        for gt_idx, gt in enumerate(gt_annotations):
            best_iou = 0.0
            best_pred_idx = -1
            
            for pred_idx, pred_bbox in enumerate(pred_bboxes):
                iou = calculate_iou(gt["bbox_norm"], pred_bbox.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            if best_iou >= iou_threshold and best_pred_idx >= 0:
                gt_matched[gt_idx] = True
                pred_cls_id = pred_classes[best_pred_idx]
                pred_cls_name = class_names[pred_cls_id] if pred_cls_id < len(class_names) else f"class_{pred_cls_id}"
                
                # Check for class confusion
                if gt["class_name"] != pred_cls_name:
                    for true_cls, expected_pred in confusion_pairs:
                        if gt["class_name"] == true_cls and pred_cls_name == expected_pred:
                            hard_examples[f"confusion_{true_cls}_to_{expected_pred}"].append({
                                "image_path": str(img_path),
                                "gt_class": gt["class_name"],
                                "gt_bbox": gt["bbox_norm"],
                                "pred_class": pred_cls_name,
                                "pred_conf": float(pred_confs[best_pred_idx]),
                                "type": "confusion"
                            })
            else:
                # False negative
                hard_examples[f"false_negatives_{gt['class_name']}"].append({
                    "image_path": str(img_path),
                    "gt_class": gt["class_name"],
                    "gt_bbox": gt["bbox_norm"],
                    "pred_class": None,
                    "pred_conf": 0.0,
                    "type": "false_negative"
                })
    
    return hard_examples


def save_gallery(
    hard_examples: Dict[str, List[Dict]],
    output_dir: Path,
    max_per_category: int = 200
):
    """Save hard example galleries as image crops and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    
    for category, examples in hard_examples.items():
        if not examples:
            continue
        
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Sort by confidence (ascending for false negatives, descending for confusions)
        if "false_negative" in category:
            examples_sorted = sorted(examples, key=lambda x: x.get("pred_conf", 0))
        else:
            examples_sorted = sorted(examples, key=lambda x: x.get("pred_conf", 0), reverse=True)
        
        # Limit per category
        examples_to_save = examples_sorted[:max_per_category]
        
        saved_count = 0
        for idx, ex in enumerate(examples_to_save):
            try:
                img = cv2.imread(ex["image_path"])
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                bbox = ex["gt_bbox"]  # [x_center, y_center, width, height] normalized
                
                # Convert to pixel coordinates
                x_center = int(bbox[0] * w)
                y_center = int(bbox[1] * h)
                box_w = int(bbox[2] * w)
                box_h = int(bbox[3] * h)
                
                # Add padding (20% on each side)
                pad_x = int(box_w * 0.2)
                pad_y = int(box_h * 0.2)
                
                x1 = max(0, x_center - box_w // 2 - pad_x)
                y1 = max(0, y_center - box_h // 2 - pad_y)
                x2 = min(w, x_center + box_w // 2 + pad_x)
                y2 = min(h, y_center + box_h // 2 + pad_y)
                
                crop = img[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Save crop
                filename = f"{idx:04d}_{Path(ex['image_path']).stem}.jpg"
                cv2.imwrite(str(category_dir / filename), crop)
                saved_count += 1
                
            except Exception as e:
                print(f"  Error processing {ex['image_path']}: {e}")
                continue
        
        summary[category] = {
            "total_found": len(examples),
            "saved": saved_count,
            "examples": examples_to_save
        }
        
        print(f"  {category}: {len(examples)} found, {saved_count} saved")
    
    # Save summary JSON
    with open(output_dir / "hard_examples_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGallery saved to: {output_dir}")
    print(f"Summary saved to: {output_dir / 'hard_examples_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Extract hard examples from YOLO validation predictions")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to trained model directory (contains weights/, args.yaml)")
    parser.add_argument("--output-dir", type=str, default="hard_examples_gallery",
                        help="Output directory for hard example galleries")
    parser.add_argument("--confusion-pairs", type=str, 
                        default="mature_pollen:midlate_pollen,young_pollen:late_microspore,mid_microspore:late_microspore,mid_microspore:young_microspore",
                        help="Comma-separated confusion pairs to track (format: true:pred)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for matching GT to predictions")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="Confidence threshold for predictions")
    parser.add_argument("--max-per-category", type=int, default=200,
                        help="Maximum examples to save per category")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    # Find best.pt
    weights_dir = model_dir / "weights"
    best_pt = None
    for name in ["best.pt", "*_best.pt"]:
        matches = list(weights_dir.glob(name))
        if matches:
            best_pt = matches[0]
            break
    
    if not best_pt:
        print(f"Error: Could not find best.pt in {weights_dir}")
        return
    
    print(f"Loading model from: {best_pt}")
    model = YOLO(str(best_pt))
    
    # Load class names
    class_names = load_class_names(model_dir)
    print(f"Class names: {class_names}")
    
    # Get validation images
    val_images = get_validation_images(model_dir)
    if not val_images:
        print("Error: Could not find validation images")
        return
    print(f"Found {len(val_images)} validation images")
    
    # Parse confusion pairs
    confusion_pairs = parse_confusion_pairs(args.confusion_pairs)
    print(f"Tracking confusion pairs: {confusion_pairs}")
    
    # Find hard examples
    hard_examples = find_hard_examples(
        model=model,
        val_images=val_images,
        class_names=class_names,
        confusion_pairs=confusion_pairs,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold
    )
    
    # Print summary
    print("\n=== Hard Examples Summary ===")
    for category, examples in hard_examples.items():
        if examples:
            print(f"  {category}: {len(examples)}")
    
    # Save galleries
    print("\nSaving galleries...")
    save_gallery(hard_examples, output_dir, args.max_per_category)


if __name__ == "__main__":
    main()
