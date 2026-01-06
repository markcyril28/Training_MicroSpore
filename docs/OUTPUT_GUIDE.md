# Training Output Interpretation Guide

This guide explains how to interpret the results, statistics, and graphs generated after training a YOLO model for microspore phenotyping.

---

## Output Folder Structure

After training, your output folder will contain organized subfolders:

```
trained_models_output/
└── <experiment_name>/
    ├── args.yaml                     # YOLO training arguments (root level)
    │
    ├── weights/                      # Model weights & configs
    │   ├── <exp_name>_best.pt        # Best performing model (dynamically named)
    │   ├── <exp_name>_last.pt        # Final epoch model (dynamically named)
    │   ├── <exp_name>_best.onnx      # ONNX export (for deployment)
    │   └── configs/                  # Training configuration files
    │       ├── <exp_name>_args.yaml  # Copy of args.yaml
    │       └── <exp_name>_args.cfg   # Human-readable config
    │
    ├── stats/                        # Training statistics
    │   ├── epoch_metrics.csv         # Per-epoch: loss, mAP, precision, recall, lr
    │   ├── training_summary.json     # Final stats: best_epoch, total_time, best_mAP
    │   ├── model_comparison.csv      # Columns for comparing multiple runs
    │   └── hardware_stats.csv        # GPU/CPU usage per epoch
    │
    ├── visualizations/               # All visualization outputs
    │   ├── curves/                   # Training curves
    │   │   ├── loss_curves.png       # Box/cls/dfl losses (train vs val)
    │   │   ├── precision_recall.png  # mAP, precision, recall vs epoch
    │   │   ├── F1_curve.png          # F1 score curve
    │   │   ├── P_curve.png           # Precision curve
    │   │   ├── R_curve.png           # Recall curve
    │   │   └── PR_curve.png          # Precision-Recall curve
    │   ├── matrices/                 # Confusion matrices
    │   │   ├── confusion_matrix.png
    │   │   ├── confusion_matrix_normalized.png
    │   │   └── labels_correlogram.png
    │   ├── samples/                  # Training/validation samples
    │   │   ├── train_batch_samples.png
    │   │   ├── val_batch_labels.png  # Ground truth
    │   │   └── val_batch_predictions.png
    │   └── overviews/                # Summary visualizations
    │       ├── results.png           # Combined training curves
    │       ├── labels.png            # Dataset labels overview
    │       └── Box_curve.png         # Box plot curve
    │
    ├── GENERAL_GUIDE_and_ANALYSIS_<timestamp>.md  # Interpretation guide with accuracy
    └── GENERAL_RESULTS_<timestamp>.md             # Concise training results
```

---

## Key Metrics Explained

### 1. **mAP (Mean Average Precision)**

The most important metrics for evaluating detection performance:

| Metric | Description | Good Value |
|--------|-------------|------------|
| **mAP50** | Mean Average Precision at IoU threshold 0.50 | > 0.70 (70%) |
| **mAP50-95** | Mean AP averaged across IoU thresholds 0.50 to 0.95 | > 0.50 (50%) |

- **mAP50**: More lenient, considers a detection correct if 50% overlap with ground truth
- **mAP50-95**: Stricter, averaged across multiple overlap thresholds (more robust metric)

### 2. **Precision & Recall**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Precision** | Of all detections made, how many were correct? | High = Few false positives |
| **Recall** | Of all ground truth objects, how many were detected? | High = Few missed objects |

**Trade-off**: Higher precision often means lower recall and vice versa.

### 3. **Loss Functions**

| Loss | Description | Expected Behavior |
|------|-------------|-------------------|
| **box_loss** | Bounding box regression loss | Should decrease |
| **cls_loss** | Classification loss | Should decrease |
| **dfl_loss** | Distribution focal loss (box quality) | Should decrease |

---

## Graph Interpretation

### results.png - Training Curves

This combined plot shows training progression across epochs:

![Training Curves Overview](https://via.placeholder.com/600x400?text=Results+Graph)

**What to look for:**
- **Losses decreasing**: All loss curves should trend downward
- **mAP increasing**: Detection accuracy should improve over epochs
- **No divergence**: Training and validation metrics should be similar
- **Smooth curves**: Jagged lines may indicate unstable training

### Confusion Matrix (confusion_matrix.png)

Shows how well the model classifies each microspore stage:

```
                 Predicted
              ┌─────────────────────────────────────┐
              │ tet  ym   mm   lm   yp   mp   mat  oth│
         tet  │ 85   5    2    0    0    0    0    8  │
         ym   │ 3    92   3    1    0    0    0    1  │
Actual   mm   │ 1    4    88   5    1    0    0    1  │
         lm   │ 0    1    6    85   5    2    0    1  │
         yp   │ 0    0    2    8    82   5    2    1  │
         mp   │ 0    0    0    3    7    83   5    2  │
         mat  │ 0    0    0    0    2    8    88   2  │
         oth  │ 5    2    1    1    1    1    2    87 │
              └─────────────────────────────────────┘
```

**Interpretation:**
- **Diagonal values**: Correct classifications (should be high)
- **Off-diagonal values**: Misclassifications (should be low)
- **Normalized version**: Shows percentages instead of counts

**Common patterns:**
- Adjacent stages often confused (e.g., young_microspore ↔ mid_microspore)
- "others" class may have more confusion if heterogeneous

### Precision-Recall Curves (BoxPR_curve.png)

Shows the trade-off between precision and recall:

- **Curve closer to top-right corner** = Better performance
- **Area Under Curve (AUC)** = mAP value
- **Per-class curves** show which classes perform best/worst

### BoxF1_curve.png

Shows F1 score (harmonic mean of precision and recall) vs confidence threshold:
- **Peak of curve**: Optimal confidence threshold
- **Higher peak**: Better overall performance

### BoxP_curve.png & BoxR_curve.png

Individual precision and recall curves at different confidence thresholds.

---

## Stats Files Explained

All statistics files are located in the `stats/` folder.

### stats/training_summary.json

Contains final training stats and configuration:

```json
{
  "experiment_name": "Dataset_2_OPTIMIZATION_yolov5nu_e5_b8_img640...",
  "generated_at": "2026-01-05T01:19:03.123456",
  "config": {
    "model": "yolov5nu.pt",       // Model architecture
    "epochs": 5,                   // Training epochs
    "batch_size": 8,               // Batch size used
    "img_size": 640,               // Image resolution
    "optimizer": "auto",           // Optimizer used
    "classes": ["tetrad", "young_microspore", ...]
  },
  "best_epoch": 3,
  "total_epochs": 5,
  "best_mAP50": 0.4401,
  "best_mAP50_95": 0.2856,
  "final_precision": 0.6234,
  "final_recall": 0.5123
}
```

### stats/epoch_metrics.csv

Per-epoch training metrics (copy of YOLO's results.csv):

| Column | Description |
|--------|-------------|
| epoch | Training epoch number |
| train/box_loss | Training bounding box loss |
| train/cls_loss | Training classification loss |
| train/dfl_loss | Training distribution focal loss |
| metrics/precision(B) | Validation precision |
| metrics/recall(B) | Validation recall |
| metrics/mAP50(B) | Validation mAP at IoU=0.50 |
| metrics/mAP50-95(B) | Validation mAP at IoU=0.50:0.95 |
| val/box_loss | Validation bounding box loss |
| val/cls_loss | Validation classification loss |

### stats/model_comparison.csv

Single-row CSV for comparing multiple training runs:

| Column | Description |
|--------|-------------|
| experiment | Experiment folder name |
| model | YOLO model used |
| epochs | Number of training epochs |
| img_size | Image resolution |
| mAP50 | Final mAP@50 value |
| mAP50_95 | Final mAP@50-95 value |
| precision | Final precision |
| recall | Final recall |
| f1 | Calculated F1 score |

### stats/hardware_stats.csv

Hardware utilization metrics (when available from monitoring logs).

---

## Visualization Files

All visualization files are organized in the `visualizations/` folder with subfolders.

### visualizations/samples/

**train_batch_samples.png**
Sample training images showing:
- Input images with augmentations applied
- Ground truth bounding boxes and labels
- Useful for verifying data loading and augmentation

**val_batch_labels.png**
Validation images with **ground truth** annotations:
- Shows what the model should detect
- Verify labels are correct

**val_batch_predictions.png**
Validation images with **model predictions**:
- Shows what the model actually detected
- Compare with labels to see detection quality
- Boxes include confidence scores

### visualizations/curves/

**loss_curves.png**
Custom generated plot showing:
- Train vs validation box loss
- Train vs validation classification loss  
- Train vs validation DFL loss

**precision_recall.png**
Custom generated plot showing mAP, precision, and recall over epochs.

**F1_curve.png, P_curve.png, R_curve.png, PR_curve.png**
Standard YOLO metric curves at different confidence thresholds.

### visualizations/matrices/

**confusion_matrix.png & confusion_matrix_normalized.png**
Classification performance per class.

**labels_correlogram.png**
Correlation between label dimensions and positions.

### visualizations/overviews/

**results.png**
Combined training curves visualization from YOLO.

**labels.png**
Overview of dataset labels showing:
- Distribution of bounding box positions
- Distribution of bounding box sizes
- Class frequency histogram

---

## Logs Folder (logs/)

Additional monitoring logs are stored in:

```
logs/<experiment_name>_<timestamp>/
├── training_summary.txt          # Human-readable summary
├── gpu_logs/                     # GPU utilization over time
│   └── gpu_<timestamp>.csv       # GPU metrics CSV
├── system_resources_log/         # CPU, RAM usage
│   └── system_<timestamp>.csv    # System metrics CSV
├── training_metrics/             # Epoch metrics
│   └── metrics.csv               # Training progress metrics
├── full_logs/                    # Complete training logs
├── errors_logs/                  # Any errors encountered
└── visualization_logs/           # Additional visualizations
```

### training_summary.txt

Quick overview of training run:
```
==============================================
  Training Log Summary
  Generated: 2026-01-05 01:19:03
==============================================

GPU Statistics:
  Log file: /path/to/gpu_logs/gpu_<timestamp>.csv
  Data points: 39
  Avg GPU Utilization: 18.1%
  Peak VRAM Used: 2470 MB
  Avg Temperature: 59.6°C

System Statistics:
  Log file: /path/to/system_resources_log/system_<timestamp>.csv
  Data points: 18
  Avg CPU Usage: 30.5%
  Peak RAM Used: 9736 MB

Error Summary:
  Errors: 0
  Warnings: 0
  OOM Events: 0

Training Metrics:
  Epochs logged: 5
  Best mAP50: 0.4401

Log Directory: /path/to/logs/<experiment_name>_<timestamp>
==============================================
```

---

## What Makes a Good Model?

### Performance Benchmarks

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| mAP50 | <0.50 | 0.50-0.70 | 0.70-0.85 | >0.85 |
| mAP50-95 | <0.30 | 0.30-0.50 | 0.50-0.65 | >0.65 |
| Precision | <0.50 | 0.50-0.70 | 0.70-0.85 | >0.85 |
| Recall | <0.50 | 0.50-0.70 | 0.70-0.85 | >0.85 |

### Signs of Good Training

✅ **Healthy training:**
- Losses decrease steadily
- mAP increases over epochs
- Training and validation losses are similar
- Precision and recall are balanced

### Signs of Problems

⚠️ **Overfitting:**
- Training loss decreases but validation loss increases
- Training mAP much higher than validation mAP
- **Solution**: Add regularization, reduce epochs, increase augmentation

⚠️ **Underfitting:**
- Both losses remain high
- Low mAP values
- **Solution**: Train longer, use larger model, reduce regularization

⚠️ **Class Imbalance:**
- Some classes have much lower mAP
- Confusion matrix shows bias toward common classes
- **Solution**: Use class weights, oversample minority classes

⚠️ **Unstable Training:**
- Losses oscillate wildly
- Metrics fluctuate significantly
- **Solution**: Reduce learning rate, increase batch size

---

## Model Files for Deployment

### weights/<exp_name>_best.pt
- Best performing model checkpoint (highest mAP50)
- Dynamically named with experiment folder name (e.g., `Dataset_2_OPTIMIZATION_yolov5nu_e5_b8_img640_lr0_001_auto_rgb_20260105_011520_best.pt`)
- Use this for inference in most cases
- PyTorch format (.pt)

### weights/<exp_name>_last.pt
- Final epoch checkpoint (dynamically named)
- Use if training was interrupted or for continued training

### weights/<exp_name>_best.onnx
- ONNX format for cross-platform deployment
- Compatible with ONNX Runtime, TensorRT, OpenVINO
- Optimized for production inference
- Named consistently with the best.pt file

### weights/configs/
- `<exp_name>_args.yaml` - Copy of YOLO training arguments
- `<exp_name>_args.cfg` - Human-readable configuration file with training parameters

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    METRICS QUICK REFERENCE                  │
├─────────────────────────────────────────────────────────────┤
│  mAP50      = How well model detects (50% overlap)          │
│  mAP50-95   = Stricter detection accuracy (main metric)     │
│  Precision  = Detection correctness (low false positives)   │
│  Recall     = Detection completeness (low missed objects)   │
│  box_loss   = Bounding box accuracy (lower = better)        │
│  cls_loss   = Classification accuracy (lower = better)      │
├─────────────────────────────────────────────────────────────┤
│                      FILE QUICK REFERENCE                   │
├─────────────────────────────────────────────────────────────┤
│  weights/<exp>_best.pt       → Best model for inference     │
│  weights/<exp>_best.onnx     → ONNX export for deployment   │
│  stats/epoch_metrics.csv     → Detailed epoch metrics       │
│  stats/training_summary.json → Complete experiment summary  │
│  visualizations/curves/      → Loss and metric curves       │
│  visualizations/matrices/    → Confusion matrices           │
│  GENERAL_RESULTS_*.md        → Quick results summary        │
└─────────────────────────────────────────────────────────────┘
```

---

## Tips for Analysis

1. **Compare experiments**: Use `stats/model_comparison.csv` to compare different training runs
2. **Check confusion matrix**: Review `visualizations/matrices/confusion_matrix.png` to identify which microspore stages are hardest to classify
3. **Validate predictions**: Review `visualizations/samples/val_batch_predictions.png` to visually verify detections
4. **Monitor resources**: Check `logs/` folder for GPU/CPU usage and optimization opportunities
5. **Track over time**: Keep `stats/training_summary.json` reports to track model improvements
6. **Quick results**: Check `GENERAL_RESULTS_*.md` for a concise accuracy summary

---

*Generated for Microspore Phenotyping Training Pipeline*
