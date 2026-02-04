#!/bin/bash

# SERVER SPECS (Dell Server with AMD ROCm):
#   GPU: AMD Instinct MI210 (Aldebaran/MI200)
#   VRAM: 64GB HBM2e
#   Architecture: gfx90a (CDNA2)
#   Driver: amdgpu
#   Compute Platform: ROCm 6.x
#   CPU Threads: 72

#===============================================================================
# YOLO VERSION SELECTION
#===============================================================================
# Model arrays are defined in common_functions.sh
# YOLOV5_MODELS, YOLOV8_MODELS, YOLOV9_MODELS, YOLOV10_MODELS, YOLO11_MODELS

# All available models for selection
# Uncomment ONE or more models to train with. All models below are pre-downloaded.
# With 64GB VRAM, you can comfortably train the largest models!
YOLO_MODELS=(
    # YOLOv5 variants - Ultralytics versions with 'u' suffix
    # "yolov5nu.pt"   # nano     - fastest, lowest accuracy
    # "yolov5su.pt"   # small    - fast, good accuracy
    # "yolov5mu.pt"   # medium   - balanced
    # "yolov5lu.pt"   # large    - slower, better accuracy
    #"yolov5xu.pt"     # xlarge   - slowest, best accuracy (OPTIMAL for MI210)
    
    # YOLOv8 variants - Recommended
    # "yolov8n.pt"    # nano     - fastest, lowest accuracy
    # "yolov8s.pt"    # small    - fast, good accuracy
    # "yolov8m.pt"    # medium   - balanced
    # "yolov8l.pt"    # large    - slower, better accuracy
    "yolov8x.pt"      # xlarge   - slowest, best accuracy (OPTIMAL for MI210)
    
    # YOLOv9 variants - GELAN/PGI architecture
    # "yolov9t.pt"    # tiny     - fastest, smallest
    # "yolov9s.pt"    # small    - fast, lightweight
    # "yolov9m.pt"    # medium   - balanced
    # "yolov9c.pt"    # compact  - efficient accuracy
    #"yolov9e.pt"      # extended - best accuracy (OPTIMAL for MI210)
    
    # YOLOv10 variants
    # "yolov10n.pt"   # nano     - fastest
    # "yolov10s.pt"   # small    - fast
    # "yolov10m.pt"   # medium   - balanced
    # "yolov10l.pt"   # large    - slower
    #"yolov10x.pt"     # xlarge   - best accuracy (OPTIMAL for MI210)
    
    # YOLO11 variants - Latest (note: named 'yolo11' not 'yolov11')
    # "yolo11n.pt"    # nano     - fastest
    # "yolo11s.pt"    # small    - fast
    # "yolo11m.pt"    # medium   - balanced
    # "yolo11l.pt"    # large    - slower
    #"yolo11x.pt"      # xlarge   - best accuracy (OPTIMAL for MI210 64GB)
)

# Select first model from the array (for quick reference)
YOLO_MODEL="${YOLO_MODELS[0]}"

#===============================================================================
# CONTINUE TRAINING FROM CUSTOM WEIGHTS (Fine-tuning)
#===============================================================================
# Use this to continue training from a previously trained model instead of
# starting from pretrained ImageNet/COCO weights.
#
# CONTINUE_FROM_CUSTOM:  true = use CUSTOM_WEIGHTS_PATH, false = use YOLO_MODELS
# AUTO_DETECT_LATEST:    true = auto-detect latest trained model (ignores CUSTOM_WEIGHTS_PATH)
# CUSTOM_WEIGHTS_PATH:   Full path to your trained .pt file (best.pt or last.pt)
#                        Only used when AUTO_DETECT_LATEST=false
#
# Example workflow:
#   1. Train initial model (500 epochs) → get best.pt at 71.4% mAP
#   2. Set CONTINUE_FROM_CUSTOM=true and AUTO_DETECT_LATEST=true
#   3. Script will find the latest trained model automatically
#   4. Train for additional epochs with lower LR for fine-tuning
#
# Tips for continued training:
#   - Use lower LR: 0.0001 or 0.00001 (10-100x lower than initial)
#   - Train for 100-500 additional epochs
#   - Consider reducing patience to 50-100
#   - Keep same IMG_SIZE as original training (1280)

CONTINUE_FROM_CUSTOM=true       # Set to true to continue from custom weights
AUTO_DETECT_LATEST=false        # Disabled - using specific best.pt from cont1
PREFER_LAST_PT=false            # false = use best.pt (best mAP checkpoint for fine-tuning)

# Full path to your trained model weights (ONLY used when AUTO_DETECT_LATEST=false)
# Using best.pt from cont1 (continue_3)
CUSTOM_WEIGHTS_PATH="/mnt/local3.5tb/home/mcmercado/Training_MicroSpore/trained_models_output/server/04_class_balancing_combination_continue_3/Dataset_2_OPTIMIZATION_Dataset_2_OPTIMIZATION_best_gray_img1280_bal-manual_auto_e300_b8_lr0_00001_20260201_101533_cont1_gray_img1280_bal-manual_auto_e400_b8_lr0_00005_20260201_170927_cont1/weights/Dataset_2_OPTIMIZATION_Dataset_2_OPTIMIZATION_best_gray_img1280_bal-manual_auto_e300_b8_lr0_00001_20260201_101533_cont1_gray_img1280_bal-manual_auto_e400_b8_lr0_00005_20260201_170927_cont1_best.pt"

# Directory to search for latest model (used when AUTO_DETECT_LATEST=true)
# This should point to the output directory where trained models are saved
AUTO_DETECT_SEARCH_DIR="/mnt/local3.5tb/home/mcmercado/Training_MicroSpore/trained_models_output/server/04_class_balancing_combination_continue_3"

# Optional filters for auto-detection (leave empty to find any latest model)
AUTO_DETECT_DATASET_FILTER=""      # e.g., "Dataset_2" to only find Dataset_2 models
AUTO_DETECT_MODEL_FILTER=""        # e.g., "yolov8x" to only find yolov8x models

# Additional epochs to train (added to model's current epoch count)
# When continuing, these epochs are ADDED to the previous training
# Previous run: best at epoch 2/31, mAP50=71.3% - needs more training with focus on confused classes
# Confusion matrix analysis shows: mature_pollen(37.9%), young_pollen(38.4%), late_microspore(47.6%), mid_microspore(49.1%)
ADDITIONAL_EPOCHS_LIST=(
    # 500                   # +500 epochs (extended fine-tuning)
    # 300                   # +300 epochs (previous run)
    200                     # +200 epochs - stabilizing training
    # 400                   # +400 epochs - previous
)

# Dataset Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Parent directory is defined in modules/common_functions.sh (synced with config.py)
# COMMON_DATASETS_DIR="DATASETS"

# List of dataset names to train on (located inside COMMON_DATASETS_DIR)
# Each dataset should contain a data.yaml file and Train/Test subdirectories
# Add multiple datasets to train models on each sequentially
DATASET_LIST=(
    # "Dataset_1_TEST"            # Test dataset
    "Dataset_2_OPTIMIZATION"      # Optimization dataset (active)
    # "Dataset_3_FINAL_RUN"       # Final run dataset (uncomment to add)
)

# Default dataset name (first in list)
DEFAULT_DATASET="${DATASET_LIST[0]}"

#===============================================================================
# TRAINING PARAMETERS (Array-based for grid search)
#===============================================================================
# GUIDE: ↑ = increase value, ↓ = decrease value
#   ↑ generally improves accuracy but costs more time/memory
#   ↓ generally speeds up training but may reduce accuracy
#
# NOTE: Parameters are now ARRAYS - uncomment multiple values to train all combinations
# Total combinations = EPOCHS × BATCH_SIZES × IMG_SIZES × LR0S × OPTIMIZERS × ...
#
# MI210 ADVANTAGE: With 64GB HBM2e, you can use much larger batch sizes and
# higher resolution images than consumer GPUs!

# Core Training Parameters
# ─────────────────────────────────────────────────────────────────────────────
# EPOCHS:     ↑ better convergence, risk of overfitting | ↓ faster, underfitting
# BATCH_SIZE: ↑ stable gradients, needs more VRAM       | ↓ noisy gradients, less VRAM
# IMG_SIZE:   ↑ detects small objects better, slower    | ↓ faster, may miss details
# PATIENCE:   ↑ waits longer before stopping            | ↓ stops earlier, saves time
# WORKERS:    ↑ faster data loading (match CPU cores)   | ↓ less CPU usage
# NOTE: When CONTINUE_FROM_CUSTOM=true, use ADDITIONAL_EPOCHS_LIST instead
# This EPOCHS_LIST is used only when training from scratch
EPOCHS_LIST=(
    # 10                    # quick test
    # 100                   # standard training
    # 150                   # optimal training (early stopping will trigger if converged)
    #200                     # long training (MI210 can handle extended training)
    500                     # standard training (ignored when continuing)
)

PATIENCE_LIST=(
    # 100                   # reduced patience for fine-tuning
    # 50                    # aggressive early stopping
    200                     # moderate patience - allow more exploration with new class focus
    # 150                   # moderate patience
    # 300                   # original patience (may be too long for fine-tuning)
)

BATCH_SIZE_LIST=(
    # 4                    # too low - noisy gradients hurt confused class learning
    8                       # moderate - better gradient stability for confused classes
    # 16                   # moderate
    # 32                   # standard for high-end GPUs
    # 64                   # optimal for MI210 64GB HBM2e (maximum throughput)
    # 128                  # very high batch size (may need gradient accumulation)
)

IMG_SIZE_LIST=(
    # 320                   # fast, low resolution
    # 512                   # medium resolution
    # 608                   # from microspores.cfg (width/height=608)
    #640                   # standard resolution
    # 800                   # high resolution
    #1024                    # very high resolution (optimal for MI210 64GB VRAM)
    1280                  # maximum (for detecting very small objects)
)

WORKERS_LIST=(
    # 2                     # low CPU
    # 4                     # standard
    # 8                     # moderate (balanced for data loading)
    16                      # server with 32 threads (optimal: ~half of available threads)
    #32                    # maximum (use all threads - may cause contention)
)

# Learning Rate & Optimizer
# ─────────────────────────────────────────────────────────────────────────────
# LR0:          ↑ faster learning, may overshoot    | ↓ slower, more stable
# LRF:          ↑ higher final LR, less fine-tuning | ↓ better fine-tuning at end
# MOMENTUM:     ↑ faster convergence, may overshoot | ↓ more stable, slower
# WEIGHT_DECAY: ↑ stronger regularization           | ↓ less regularization, may overfit
# OPTIMIZER:    SGD=stable, Adam/AdamW=faster convergence, auto=recommended
LR0_LIST=(
    # For continued training, use LOWER learning rates (10-100x lower):
    # 0.001                 # original training LR (too high for fine-tuning)
    # 0.0001                # moderate - may overshoot on confused classes
    0.00001                 # ultra fine-tuning - to prevent degradation
    # 0.00005               # previous - too high, caused degradation?
    # With stronger class focus on confused classes, slightly higher LR helps
)

LRF_LIST=(
    #0.1                   # higher final LR
    0.01                    # standard final LR ratio
    #0.001                 # very low final LR
)

MOMENTUM_LIST=(
    # 0.9                   # lower momentum
    # 0.937                 # standard momentum
    0.949                   # from microspores.cfg (momentum=0.949)
)

WEIGHT_DECAY_LIST=(
    0.0005                  # standard weight decay - reduce overfitting on majority classes
    # 0.0001                # low regularization
    # 0.001                 # high regularization
    # 0.01                  # too high - was causing plateau (20x regularization)
)

OPTIMIZER_LIST=(
    "auto"                  # auto-select (recommended)
    #"SGD"                 # Stochastic Gradient Descent
    #"Adam"                # Adam optimizer
    #"AdamW"               # Adam with weight decay
    #"NAdam"               # Nesterov Adam
    #"RAdam"               # Rectified Adam
)

# Grayscale Configuration
# Color Mode Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Select image color mode for training
# 'RGB' = color (3 channels), 'grayscale' = grayscale (converted to 3-channel gray)
COLOR_MODE_LIST=(
    #"RGB"                   # RGB color images (default)
    "grayscale"           # grayscale images
)

# Class Focus Configuration (Address Class Imbalance)
# ─────────────────────────────────────────────────────────────────────────────
# Focus training on specific underrepresented classes by oversampling them.
# Distribution from Dataset_2_OPTIMIZATION (Train counts):
#   midlate_pollen: 2840, young_microspore: 2175, late_microspore: 1562
#   mid_microspore: 1454, others: 1296, Blank: 1204, young_pollen: 1098
#   mature_pollen: 947, tetrad: 853
#
# CLASS_FOCUS_MODE:
#   "none"      - No class focus, use original distribution
#   "manual"    - Manually specify classes and fold multipliers
#   "auto"      - Auto-balance based on distribution.txt (target: equal representation)
#   "sqrt"      - Square root balancing (softer than full equalization)
#
# CLASS_FOCUS_CLASSES: Classes to focus on (used in "manual" mode)
#   Specify class names from: tetrad, young_microspore, mid_microspore,
#   late_microspore, young_pollen, midlate_pollen, mature_pollen, others, Blank
#
# CLASS_FOCUS_FOLD: Oversampling multiplier for focused classes
#   In "manual" mode: Apply this fold to all specified classes
#   In "auto"/"sqrt" mode: Maximum fold to apply (caps the multiplier)
#
# Example: To oversample tetrad (801) to match midlate_pollen (2740), use fold ~3.4

CLASS_FOCUS_MODE_LIST=(
    #"none"                  # No class focus (original distribution)
    #"auto"                # Auto-equalize all classes (recommended for production)
    #"sqrt"                # Square root balancing (gentler, good for mild imbalance)
    "manual"              # Manual class selection with specified fold
)

# Classes to focus on in "manual" mode (comma-separated, no spaces)
# ANALYSIS from confusion_matrix_normalized.csv (diagonal accuracy):
#   - mature_pollen: 37.9% - WORST (confused with midlate_pollen 23.8%, background 36.7%)
#   - young_pollen: 38.4% - VERY POOR (confused with late_microspore 17.1%, background 27.4%)
#   - late_microspore: 47.6% - POOR (confused with mid_microspore 12.3%, young_pollen 6.8%)
#   - mid_microspore: 49.1% - POOR (confused with young_microspore 12.3%, late_microspore 12%)
#   - others: 53.8% - MODERATE (confused with background 33.1%)
#   - tetrad: 91.6% - EXCELLENT (no focus needed)
CLASS_FOCUS_CLASSES_LIST=(
    "mature_pollen,young_pollen,late_microspore,mid_microspore"  # Worst 4 classes by accuracy
    # "young_pollen,mature_pollen,others,late_microspore,mid_microspore"  # All 5 confused classes
    # "mature_pollen,young_pollen"          # Focus on worst 2 only
    # "all"                                 # Apply to all classes (for auto/sqrt modes)
)

# Oversampling fold multiplier
# In "manual": Multiplies the specified classes by this factor
# In "auto"/"sqrt": Maximum fold cap to prevent extreme oversampling
# Targeting worst classes: mature_pollen (947 samples), young_pollen (1098), need ~3-5x to match midlate_pollen (2840)
CLASS_FOCUS_FOLD_LIST=(
    # 2.0                   # 2x oversampling (moderate boost)
    # 3.0                   # 3x oversampling (previous runs)
    # 4.0                   # 4x oversampling (previous config)
    5.0                     # 5x oversampling - aggressive boost for worst classes (37-49% accuracy)
    # 6.0                   # 6x oversampling (may cause overfitting)
)

# Target class for ratio calculation in "auto" mode
# The class that others will be balanced towards
# "max" = balance towards the largest class
# "median" = balance towards median class count
# "mean" = balance towards mean class count
CLASS_FOCUS_TARGET_LIST=(
    "median"                # Balance towards median (recommended)
    # "max"                 # Balance towards largest class
    # "mean"                # Balance towards mean count
)

# Distribution file path (relative to dataset directory)
# This file contains class distribution statistics for dynamic fold calculation
CLASS_DISTRIBUTION_FILE="Distribution/1_class_distribution/distribution.txt"

# Augmentation Parameters
# ─────────────────────────────────────────────────────────────────────────────
# Higher values = more aggressive augmentation = better generalization but slower
# Set to 0.0 to disable specific augmentation
# For microscopy: consider lower HSV, enable rotation if objects have no orientation
HSV_H_LIST=(
    # 0.015                 # standard hue shift
    0.1                     # from microspores.cfg (hue=0.1)
    # 0.0                   # no hue shift
)

HSV_S_LIST=(
    0.7                     # standard saturation
    # 0.0                   # no saturation change
    # 0.4                   # moderate saturation
)

HSV_V_LIST=(
    0.4                     # standard brightness
    # 0.0                   # no brightness change
    # 0.2                   # low brightness variation
)

DEGREES_LIST=(
    # 0.0                   # no rotation (faster augmentation)
    15.0
    #45.0                    # moderate rotation
    # 90.0                  # quarter rotation
    # 180.0                 # half rotation (orientation-invariant)
)

TRANSLATE_LIST=(
    0.1                     # standard translation
    # 0.0                   # no translation
    # 0.2                   # moderate translation
)

SCALE_LIST=(
    # 0.0                   # no scaling
    # 0.3                   # low scale variation
    0.5                     # standard scale variation
    # 0.9                   # high scale variation
)

SHEAR_LIST=(
    0.0                     # no shear
    # 5.0                   # moderate shear
    # 10.0                  # high shear
)

PERSPECTIVE_LIST=(
    0.0                     # no perspective warp (faster augmentation)
    # 0.0005                # slight perspective
    # 0.001                 # moderate perspective
)

FLIPUD_LIST=(
    0.5                     # 50% vertical flip (microscopy)
    # 0.0                   # no vertical flip
)

FLIPLR_LIST=(
    0.5                     # 50% horizontal flip
    # 0.0                   # no horizontal flip
)

MOSAIC_LIST=(
    1.0                     # always mosaic
    # 0.0                   # no mosaic
    # 0.5                   # 50% mosaic
)

MIXUP_LIST=(
    # 0.0                   # no mixup
    # 0.2                   # previous config - boundary confusion persists
    0.3                     # increased mixup - better handling of transition classes
    # 0.5                   # heavy mixup (may blur class boundaries too much)
)

COPY_PASTE_LIST=(
    # 0.0                   # no copy-paste (previous config)
    0.3                     # moderate copy-paste - address background confusion
    # 0.15                  # light copy-paste
)

# Warmup Parameters (equivalent to Darknet's burn_in)
# ─────────────────────────────────────────────────────────────────────────────
# WARMUP_EPOCHS:      Number of warmup epochs (gradual LR increase)
# WARMUP_MOMENTUM:    Initial momentum during warmup
# WARMUP_BIAS_LR:     Initial learning rate for bias during warmup
# From microspores.cfg: burn_in=1000 batches ≈ 3 epochs with batch 64
WARMUP_EPOCHS_LIST=(
    # 3.0                   # standard warmup (from cfg: burn_in=1000)
    # 0.0                   # no warmup
    5.0                     # extended warmup
)

WARMUP_MOMENTUM_LIST=(
    0.8                     # standard warmup momentum
    # 0.5                   # lower warmup momentum
)

WARMUP_BIAS_LR_LIST=(
    0.1                     # standard warmup bias LR
    # 0.01                  # lower warmup bias LR
)

# Loss Function Weights (from microspores.cfg: iou_normalizer, cls_normalizer)
# ─────────────────────────────────────────────────────────────────────────────
# BOX:    Box loss weight (higher = more focus on localization)
# CLS:    Classification loss weight (higher = more focus on classification)
# DFL:    Distribution focal loss weight (for anchor-free models)
# From microspores.cfg: iou_normalizer=0.07, cls_normalizer=1.0
BOX_LOSS_LIST=(
    7.5                     # standard box loss weight
    # 0.07                  # from cfg iou_normalizer (very low)
    # 5.0                   # lower box loss
)

CLS_LOSS_LIST=(
    # 0.5                   # standard classification loss weight
    # 1.0                   # previous run - classification still struggling
    2.0                     # high cls weight - force model to learn class distinction
    # 1.5                   # previous
)

DFL_LOSS_LIST=(
    1.5                     # standard DFL loss weight
    # 1.0                   # lower DFL loss
)

# IoU and NMS Configuration (from microspores.cfg: iou_thresh, beta_nms)
# ─────────────────────────────────────────────────────────────────────────────
# IOU_THRESHOLD:     IoU threshold for training (matching positive samples)
# NMS_THRESHOLD:     NMS IoU threshold for inference
# From microspores.cfg: iou_thresh=0.213, beta_nms=0.6
IOU_THRESHOLD_LIST=(
    0.7                     # standard IoU threshold
    # 0.213                 # from cfg iou_thresh
    # 0.5                   # lower threshold (more positives)
)

# Label Smoothing (regularization technique)
# ─────────────────────────────────────────────────────────────────────────────
# Helps with confused classes by softening hard labels
# Higher values help when classes are easily confused (mature_pollen↔midlate_pollen, young_pollen↔late_microspore)
LABEL_SMOOTHING_LIST=(
    # 0.0                   # no label smoothing
    # 0.1                   # light label smoothing
    # 0.15                  # previous config
    0.2                     # stronger label smoothing - helps reduce overconfidence on similar classes
)

# Close Mosaic (disable mosaic augmentation near end of training)
# ─────────────────────────────────────────────────────────────────────────────
# Number of final epochs to disable mosaic augmentation for fine-tuning
CLOSE_MOSAIC_LIST=(
    # 10                    # disable mosaic for last 10 epochs
    # 0                     # never disable mosaic
    # 20                    # disable mosaic for last 20 epochs
    50                      # disable mosaic for last 50 epochs - longer clean training
)

# Multi-scale Training
# ─────────────────────────────────────────────────────────────────────────────
# MULTI_SCALE: Train with varying image sizes (+/- 50%)
# RECT:        Rectangular training (non-square images, faster)
MULTI_SCALE_LIST=(
    #false                   # fixed image size
    true                  # multi-scale training (MI210 can handle this well)
)

RECT_LIST=(
    false                   # square images (standard)
    # true                  # rectangular images (faster)
)

# Model & Output Configuration
# ─────────────────────────────────────────────────────────────────────────────
# PRETRAINED: true=faster training, better results | false=train from scratch
# CACHE:      ram=fastest (needs RAM), disk=slower, false=no cache (slowest)
# AMP:        true=uses less VRAM, faster          | false=full precision
# FREEZE:     ↑ freezes more layers, faster, less adaptation | 0=train all layers
PRETRAINED_LIST=(
    true                    # use pretrained weights (recommended)
    # false                 # train from scratch
)

RESUME=false                # Resume training from last checkpoint

CACHE_LIST=(
    # "disk"                # disk cache (use if RAM limited)
    "ram"                   # RAM cache (fastest - server likely has plenty of RAM)
    # false                 # no cache (slowest)
)

AMP_LIST=(
    #true                    # mixed precision (recommended - faster on MI210)
    false                 # full precision (use if numerical stability issues)
)

FREEZE_LIST=(
    0                       # train all layers
    # 10                    # freeze backbone (transfer learning)
    # 20                    # freeze more layers
)

# Device Configuration
# ─────────────────────────────────────────────────────────────────────────────
# For AMD ROCm: Use HIP_VISIBLE_DEVICES environment variable (set above)
# Device 0 is the first GPU (MI210)
# For multi-GPU: DEVICE="0,1" (requires larger batch size)
DEVICE=0                    # GPU device: 0, 1, 2... or "cpu" or "0,1" for multi-GPU
