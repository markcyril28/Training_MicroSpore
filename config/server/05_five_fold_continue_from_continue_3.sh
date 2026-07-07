#!/bin/bash
# shellcheck disable=SC2034

#===============================================================================
# FIVE-FOLD CONTINUATION CONFIG
#===============================================================================
# Trains one continuation run per fold from:
#   trained_models_output/server/04_class_balancing_combination_continue_3
#
# Dataset layout:
#   TRAINING_WD/Datasets_4_Five_Fold_Strategy/Five_Fold/fold_1/data.yaml
#   ...
#   TRAINING_WD/Datasets_4_Five_Fold_Strategy/Five_Fold/fold_5/data.yaml
#
# Each fold uses its own train/val split. After training completes, compare the
# five output folders and select the best-performing fold/model.
#===============================================================================

#===============================================================================
# MODEL / CHECKPOINT
#===============================================================================

YOLO_MODELS=(
    "yolov8x.pt"
)

YOLO_MODEL="${YOLO_MODELS[0]}"

CONTINUE_FROM_CUSTOM=true
AUTO_DETECT_LATEST=true
PREFER_LAST_PT=false

CONTINUE_3_OUTPUT_DIR="${SCRIPT_DIR}/trained_models_output/server/04_class_balancing_combination_continue_3"
CONTINUE_3_WEIGHTS_DIR="${CONTINUE_3_OUTPUT_DIR}/continue_3/weights"
CONTINUE_3_WEIGHTS_FILE="Dataset_2_OPTIMIZATION_Dataset_2_OPTIMIZATION_best_gray_img1280_bal-manual_auto_e300_b8_lr0_00001_20260201_101533_cont1_gray_img1280_bal-manual_auto_e400_b8_lr0_00005_20260201_170927_cont1_best.pt"

# Fallback if auto-detection is disabled or finds no checkpoint.
CUSTOM_WEIGHTS_PATH="${CONTINUE_3_WEIGHTS_DIR}/${CONTINUE_3_WEIGHTS_FILE}"

# Keep this relative to the training root so the mirrored server tree works.
AUTO_DETECT_SEARCH_DIR="${CONTINUE_3_OUTPUT_DIR}"
AUTO_DETECT_DATASET_FILTER=""
AUTO_DETECT_MODEL_FILTER=""

#===============================================================================
# FIVE-FOLD DATASETS
#===============================================================================

DATASET_LIST=(
    "Datasets_4_Five_Fold_Strategy/Five_Fold/fold_1"
    "Datasets_4_Five_Fold_Strategy/Five_Fold/fold_2"
    "Datasets_4_Five_Fold_Strategy/Five_Fold/fold_3"
    "Datasets_4_Five_Fold_Strategy/Five_Fold/fold_4"
    "Datasets_4_Five_Fold_Strategy/Five_Fold/fold_5"
)

DEFAULT_DATASET="${DATASET_LIST[0]}"

#===============================================================================
# TRAINING PARAMETERS
#===============================================================================

ADDITIONAL_EPOCHS_LIST=(
    200
)

EPOCHS_LIST=(
    500
)

PATIENCE_LIST=(
    100
)

BATCH_SIZE_LIST=(
    16
)

IMG_SIZE_LIST=(
    1280
)

WORKERS_LIST=(
    32
)

LR0_LIST=(
    0.000008
)

LRF_LIST=(
    0.15
)

MOMENTUM_LIST=(
    0.949
)

WEIGHT_DECAY_LIST=(
    0.0005
)

OPTIMIZER_LIST=(
    "auto"
)

COLOR_MODE_LIST=(
    "grayscale"
)

#===============================================================================
# CLASS FOCUS
#===============================================================================
# The five-fold dataset currently has no per-fold
# Distribution/1_class_distribution/distribution.txt files. Use "none" so the
# five-fold run is explicit and comparable instead of silently calculating empty
# balancing weights. Add per-fold distribution files before enabling auto/manual.

CLASS_FOCUS_MODE_LIST=(
    "none"
)

CLASS_FOCUS_CLASSES_LIST=(
    "mature_pollen,young_pollen,late_microspore,mid_microspore"
)

CLASS_FOCUS_FOLD_LIST=(
    3.0
)

CLASS_FOCUS_TARGET_LIST=(
    "median"
)

CLASS_DISTRIBUTION_FILE="Distribution/1_class_distribution/distribution.txt"

#===============================================================================
# AUGMENTATION
#===============================================================================

HSV_H_LIST=(
    0.08
)

HSV_S_LIST=(
    0.5
)

HSV_V_LIST=(
    0.35
)

DEGREES_LIST=(
    30.0
)

TRANSLATE_LIST=(
    0.1
)

SCALE_LIST=(
    0.4
)

SHEAR_LIST=(
    0.0
)

PERSPECTIVE_LIST=(
    0.0
)

FLIPUD_LIST=(
    0.5
)

FLIPLR_LIST=(
    0.5
)

MOSAIC_LIST=(
    0.3
)

MIXUP_LIST=(
    0.0
)

COPY_PASTE_LIST=(
    0.15
)

#===============================================================================
# WARMUP / LOSSES / VALIDATION
#===============================================================================

WARMUP_EPOCHS_LIST=(
    5.0
)

WARMUP_MOMENTUM_LIST=(
    0.8
)

WARMUP_BIAS_LR_LIST=(
    0.1
)

BOX_LOSS_LIST=(
    7.5
)

CLS_LOSS_LIST=(
    1.2
)

DFL_LOSS_LIST=(
    1.5
)

IOU_THRESHOLD_LIST=(
    0.55
)

LABEL_SMOOTHING_LIST=(
    0.02
)

CLOSE_MOSAIC_LIST=(
    50
)

MULTI_SCALE_LIST=(
    false
)

RECT_LIST=(
    false
)

#===============================================================================
# RUNTIME
#===============================================================================

PRETRAINED_LIST=(
    true
)

RESUME=false

CACHE_LIST=(
    "ram"
)

AMP_LIST=(
    true
)

FREEZE_LIST=(
    0
)

DEVICE=0
