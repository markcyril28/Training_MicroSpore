#!/bin/bash

# SERVER SPECS (Dell Server with AMD ROCm):
#   GPU: AMD Instinct MI210 (Aldebaran/MI200)
#   VRAM: 64GB HBM2e
#   Architecture: gfx90a (CDNA2)
#   Driver: amdgpu
#   Compute Platform: ROCm 6.x
#   CPU Threads: 72

#===============================================================================
# PHASE 3: HARD-EXAMPLE MINING & REFINEMENT
#===============================================================================
# This phase runs AFTER Phase 2 and focuses on:
#   1. Training on a dataset augmented with curated hard examples
#   2. Minimal augmentation to preserve learned boundaries
#   3. Very conservative LR to avoid forgetting
#
# PREREQUISITES before running Phase 3:
#   1. Run extract_hard_examples.py (see docs/ANALYSIS_AND_RECOMMENDATIONS_*.md)
#   2. Review and clean the extracted error galleries
#   3. Add verified hard examples to Dataset_2_OPTIMIZATION_HARDEX/Train/
#   4. Update DATASET_LIST below to point to the augmented dataset
#
# Expected improvements:
#   - Reduce remaining class↔class confusions (mature_pollen↔midlate_pollen)
#   - Improve decision boundaries on morphological transition cases
#===============================================================================

#===============================================================================
# YOLO VERSION SELECTION
#===============================================================================
YOLO_MODELS=(
    "yolov8x.pt"      # xlarge - same as Phase 1 & 2
)

YOLO_MODEL="${YOLO_MODELS[0]}"

#===============================================================================
# CONTINUE TRAINING FROM PHASE 2 OUTPUT
#===============================================================================
CONTINUE_FROM_CUSTOM=true       # Continue from Phase 2 best.pt
AUTO_DETECT_LATEST=true         # Auto-detect latest trained model from Phase 2 output
PREFER_LAST_PT=false            # Prefer best.pt (more stable)

# Fallback path (only used if AUTO_DETECT_LATEST=false)
CUSTOM_WEIGHTS_PATH=""

# Directory to search for Phase 2 output (auto-detection)
# Points to Phase 2's output directory (use absolute path for reliability)
AUTO_DETECT_SEARCH_DIR="/mnt/local3.5tb/home/mcmercado/Training_MicroSpore/trained_models_output/server/04_class_balancing_combination_continue_4_5.2_phase_2"

AUTO_DETECT_DATASET_FILTER=""
AUTO_DETECT_MODEL_FILTER=""


#===============================================================================
# DATASET - Use hard-example augmented dataset
#===============================================================================
# IMPORTANT: Before Phase 3, create a new dataset with hard examples added:
#   1. Copy Dataset_2_OPTIMIZATION to Dataset_2_OPTIMIZATION_HARDEX
#   2. Run extract_hard_examples.py to identify misclassified samples
#   3. Add verified hard examples to Train/ folder
#   4. Regenerate distribution.txt
#
# Alternatively, use the original dataset if you've added hard examples in-place
DATASET_LIST=(
    # "Dataset_2_OPTIMIZATION_HARDEX"   # Dataset with added hard examples (preferred)
    "Dataset_2_OPTIMIZATION"            # Original dataset (use if hard examples added in-place)
)

DEFAULT_DATASET="${DATASET_LIST[0]}"

#===============================================================================
# TRAINING PARAMETERS - Very conservative for refinement
#===============================================================================

#===============================================================================
# EPOCHS - Short refinement phase
#===============================================================================
ADDITIONAL_EPOCHS_LIST=(
    # Phase 3 goal: refine on hard examples with extended training
    # Phase 2 peaked at epoch 125/200 then plateaued - give more room
    200
)

EPOCHS_LIST=(
    500                     # Ignored when continuing
)

PATIENCE_LIST=(
    125                       # Tighter early stopping - Phase 2 plateaued after 75 epochs
)

BATCH_SIZE_LIST=(
    16                       # Keep consistent with Phase 1 & 2
)

IMG_SIZE_LIST=(
    1280                    # Same as Phase 1 & 2
)

WORKERS_LIST=(
    32                      # Match server capacity
)

#===============================================================================
# LEARNING RATE - Slightly higher to enable learning
#===============================================================================
LR0_LIST=(
    # Phase 3: slightly higher than Phase 2 (5e-6) to escape plateau
    # Phase 2 used 5e-6 and plateaued after epoch 125
    # Try 8e-6 as middle ground - not too aggressive
    0.000008                # 8e-6 (balanced approach)
)

LRF_LIST=(
    0.15                    # Higher final LR ratio to maintain learning momentum
)

MOMENTUM_LIST=(
    0.949                   # Same as Phase 1 & 2
)

WEIGHT_DECAY_LIST=(
    0.0005                  # Standard weight decay
)

OPTIMIZER_LIST=(
    "auto"                  # Auto-select
)

#===============================================================================
# COLOR MODE
#===============================================================================
COLOR_MODE_LIST=(
    "grayscale"             # Same as Phase 1 & 2
)

#===============================================================================
# CLASS FOCUS - Reduced oversampling
#===============================================================================
# In Phase 3, we rely on the curated hard examples rather than heavy oversampling
CLASS_FOCUS_MODE_LIST=(
    "manual"                # Still focus on weak classes
)

CLASS_FOCUS_CLASSES_LIST=(
    "mature_pollen,young_pollen,late_microspore,mid_microspore"  # Same 4 weak classes
)

CLASS_FOCUS_FOLD_LIST=(
    3.0                     # Moderate fold - Phase 2 showed mature_pollen/young_pollen still struggling
)

CLASS_FOCUS_TARGET_LIST=(
    "median"
)

CLASS_DISTRIBUTION_FILE="Distribution/1_class_distribution/distribution.txt"

#===============================================================================
# AUGMENTATION - Minimal to preserve learned boundaries
#===============================================================================
# Phase 3 uses very gentle augmentation to avoid blurring the hard-learned
# decision boundaries from Phase 2

HSV_H_LIST=(
    0.08                    # Slightly higher hue variation for diversity
)

HSV_S_LIST=(
    0.5                     # Moderate saturation variation
)

HSV_V_LIST=(
    0.35                    # Moderate brightness variation
)

DEGREES_LIST=(
    30.0                    # Reduced rotation
)

TRANSLATE_LIST=(
    0.1                     # Standard translation
)

SCALE_LIST=(
    0.4                     # Moderate scale variation for size invariance
)

SHEAR_LIST=(
    0.0                     # No shear
)

PERSPECTIVE_LIST=(
    0.0                     # No perspective
)

FLIPUD_LIST=(
    0.5                     # Keep flips (microscopy is rotation-invariant)
)

FLIPLR_LIST=(
    0.5                     # Keep flips
)

# Mosaic and Mixup - Light augmentation to prevent overfitting
MOSAIC_LIST=(
    0.3                     # Light mosaic - prevents overfitting on hard examples
)

MIXUP_LIST=(
    0.0                     # No mixup - preserve class boundaries
)

COPY_PASTE_LIST=(
    0.25                    # Moderate copy-paste for weak class recall (mature_pollen, young_pollen)
)

#===============================================================================
# WARMUP - Short warmup
#===============================================================================
WARMUP_EPOCHS_LIST=(
    5.0                     # Standard warmup to stabilize gradients
)

WARMUP_MOMENTUM_LIST=(
    0.8
)

WARMUP_BIAS_LR_LIST=(
    0.1
)

#===============================================================================
# LOSS WEIGHTS
#===============================================================================
BOX_LOSS_LIST=(
    7.5                     # Standard
)

CLS_LOSS_LIST=(
    1.2                     # Slightly increased from Phase 2 (1.0) to reduce class confusions
                            # Phase 2: mature_pollen→midlate_pollen=23%, young_pollen→late_microspore=18%
)

DFL_LOSS_LIST=(
    1.5                     # Standard
)

#===============================================================================
# IoU THRESHOLD
#===============================================================================
IOU_THRESHOLD_LIST=(
    0.55                    # Slightly stricter than Phase 2 (0.6) to improve precision
                            # Phase 2 recall was good (71.2%), now focus on precision (64.9%)
)

#===============================================================================
# LABEL SMOOTHING - None
#===============================================================================
LABEL_SMOOTHING_LIST=(
    0.02                    # Very light smoothing to prevent overconfidence on hard examples
)

#===============================================================================
# CLOSE MOSAIC - N/A (mosaic disabled)
#===============================================================================
CLOSE_MOSAIC_LIST=(
    50                      # Disable mosaic for last 50 epochs for clean fine-tuning
)

#===============================================================================
# OTHER SETTINGS
#===============================================================================
MULTI_SCALE_LIST=(
    false
)

RECT_LIST=(
    false
)

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
    0                       # Train all layers
)

DEVICE=0
