#!/bin/bash
#===============================================================================
# Microspore Phenotyping - YOLO Training Script (SERVER VERSION)
# Main script for training YOLO models on microspore phenotyping dataset.
# Uses common_functions.sh for shared utilities (DRY principle).
#
# SERVER SPECS (Dell Server with AMD ROCm):
#   GPU: AMD Instinct MI210 (Aldebaran/MI200)
#   VRAM: 64GB HBM2e
#   Architecture: gfx90a (CDNA2)
#   Driver: amdgpu
#   Compute Platform: ROCm 6.x
#   PyTorch: ROCm version (torch+rocm)
#   Allowed no. of threads: 32
#
# This configuration uses:
#   - Batch size 64 (optimal for 64GB HBM2e VRAM)
#   - RAM caching (fastest loading)
#   - 16 workers (optimal for 32 threads)
#   - Mixed precision (AMP) enabled for faster training
#===============================================================================

#set -e  # Exit on error

#===============================================================================
# FEATURES & TOGGLES
#===============================================================================
REST_TIME_PER_RUN=60          # GPU cooldown between runs (seconds, 0=disabled)
                              # Reference: 60=1min (MI210 has excellent thermals with HBM2e)
CLEAR_LOGS_ON_START=false     # Delete previous logs before training
CLEAR_OUTPUT_ON_START=false   # Delete previous model outputs before training
SKIP_EXISTING=true            # Skip if model with same params already exists
#===============================================================================

# Get script directory (modules are in the same directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common functions from modules/config/ (DRY principle)
if [ -f "${SCRIPT_DIR}/modules/config/common_functions.sh" ]; then
    source "${SCRIPT_DIR}/modules/config/common_functions.sh"
else
    echo "ERROR: modules/config/common_functions.sh not found. Please ensure it exists."
    exit 1
fi

# Source logging utilities from modules/logging/
if [ -f "${SCRIPT_DIR}/modules/logging/logging_utils.sh" ]; then
    source "${SCRIPT_DIR}/modules/logging/logging_utils.sh"
    LOGGING_ENABLED=true
else
    echo "WARNING: modules/logging/logging_utils.sh not found. Running without logging."
    LOGGING_ENABLED=false
fi

# Use shared configuration from common_functions.sh
ENV_NAME="${COMMON_ENV_NAME}"

#===============================================================================
# AMD ROCm ENVIRONMENT SETUP
#===============================================================================
# Ensure ROCm environment variables are set for MI210
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${ROCM_PATH}"
export HSA_OVERRIDE_GFX_VERSION=9.0.10  # For gfx90a (MI210)
export HIP_VISIBLE_DEVICES=0             # Single GPU training
export MIOPEN_LOG_LEVEL=1                # Reduce MIOpen warnings

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
    "yolov5xu.pt"     # xlarge   - slowest, best accuracy (OPTIMAL for MI210)
    
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
    "yolov9e.pt"      # extended - best accuracy (OPTIMAL for MI210)
    
    # YOLOv10 variants
    # "yolov10n.pt"   # nano     - fastest
    # "yolov10s.pt"   # small    - fast
    # "yolov10m.pt"   # medium   - balanced
    # "yolov10l.pt"   # large    - slower
    "yolov10x.pt"     # xlarge   - best accuracy (OPTIMAL for MI210)
    
    # YOLO11 variants - Latest (note: named 'yolo11' not 'yolov11')
    # "yolo11n.pt"    # nano     - fastest
    # "yolo11s.pt"    # small    - fast
    # "yolo11m.pt"    # medium   - balanced
    # "yolo11l.pt"    # large    - slower
    "yolo11x.pt"      # xlarge   - best accuracy (OPTIMAL for MI210 64GB)
)

# Select first model from the array (for quick reference)
YOLO_MODEL="${YOLO_MODELS[0]}"

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
EPOCHS_LIST=(
    # 10                    # quick test
    # 100                   # standard training
    # 150                   # optimal training (early stopping will trigger if converged)
    200                     # long training (MI210 can handle extended training)
    # 300                   # maximum training
)

PATIENCE_LIST=(
    # 50                    # standard patience
    # 25                    # quick stopping
    150                     # balanced patience (optimal for convergence detection)
)

BATCH_SIZE_LIST=(
    # 8                     # low (for debugging)
    # 16                    # moderate
    #32                    # standard for high-end GPUs
    64                      # optimal for MI210 64GB HBM2e (maximum throughput)
    # 128                   # very high batch size (may need gradient accumulation)
)

IMG_SIZE_LIST=(
    # 320                   # fast, low resolution
    # 512                   # medium resolution
    # 608                   # from microspores.cfg (width/height=608)
    640                   # standard resolution
    # 800                   # high resolution
    #1024                    # very high resolution (optimal for MI210 64GB VRAM)
    # 1280                  # maximum (for detecting very small objects)
)

WORKERS_LIST=(
    # 2                     # low CPU
    # 4                     # standard
    # 8                     # moderate (balanced for data loading)
    16                      # server with 32 threads (optimal: ~half of available threads)
    # 32                    # maximum (use all threads - may cause contention)
)

# Learning Rate & Optimizer
# ─────────────────────────────────────────────────────────────────────────────
# LR0:          ↑ faster learning, may overshoot    | ↓ slower, more stable
# LRF:          ↑ higher final LR, less fine-tuning | ↓ better fine-tuning at end
# MOMENTUM:     ↑ faster convergence, may overshoot | ↓ more stable, slower
# WEIGHT_DECAY: ↑ stronger regularization           | ↓ less regularization, may overfit
# OPTIMIZER:    SGD=stable, Adam/AdamW=faster convergence, auto=recommended
LR0_LIST=(
    0.001                   # from microspores.cfg (learning_rate=0.001)
    # 0.005                 # medium-low learning rate
    # 0.01                  # standard learning rate
    # 0.02                  # high learning rate
)

LRF_LIST=(
    # 0.001                 # very low final LR
    0.01                    # standard final LR ratio
    # 0.1                   # higher final LR
)

MOMENTUM_LIST=(
    # 0.9                   # lower momentum
    # 0.937                 # standard momentum
    0.949                   # from microspores.cfg (momentum=0.949)
)

WEIGHT_DECAY_LIST=(
    0.0005                  # standard weight decay
    # 0.0001                # low regularization
    # 0.001                 # high regularization
)

OPTIMIZER_LIST=(
    "auto"                  # auto-select (recommended)
    # "SGD"                 # Stochastic Gradient Descent
    # "Adam"                # Adam optimizer
    # "AdamW"               # Adam with weight decay
    # "NAdam"               # Nesterov Adam
    # "RAdam"               # Rectified Adam
)

# Grayscale Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Train with grayscale images (useful for microscopy where color is not informative)
# 'RGB' = color (3 channels), 'grayscale' = grayscale (converted to 3-channel gray)
GRAYSCALE_LIST=(
    "RGB"                   # RGB color images (default)
    "grayscale"             # grayscale images
)

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
    45.0                    # moderate rotation
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
    0.0                     # no mixup
    # 0.1                   # light mixup
    # 0.5                   # moderate mixup
)

COPY_PASTE_LIST=(
    0.0                     # no copy-paste
    # 0.1                   # light copy-paste
    # 0.5                   # moderate copy-paste
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
    0.5                     # standard classification loss weight
    # 1.0                   # from cfg cls_normalizer
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
LABEL_SMOOTHING_LIST=(
    0.0                     # no label smoothing
    # 0.1                   # light label smoothing
)

# Close Mosaic (disable mosaic augmentation near end of training)
# ─────────────────────────────────────────────────────────────────────────────
# Number of final epochs to disable mosaic augmentation for fine-tuning
CLOSE_MOSAIC_LIST=(
    # 10                    # disable mosaic for last 10 epochs
    # 0                     # never disable mosaic
    20                      # disable mosaic for last 20 epochs (better fine-tuning)
)

# Multi-scale Training
# ─────────────────────────────────────────────────────────────────────────────
# MULTI_SCALE: Train with varying image sizes (+/- 50%)
# RECT:        Rectangular training (non-square images, faster)
MULTI_SCALE_LIST=(
    false                   # fixed image size
    # true                  # multi-scale training (MI210 can handle this well)
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
    true                    # mixed precision (recommended - faster on MI210)
    # false                 # full precision (use if numerical stability issues)
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


#===============================================================================
# DEFAULT VALUES (first element of each array - used for naming and quick access)
#===============================================================================
EPOCHS="${EPOCHS_LIST[0]}"
BATCH_SIZE="${BATCH_SIZE_LIST[0]}"
IMG_SIZE="${IMG_SIZE_LIST[0]}"
PATIENCE="${PATIENCE_LIST[0]}"
WORKERS="${WORKERS_LIST[0]}"
LR0="${LR0_LIST[0]}"
LRF="${LRF_LIST[0]}"
MOMENTUM="${MOMENTUM_LIST[0]}"
WEIGHT_DECAY="${WEIGHT_DECAY_LIST[0]}"
OPTIMIZER="${OPTIMIZER_LIST[0]}"
HSV_H="${HSV_H_LIST[0]}"
HSV_S="${HSV_S_LIST[0]}"
HSV_V="${HSV_V_LIST[0]}"
DEGREES="${DEGREES_LIST[0]}"
TRANSLATE="${TRANSLATE_LIST[0]}"
SCALE="${SCALE_LIST[0]}"
SHEAR="${SHEAR_LIST[0]}"
PERSPECTIVE="${PERSPECTIVE_LIST[0]}"
FLIPUD="${FLIPUD_LIST[0]}"
FLIPLR="${FLIPLR_LIST[0]}"
MOSAIC="${MOSAIC_LIST[0]}"
MIXUP="${MIXUP_LIST[0]}"
COPY_PASTE="${COPY_PASTE_LIST[0]}"
GRAYSCALE="${GRAYSCALE_LIST[0]}"
PRETRAINED="${PRETRAINED_LIST[0]}"
CACHE="${CACHE_LIST[0]}"
AMP="${AMP_LIST[0]}"
FREEZE="${FREEZE_LIST[0]}"

# New parameters from microspores.cfg
WARMUP_EPOCHS="${WARMUP_EPOCHS_LIST[0]}"
WARMUP_MOMENTUM="${WARMUP_MOMENTUM_LIST[0]}"
WARMUP_BIAS_LR="${WARMUP_BIAS_LR_LIST[0]}"
BOX_LOSS="${BOX_LOSS_LIST[0]}"
CLS_LOSS="${CLS_LOSS_LIST[0]}"
DFL_LOSS="${DFL_LOSS_LIST[0]}"
IOU_THRESHOLD="${IOU_THRESHOLD_LIST[0]}"
LABEL_SMOOTHING="${LABEL_SMOOTHING_LIST[0]}"
CLOSE_MOSAIC="${CLOSE_MOSAIC_LIST[0]}"
MULTI_SCALE="${MULTI_SCALE_LIST[0]}"
RECT="${RECT_LIST[0]}"

#===============================================================================
# PATHS (Using common configuration - DRY principle)
#===============================================================================

# Base paths (dataset-independent)
WEIGHTS_DIR="${SCRIPT_DIR}/${COMMON_WEIGHTS_SUBDIR}"
OUTPUT_DIR="${SCRIPT_DIR}/${COMMON_TRAINED_MODELS_SUBDIR}"

# Create directories using common function
ensure_dir "${WEIGHTS_DIR}"
ensure_dir "${OUTPUT_DIR}"

# Note: DATA_YAML, YOLO_MODEL_PATH, MODEL_NAME, and EXP_NAME are now set 
# dynamically within the training loop to support multiple datasets

#===============================================================================
# CONDA ENVIRONMENT ACTIVATION
#===============================================================================

print_header "Microspore Phenotyping - YOLO Training (Server AMD ROCm)"

echo "Server Configuration:"
echo "  - GPU: AMD Instinct MI210 (64GB HBM2e)"
echo "  - Architecture: gfx90a (CDNA2)"
echo "  - Compute: ROCm 6.x"
echo ""

echo "Models selected for training:"
for model in "${YOLO_MODELS[@]}"; do
    echo "  - ${model}"
done
echo ""

# Initialize and activate conda using common functions
if ! init_conda_shell; then
    exit 1
fi

if ! activate_conda_env "${ENV_NAME}"; then
    print_error "Please run setup_conda_training_server.sh first."
    exit 1
fi

echo ""

#===============================================================================
# VERIFY GPU AVAILABILITY (AMD ROCm)
#===============================================================================

print_subheader "Checking AMD GPU Availability (ROCm)"

# Check if rocm-smi is available
if command -v rocm-smi &> /dev/null; then
    if rocm-smi --showproductname &> /dev/null; then
        print_success "AMD GPU detected via rocm-smi"
        echo ""
        rocm-smi --showproductname --showmeminfo vram 2>/dev/null | grep -E "GPU|VRAM" | head -6 || true
        echo ""
        
        # Show temperature
        echo "GPU Temperature:"
        rocm-smi --showtemp 2>/dev/null | grep -E "GPU|junction" | head -2 || true
        echo ""
    else
        print_warning "rocm-smi available but cannot detect GPU"
    fi
else
    print_warning "rocm-smi not found - GPU monitoring unavailable"
fi

# Verify PyTorch can see the GPU
echo "Verifying PyTorch ROCm support..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  PyTorch version: {torch.__version__}')
    print(f'  ROCm/HIP version: {torch.version.hip}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('  WARNING: PyTorch cannot detect ROCm GPU!')
    print('  Training will run on CPU (much slower)')
" 2>/dev/null || print_warning "Could not verify PyTorch ROCm support"

echo ""

#===============================================================================
# LOG MANAGEMENT - Clear or Keep Old Logs
#===============================================================================

LOGS_DIR="${SCRIPT_DIR}/logs"

if [ "$CLEAR_LOGS_ON_START" = true ]; then
    if [ -d "${LOGS_DIR}" ] && [ "$(ls -A ${LOGS_DIR} 2>/dev/null)" ]; then
        # Count existing log directories
        LOG_COUNT=$(find "${LOGS_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        
        if [ "$LOG_COUNT" -gt 0 ]; then
            print_warning "CLEAR_LOGS_ON_START is enabled. Clearing ${LOG_COUNT} existing log directory(ies)..."
            rm -rf "${LOGS_DIR:?}"/*
            print_success "Old logs cleared successfully."
        fi
    else
        print_info "No existing logs to clear."
    fi
else
    print_info "CLEAR_LOGS_ON_START is disabled. Old logs will be preserved."
fi
echo ""

#===============================================================================
# OUTPUT MANAGEMENT - Clear or Keep Old Trained Models
#===============================================================================

OUTPUT_DIR="${SCRIPT_DIR}/trained_models_output"

if [ "$CLEAR_OUTPUT_ON_START" = true ]; then
    if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A ${OUTPUT_DIR} 2>/dev/null)" ]; then
        # Count existing output directories
        OUTPUT_COUNT=$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        
        if [ "$OUTPUT_COUNT" -gt 0 ]; then
            print_warning "CLEAR_OUTPUT_ON_START is enabled. Clearing ${OUTPUT_COUNT} existing trained model(s)..."
            rm -rf "${OUTPUT_DIR:?}"/*
            print_success "Old trained models cleared successfully."
        fi
    else
        print_info "No existing trained models to clear."
    fi
else
    print_info "CLEAR_OUTPUT_ON_START is disabled. Existing trained models will be preserved."
fi
echo ""

#===============================================================================
# CALCULATE TOTAL COMBINATIONS
#===============================================================================

# Calculate total number of training combinations
calculate_total_combinations() {
    local total=1
    total=$((total * ${#DATASET_LIST[@]}))
    total=$((total * ${#YOLO_MODELS[@]}))
    total=$((total * ${#EPOCHS_LIST[@]}))
    total=$((total * ${#BATCH_SIZE_LIST[@]}))
    total=$((total * ${#IMG_SIZE_LIST[@]}))
    total=$((total * ${#LR0_LIST[@]}))
    total=$((total * ${#OPTIMIZER_LIST[@]}))
    total=$((total * ${#GRAYSCALE_LIST[@]}))
    echo $total
}

TOTAL_COMBINATIONS=$(calculate_total_combinations)

print_info "Parameter Grid Summary:"
echo "  - Datasets:    ${#DATASET_LIST[@]} (${DATASET_LIST[*]})"
echo "  - Models:      ${#YOLO_MODELS[@]} (${YOLO_MODELS[*]})"
echo "  - Epochs:      ${#EPOCHS_LIST[@]} (${EPOCHS_LIST[*]})"
echo "  - Batch Sizes: ${#BATCH_SIZE_LIST[@]} (${BATCH_SIZE_LIST[*]})"
echo "  - Image Sizes: ${#IMG_SIZE_LIST[@]} (${IMG_SIZE_LIST[*]})"
echo "  - LR0 values:  ${#LR0_LIST[@]} (${LR0_LIST[*]})"
echo "  - Optimizers:  ${#OPTIMIZER_LIST[@]} (${OPTIMIZER_LIST[*]})"
echo "  - Grayscale:   ${#GRAYSCALE_LIST[@]} (${GRAYSCALE_LIST[*]})"
echo ""
print_warning "Total combinations to train: ${TOTAL_COMBINATIONS}"
echo ""

#===============================================================================
# TRAINING LOOP - Train all selected models and parameter combinations
#===============================================================================

CURRENT_RUN=0
SUCCESSFUL_RUNS=()
FAILED_RUNS=()
SKIPPED_RUNS=()

# Function to generate experiment name based on current parameters
generate_exp_name() {
    local dataset_name="$1"
    local model_name="$2"
    local epochs="$3"
    local batch_size="$4"
    local img_size="$5"
    local lr0="$6"
    local optimizer="$7"
    local grayscale="$8"
    local timestamp="$9"
    
    # Format LR0 for filename (remove decimal point)
    local lr0_str=$(echo "${lr0}" | sed 's/\./_/g')
    
    # Add grayscale indicator
    local gray_str="rgb"
    if [ "$grayscale" = "grayscale" ]; then
        gray_str="gray"
    fi
    
    echo "${dataset_name}_${model_name}_e${epochs}_b${batch_size}_img${img_size}_lr${lr0_str}_${optimizer}_${gray_str}_${timestamp}"
}

# Function to check if training already completed for given params
check_training_exists() {
    local dataset_name="$1"
    local model_name="$2"
    local epochs="$3"
    local batch_size="$4"
    local img_size="$5"
    local lr0="$6"
    local optimizer="$7"
    local grayscale="$8"
    
    # Format LR0 for filename (remove decimal point)
    local lr0_str=$(echo "${lr0}" | sed 's/\./_/g')
    
    # Add grayscale indicator
    local gray_str="rgb"
    if [ "$grayscale" = "grayscale" ]; then
        gray_str="gray"
    fi
    local pattern="${dataset_name}_${model_name}_e${epochs}_b${batch_size}_img${img_size}_lr${lr0_str}_${optimizer}_${gray_str}_*"
    
    # Look for matching directories with completed training (best.pt exists)
    for dir in "${OUTPUT_DIR}"/${pattern}; do
        if [ -d "$dir" ] && [ -f "${dir}/weights/best.pt" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

# Nested loops for all parameter combinations
# Primary loop parameters (commonly varied for grid search)
for DATASET_NAME in "${DATASET_LIST[@]}"; do
for YOLO_MODEL in "${YOLO_MODELS[@]}"; do
for EPOCHS in "${EPOCHS_LIST[@]}"; do
for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
for IMG_SIZE in "${IMG_SIZE_LIST[@]}"; do
for LR0 in "${LR0_LIST[@]}"; do
for OPTIMIZER in "${OPTIMIZER_LIST[@]}"; do
for GRAYSCALE in "${GRAYSCALE_LIST[@]}"; do

    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    # Set dataset-specific paths (using COMMON_DATASETS_DIR from config)
    DATASET_PATH="${SCRIPT_DIR}/${COMMON_DATASETS_DIR}/${DATASET_NAME}"
    DATA_YAML="${DATASET_PATH}/data.yaml"
    
    # Validate dataset exists
    if [ ! -f "${DATA_YAML}" ]; then
        print_error "Dataset data.yaml not found: ${DATA_YAML}"
        print_warning "Skipping dataset: ${DATASET_NAME}"
        SKIPPED_RUNS+=("${DATASET_NAME}_*")
        continue
    fi
    
    # =========================================================================
    # PORTABILITY: Update data.yaml path to current location
    # This ensures the project works on any machine/directory
    # =========================================================================
    update_data_yaml_path "${DATA_YAML}" "${DATASET_PATH}"
    
    # Use first values for less commonly varied parameters
    PATIENCE="${PATIENCE_LIST[0]}"
    WORKERS="${WORKERS_LIST[0]}"
    LRF="${LRF_LIST[0]}"
    MOMENTUM="${MOMENTUM_LIST[0]}"
    WEIGHT_DECAY="${WEIGHT_DECAY_LIST[0]}"
    HSV_H="${HSV_H_LIST[0]}"
    HSV_S="${HSV_S_LIST[0]}"
    HSV_V="${HSV_V_LIST[0]}"
    DEGREES="${DEGREES_LIST[0]}"
    TRANSLATE="${TRANSLATE_LIST[0]}"
    SCALE="${SCALE_LIST[0]}"
    SHEAR="${SHEAR_LIST[0]}"
    PERSPECTIVE="${PERSPECTIVE_LIST[0]}"
    FLIPUD="${FLIPUD_LIST[0]}"
    FLIPLR="${FLIPLR_LIST[0]}"
    MOSAIC="${MOSAIC_LIST[0]}"
    MIXUP="${MIXUP_LIST[0]}"
    COPY_PASTE="${COPY_PASTE_LIST[0]}"
    PRETRAINED="${PRETRAINED_LIST[0]}"
    CACHE="${CACHE_LIST[0]}"
    AMP="${AMP_LIST[0]}"
    FREEZE="${FREEZE_LIST[0]}"
    
    # New parameters from microspores.cfg
    WARMUP_EPOCHS="${WARMUP_EPOCHS_LIST[0]}"
    WARMUP_MOMENTUM="${WARMUP_MOMENTUM_LIST[0]}"
    WARMUP_BIAS_LR="${WARMUP_BIAS_LR_LIST[0]}"
    BOX_LOSS="${BOX_LOSS_LIST[0]}"
    CLS_LOSS="${CLS_LOSS_LIST[0]}"
    DFL_LOSS="${DFL_LOSS_LIST[0]}"
    IOU_THRESHOLD="${IOU_THRESHOLD_LIST[0]}"
    LABEL_SMOOTHING="${LABEL_SMOOTHING_LIST[0]}"
    CLOSE_MOSAIC="${CLOSE_MOSAIC_LIST[0]}"
    MULTI_SCALE="${MULTI_SCALE_LIST[0]}"
    RECT="${RECT_LIST[0]}"
    
    # Check if model exists locally, otherwise will be downloaded
    if [ -f "${WEIGHTS_DIR}/${YOLO_MODEL}" ]; then
        YOLO_MODEL_PATH="${WEIGHTS_DIR}/${YOLO_MODEL}"
    else
        YOLO_MODEL_PATH="${YOLO_MODEL}"
    fi
    
    # Extract model name without extension for naming
    MODEL_NAME=$(basename "${YOLO_MODEL}" .pt)
    
    # Grayscale indicator for display
    if [ "$GRAYSCALE" = "grayscale" ]; then
        GRAY_DISPLAY="gray"
    else
        GRAY_DISPLAY="rgb"
    fi
    
    # Create run identifier for tracking (includes dataset name)
    RUN_ID="${DATASET_NAME}_${MODEL_NAME}_e${EPOCHS}_b${BATCH_SIZE}_img${IMG_SIZE}_lr${LR0}_${OPTIMIZER}_${GRAY_DISPLAY}"
    
    #===========================================================================
    # CHECK IF TRAINING ALREADY EXISTS (SKIP IF ENABLED)
    #===========================================================================
    if [ "$SKIP_EXISTING" = true ]; then
        existing_dir=$(check_training_exists "$DATASET_NAME" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$IMG_SIZE" "$LR0" "$OPTIMIZER" "$GRAYSCALE")
        if [ $? -eq 0 ]; then
            print_warning "Skipping ${RUN_ID} - already trained: $(basename "$existing_dir")"
            SKIPPED_RUNS+=("${RUN_ID}")
            continue
        fi
    fi
    
    # Generate experiment name with timestamp and parameters
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXP_NAME=$(generate_exp_name "$DATASET_NAME" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$IMG_SIZE" "$LR0" "$OPTIMIZER" "$GRAYSCALE" "$TIMESTAMP")
    
    #===========================================================================
    # INITIALIZE LOGGING FOR THIS RUN
    #===========================================================================
    if [ "$LOGGING_ENABLED" = true ]; then
        # Call directly (not in subshell) so variables are set in this shell
        init_logging_dirs "${EXP_NAME}"
        trap_logging_cleanup
        start_gpu_monitor 5
        start_system_monitor 10
        log_info "Starting training run ${CURRENT_RUN}/${TOTAL_COMBINATIONS}: ${RUN_ID}"
    fi
    
    print_header "Training Run ${CURRENT_RUN}/${TOTAL_COMBINATIONS}"
    
    echo "Configuration:"
    echo "  - Dataset:    ${DATASET_NAME}"
    echo "  - Data:       ${DATA_YAML}"
    echo "  - Model:      ${YOLO_MODEL}"
    echo "  - Weights:    ${WEIGHTS_DIR}"
    echo "  - Epochs:     ${EPOCHS}"
    echo "  - Batch Size: ${BATCH_SIZE}"
    echo "  - Image Size: ${IMG_SIZE}"
    echo "  - LR0:        ${LR0}"
    echo "  - Optimizer:  ${OPTIMIZER}"
    echo "  - Grayscale:  ${GRAYSCALE}"
    echo "  - GPU:        AMD Instinct MI210 (ROCm)"
    echo "  - Output:     ${OUTPUT_DIR}/${EXP_NAME}"
    echo ""
    
    # Run training using the modules/training/train.py script
    # Capture both stdout and stderr for comprehensive error logging
    TRAINING_OUTPUT_FILE=$(mktemp)
    
    if python -m modules.training.train \
        --data-yaml "${DATA_YAML}" \
        --model "${YOLO_MODEL}" \
        --weights-dir "${WEIGHTS_DIR}" \
        --project-dir "${OUTPUT_DIR}" \
        --exp-name "${EXP_NAME}" \
        --dataset-dir "${DATASET_PATH}" \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --img-size ${IMG_SIZE} \
        --patience ${PATIENCE} \
        --workers ${WORKERS} \
        --lr0 ${LR0} \
        --lrf ${LRF} \
        --momentum ${MOMENTUM} \
        --weight-decay ${WEIGHT_DECAY} \
        --optimizer "${OPTIMIZER}" \
        --hsv-h ${HSV_H} \
        --hsv-s ${HSV_S} \
        --hsv-v ${HSV_V} \
        --degrees ${DEGREES} \
        --translate ${TRANSLATE} \
        --scale ${SCALE} \
        --shear ${SHEAR} \
        --perspective ${PERSPECTIVE} \
        --flipud ${FLIPUD} \
        --fliplr ${FLIPLR} \
        --mosaic ${MOSAIC} \
        --mixup ${MIXUP} \
        --copy-paste ${COPY_PASTE} \
        --grayscale ${GRAYSCALE} \
        --pretrained ${PRETRAINED} \
        --resume ${RESUME} \
        --cache "${CACHE}" \
        --amp ${AMP} \
        --freeze ${FREEZE} \
        --device ${DEVICE} \
        --warmup-epochs ${WARMUP_EPOCHS} \
        --warmup-momentum ${WARMUP_MOMENTUM} \
        --warmup-bias-lr ${WARMUP_BIAS_LR} \
        --box-loss ${BOX_LOSS} \
        --cls-loss ${CLS_LOSS} \
        --dfl-loss ${DFL_LOSS} \
        --iou-threshold ${IOU_THRESHOLD} \
        --label-smoothing ${LABEL_SMOOTHING} \
        --close-mosaic ${CLOSE_MOSAIC} \
        --multi-scale ${MULTI_SCALE} \
        --rect ${RECT} \
        --log-dir "${EXPERIMENT_LOG_DIR:-}" 2>&1 | tee "${TRAINING_OUTPUT_FILE}"; then
        
        SUCCESSFUL_RUNS+=("${RUN_ID}")
        print_success "Run ${RUN_ID} trained successfully!"
        print_success "Saved to: ${OUTPUT_DIR}/${EXP_NAME}"
        
        if [ "$LOGGING_ENABLED" = true ]; then
            log_info "Training completed successfully for ${RUN_ID}"
        fi
        
        # Copy logs and trained model to dataset-specific folders
        copy_outputs_to_dataset "${EXPERIMENT_LOG_DIR:-}" "${OUTPUT_DIR}" "${EXP_NAME}" "${DATASET_PATH}"
    else
        TRAINING_EXIT_CODE=$?
        FAILED_RUNS+=("${RUN_ID}")
        print_error "Run ${RUN_ID} training failed!"
        
        if [ "$LOGGING_ENABLED" = true ]; then
            # Log comprehensive error details
            log_error "Training failed for ${RUN_ID}"
            log_error "Exit code: ${TRAINING_EXIT_CODE}"
            log_error "Command: python -m modules.training.train"
            log_error "Parameters:"
            log_error "  --data-yaml: ${DATA_YAML}"
            log_error "  --model: ${YOLO_MODEL}"
            log_error "  --epochs: ${EPOCHS}"
            log_error "  --batch-size: ${BATCH_SIZE}"
            log_error "  --img-size: ${IMG_SIZE}"
            log_error "  --device: ${DEVICE}"
            
            # Capture and log the Python error output
            if [ -f "${TRAINING_OUTPUT_FILE}" ] && [ -s "${TRAINING_OUTPUT_FILE}" ]; then
                log_error "=== Python Error Output ==="
                # Log the last 100 lines which typically contain the error
                tail -n 100 "${TRAINING_OUTPUT_FILE}" >> "${CURRENT_ERROR_LOG}"
                log_error "=== End of Error Output ==="
            fi
        fi
    fi
    
    # Clean up temp file
    rm -f "${TRAINING_OUTPUT_FILE}"
    
    # Stop monitors for this run and generate summary
    if [ "$LOGGING_ENABLED" = true ]; then
        stop_all_monitors
        generate_log_summary
    fi
    
    echo ""
    
    # Rest between runs to let GPU cool down (only if multiple combinations)
    if [ ${CURRENT_RUN} -lt ${TOTAL_COMBINATIONS} ] && [ ${TOTAL_COMBINATIONS} -gt 1 ] && [ ${REST_TIME_PER_RUN} -gt 0 ]; then
        # Convert seconds to human-readable format
        REST_MINUTES=$((REST_TIME_PER_RUN / 60))
        REST_SECONDS=$((REST_TIME_PER_RUN % 60))
        if [ ${REST_MINUTES} -gt 0 ]; then
            print_info "Resting for ${REST_MINUTES} minute(s) ${REST_SECONDS} second(s) before next run (GPU cooldown)..."
        else
            print_info "Resting for ${REST_SECONDS} second(s) before next run (GPU cooldown)..."
        fi
        sleep ${REST_TIME_PER_RUN}
    fi

done  # GRAYSCALE
done  # OPTIMIZER
done  # LR0
done  # IMG_SIZE
done  # BATCH_SIZE
done  # EPOCHS
done  # YOLO_MODEL
done  # DATASET_NAME

#===============================================================================
# TRAINING SUMMARY
#===============================================================================

echo ""
print_header "Training Complete - Summary (Server AMD ROCm)"

echo "Total combinations: ${TOTAL_COMBINATIONS}"
echo "GPU: AMD Instinct MI210 (64GB HBM2e)"
echo ""

if [ ${#SKIPPED_RUNS[@]} -gt 0 ]; then
    print_warning "Skipped - already trained (${#SKIPPED_RUNS[@]}):"
    for run in "${SKIPPED_RUNS[@]}"; do
        echo "  ⊘ ${run}"
    done
    echo ""
fi

if [ ${#SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    print_success "Successful (${#SUCCESSFUL_RUNS[@]}):"
    for run in "${SUCCESSFUL_RUNS[@]}"; do
        echo "  ✓ ${run}"
    done
    echo ""
fi

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    print_error "Failed (${#FAILED_RUNS[@]}):"
    for run in "${FAILED_RUNS[@]}"; do
        echo "  ✗ ${run}"
    done
    echo ""
fi

print_info "All trained models saved to: ${OUTPUT_DIR}"
print_info "Copies also saved to each dataset's 'logs/' and 'trained_models_output/' folders"
echo ""
print_info "To compare models, check the training_stats.json in each experiment folder."
echo ""
print_info "Experiment naming format: {dataset}_{model}_e{epochs}_b{batch}_img{size}_lr{lr0}_{optimizer}_{rgb|gray}_{timestamp}"
echo ""
