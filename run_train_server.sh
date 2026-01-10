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

# Source training functions from modules/training/ (DRY principle)
if [ -f "${SCRIPT_DIR}/modules/training/training_functions.sh" ]; then
    source "${SCRIPT_DIR}/modules/training/training_functions.sh"
else
    echo "ERROR: modules/training/training_functions.sh not found. Please ensure it exists."
    exit 1
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
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation

#===============================================================================
# CONFIG SCRIPTS SELECTION
#===============================================================================
# List of config scripts to run sequentially. Each config contains all training
# parameters (YOLO_MODELS, EPOCHS_LIST, BATCH_SIZE_LIST, etc.)
# 
# Comment/uncomment configs to include/exclude them from training.
# Training will loop through each enabled config and run all its combinations.
#
# Config scripts are located in: config/server/
# Each config should define all required parameter arrays.

CONFIG_DIR="${SCRIPT_DIR}/config/server"

CONFIG_SCRIPTS=(
    # Uncomment the configs you want to run:
    "01_yolo_combination.sh"           # YOLO model + color mode combinations
    #"02_color_combination.sh"                # Color mode variations (RGB, grayscale)
    "03_img_size_combination_a.sh"              # Image size variations
    "03_img_size_combination_b.sh"
    #"04_class_balancing_combination.sh"      # Class balancing strategies
    #"05_optimizer_combination.sh"            # Optimizer variations
    #"z_epoch_combination.sh"                # Epoch variations
)

# Device Configuration (applies to all configs)
# ─────────────────────────────────────────────────────────────────────────────
# For AMD ROCm: Use HIP_VISIBLE_DEVICES environment variable (set above)
# Device 0 is the first GPU (MI210)
# For multi-GPU: DEVICE="0,1" (requires larger batch size)
DEVICE=0                    # GPU device: 0, 1, 2... or "cpu" or "0,1" for multi-GPU

# Resume training from last checkpoint (applies to all configs)
RESUME=false

# Distribution file path (relative to dataset directory)
# This file contains class distribution statistics for dynamic fold calculation
CLASS_DISTRIBUTION_FILE="Distribution/distribution.txt"

#===============================================================================
# ENVIRONMENT TYPE
#===============================================================================
# Identifies which environment this script runs on for output organization
# Used to organize outputs into: trained_models_output/server/<config_type>/
ENVIRONMENT_TYPE="server"

#===============================================================================
# PATHS (Using common configuration - DRY principle)
#===============================================================================

# Base paths (dataset-independent)
WEIGHTS_DIR="${SCRIPT_DIR}/${COMMON_WEIGHTS_SUBDIR}"
# Base output directory - will be further organized by environment and config type
OUTPUT_BASE_DIR="${SCRIPT_DIR}/${COMMON_TRAINED_MODELS_SUBDIR}"

# Create directories using common function
ensure_dir "${WEIGHTS_DIR}"
ensure_dir "${OUTPUT_BASE_DIR}"

# Note: DATA_YAML, YOLO_MODEL_PATH, MODEL_NAME, and EXP_NAME are now set 
# dynamically within the training loop to support multiple datasets

#===============================================================================
# CONDA ENVIRONMENT ACTIVATION
#===============================================================================

print_header "Microspore Phenotyping - YOLO Training (Server AMD ROCm - Config-Based)"

echo "Server Configuration:"
echo "  - GPU: AMD Instinct MI210 (64GB HBM2e)"
echo "  - Architecture: gfx90a (CDNA2)"
echo "  - Compute: ROCm 6.x"
echo ""

# Validate config scripts exist
echo "Config scripts to run:"
VALID_CONFIGS=()
for config in "${CONFIG_SCRIPTS[@]}"; do
    config_path="${CONFIG_DIR}/${config}"
    if [ -f "${config_path}" ]; then
        echo "  ✓ ${config}"
        VALID_CONFIGS+=("${config}")
    else
        echo "  ✗ ${config} (NOT FOUND)"
    fi
done
echo ""

if [ ${#VALID_CONFIGS[@]} -eq 0 ]; then
    print_error "No valid config scripts found. Please check CONFIG_SCRIPTS in the script."
    exit 1
fi

print_info "Total configs to process: ${#VALID_CONFIGS[@]}"
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

# Note: OUTPUT_BASE_DIR is already set above, used for clearing
# Actual OUTPUT_DIR will be set per-config inside the training loop

if [ "$CLEAR_OUTPUT_ON_START" = true ]; then
    if [ -d "${OUTPUT_BASE_DIR}" ] && [ "$(ls -A ${OUTPUT_BASE_DIR} 2>/dev/null)" ]; then
        # Count existing output directories
        OUTPUT_COUNT=$(find "${OUTPUT_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        
        if [ "$OUTPUT_COUNT" -gt 0 ]; then
            print_warning "CLEAR_OUTPUT_ON_START is enabled. Clearing ${OUTPUT_COUNT} existing trained model(s)..."
            rm -rf "${OUTPUT_BASE_DIR:?}"/*
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
# CLASS FOCUS HELPER FUNCTIONS
# Note: Core functions moved to modules/training/training_functions.sh
# The following functions are sourced from there:
#   - parse_distribution_file()
#   - calculate_class_weights()
#   - display_class_focus_info()
#   - calculate_total_combinations()
#   - generate_exp_name()
#   - check_training_exists()
#   - get_color_mode_str()
#   - get_balance_str()
#   - format_lr_for_filename()
#===============================================================================

#===============================================================================
# MAIN CONFIG LOOP - Process each config script sequentially
#===============================================================================

GLOBAL_SUCCESSFUL_RUNS=()
GLOBAL_FAILED_RUNS=()
GLOBAL_SKIPPED_RUNS=()
CONFIG_INDEX=0

for CURRENT_CONFIG in "${VALID_CONFIGS[@]}"; do
    CONFIG_INDEX=$((CONFIG_INDEX + 1))
    CONFIG_PATH="${CONFIG_DIR}/${CURRENT_CONFIG}"
    
    # Extract config combination name for output organization
    # e.g., "01_yolo_color_combination.sh" -> "01_yolo_color_combination"
    CONFIG_COMBINATION_NAME=$(get_config_combination_name "${CURRENT_CONFIG}")
    
    # Set OUTPUT_DIR for this config: trained_models_output/server/01_yolo_color_combination/
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${ENVIRONMENT_TYPE}/${CONFIG_COMBINATION_NAME}"
    ensure_dir "${OUTPUT_DIR}"
    
    print_header "Config ${CONFIG_INDEX}/${#VALID_CONFIGS[@]}: ${CURRENT_CONFIG}"
    print_info "Output directory: ${OUTPUT_DIR}"
    
    # Source the config script to load all parameter arrays
    print_info "Loading config: ${CONFIG_PATH}"
    source "${CONFIG_PATH}"
    
    # Set default values from arrays (first element of each array)
    YOLO_MODEL="${YOLO_MODELS[0]}"
    DEFAULT_DATASET="${DATASET_LIST[0]}"
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
    COLOR_MODE="${COLOR_MODE_LIST[0]}"
    PRETRAINED="${PRETRAINED_LIST[0]}"
    CACHE="${CACHE_LIST[0]}"
    AMP="${AMP_LIST[0]}"
    FREEZE="${FREEZE_LIST[0]}"
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
    CLASS_FOCUS_MODE="${CLASS_FOCUS_MODE_LIST[0]}"
    CLASS_FOCUS_CLASSES="${CLASS_FOCUS_CLASSES_LIST[0]}"
    CLASS_FOCUS_FOLD="${CLASS_FOCUS_FOLD_LIST[0]}"
    CLASS_FOCUS_TARGET="${CLASS_FOCUS_TARGET_LIST[0]}"
    
    # Display models from this config
    echo ""
    echo "Models in this config:"
    for model in "${YOLO_MODELS[@]}"; do
        echo "  - ${model}"
    done
    echo ""

TOTAL_COMBINATIONS=$(calculate_total_combinations)

print_info "Parameter Grid Summary:"
echo "  - Datasets:    ${#DATASET_LIST[@]} (${DATASET_LIST[*]})"
echo "  - Models:      ${#YOLO_MODELS[@]} (${YOLO_MODELS[*]})"
echo "  - Epochs:      ${#EPOCHS_LIST[@]} (${EPOCHS_LIST[*]})"
echo "  - Batch Sizes: ${#BATCH_SIZE_LIST[@]} (${BATCH_SIZE_LIST[*]})"
echo "  - Image Sizes: ${#IMG_SIZE_LIST[@]} (${IMG_SIZE_LIST[*]})"
echo "  - LR0 values:  ${#LR0_LIST[@]} (${LR0_LIST[*]})"
echo "  - Optimizers:  ${#OPTIMIZER_LIST[@]} (${OPTIMIZER_LIST[*]})"
echo "  - Color Mode:  ${#COLOR_MODE_LIST[@]} (${COLOR_MODE_LIST[*]})"
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

# Nested loops for all parameter combinations
# Primary loop parameters (commonly varied for grid search)
for DATASET_NAME in "${DATASET_LIST[@]}"; do
for YOLO_MODEL in "${YOLO_MODELS[@]}"; do
for EPOCHS in "${EPOCHS_LIST[@]}"; do
for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
for IMG_SIZE in "${IMG_SIZE_LIST[@]}"; do
for LR0 in "${LR0_LIST[@]}"; do
for OPTIMIZER in "${OPTIMIZER_LIST[@]}"; do
for COLOR_MODE in "${COLOR_MODE_LIST[@]}"; do
for CLASS_FOCUS_MODE in "${CLASS_FOCUS_MODE_LIST[@]}"; do

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
    
    # Class focus parameters (CLASS_FOCUS_MODE is from the loop)
    CLASS_FOCUS_CLASSES="${CLASS_FOCUS_CLASSES_LIST[0]}"
    CLASS_FOCUS_FOLD="${CLASS_FOCUS_FOLD_LIST[0]}"
    CLASS_FOCUS_TARGET="${CLASS_FOCUS_TARGET_LIST[0]}"
    
    # Calculate class weights based on distribution file
    DIST_FILE="${DATASET_PATH}/${CLASS_DISTRIBUTION_FILE}"
    CLASS_WEIGHTS_JSON=$(calculate_class_weights "${DIST_FILE}" "${CLASS_FOCUS_MODE}" "${CLASS_FOCUS_CLASSES}" "${CLASS_FOCUS_FOLD}" "${CLASS_FOCUS_TARGET}")
    
    # Check if model exists locally, otherwise will be downloaded
    if [ -f "${WEIGHTS_DIR}/${YOLO_MODEL}" ]; then
        YOLO_MODEL_PATH="${WEIGHTS_DIR}/${YOLO_MODEL}"
    else
        YOLO_MODEL_PATH="${YOLO_MODEL}"
    fi
    
    # Extract model name without extension for naming
    MODEL_NAME=$(basename "${YOLO_MODEL}" .pt)
    
    # Use shared utility functions for consistent naming (from training_functions.sh)
    GRAY_DISPLAY=$(get_color_mode_str "${COLOR_MODE}")
    LR0_DISPLAY=$(format_lr_for_filename "${LR0}")
    BALANCE_DISPLAY=$(get_balance_str "${CLASS_FOCUS_MODE}")
    
    # Create run identifier for tracking using unified naming scheme
    # Format: {dataset}_{model}_{color}_img{size}_{balance}_{optimizer}_e{epochs}_b{batch}_lr{lr0}
    RUN_ID="${DATASET_NAME}_${MODEL_NAME}_${GRAY_DISPLAY}_img${IMG_SIZE}_${BALANCE_DISPLAY}_${OPTIMIZER}_e${EPOCHS}_b${BATCH_SIZE}_lr${LR0_DISPLAY}"
    
    #===========================================================================
    # CHECK IF TRAINING ALREADY EXISTS (SKIP IF ENABLED)
    #===========================================================================
    if [ "$SKIP_EXISTING" = true ]; then
        # Use || true to prevent set -e from exiting when no match found (returns 1)
        # Pass OUTPUT_DIR as the 10th parameter for the shared function
        existing_dir=$(check_training_exists "$DATASET_NAME" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$IMG_SIZE" "$LR0" "$OPTIMIZER" "$COLOR_MODE" "$CLASS_FOCUS_MODE" "$OUTPUT_DIR") || true
        if [ -n "$existing_dir" ]; then
            print_warning "Skipping ${RUN_ID} - already trained: $(basename "$existing_dir")"
            SKIPPED_RUNS+=("${RUN_ID}")
            continue
        fi
    fi
    
    # Generate experiment name with timestamp and parameters
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXP_NAME=$(generate_exp_name "$DATASET_NAME" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$IMG_SIZE" "$LR0" "$OPTIMIZER" "$COLOR_MODE" "$TIMESTAMP" "$CLASS_FOCUS_MODE")
    
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
    echo "  - Color Mode: ${COLOR_MODE}"
    echo "  - GPU:        AMD Instinct MI210 (ROCm)"
    echo "  - Output:     ${OUTPUT_DIR}/${EXP_NAME}"
    echo ""
    
    # Display class focus configuration
    display_class_focus_info "${CLASS_FOCUS_MODE}" "${CLASS_FOCUS_CLASSES}" "${CLASS_FOCUS_FOLD}" "${CLASS_FOCUS_TARGET}" "${CLASS_WEIGHTS_JSON}"
    echo ""
    
    # Run training using the modules/training/train.py script
    # Capture both stdout and stderr for comprehensive logging
    # Use the logging module's training output file if available, otherwise use temp file
    if [ "$LOGGING_ENABLED" = true ] && [ -n "${CURRENT_TRAINING_OUTPUT:-}" ]; then
        TRAINING_OUTPUT_FILE="${CURRENT_TRAINING_OUTPUT}"
    else
        TRAINING_OUTPUT_FILE=$(mktemp)
    fi
    
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
        --grayscale ${COLOR_MODE} \
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
        --class-focus-mode "${CLASS_FOCUS_MODE}" \
        --class-weights "${CLASS_WEIGHTS_JSON}" \
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
    
    # Clean up temp file only if not using the logging module's file
    if [ "$LOGGING_ENABLED" != true ] || [ -z "${CURRENT_TRAINING_OUTPUT:-}" ]; then
        rm -f "${TRAINING_OUTPUT_FILE}"
    fi
    
    # Append training output to the main full log, then stop monitors
    if [ "$LOGGING_ENABLED" = true ]; then
        append_training_output_to_full_log
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

done  # CLASS_FOCUS_MODE
done  # COLOR_MODE
done  # OPTIMIZER
done  # LR0
done  # IMG_SIZE
done  # BATCH_SIZE
done  # EPOCHS
done  # YOLO_MODEL
done  # DATASET_NAME

# Accumulate results from this config to global results
GLOBAL_SUCCESSFUL_RUNS+=("${SUCCESSFUL_RUNS[@]}")
GLOBAL_FAILED_RUNS+=("${FAILED_RUNS[@]}")
GLOBAL_SKIPPED_RUNS+=("${SKIPPED_RUNS[@]}")

# Config-level summary
echo ""
print_subheader "Config '${CURRENT_CONFIG}' Summary"
echo "  Combinations: ${TOTAL_COMBINATIONS}"
echo "  Successful:   ${#SUCCESSFUL_RUNS[@]}"
echo "  Failed:       ${#FAILED_RUNS[@]}"
echo "  Skipped:      ${#SKIPPED_RUNS[@]}"
echo ""

done  # CURRENT_CONFIG (config loop)

#===============================================================================
# TRAINING SUMMARY (All Configs)
#===============================================================================

echo ""
print_header "Training Complete - Final Summary (Server AMD ROCm - All Configs)"

echo "Total configs processed: ${#VALID_CONFIGS[@]}"
for config in "${VALID_CONFIGS[@]}"; do
    config_name=$(get_config_combination_name "${config}")
    echo "  - ${config} -> ${ENVIRONMENT_TYPE}/${config_name}/"
done
echo ""
echo "GPU: AMD Instinct MI210 (64GB HBM2e)"
echo ""

if [ ${#GLOBAL_SKIPPED_RUNS[@]} -gt 0 ]; then
    print_warning "Skipped - already trained (${#GLOBAL_SKIPPED_RUNS[@]}):"
    for run in "${GLOBAL_SKIPPED_RUNS[@]}"; do
        echo "  ⊘ ${run}"
    done
    echo ""
fi

if [ ${#GLOBAL_SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    print_success "Successful (${#GLOBAL_SUCCESSFUL_RUNS[@]}):"
    for run in "${GLOBAL_SUCCESSFUL_RUNS[@]}"; do
        echo "  ✓ ${run}"
    done
    echo ""
fi

if [ ${#GLOBAL_FAILED_RUNS[@]} -gt 0 ]; then
    print_error "Failed (${#GLOBAL_FAILED_RUNS[@]}):"
    for run in "${GLOBAL_FAILED_RUNS[@]}"; do
        echo "  ✗ ${run}"
    done
    echo ""
fi

print_info "Output directory structure: ${OUTPUT_BASE_DIR}/${ENVIRONMENT_TYPE}/<config_type>/"
print_info "Copies also saved to each dataset's 'logs/' and 'trained_models_output/' folders"
print_info "Logs included in each training output folder"
echo ""
print_info "To compare models, check the training_stats.json in each experiment folder."
echo ""
print_info "Experiment naming format: {dataset}_{model}_{rgb|gray}_img{size}_{balance}_{optimizer}_e{epochs}_b{batch}_lr{lr0}_{timestamp}"
echo ""
