#!/bin/bash
#===============================================================================
# Microspore Phenotyping - Training Functions Module
# Shared training functions used across local, server, and workstation scripts
#
# Location: modules/training/training_functions.sh
# Source:   source "${SCRIPT_DIR}/modules/training/training_functions.sh"
#
# This module provides:
#   - Experiment naming functions (unified scheme)
#   - Training existence check functions
#   - Class focus/balancing helper functions
#   - Combination calculation functions
#===============================================================================

#===============================================================================
# NAMING SCHEME CONFIGURATION
#===============================================================================
# Unified naming scheme for experiment outputs:
# Format: {dataset}_{model}_{color}_img{size}_{optimizer}_{balance}_e{epochs}_b{batch}_lr{lr0}_{timestamp}
#
# Example: Dataset_1_yolov8n_rgb_img640_auto_bal-none_e100_b16_lr0_001_20260110_143022
#
# This format prioritizes:
# 1. Dataset identification
# 2. Model architecture
# 3. Key differentiating parameters (color, img size, optimizer, balance)
# 4. Training parameters (epochs, batch, learning rate)
# 5. Timestamp for uniqueness

#===============================================================================
# COLOR MODE UTILITIES
#===============================================================================

# Convert color mode to short string for filenames
# Usage: get_color_mode_str "grayscale" -> "gray"
get_color_mode_str() {
    local color_mode="$1"
    if [ "$color_mode" = "grayscale" ]; then
        echo "gray"
    else
        echo "rgb"
    fi
}

#===============================================================================
# CLASS FOCUS UTILITIES
#===============================================================================

# Convert class focus mode to short string for filenames
# Usage: get_balance_str "auto" -> "bal-auto"
get_balance_str() {
    local class_focus_mode="$1"
    case "$class_focus_mode" in
        "auto") echo "bal-auto" ;;
        "sqrt") echo "bal-sqrt" ;;
        "manual") echo "bal-manual" ;;
        *) echo "bal-none" ;;
    esac
}

# Format learning rate for filename (replace decimal point with underscore)
# Usage: format_lr_for_filename "0.001" -> "0_001"
format_lr_for_filename() {
    local lr="$1"
    echo "$lr" | sed 's/\./_/g'
}

#===============================================================================
# EXPERIMENT NAME GENERATION
#===============================================================================

# Generate unified experiment name based on current parameters
# This function uses a consistent naming scheme across all environments
#
# Usage: generate_exp_name "dataset" "model" "epochs" "batch" "img_size" "lr0" "optimizer" "color_mode" "timestamp" "class_focus_mode"
#
# Output format: {dataset}_{model}_{color}_img{size}_{optimizer}_{balance}_e{epochs}_b{batch}_lr{lr0}_{timestamp}
generate_exp_name() {
    local dataset_name="$1"
    local model_name="$2"
    local epochs="$3"
    local batch_size="$4"
    local img_size="$5"
    local lr0="$6"
    local optimizer="$7"
    local color_mode="$8"
    local timestamp="$9"
    local class_focus_mode="${10}"
    
    # Format LR0 for filename (remove decimal point)
    local lr0_str
    lr0_str=$(format_lr_for_filename "${lr0}")
    
    # Get color mode indicator
    local color_str
    color_str=$(get_color_mode_str "${color_mode}")
    
    # Get class focus mode indicator
    local balance_str
    balance_str=$(get_balance_str "${class_focus_mode}")
    
    # Unified naming format: {dataset}_{model}_{color}_img{size}_{optimizer}_{balance}_e{epochs}_b{batch}_lr{lr0}_{timestamp}
    echo "${dataset_name}_${model_name}_${color_str}_img${img_size}_${optimizer}_${balance_str}_e${epochs}_b${batch_size}_lr${lr0_str}_${timestamp}"
}

#===============================================================================
# TRAINING EXISTENCE CHECK
#===============================================================================

# Check if training already completed for given parameters
# Returns 0 (true) if exists, 1 (false) if not
#
# Usage: check_training_exists "dataset" "model" "epochs" "batch" "img_size" "lr0" "optimizer" "color_mode" "class_focus_mode" "output_dir"
#
# Note: This uses the unified naming scheme pattern
check_training_exists() {
    local dataset_name="$1"
    local model_name="$2"
    local epochs="$3"
    local batch_size="$4"
    local img_size="$5"
    local lr0="$6"
    local optimizer="$7"
    local color_mode="$8"
    local class_focus_mode="$9"
    local output_dir="${10}"
    
    # Format LR0 for filename (remove decimal point)
    local lr0_str
    lr0_str=$(format_lr_for_filename "${lr0}")
    
    # Get color mode indicator
    local color_str
    color_str=$(get_color_mode_str "${color_mode}")
    
    # Get class focus mode indicator
    local balance_str
    balance_str=$(get_balance_str "${class_focus_mode}")
    
    # Build pattern to match using unified naming scheme:
    # {dataset}_{model}_{color}_img{size}_{optimizer}_{balance}_e{epochs}_b{batch}_lr{lr0}_*
    local pattern="${dataset_name}_${model_name}_${color_str}_img${img_size}_${optimizer}_${balance_str}_e${epochs}_b${batch_size}_lr${lr0_str}_"
    
    # Look for matching directories with completed training
    # Use find for more reliable file detection (works better in WSL/cross-platform)
    if [ -d "${output_dir}" ]; then
        while IFS= read -r dir; do
            # Check if this directory has a completed best.pt weight file
            if [ -d "${dir}/weights" ]; then
                # Use find to check for *_best.pt files (more reliable than ls glob)
                if find "${dir}/weights" -maxdepth 1 -name "*_best.pt" -type f 2>/dev/null | grep -q .; then
                    echo "$dir"
                    return 0
                fi
            fi
        done < <(find "${output_dir}" -maxdepth 1 -type d -name "${pattern}*" 2>/dev/null)
    fi
    return 1
}

#===============================================================================
# CLASS FOCUS HELPER FUNCTIONS
#===============================================================================

# Parse distribution.txt and extract class counts for Train set
# Returns associative array-like output: "class_name:count" lines
#
# Usage: parse_distribution_file "/path/to/distribution.txt"
parse_distribution_file() {
    local dist_file="$1"
    
    if [ ! -f "${dist_file}" ]; then
        echo ""
        return 1
    fi
    
    # Parse the sorted distribution section to get Train counts
    # Format: Rank   Class                     Train       Test      Total     Pooled       Temp
    awk '
    /^[0-9]+[[:space:]]+[a-zA-Z_]+[[:space:]]+[0-9]+/ {
        # Skip header lines, extract class name and train count
        class_name = $2
        train_count = $3
        if (train_count ~ /^[0-9]+$/) {
            print class_name ":" train_count
        }
    }
    ' "${dist_file}"
}

# Calculate class weights based on distribution and focus mode
# Arguments: dist_file, mode, focus_classes, max_fold, target
# Outputs: JSON-like string for Python consumption
#
# Usage: calculate_class_weights "/path/to/distribution.txt" "auto" "all" "10" "median"
calculate_class_weights() {
    local dist_file="$1"
    local mode="$2"
    local focus_classes="$3"
    local max_fold="$4"
    local target="$5"
    
    if [ "$mode" = "none" ] || [ ! -f "${dist_file}" ]; then
        echo "{}"
        return
    fi
    
    # Parse distribution file
    local class_counts
    class_counts=$(parse_distribution_file "${dist_file}")
    
    if [ -z "${class_counts}" ]; then
        echo "{}"
        return
    fi
    
    # Convert to Python and calculate weights
    python3 << PYTHON_SCRIPT
import sys

# Parse class counts
class_counts = {}
for line in """${class_counts}""".strip().split('\n'):
    if ':' in line:
        name, count = line.split(':')
        class_counts[name.strip()] = int(count.strip())

if not class_counts:
    print("{}")
    sys.exit(0)

mode = "${mode}"
focus_classes_str = "${focus_classes}"
max_fold = float("${max_fold}")
target = "${target}"

# Parse focus classes
if focus_classes_str.lower() == "all":
    focus_classes = list(class_counts.keys())
else:
    focus_classes = [c.strip() for c in focus_classes_str.split(',')]

# Calculate target count based on mode
counts = list(class_counts.values())
if target == "max":
    target_count = max(counts)
elif target == "median":
    sorted_counts = sorted(counts)
    mid = len(sorted_counts) // 2
    target_count = sorted_counts[mid] if len(sorted_counts) % 2 else (sorted_counts[mid-1] + sorted_counts[mid]) // 2
else:  # mean
    target_count = sum(counts) // len(counts)

# Calculate weights
weights = {}
for cls, count in class_counts.items():
    if mode == "manual":
        # Apply fold only to specified classes
        if cls in focus_classes:
            weights[cls] = min(max_fold, max(1.0, target_count / count if count > 0 else 1.0))
        else:
            weights[cls] = 1.0
    elif mode == "auto":
        # Auto-balance all classes towards target
        if count > 0:
            fold = target_count / count
            weights[cls] = min(max_fold, max(1.0, fold))
        else:
            weights[cls] = 1.0
    elif mode == "sqrt":
        # Square root balancing (softer)
        if count > 0:
            fold = (target_count / count) ** 0.5
            weights[cls] = min(max_fold, max(1.0, fold))
        else:
            weights[cls] = 1.0
    else:
        weights[cls] = 1.0

# Output as JSON-like string
import json
print(json.dumps(weights))
PYTHON_SCRIPT
}

# Display class focus configuration in a formatted way
#
# Usage: display_class_focus_info "auto" "all" "10" "median" '{"class1": 2.0}'
display_class_focus_info() {
    local mode="$1"
    local focus_classes="$2"
    local fold="$3"
    local target="$4"
    local weights_json="$5"
    
    echo "  Class Focus Configuration:"
    echo "    - Mode:           ${mode}"
    
    if [ "$mode" != "none" ]; then
        echo "    - Target Classes: ${focus_classes}"
        echo "    - Max Fold:       ${fold}x"
        echo "    - Balance Target: ${target}"
        
        if [ -n "${weights_json}" ] && [ "${weights_json}" != "{}" ]; then
            echo "    - Calculated Weights:"
            echo "${weights_json}" | python3 -c "
import sys, json
weights = json.load(sys.stdin)
for cls, weight in sorted(weights.items(), key=lambda x: -x[1]):
    if weight > 1.0:
        print(f'        {cls}: {weight:.2f}x (boosted)')
    else:
        print(f'        {cls}: {weight:.2f}x')
" 2>/dev/null || echo "        (Unable to parse weights)"
        fi
    fi
}

#===============================================================================
# COMBINATION CALCULATION
#===============================================================================

# Calculate total number of training combinations from parameter arrays
# Requires these arrays to be defined: DATASET_LIST, YOLO_MODELS, EPOCHS_LIST,
# BATCH_SIZE_LIST, IMG_SIZE_LIST, LR0_LIST, OPTIMIZER_LIST, COLOR_MODE_LIST,
# CLASS_FOCUS_MODE_LIST
#
# Usage: total=$(calculate_total_combinations)
calculate_total_combinations() {
    local total=1
    total=$((total * ${#DATASET_LIST[@]}))
    total=$((total * ${#YOLO_MODELS[@]}))
    total=$((total * ${#EPOCHS_LIST[@]}))
    total=$((total * ${#BATCH_SIZE_LIST[@]}))
    total=$((total * ${#IMG_SIZE_LIST[@]}))
    total=$((total * ${#LR0_LIST[@]}))
    total=$((total * ${#OPTIMIZER_LIST[@]}))
    total=$((total * ${#COLOR_MODE_LIST[@]}))
    total=$((total * ${#CLASS_FOCUS_MODE_LIST[@]}))
    echo $total
}

#===============================================================================
# LOG AND OUTPUT MANAGEMENT
#===============================================================================

# Clear logs directory if enabled
# Usage: clear_logs_if_enabled "true" "/path/to/logs"
clear_logs_if_enabled() {
    local clear_enabled="$1"
    local logs_dir="$2"
    
    if [ "$clear_enabled" = true ]; then
        if [ -d "${logs_dir}" ] && [ "$(ls -A ${logs_dir} 2>/dev/null)" ]; then
            local LOG_COUNT
            LOG_COUNT=$(find "${logs_dir}" -mindepth 1 -maxdepth 1 | wc -l)
            if [ "$LOG_COUNT" -gt 0 ]; then
                print_warning "CLEAR_LOGS_ON_START is enabled. Clearing ${LOG_COUNT} existing log(s)..."
                rm -rf "${logs_dir:?}"/*
                print_success "Old logs cleared successfully."
            fi
        else
            print_info "No existing logs to clear."
        fi
    else
        print_info "CLEAR_LOGS_ON_START is disabled. Old logs will be preserved."
    fi
}

# Clear output directory if enabled
# Usage: clear_output_if_enabled "true" "/path/to/output"
clear_output_if_enabled() {
    local clear_enabled="$1"
    local output_base_dir="$2"
    
    if [ "$clear_enabled" = true ]; then
        if [ -d "${output_base_dir}" ] && [ "$(ls -A ${output_base_dir} 2>/dev/null)" ]; then
            local OUTPUT_COUNT
            OUTPUT_COUNT=$(find "${output_base_dir}" -mindepth 1 -maxdepth 1 | wc -l)
            if [ "$OUTPUT_COUNT" -gt 0 ]; then
                print_warning "CLEAR_OUTPUT_ON_START is enabled. Clearing ${OUTPUT_COUNT} existing trained model(s)..."
                rm -rf "${output_base_dir:?}"/*
                print_success "Old trained models cleared successfully."
            fi
        else
            print_info "No existing trained models to clear."
        fi
    else
        print_info "CLEAR_OUTPUT_ON_START is disabled. Existing trained models will be preserved."
    fi
}

#===============================================================================
# PARAMETER GRID DISPLAY
#===============================================================================

# Display parameter grid summary in a formatted way
#
# Usage: display_parameter_grid_summary
display_parameter_grid_summary() {
    local total_combinations
    total_combinations=$(calculate_total_combinations)
    
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
    print_warning "Total combinations to train: ${total_combinations}"
    echo ""
    
    echo $total_combinations
}

#===============================================================================
# CONFIG VALIDATION
#===============================================================================

# Validate config scripts and return list of valid configs
# Usage: validate_config_scripts CONFIG_SCRIPTS[@] config_dir
validate_config_scripts() {
    local -n _config_scripts=$1  # nameref to array
    local config_dir="$2"
    
    echo "Config scripts to run:"
    VALID_CONFIGS=()
    for config in "${_config_scripts[@]}"; do
        local config_path="${config_dir}/${config}"
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
        return 1
    fi
    
    print_info "Total configs to process: ${#VALID_CONFIGS[@]}"
    echo ""
    return 0
}

#===============================================================================
# SET DEFAULT VALUES FROM CONFIG
#===============================================================================

# Set default values from config arrays (first element of each array)
# This function expects all parameter arrays to be defined
#
# Usage: set_defaults_from_config
set_defaults_from_config() {
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
}

#===============================================================================
# TRAINING SUMMARY DISPLAY
#===============================================================================

# Display config-level summary
# Usage: display_config_summary "config_name" total successful_count failed_count skipped_count
display_config_summary() {
    local config_name="$1"
    local total="$2"
    local successful="$3"
    local failed="$4"
    local skipped="$5"
    
    echo ""
    print_subheader "Config '${config_name}' Summary"
    echo "  Combinations: ${total}"
    echo "  Successful:   ${successful}"
    echo "  Failed:       ${failed}"
    echo "  Skipped:      ${skipped}"
    echo ""
}

# Display final training summary for all configs
# Usage: display_final_summary env_type valid_configs[@] skipped[@] successful[@] failed[@] output_base_dir
display_final_summary() {
    local env_type="$1"
    local output_base_dir="$2"
    shift 2
    
    echo ""
    print_header "Training Complete - Final Summary (All Configs)"
    
    print_info "Output directory structure: ${output_base_dir}/${env_type}/<config_type>/"
    print_info "Copies also saved to each dataset's 'logs/' and 'trained_models_output/' folders"
    print_info "Logs included in each training output folder"
    echo ""
    print_info "To compare models, check the training_stats.json in each experiment folder."
    echo ""
    print_info "Experiment naming format: {dataset}_{model}_{rgb|gray}_img{size}_{optimizer}_{balance}_e{epochs}_b{batch}_lr{lr0}_{timestamp}"
    echo ""
}
