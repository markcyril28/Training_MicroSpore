#!/bin/bash
#===============================================================================
# Microspore Phenotyping - Common Shell Functions
# Shared configuration and utilities used across all shell scripts
#
# Location: modules/config/common_functions.sh
# Source:   source "${BASE_DIR}/modules/config/common_functions.sh"
#===============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

#===============================================================================
# SHARED CONFIGURATION
#===============================================================================
# These values are used across all scripts for consistency

# Conda environment name
COMMON_ENV_NAME="train"

# Python version
COMMON_PYTHON_VERSION="3.10"

# Directory structure
get_script_dir() {
    echo "$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
}

# Weights directory (relative to script dir)
COMMON_WEIGHTS_SUBDIR="modules/yolo_models_weights"

# Trained models directory (relative to script dir)
COMMON_TRAINED_MODELS_SUBDIR="trained_models_output"

# Datasets parent directory (contains all dataset folders)
COMMON_DATASETS_DIR="TRAINING_WD"

# Default dataset name (can be overridden)
COMMON_DATASET_NAME="Dataset_1"

# Full dataset path (default - can be overridden)
COMMON_DATASET_DIR="${COMMON_DATASETS_DIR}/${COMMON_DATASET_NAME}"

#===============================================================================
# PRINT FUNCTIONS
#===============================================================================

print_header() {
    echo -e "${BLUE}=============================================="
    echo -e "  $1"
    echo -e "==============================================${NC}"
    echo ""
}

print_subheader() {
    echo -e "${CYAN}----------------------------------------------"
    echo -e "  $1"
    echo -e "----------------------------------------------${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_debug() {
    if [ "${DEBUG:-false}" = "true" ]; then
        echo -e "${MAGENTA}[DEBUG] $1${NC}"
    fi
}

#===============================================================================
# CONDA UTILITIES
#===============================================================================

# Initialize conda for the current shell
init_conda_shell() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        echo ""
        echo "Please install Miniconda first:"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh"
        echo "  source ~/.bashrc"
        return 1
    fi
    
    eval "$(conda shell.bash hook)"
    print_success "Conda initialized"
    return 0
}

# Check if conda environment exists
check_conda_env_exists() {
    local env_name="${1:-$COMMON_ENV_NAME}"
    if conda env list | grep -q "^${env_name} "; then
        return 0
    else
        return 1
    fi
}

# Activate conda environment
activate_conda_env() {
    local env_name="${1:-$COMMON_ENV_NAME}"
    
    if ! check_conda_env_exists "$env_name"; then
        print_error "Conda environment '${env_name}' not found"
        return 1
    fi
    
    conda activate "$env_name"
    print_success "Activated conda environment: ${env_name}"
    return 0
}

#===============================================================================
# GPU UTILITIES
#===============================================================================

# Check if NVIDIA GPU is available
check_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Get GPU information as variables
get_gpu_info() {
    if check_nvidia_gpu; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        GPU_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        GPU_CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null || echo "Unknown")
        GPU_AVAILABLE=true
    else
        GPU_NAME=""
        GPU_MEMORY=""
        GPU_DRIVER_VERSION=""
        GPU_CUDA_VERSION=""
        GPU_AVAILABLE=false
    fi
}

# Print GPU information
print_gpu_info() {
    get_gpu_info
    if [ "$GPU_AVAILABLE" = true ]; then
        print_success "NVIDIA GPU detected"
        echo "    GPU: $GPU_NAME"
        echo "    Memory: $GPU_MEMORY"
        echo "    Driver: $GPU_DRIVER_VERSION"
        echo "    CUDA: $GPU_CUDA_VERSION"
    else
        print_warning "No NVIDIA GPU detected"
    fi
}

# Determine PyTorch CUDA version based on driver and GPU architecture
get_pytorch_cuda_version() {
    get_gpu_info
    if [ "$GPU_AVAILABLE" = true ]; then
        local driver_major=$(echo "$GPU_DRIVER_VERSION" | cut -d. -f1)
        # Check for Blackwell architecture (RTX 50-series) - requires nightly with CUDA 12.8
        if echo "$GPU_NAME" | grep -qiE "RTX 50[0-9][0-9]"; then
            echo "nightly/cu128"
        # Driver 570+ with CUDA 12.8 support (RTX 4000 Ada workstation)
        elif [ "$driver_major" -ge 570 ]; then
            echo "cu124"  # CUDA 12.4 for latest stable compatibility
        elif [ "$driver_major" -ge 525 ]; then
            echo "cu121"
        elif [ "$driver_major" -ge 450 ]; then
            echo "cu118"
        else
            echo "cu118"  # Fallback
        fi
    else
        echo "cpu"
    fi
}

# Check if GPU requires nightly PyTorch build
requires_nightly_pytorch() {
    get_gpu_info
    if [ "$GPU_AVAILABLE" = true ]; then
        # Blackwell architecture (RTX 50-series) requires nightly build
        if echo "$GPU_NAME" | grep -qiE "RTX 50[0-9][0-9]"; then
            return 0  # true
        fi
    fi
    return 1  # false
}

#===============================================================================
# WSL DETECTION
#===============================================================================

is_wsl() {
    if grep -qi microsoft /proc/version 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

#===============================================================================
# YOLO MODEL DEFINITIONS (Ultralytics-supported models)
#===============================================================================
# Centralized model lists used across scripts
#
# NOTE: YOLOv4 is NOT supported by Ultralytics. Use run_train_yolov4_*.sh
#       scripts for YOLOv4 training with the Darknet-compatible pipeline.
#       YOLOv4 models: yolov4-tiny, yolov4, yolov4-csp, yolov4x-mish

# Ultralytics-supported model arrays
YOLOV5_MODELS=("yolov5nu.pt" "yolov5su.pt" "yolov5mu.pt" "yolov5lu.pt" "yolov5xu.pt")
YOLOV8_MODELS=("yolov8n.pt" "yolov8s.pt" "yolov8m.pt" "yolov8l.pt" "yolov8x.pt")
YOLOV9_MODELS=("yolov9t.pt" "yolov9s.pt" "yolov9m.pt" "yolov9c.pt" "yolov9e.pt")
YOLOV10_MODELS=("yolov10n.pt" "yolov10s.pt" "yolov10m.pt" "yolov10l.pt" "yolov10x.pt")
YOLO11_MODELS=("yolo11n.pt" "yolo11s.pt" "yolo11m.pt" "yolo11l.pt" "yolo11x.pt")

# Get all Ultralytics models as single array
get_all_yolo_models() {
    local all_models=()
    all_models+=("${YOLOV5_MODELS[@]}")
    all_models+=("${YOLOV8_MODELS[@]}")
    all_models+=("${YOLOV9_MODELS[@]}")
    all_models+=("${YOLOV10_MODELS[@]}")
    all_models+=("${YOLO11_MODELS[@]}")
    echo "${all_models[@]}"
}

#===============================================================================
# PATH UTILITIES
#===============================================================================

# Get absolute path from relative path
get_abs_path() {
    local path="$1"
    echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
}

# Ensure directory exists
ensure_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_debug "Created directory: $dir"
    fi
}

#===============================================================================
# VALIDATION UTILITIES
#===============================================================================

# Check if a file exists
require_file() {
    local file="$1"
    local description="${2:-File}"
    if [ ! -f "$file" ]; then
        print_error "$description not found: $file"
        return 1
    fi
    return 0
}

# Check if a directory exists
require_dir() {
    local dir="$1"
    local description="${2:-Directory}"
    if [ ! -d "$dir" ]; then
        print_error "$description not found: $dir"
        return 1
    fi
    return 0
}

#===============================================================================
# CONFIG NAME UTILITIES
#===============================================================================
# Helper functions for extracting config combination names

# Extract config combination name from config script filename
# e.g., "01_yolo_color_combination.sh" -> "01_yolo_color_combination"
# Usage: get_config_combination_name "config_script_filename"
get_config_combination_name() {
    local config_file="$1"
    # Remove .sh extension
    echo "${config_file%.sh}"
}

#===============================================================================
# DATASET OUTPUT COPY UTILITIES
#===============================================================================
# Functions to copy logs and trained models to dataset-specific directories

# Copy logs to dataset-specific logs folder
# Usage: copy_logs_to_dataset "experiment_log_dir" "dataset_path" "exp_name"
copy_logs_to_dataset() {
    local experiment_log_dir="$1"
    local dataset_path="$2"
    local exp_name="$3"
    
    if [ -z "$experiment_log_dir" ] || [ ! -d "$experiment_log_dir" ]; then
        print_warning "No experiment log directory to copy"
        return 1
    fi
    
    # Create dataset logs directory
    local dataset_logs_dir="${dataset_path}/logs"
    ensure_dir "$dataset_logs_dir"
    
    # Copy logs to dataset-specific folder
    local target_log_dir="${dataset_logs_dir}/${exp_name}"
    if [ -d "$experiment_log_dir" ]; then
        cp -r "$experiment_log_dir" "$target_log_dir"
        print_success "Logs copied to: ${target_log_dir}"
        return 0
    else
        print_warning "Source log directory not found: $experiment_log_dir"
        return 1
    fi
}

# Copy trained model to dataset-specific trained_models_output folder
# Usage: copy_model_to_dataset "output_dir" "exp_name" "dataset_path"
copy_model_to_dataset() {
    local output_dir="$1"
    local exp_name="$2"
    local dataset_path="$3"
    
    local source_model_dir="${output_dir}/${exp_name}"
    
    if [ ! -d "$source_model_dir" ]; then
        print_warning "Source model directory not found: $source_model_dir"
        return 1
    fi
    
    # Create dataset trained_models_output directory
    local dataset_models_dir="${dataset_path}/trained_models_output"
    ensure_dir "$dataset_models_dir"
    
    # Copy model to dataset-specific folder
    local target_model_dir="${dataset_models_dir}/${exp_name}"
    cp -r "$source_model_dir" "$target_model_dir"
    print_success "Model copied to: ${target_model_dir}"
    return 0
}

# Copy logs into the training output folder
# Usage: copy_logs_to_output "experiment_log_dir" "output_dir" "exp_name"
copy_logs_to_output() {
    local experiment_log_dir="$1"
    local output_dir="$2"
    local exp_name="$3"
    
    if [ -z "$experiment_log_dir" ] || [ ! -d "$experiment_log_dir" ]; then
        print_warning "No experiment log directory to copy to output"
        return 1
    fi
    
    local target_output_dir="${output_dir}/${exp_name}"
    if [ ! -d "$target_output_dir" ]; then
        print_warning "Target output directory not found: $target_output_dir"
        return 1
    fi
    
    # Create logs subdirectory in the training output folder
    local target_logs_dir="${target_output_dir}/logs"
    ensure_dir "$target_logs_dir"
    
    # Copy all log files to the logs subdirectory
    if [ -d "$experiment_log_dir" ]; then
        cp -r "${experiment_log_dir}"/* "$target_logs_dir/" 2>/dev/null || true
        print_success "Logs copied to output folder: ${target_logs_dir}"
        return 0
    else
        print_warning "Source log directory not found: $experiment_log_dir"
        return 1
    fi
}

# Copy both logs and trained model to dataset folder
# Usage: copy_outputs_to_dataset "experiment_log_dir" "output_dir" "exp_name" "dataset_path"
copy_outputs_to_dataset() {
    local experiment_log_dir="$1"
    local output_dir="$2"
    local exp_name="$3"
    local dataset_path="$4"
    
    print_subheader "Copying outputs to dataset folder"
    
    # Copy logs to dataset-specific logs folder
    copy_logs_to_dataset "$experiment_log_dir" "$dataset_path" "$exp_name"
    
    # Copy trained model to dataset-specific trained_models_output folder
    copy_model_to_dataset "$output_dir" "$exp_name" "$dataset_path"
    
    # Also copy logs into the training output folder itself
    copy_logs_to_output "$experiment_log_dir" "$output_dir" "$exp_name"
    
    echo ""
}

#===============================================================================
# PORTABILITY UTILITIES
#===============================================================================

# Update data.yaml path to current location for portability
# This ensures the project works when moved to a different directory/computer
# Usage: update_data_yaml_path "data_yaml_path" ["dataset_dir"]
update_data_yaml_path() {
    local data_yaml="$1"
    local dataset_dir="${2:-$(dirname "$data_yaml")}"
    
    if [ ! -f "$data_yaml" ]; then
        print_error "data.yaml not found: $data_yaml"
        return 1
    fi
    
    # Get absolute path of dataset directory
    local abs_dataset_dir
    abs_dataset_dir="$(cd "$dataset_dir" && pwd)"
    
    # Read current path from data.yaml
    local current_path
    current_path=$(grep "^path:" "$data_yaml" | sed 's/path:[[:space:]]*//' | sed 's/#.*//' | tr -d '[:space:]')
    
    # Check if update is needed
    if [ "$current_path" = "$abs_dataset_dir" ]; then
        print_info "data.yaml path already correct: $abs_dataset_dir"
        return 0
    fi
    
    # Update the path in data.yaml using sed
    # Handle both 'path: value' and 'path: value  # comment' formats
    if sed -i "s|^path:.*|path: ${abs_dataset_dir}|" "$data_yaml"; then
        print_success "Updated data.yaml path:"
        echo "    Old: $current_path"
        echo "    New: $abs_dataset_dir"
        return 0
    else
        print_error "Failed to update data.yaml path"
        return 1
    fi
}

# Ensure data.yaml is portable (auto-detect dataset directory)
# Usage: ensure_portable_data_yaml "data_yaml_path"
ensure_portable_data_yaml() {
    local data_yaml="$1"
    update_data_yaml_path "$data_yaml"
}
