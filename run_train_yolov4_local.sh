#!/bin/bash
#===============================================================================
# Microspore Phenotyping - YOLOv4 Training Script (LOCAL)
#===============================================================================
# This script trains YOLOv4 models using the Darknet framework.
# YOLOv4 is NOT supported by Ultralytics, so this separate pipeline is required.
#
# Prerequisites:
#   1. Darknet compiled with GPU support (CUDA/cuDNN or ROCm)
#   2. Pre-trained weights downloaded
#   3. Dataset prepared in YOLO format
#
# Usage:
#   ./run_train_yolov4_local.sh              # Interactive mode
#   ./run_train_yolov4_local.sh --train      # Start training directly
#   ./run_train_yolov4_local.sh --setup      # Setup Darknet only
#===============================================================================

set -e  # Exit on error

#===============================================================================
# CONFIGURATION
#===============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/local/01_yolov4_config.sh"

# Source common functions
if [ -f "${SCRIPT_DIR}/modules/config/common_functions.sh" ]; then
    source "${SCRIPT_DIR}/modules/config/common_functions.sh"
else
    echo "ERROR: modules/config/common_functions.sh not found"
    exit 1
fi

# Source YOLOv4 configuration
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Darknet directory
DARKNET_DIR="${SCRIPT_DIR}/modules/darknet"
WEIGHTS_DIR="${SCRIPT_DIR}/modules/yolov4_weights"

#===============================================================================
# FUNCTIONS
#===============================================================================

check_dependencies() {
    print_header "Checking Dependencies"
    
    local missing=()
    
    # Check for required tools
    for cmd in git make gcc g++ wget; do
        if ! command -v $cmd &> /dev/null; then
            missing+=("$cmd")
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        print_error "Missing dependencies: ${missing[*]}"
        echo "Install with: sudo apt-get install ${missing[*]}"
        return 1
    fi
    
    print_success "All dependencies found"
    
    # Check for CUDA/GPU
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    elif command -v rocm-smi &> /dev/null; then
        print_success "AMD GPU (ROCm) detected"
        rocm-smi --showproductname
    else
        print_warning "No GPU detected - training will be slow on CPU"
    fi
    
    echo ""
}

setup_darknet() {
    print_header "Setting up Darknet"
    
    if [ -d "$DARKNET_DIR" ] && [ -f "${DARKNET_DIR}/darknet" ]; then
        print_success "Darknet already compiled"
        return 0
    fi
    
    # Clone Darknet (AlexeyAB fork with YOLOv4 support)
    if [ ! -d "$DARKNET_DIR" ]; then
        print_info "Cloning Darknet repository..."
        git clone https://github.com/AlexeyAB/darknet.git "$DARKNET_DIR"
    fi
    
    cd "$DARKNET_DIR"
    
    # Configure Makefile for GPU
    print_info "Configuring Makefile..."
    
    if command -v nvidia-smi &> /dev/null; then
        # NVIDIA GPU
        sed -i 's/GPU=0/GPU=1/' Makefile
        sed -i 's/CUDNN=0/CUDNN=1/' Makefile
        sed -i 's/OPENCV=0/OPENCV=1/' Makefile
        sed -i 's/LIBSO=0/LIBSO=1/' Makefile
    else
        # CPU only or AMD (needs custom compilation)
        print_warning "NVIDIA GPU not detected. Compiling for CPU..."
        sed -i 's/OPENCV=0/OPENCV=1/' Makefile
        sed -i 's/LIBSO=0/LIBSO=1/' Makefile
    fi
    
    # Compile
    print_info "Compiling Darknet (this may take a few minutes)..."
    make -j$(nproc)
    
    if [ -f "${DARKNET_DIR}/darknet" ]; then
        print_success "Darknet compiled successfully"
    else
        print_error "Darknet compilation failed"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
}

download_pretrained_weights() {
    print_header "Downloading Pre-trained Weights"
    
    mkdir -p "$WEIGHTS_DIR"
    
    local weights_file=""
    local weights_url=""
    
    case "$YOLOV4_MODEL" in
        "yolov4-tiny")
            weights_file="yolov4-tiny.conv.29"
            weights_url="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4-tiny.conv.29"
            ;;
        "yolov4")
            weights_file="yolov4.conv.137"
            weights_url="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
            ;;
        "yolov4-csp")
            weights_file="yolov4-csp.conv.142"
            weights_url="https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-csp.conv.142"
            ;;
        "yolov4x-mish")
            weights_file="yolov4x-mish.conv.166"
            weights_url="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.conv.166"
            ;;
        *)
            print_error "Unknown model: $YOLOV4_MODEL"
            return 1
            ;;
    esac
    
    local weights_path="${WEIGHTS_DIR}/${weights_file}"
    
    if [ -f "$weights_path" ]; then
        print_success "Weights already downloaded: $weights_file"
    else
        print_info "Downloading $weights_file..."
        wget -O "$weights_path" "$weights_url"
        print_success "Downloaded: $weights_file"
    fi
}

prepare_dataset() {
    print_header "Preparing Dataset"
    
    local dataset_path="${SCRIPT_DIR}/${DATASETS_DIR}/${DATASET_NAME}"
    
    if [ ! -d "$dataset_path" ]; then
        print_error "Dataset not found: $dataset_path"
        return 1
    fi
    
    print_info "Dataset: $dataset_path"
    
    # Check for data.yaml
    if [ ! -f "${dataset_path}/data.yaml" ]; then
        print_error "data.yaml not found in dataset"
        return 1
    fi
    
    # Create Darknet-compatible data file
    local data_file="${dataset_path}/yolov4.data"
    local names_file="${dataset_path}/classes.names"
    local train_file="${dataset_path}/train.txt"
    local valid_file="${dataset_path}/valid.txt"
    
    # Extract class names from data.yaml
    print_info "Extracting class names from data.yaml..."
    python3 << EOF
import yaml
import os

with open("${dataset_path}/data.yaml", 'r') as f:
    data = yaml.safe_load(f)

names = data.get('names', [])
if isinstance(names, dict):
    names = [names[i] for i in sorted(names.keys())]

# Write classes.names
with open("${names_file}", 'w') as f:
    for name in names:
        f.write(f"{name}\n")

print(f"Classes: {len(names)}")
for i, name in enumerate(names):
    print(f"  {i}: {name}")
EOF
    
    # Generate train.txt and valid.txt (list of image paths)
    print_info "Generating image lists..."
    
    find "${dataset_path}/Train/images" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) > "$train_file"
    find "${dataset_path}/Test/images" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) > "$valid_file"
    
    local train_count=$(wc -l < "$train_file")
    local valid_count=$(wc -l < "$valid_file")
    
    print_info "Train images: $train_count"
    print_info "Valid images: $valid_count"
    
    # Count classes
    local num_classes=$(wc -l < "$names_file")
    
    # Create .data file
    cat > "$data_file" << EOF
classes = ${num_classes}
train = ${train_file}
valid = ${valid_file}
names = ${names_file}
backup = ${SCRIPT_DIR}/${OUTPUT_DIR}/yolov4_backup
EOF
    
    # Create backup directory
    mkdir -p "${SCRIPT_DIR}/${OUTPUT_DIR}/yolov4_backup"
    
    print_success "Dataset prepared for Darknet"
    echo ""
}

generate_cfg_file() {
    print_header "Generating YOLOv4 Configuration"
    
    local dataset_path="${SCRIPT_DIR}/${DATASETS_DIR}/${DATASET_NAME}"
    local names_file="${dataset_path}/classes.names"
    local num_classes=$(wc -l < "$names_file")
    local cfg_file="${dataset_path}/${YOLOV4_MODEL}-custom.cfg"
    
    # Calculate filters for [yolo] layers: filters = (classes + 5) * 3
    local filters=$(( (num_classes + 5) * 3 ))
    
    print_info "Number of classes: $num_classes"
    print_info "Filters for YOLO layers: $filters"
    
    # Copy base cfg and modify
    local base_cfg="${DARKNET_DIR}/cfg/${YOLOV4_MODEL}.cfg"
    
    if [ ! -f "$base_cfg" ]; then
        print_error "Base config not found: $base_cfg"
        return 1
    fi
    
    cp "$base_cfg" "$cfg_file"
    
    # Modify configuration
    sed -i "s/^batch=.*/batch=${BATCH_SIZE}/" "$cfg_file"
    sed -i "s/^subdivisions=.*/subdivisions=${SUBDIVISIONS}/" "$cfg_file"
    sed -i "s/^width=.*/width=${IMG_SIZE}/" "$cfg_file"
    sed -i "s/^height=.*/height=${IMG_SIZE}/" "$cfg_file"
    sed -i "s/^max_batches=.*/max_batches=${MAX_BATCHES}/" "$cfg_file"
    sed -i "s/^learning_rate=.*/learning_rate=${LEARNING_RATE}/" "$cfg_file"
    sed -i "s/^burn_in=.*/burn_in=${BURN_IN}/" "$cfg_file"
    sed -i "s/^momentum=.*/momentum=${MOMENTUM}/" "$cfg_file"
    sed -i "s/^decay=.*/decay=${DECAY}/" "$cfg_file"
    sed -i "s/^angle=.*/angle=${ANGLE}/" "$cfg_file"
    sed -i "s/^saturation=.*/saturation=${SATURATION}/" "$cfg_file"
    sed -i "s/^exposure=.*/exposure=${EXPOSURE}/" "$cfg_file"
    sed -i "s/^hue=.*/hue=${HUE}/" "$cfg_file"
    sed -i "s/^mosaic=.*/mosaic=${MOSAIC}/" "$cfg_file"
    sed -i "s/^steps=.*/steps=${STEPS}/" "$cfg_file"
    sed -i "s/^scales=.*/scales=${SCALES}/" "$cfg_file"
    
    # Update classes and filters in [yolo] and preceding [convolutional] layers
    # This is a simplified approach - for production, use a proper parser
    sed -i "s/^classes=.*/classes=${num_classes}/" "$cfg_file"
    
    # Update filters before each [yolo] layer (requires careful regex)
    python3 << EOF
import re

with open("${cfg_file}", 'r') as f:
    content = f.read()

# Find all [convolutional] sections before [yolo] and update filters
# This is a simplified approach
lines = content.split('\n')
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    # Check if this is a filters line before a [yolo] section
    if line.startswith('filters=') and i + 3 < len(lines):
        # Look ahead for [yolo]
        for j in range(i+1, min(i+10, len(lines))):
            if lines[j].strip() == '[yolo]':
                new_lines[-1] = f'filters=${filters}'
                break

with open("${cfg_file}", 'w') as f:
    f.write('\n'.join(new_lines))

print("Configuration updated")
EOF
    
    print_success "Generated: $cfg_file"
    echo ""
}

start_training() {
    print_header "Starting YOLOv4 Training"
    
    local dataset_path="${SCRIPT_DIR}/${DATASETS_DIR}/${DATASET_NAME}"
    local data_file="${dataset_path}/yolov4.data"
    local cfg_file="${dataset_path}/${YOLOV4_MODEL}-custom.cfg"
    local weights_file=""
    
    case "$YOLOV4_MODEL" in
        "yolov4-tiny") weights_file="${WEIGHTS_DIR}/yolov4-tiny.conv.29" ;;
        "yolov4") weights_file="${WEIGHTS_DIR}/yolov4.conv.137" ;;
        "yolov4-csp") weights_file="${WEIGHTS_DIR}/yolov4-csp.conv.142" ;;
        "yolov4x-mish") weights_file="${WEIGHTS_DIR}/yolov4x-mish.conv.166" ;;
    esac
    
    print_info "Model: $YOLOV4_MODEL"
    print_info "Data file: $data_file"
    print_info "Config file: $cfg_file"
    print_info "Weights: $weights_file"
    print_info "GPU ID: $GPU_ID"
    echo ""
    
    # Check files exist
    for f in "$data_file" "$cfg_file" "$weights_file"; do
        if [ ! -f "$f" ]; then
            print_error "File not found: $f"
            return 1
        fi
    done
    
    # Create log file
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="${SCRIPT_DIR}/logs/yolov4_${YOLOV4_MODEL}_${timestamp}.log"
    mkdir -p "${SCRIPT_DIR}/logs"
    
    print_info "Log file: $log_file"
    print_info "Starting training..."
    echo ""
    
    # Run Darknet training
    cd "$DARKNET_DIR"
    
    if [ "$GPU_ID" -ge 0 ]; then
        ./darknet detector train "$data_file" "$cfg_file" "$weights_file" \
            -gpus $GPU_ID \
            -map \
            2>&1 | tee "$log_file"
    else
        ./darknet detector train "$data_file" "$cfg_file" "$weights_file" \
            -map \
            2>&1 | tee "$log_file"
    fi
    
    cd "$SCRIPT_DIR"
    
    print_success "Training complete!"
}

show_menu() {
    echo ""
    echo "YOLOv4 Training Pipeline (LOCAL)"
    echo "================================"
    echo ""
    echo "Model: $YOLOV4_MODEL"
    echo "Dataset: $DATASET_NAME"
    echo ""
    echo "  1) Setup Darknet (compile from source)"
    echo "  2) Download pre-trained weights"
    echo "  3) Prepare dataset"
    echo "  4) Generate config file"
    echo "  5) Start training"
    echo "  6) Run full pipeline (1-5)"
    echo ""
    echo "  c) Check dependencies"
    echo "  q) Quit"
    echo ""
}

#===============================================================================
# MAIN
#===============================================================================

main() {
    print_header "YOLOv4 Training Pipeline (LOCAL)"
    
    print_info "Configuration: $CONFIG_FILE"
    print_info "Model: $YOLOV4_MODEL"
    print_info "Dataset: $DATASET_NAME"
    echo ""
    
    # Parse command line arguments
    case "${1:-}" in
        --setup)
            check_dependencies
            setup_darknet
            download_pretrained_weights
            ;;
        --train)
            check_dependencies
            setup_darknet
            download_pretrained_weights
            prepare_dataset
            generate_cfg_file
            start_training
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (no args)    Interactive mode"
            echo "  --setup      Setup Darknet and download weights"
            echo "  --train      Run full training pipeline"
            echo "  --help       Show this help"
            ;;
        "")
            # Interactive mode
            while true; do
                show_menu
                read -p "Enter choice: " choice
                
                case $choice in
                    1) setup_darknet ;;
                    2) download_pretrained_weights ;;
                    3) prepare_dataset ;;
                    4) generate_cfg_file ;;
                    5) start_training ;;
                    6)
                        check_dependencies
                        setup_darknet
                        download_pretrained_weights
                        prepare_dataset
                        generate_cfg_file
                        start_training
                        ;;
                    c|C) check_dependencies ;;
                    q|Q)
                        print_success "Done!"
                        exit 0
                        ;;
                    *)
                        print_warning "Invalid choice"
                        ;;
                esac
            done
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

main "$@"
