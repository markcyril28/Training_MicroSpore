#!/bin/bash
#===============================================================================
# Microspore Phenotyping - YOLO Models Setup Script
# Downloads and sets up all available YOLO model variants.
# Uses common_functions.sh for shared utilities (DRY principle).
#===============================================================================

set -e  # Exit on error

# Get script directory (modules are in the same directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common functions from modules/config/ (DRY principle)
if [ -f "${SCRIPT_DIR}/modules/config/common_functions.sh" ]; then
    source "${SCRIPT_DIR}/modules/config/common_functions.sh"
else
    echo "ERROR: modules/config/common_functions.sh not found. Please ensure it exists."
    exit 1
fi

# Use shared configuration from common_functions.sh
ENV_NAME="${COMMON_ENV_NAME}"
WEIGHTS_DIR="${SCRIPT_DIR}/${COMMON_WEIGHTS_SUBDIR}"

# Note: Model arrays are defined in common_functions.sh
# YOLOV5_MODELS, YOLOV8_MODELS, YOLOV9_MODELS, YOLOV10_MODELS, YOLO11_MODELS

#===============================================================================
# FUNCTIONS
#===============================================================================

check_conda() {
    print_header "Checking Conda Environment"
    
    # Use common function
    if ! init_conda_shell; then
        exit 1
    fi
    
    # Use common function to check and activate
    if ! activate_conda_env "${ENV_NAME}"; then
        print_info "Run setup_conda_training.sh first."
        exit 1
    fi
    echo ""
}

download_model() {
    local model_name=$1
    local model_path="${WEIGHTS_DIR}/${model_name}"
    
    if [ -f "$model_path" ]; then
        print_info "Already exists: $model_name"
        return 0
    fi
    
    echo -n "  Downloading $model_name... "
    
    python -c "
from ultralytics import YOLO
from pathlib import Path
import shutil

try:
    # Load model (triggers download)
    model = YOLO('${model_name}')
    
    # Get the downloaded model path
    from ultralytics import settings
    src_path = Path(settings['weights_dir']) / '${model_name}'
    
    # Also check current directory
    if not src_path.exists():
        src_path = Path('${model_name}')
    
    # Copy to our weights directory
    dst_path = Path('${model_path}')
    if src_path.exists() and not dst_path.exists():
        shutil.copy(src_path, dst_path)
    
    # Clean up if downloaded to current directory
    local_file = Path('${model_name}')
    if local_file.exists() and str(local_file.absolute()) != str(dst_path.absolute()):
        local_file.unlink()
        
    print('OK')
except Exception as e:
    print(f'FAILED: {e}')
" 2>/dev/null
}

download_model_set() {
    local set_name=$1
    shift
    local models=("$@")
    
    print_header "Downloading $set_name Models"
    
    for model in "${models[@]}"; do
        download_model "$model"
    done
    
    echo ""
}

list_models() {
    print_header "Available YOLO Models"
    
    echo "YOLOv4 (5 variants):"
    for m in "${YOLOV4_MODELS[@]}"; do echo "  - $m"; done
    echo ""
    
    echo "YOLOv5 (5 variants):"
    for m in "${YOLOV5_MODELS[@]}"; do echo "  - $m"; done
    echo ""
    
    echo "YOLOv8 (5 variants) - Recommended:"
    for m in "${YOLOV8_MODELS[@]}"; do echo "  - $m"; done
    echo ""
    
    echo "YOLOv9 (5 variants):"
    for m in "${YOLOV9_MODELS[@]}"; do echo "  - $m"; done
    echo ""
    
    echo "YOLOv10 (5 variants):"
    for m in "${YOLOV10_MODELS[@]}"; do echo "  - $m"; done
    echo ""
    
    echo "YOLO11 (5 variants) - Latest:"
    for m in "${YOLO11_MODELS[@]}"; do echo "  - $m"; done
    echo ""
}

show_downloaded() {
    print_header "Downloaded Models"
    
    if [ -d "$WEIGHTS_DIR" ]; then
        local count=$(ls -1 "$WEIGHTS_DIR"/*.pt 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            echo "Location: $WEIGHTS_DIR"
            echo ""
            ls -lh "$WEIGHTS_DIR"/*.pt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
            echo ""
            echo "Total: $count model(s)"
        else
            print_warning "No models downloaded yet."
        fi
    else
        print_warning "Weights directory does not exist."
    fi
    echo ""
}

show_menu() {
    echo ""
    echo "Select models to download:"
    echo ""
    echo "  1) YOLOv8 small only (yolov8s.pt) - Quick start, recommended"
    echo "  2) YOLOv8 all variants (5 models)"
    echo "  3) All YOLOv4 models (5 models)"
    echo "  4) All YOLOv5 models (5 models)"
    echo "  5) All YOLOv9 models (5 models)"
    echo "  6) All YOLOv10 models (5 models)"
    echo "  7) All YOLO11 models (5 models) - Latest"
    echo "  8) All small/nano models (12 models) - Best for 8GB VRAM"
    echo "  9) ALL models (30 models) - Full download"
    echo ""
    echo "  l) List all available models"
    echo "  s) Show downloaded models"
    echo "  q) Quit"
    echo ""
}

#===============================================================================
# MAIN
#===============================================================================

main() {
    print_header "YOLO Models Setup"
    
    # Create weights directory
    mkdir -p "$WEIGHTS_DIR"
    print_info "Weights directory: $WEIGHTS_DIR"
    echo ""
    
    # Check conda environment
    check_conda
    
    # Interactive mode or command line
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Enter choice: " choice
            
            case $choice in
                1)
                    download_model "yolov8s.pt"
                    ;;
                2)
                    download_model_set "YOLOv8" "${YOLOV8_MODELS[@]}"
                    ;;
                3)
                    download_model_set "YOLOv4" "${YOLOV4_MODELS[@]}"
                    ;;
                4)
                    download_model_set "YOLOv5" "${YOLOV5_MODELS[@]}"
                    ;;
                5)
                    download_model_set "YOLOv9" "${YOLOV9_MODELS[@]}"
                    ;;
                6)
                    download_model_set "YOLOv10" "${YOLOV10_MODELS[@]}"
                    ;;
                7)
                    download_model_set "YOLO11" "${YOLO11_MODELS[@]}"
                    ;;
                8)
                    print_header "Downloading Small/Nano Models (Best for 8GB VRAM)"
                    download_model "yolov4-tiny.pt"
                    download_model "yolov4s-mish.pt"
                    download_model "yolov5nu.pt"
                    download_model "yolov5su.pt"
                    download_model "yolov8n.pt"
                    download_model "yolov8s.pt"
                    download_model "yolov9t.pt"
                    download_model "yolov9s.pt"
                    download_model "yolov10n.pt"
                    download_model "yolov10s.pt"
                    download_model "yolo11n.pt"
                    download_model "yolo11s.pt"
                    ;;
                9)
                    download_model_set "YOLOv4" "${YOLOV4_MODELS[@]}"
                    download_model_set "YOLOv5" "${YOLOV5_MODELS[@]}"
                    download_model_set "YOLOv8" "${YOLOV8_MODELS[@]}"
                    download_model_set "YOLOv9" "${YOLOV9_MODELS[@]}"
                    download_model_set "YOLOv10" "${YOLOV10_MODELS[@]}"
                    download_model_set "YOLO11" "${YOLO11_MODELS[@]}"
                    ;;
                l|L)
                    list_models
                    ;;
                s|S)
                    show_downloaded
                    ;;
                q|Q)
                    print_success "Done!"
                    exit 0
                    ;;
                *)
                    print_warning "Invalid choice. Please try again."
                    ;;
            esac
        done
    else
        # Command line mode
        case $1 in
            --all)
                download_model_set "YOLOv4" "${YOLOV4_MODELS[@]}"
                download_model_set "YOLOv5" "${YOLOV5_MODELS[@]}"
                download_model_set "YOLOv8" "${YOLOV8_MODELS[@]}"
                download_model_set "YOLOv9" "${YOLOV9_MODELS[@]}"
                download_model_set "YOLOv10" "${YOLOV10_MODELS[@]}"
                download_model_set "YOLO11" "${YOLO11_MODELS[@]}"
                ;;
            --yolov4)
                download_model_set "YOLOv4" "${YOLOV4_MODELS[@]}"
                ;;
            --yolov5)
                download_model_set "YOLOv5" "${YOLOV5_MODELS[@]}"
                ;;
            --yolov8)
                download_model_set "YOLOv8" "${YOLOV8_MODELS[@]}"
                ;;
            --yolov9)
                download_model_set "YOLOv9" "${YOLOV9_MODELS[@]}"
                ;;
            --yolov10)
                download_model_set "YOLOv10" "${YOLOV10_MODELS[@]}"
                ;;
            --yolo11)
                download_model_set "YOLO11" "${YOLO11_MODELS[@]}"
                ;;
            --small)
                print_header "Downloading Small/Nano Models"
                download_model "yolov4-tiny.pt"
                download_model "yolov4s-mish.pt"
                download_model "yolov5nu.pt"
                download_model "yolov5su.pt"
                download_model "yolov8n.pt"
                download_model "yolov8s.pt"
                download_model "yolov9t.pt"
                download_model "yolov9s.pt"
                download_model "yolov10n.pt"
                download_model "yolov10s.pt"
                download_model "yolo11n.pt"
                download_model "yolo11s.pt"
                ;;
            --list)
                list_models
                ;;
            --status)
                show_downloaded
                ;;
            *)
                # Try to download specific model
                if [[ "$1" == *.pt ]]; then
                    download_model "$1"
                else
                    echo "Usage: $0 [OPTIONS]"
                    echo ""
                    echo "Options:"
                    echo "  (no args)    Interactive mode"
                    echo "  --all        Download all models (30 models)"
                    echo "  --yolov4     Download all YOLOv4 models"
                    echo "  --yolov5     Download all YOLOv5 models"
                    echo "  --yolov8     Download all YOLOv8 models"
                    echo "  --yolov9     Download all YOLOv9 models"
                    echo "  --yolov10    Download all YOLOv10 models"
                    echo "  --yolo11     Download all YOLO11 models"
                    echo "  --small      Download small/nano models (for 8GB VRAM)"
                    echo "  --list       List all available models"
                    echo "  --status     Show downloaded models"
                    echo "  <model>.pt   Download specific model"
                fi
                ;;
        esac
    fi
    
    echo ""
    show_downloaded
    print_success "YOLO Models Setup Complete!"
}

# Run main
main "$@"
