#!/bin/bash
#===============================================================================
# Microspore Phenotyping - Conda Training Setup Core Functions
# Shared functions for conda environment setup across local and workstation.
# This file is sourced by setup_conda_training_local.sh and 
# setup_conda_training_workstation.sh
#
# Location: modules/setup/setup_conda_core.sh
# Requires: source "${BASE_DIR}/modules/config/common_functions.sh" first
#===============================================================================

# Ensure common_functions.sh is sourced first
if [ -z "$COMMON_ENV_NAME" ]; then
    echo "ERROR: common_functions.sh must be sourced before setup_conda_core.sh"
    exit 1
fi

# Use shared configuration
ENV_NAME="${COMMON_ENV_NAME}"
PYTHON_VERSION="${COMMON_PYTHON_VERSION}"

#===============================================================================
# PRE-FLIGHT GPU CHECK
#===============================================================================

preflight_gpu_check() {
    echo "Checking GPU availability..."
    echo ""
    
    get_gpu_info
    
    if [ "$GPU_AVAILABLE" = true ]; then
        print_success "NVIDIA GPU detected"
        echo "    GPU: $GPU_NAME"
        echo "    Memory: $GPU_MEMORY"
        echo "    Driver: $GPU_DRIVER_VERSION"
        echo "    CUDA: $GPU_CUDA_VERSION"
        echo ""
        
        PYTORCH_CUDA=$(get_pytorch_cuda_version)
        print_info "Using PyTorch with ${PYTORCH_CUDA}"
    else
        print_warning "nvidia-smi not found or GPU not accessible"
    fi
}

#===============================================================================
# OFFER GPU SETUP
#===============================================================================

offer_gpu_setup() {
    local gpu_setup_script="${1:-gpu_setup_local.sh}"
    
    if [ "$GPU_AVAILABLE" = false ]; then
        echo ""
        print_warning "GPU not detected or not properly configured."
        echo ""
        
        if [ -f "${SCRIPT_DIR}/${gpu_setup_script}" ]; then
            read -p "Would you like to run ${gpu_setup_script} first? (y/n): " run_gpu_setup
            if [[ "$run_gpu_setup" == "y" || "$run_gpu_setup" == "Y" ]]; then
                chmod +x "${SCRIPT_DIR}/${gpu_setup_script}"
                "${SCRIPT_DIR}/${gpu_setup_script}"
                
                get_gpu_info
                if [ "$GPU_AVAILABLE" = true ]; then
                    PYTORCH_CUDA=$(get_pytorch_cuda_version)
                fi
            fi
        else
            echo "Run ${gpu_setup_script} to configure GPU support."
        fi
        
        if [ "$GPU_AVAILABLE" = false ]; then
            read -p "Continue with CPU-only installation? (y/n): " continue_cpu
            if [[ "$continue_cpu" != "y" && "$continue_cpu" != "Y" ]]; then
                echo "Setup cancelled."
                exit 0
            fi
            PYTORCH_CUDA="cpu"
            print_warning "Installing CPU-only version of PyTorch"
        fi
    fi
    
    echo ""
}

#===============================================================================
# CONDA INITIALIZATION
#===============================================================================

init_conda_for_setup() {
    print_header "Checking Conda Installation"
    
    if ! init_conda_shell; then
        exit 1
    fi
    
    eval "$(conda shell.bash hook)"
    
    if command -v mamba &> /dev/null; then
        PKG_MANAGER="mamba"
        print_success "Using mamba (faster package manager)"
    else
        PKG_MANAGER="conda"
        print_info "Using conda (install mamba for faster operations: conda install -n base -c conda-forge mamba)"
    fi
}

#===============================================================================
# HANDLE EXISTING ENVIRONMENT
#===============================================================================

handle_existing_env() {
    SKIP_TO_VERIFY=false
    
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "Environment '${ENV_NAME}' already exists."
        echo ""
        echo "Options:"
        echo "  [1] Update - Update packages in existing environment"
        echo "  [2] Recreate - Remove and recreate environment from scratch"
        echo "  [3] Skip - Use existing environment as-is"
        echo "  [4] Cancel - Exit setup"
        echo ""
        read -p "Select option (1/2/3/4): " choice
        
        case "$choice" in
            1|update|Update|UPDATE)
                print_info "Updating existing environment..."
                conda activate ${ENV_NAME}
                
                echo "Updating packages with ${PKG_MANAGER}..."
                ${PKG_MANAGER} update --all -y || conda update --all -y
                
                echo "Updating pip packages..."
                pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
                pip install --upgrade ultralytics
                pip install --upgrade \
                    opencv-python \
                    numpy \
                    matplotlib \
                    seaborn \
                    pandas \
                    scikit-learn \
                    tensorboard \
                    albumentations \
                    onnx \
                    onnxruntime-gpu \
                    pyyaml
                
                if [ -d "${SCRIPT_DIR}/modules" ]; then
                    pip install -e "${SCRIPT_DIR}" --quiet --upgrade
                fi
                
                print_success "Environment updated successfully!"
                SKIP_TO_VERIFY=true
                ;;
            2|recreate|Recreate|RECREATE)
                echo "Removing existing environment..."
                ${PKG_MANAGER} env remove -n ${ENV_NAME} -y || conda env remove -n ${ENV_NAME} -y
                SKIP_TO_VERIFY=false
                ;;
            3|skip|Skip|SKIP)
                echo "Activating existing environment..."
                conda activate ${ENV_NAME}
                
                MISSING_PKGS=""
                python -c "import torch" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} torch"
                python -c "import ultralytics" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} ultralytics"
                python -c "import yaml" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} pyyaml"
                
                if [ -n "$MISSING_PKGS" ]; then
                    print_error "Missing critical packages:${MISSING_PKGS}"
                    echo ""
                    echo "Please select option [1] Update or [2] Recreate instead."
                    exit 1
                fi
                
                print_success "Environment ready!"
                exit 0
                ;;
            4|cancel|Cancel|CANCEL|*)
                echo "Setup cancelled."
                exit 0
                ;;
        esac
    fi
}

#===============================================================================
# CREATE CONDA ENVIRONMENT
#===============================================================================

create_conda_env() {
    print_header "Creating Conda Environment"
    
    echo "Environment: ${ENV_NAME}"
    echo "Python version: ${PYTHON_VERSION}"
    echo ""
    
    ${PKG_MANAGER} create -n ${ENV_NAME} python=${PYTHON_VERSION} -y || \
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    
    conda activate ${ENV_NAME}
    
    print_success "Conda environment created and activated"
    echo ""
}

#===============================================================================
# INSTALL NCURSES
#===============================================================================

install_ncurses() {
    print_header "Installing ncurses (fixes libtinfo compatibility)"
    conda install -c conda-forge ncurses -y
    print_success "ncurses installed"
    echo ""
}

#===============================================================================
# INSTALL PYTORCH
#===============================================================================

install_pytorch() {
    print_header "Installing PyTorch"
    
    echo "Installing PyTorch with ${PYTORCH_CUDA}..."
    echo ""
    
    if [ "$PYTORCH_CUDA" = "cpu" ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    elif [[ "$PYTORCH_CUDA" == nightly/* ]]; then
        print_warning "Detected RTX 50-series GPU (Blackwell architecture)"
        print_info "Installing PyTorch nightly build with CUDA 12.8 support..."
        pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
    fi
    
    print_success "PyTorch installed"
    echo ""
}

#===============================================================================
# INSTALL ULTRALYTICS
#===============================================================================

install_ultralytics() {
    print_header "Installing Ultralytics YOLO"
    pip install ultralytics
    print_success "Ultralytics YOLO installed"
    echo ""
}

#===============================================================================
# INSTALL ADDITIONAL DEPENDENCIES
#===============================================================================

install_dependencies() {
    print_header "Installing Additional Dependencies"
    
    pip install \
        opencv-python \
        numpy \
        matplotlib \
        seaborn \
        pandas \
        scikit-learn \
        tensorboard \
        albumentations \
        onnx \
        onnxruntime-gpu \
        pyyaml
    
    print_success "Additional dependencies installed"
    echo ""
}

#===============================================================================
# INSTALL LOCAL MODULES
#===============================================================================

install_local_modules() {
    print_header "Setting up Local Modules"
    
    if [ -d "${SCRIPT_DIR}/modules" ]; then
        if [ ! -f "${SCRIPT_DIR}/modules/setup.py" ]; then
            cat > "${SCRIPT_DIR}/modules/setup.py" << 'SETUP_PY'
from setuptools import setup, find_packages

setup(
    name="microspore_training",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    author="Microspore Phenotyping Team",
    description="Training modules for microspore phenotyping YOLO models",
)
SETUP_PY
            print_info "Created setup.py for local modules"
        fi
        
        pip install -e "${SCRIPT_DIR}/modules" --quiet
        print_success "Local modules installed in development mode"
    else
        print_warning "modules folder not found, skipping local module installation"
    fi
    
    echo ""
}

#===============================================================================
# VERIFY INSTALLATION
#===============================================================================

verify_installation() {
    print_header "Verifying Installation"
    
    echo "Checking installed packages..."
    echo ""
    
    PYTORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    print_success "PyTorch version: $PYTORCH_VER"
    
    CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAIL" = "True" ]; then
        CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_success "CUDA available: $CUDA_VER"
        print_success "GPU: $GPU_NAME"
    else
        if [ "$GPU_AVAILABLE" = true ]; then
            print_warning "CUDA not available in PyTorch (check installation)"
        else
            print_info "CUDA not available (CPU-only installation)"
        fi
    fi
    
    ULTRA_VER=$(python -c "import ultralytics; print(ultralytics.__version__)" 2>/dev/null)
    print_success "Ultralytics version: $ULTRA_VER"
    
    if python -c "import modules" 2>/dev/null; then
        print_success "Local modules: Available"
    else
        print_warning "Local modules: Not installed"
    fi
    
    echo ""
}

#===============================================================================
# GENERATE DATASET STATISTICS
#===============================================================================

generate_dataset_stats() {
    print_header "Generating Dataset Statistics"
    
    python << STATS_SCRIPT
import sys
sys.path.insert(0, '.')

try:
    from modules.stats import DatasetStats
    from modules.utils import save_json
    
    dataset_path = "${COMMON_DATASETS_DIR}/${COMMON_DATASET_NAME}"
    dataset = DatasetStats(dataset_path)
    stats = dataset.get_full_stats()
    
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Classes: {len(stats['classes'])}")
    
    for split, split_stats in stats.get('splits', {}).items():
        print(f"  {split.capitalize()}: {split_stats['images']} images, {split_stats['total_annotations']} annotations")
    
    save_json(stats, f"{dataset_path}/dataset_stats.json")
    print(f"\n  Stats saved to: {dataset_path}/dataset_stats.json")
    
except Exception as e:
    print(f"  Could not generate stats: {e}")
STATS_SCRIPT
    
    DATA_DIR="${COMMON_DATASETS_DIR}/${COMMON_DATASET_NAME}"
    if [ -f "${DATA_DIR}/tally.py" ]; then
        print_info "Running tally.py for class distribution..."
        cd "${DATA_DIR}"
        python tally.py || print_warning "tally.py failed (check classes.txt and label files)"
        cd "${SCRIPT_DIR}"
    fi
    
    echo ""
}

#===============================================================================
# PRINT COMPLETION MESSAGE
#===============================================================================

print_completion() {
    local run_train_script="${1:-run_train_local.sh}"
    
    print_header "Environment Setup Complete!"
    
    echo "To activate the environment, run:"
    echo "    conda activate ${ENV_NAME}"
    echo ""
    echo "To verify PyTorch CUDA availability, run:"
    echo "    python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
    echo ""
    echo "To download YOLO models, run:"
    echo "    chmod +x yolo_models_setup.sh"
    echo "    ./yolo_models_setup.sh"
    echo ""
    echo "To start training, run:"
    echo "    chmod +x ${run_train_script}"
    echo "    ./${run_train_script}"
    echo ""
    
    CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAIL" = "True" ]; then
        print_success "GPU training is ready!"
    else
        print_warning "Training will run on CPU (slower)"
    fi
    echo ""
}

#===============================================================================
# MAIN SETUP EXECUTION
#===============================================================================

run_conda_setup() {
    local gpu_setup_script="${1:-gpu_setup_local.sh}"
    local run_train_script="${2:-run_train_local.sh}"
    
    print_header "Microspore Phenotyping Training Setup (Conda)"
    
    preflight_gpu_check
    offer_gpu_setup "$gpu_setup_script"
    init_conda_for_setup
    handle_existing_env
    
    if [ "$SKIP_TO_VERIFY" = true ]; then
        : # Skip to verification
    else
        create_conda_env
        install_ncurses
        install_pytorch
        install_ultralytics
        install_dependencies
        install_local_modules
    fi
    
    verify_installation
    generate_dataset_stats
    print_completion "$run_train_script"
}
