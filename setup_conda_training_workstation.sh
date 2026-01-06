#!/bin/bash
#===============================================================================
# Microspore Phenotyping - Conda Environment Setup Script (WORKSTATION VERSION)
# Creates and configures the 'training' conda environment for YOLO training
# Uses common_functions.sh for shared utilities (DRY principle)
#
# WORKSTATION SPECS (Target Configuration):
#   GPU: NVIDIA RTX 4000 Ada Generation
#   VRAM: 20GB (20475 MiB)
#   Driver: 573.44
#   CUDA: 12.8
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
PYTHON_VERSION="${COMMON_PYTHON_VERSION}"

print_header "Microspore Phenotyping Training Setup (Conda)"

#===============================================================================
# PRE-FLIGHT GPU CHECK
#===============================================================================

echo "Checking GPU availability..."
echo ""

# Use common function to get GPU info
get_gpu_info

if [ "$GPU_AVAILABLE" = true ]; then
    print_success "NVIDIA GPU detected"
    echo "    GPU: $GPU_NAME"
    echo "    Memory: $GPU_MEMORY"
    echo "    Driver: $GPU_DRIVER_VERSION"
    echo "    CUDA: $GPU_CUDA_VERSION"
    echo ""
    
    # Get appropriate PyTorch CUDA version using common function
    PYTORCH_CUDA=$(get_pytorch_cuda_version)
    print_info "Using PyTorch with ${PYTORCH_CUDA}"
else
    print_warning "nvidia-smi not found or GPU not accessible"
fi

# Offer to run GPU setup if not available
if [ "$GPU_AVAILABLE" = false ]; then
    echo ""
    print_warning "GPU not detected or not properly configured."
    echo ""
    
    if [ -f "${SCRIPT_DIR}/gpu_setup.sh" ]; then
        read -p "Would you like to run gpu_setup.sh first? (y/n): " run_gpu_setup
        if [[ "$run_gpu_setup" == "y" || "$run_gpu_setup" == "Y" ]]; then
            chmod +x "${SCRIPT_DIR}/gpu_setup.sh"
            "${SCRIPT_DIR}/gpu_setup.sh"
            
            # Re-check GPU after setup
            get_gpu_info
            if [ "$GPU_AVAILABLE" = true ]; then
                PYTORCH_CUDA=$(get_pytorch_cuda_version)
            fi
        fi
    else
        echo "Run gpu_setup.sh to configure GPU support."
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

#===============================================================================
# CONDA CHECK
#===============================================================================

print_header "Checking Conda Installation"

# Use common function to initialize conda
if ! init_conda_shell; then
    exit 1
fi

# Initialize conda for bash if needed
eval "$(conda shell.bash hook)"

# Check for mamba (faster alternative to conda)
if command -v mamba &> /dev/null; then
    PKG_MANAGER="mamba"
    print_success "Using mamba (faster package manager)"
else
    PKG_MANAGER="conda"
    print_info "Using conda (install mamba for faster operations: conda install -n base -c conda-forge mamba)"
fi

# Check if environment already exists
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
            
            # Update packages
            echo "Updating packages with ${PKG_MANAGER}..."
            ${PKG_MANAGER} update --all -y || conda update --all -y
            
            # Update pip packages
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
            
            # Reinstall local modules
            if [ -d "${SCRIPT_DIR}/modules" ]; then
                pip install -e "${SCRIPT_DIR}" --quiet --upgrade
            fi
            
            print_success "Environment updated successfully!"
            
            # Jump to verification
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
            
            # Verify critical packages are installed
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

# Skip installation if we just updated
if [ "$SKIP_TO_VERIFY" = true ]; then
    # Jump directly to verification section
    :
else

echo ""

#===============================================================================
# CREATE CONDA ENVIRONMENT
#===============================================================================

print_header "Creating Conda Environment"

echo "Environment: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo ""

# Create the environment (mamba if available, fallback to conda)
${PKG_MANAGER} create -n ${ENV_NAME} python=${PYTHON_VERSION} -y || conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate the environment
conda activate ${ENV_NAME}

print_success "Conda environment created and activated"
echo ""

#===============================================================================
# INSTALL SYSTEM UTILITIES
#===============================================================================

print_header "Installing system utilities (ncurses, pigz)"

mamba install -c conda-forge ncurses pigz -y

print_success "ncurses and pigz installed"
echo ""

#===============================================================================
# INSTALL PYTORCH
#===============================================================================

print_header "Installing PyTorch"

echo "Installing PyTorch with ${PYTORCH_CUDA}..."
echo ""

# Install PyTorch with appropriate CUDA version (detected earlier)
if [ "$PYTORCH_CUDA" = "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [[ "$PYTORCH_CUDA" == nightly/* ]]; then
    # RTX 50-series (Blackwell) requires nightly build with CUDA 12.8+
    print_warning "Detected RTX 50-series GPU (Blackwell architecture)"
    print_info "Installing PyTorch nightly build with CUDA 12.8 support..."
    pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
fi

print_success "PyTorch installed"
echo ""

#===============================================================================
# INSTALL ULTRALYTICS YOLO
#===============================================================================

print_header "Installing Ultralytics YOLO"

# Install ultralytics (includes YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11)
pip install ultralytics

print_success "Ultralytics YOLO installed"
echo ""

#===============================================================================
# INSTALL ADDITIONAL DEPENDENCIES
#===============================================================================

print_header "Installing Additional Dependencies"

# Additional useful packages
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

#===============================================================================
# INSTALL LOCAL MODULES
#===============================================================================

print_header "Setting up Local Modules"

# Install the local modules package in development mode
if [ -d "${SCRIPT_DIR}/modules" ]; then
    # Create a minimal setup.py if it doesn't exist
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

fi  # End of SKIP_TO_VERIFY check

#===============================================================================
# VERIFICATION
#===============================================================================

print_header "Verifying Installation"

echo "Checking installed packages..."
echo ""

# Verify PyTorch
PYTORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
print_success "PyTorch version: $PYTORCH_VER"

# Verify CUDA
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

# Verify Ultralytics
ULTRA_VER=$(python -c "import ultralytics; print(ultralytics.__version__)" 2>/dev/null)
print_success "Ultralytics version: $ULTRA_VER"

# Verify local modules
if python -c "import modules" 2>/dev/null; then
    print_success "Local modules: Available"
else
    print_warning "Local modules: Not installed"
fi

echo ""

#===============================================================================
# GENERATE DATASET STATISTICS
#===============================================================================

print_header "Generating Dataset Statistics"

# Use the external Python script (DRY principle - no embedded Python)
DATA_DIR="${COMMON_DATASETS_DIR}/${COMMON_DATASET_NAME}"
python -m modules.scripts.generate_dataset_stats --dataset-path "${DATA_DIR}" || print_warning "Could not generate dataset stats"

# Run tally.py for class distribution if it exists
DATA_DIR="${COMMON_DATASETS_DIR}/${COMMON_DATASET_NAME}"
if [ -f "${DATA_DIR}/tally.py" ]; then
    print_info "Running tally.py for class distribution..."
    cd "${DATA_DIR}"
    python tally.py || print_warning "tally.py failed (check classes.txt and label files)"
    cd "${SCRIPT_DIR}"
fi

echo ""

#===============================================================================
# SETUP COMPLETE
#===============================================================================

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
echo "    chmod +x run_train.sh"
echo "    ./run_train.sh"
echo ""

if [ "$CUDA_AVAIL" = "True" ]; then
    print_success "GPU training is ready!"
else
    print_warning "Training will run on CPU (slower)"
fi
echo ""
