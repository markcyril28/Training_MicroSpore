#!/bin/bash 
#===============================================================================
# Microspore Phenotyping - Conda Environment Setup Script (SERVER VERSION)
# Creates and configures the 'training' conda environment for YOLO training
# Uses common_functions.sh for shared utilities (DRY principle)
#
# SERVER SPECS (Dell Server with AMD ROCm):
#   GPU: AMD Instinct MI210 (Aldebaran/MI200)
#   VRAM: 64GB HBM2e
#   Architecture: gfx90a (CDNA2)
#   Driver: amdgpu
#   Compute Platform: ROCm 6.x
#   CPU Threads: 32
#   PyTorch: ROCm version (torch+rocm)
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

print_header "Microspore Phenotyping Training Setup (Server - AMD ROCm)"

#===============================================================================
# AMD GPU UTILITIES
#===============================================================================

# Check if AMD GPU is available via rocm-smi
check_amd_gpu() {
    if command -v rocm-smi &> /dev/null; then
        if rocm-smi --showproductname &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Get AMD GPU information
get_amd_gpu_info() {
    if check_amd_gpu; then
        AMD_GPU_NAME="AMD Instinct MI210"
        AMD_GPU_MEMORY="64GB HBM2e"
        AMD_GPU_ARCH="gfx90a (CDNA2)"
        AMD_ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || echo "Unknown")
        AMD_GPU_AVAILABLE=true
    else
        AMD_GPU_NAME=""
        AMD_GPU_MEMORY=""
        AMD_GPU_ARCH=""
        AMD_ROCM_VERSION=""
        AMD_GPU_AVAILABLE=false
    fi
}

#===============================================================================
# PRE-FLIGHT GPU CHECK
#===============================================================================

echo "Checking GPU availability (AMD ROCm)..."
echo ""

get_amd_gpu_info

if [ "$AMD_GPU_AVAILABLE" = true ]; then
    print_success "AMD GPU detected"
    echo "    GPU: $AMD_GPU_NAME"
    echo "    Memory: $AMD_GPU_MEMORY"
    echo "    Architecture: $AMD_GPU_ARCH"
    echo "    ROCm: $AMD_ROCM_VERSION"
    echo ""
    
    # Set PyTorch variant for ROCm
    PYTORCH_ROCM_VERSION="rocm6.2.4"  # Adjust based on installed ROCm version
    print_info "Will install PyTorch with ROCm support"
else
    print_warning "rocm-smi not found or AMD GPU not accessible"
fi

# Offer to run GPU setup if not available
if [ "$AMD_GPU_AVAILABLE" = false ]; then
    echo ""
    print_warning "AMD GPU not detected or not properly configured."
    echo ""
    
    if [ -f "${SCRIPT_DIR}/gpu_setup_server.sh" ]; then
        read -p "Would you like to run gpu_setup_server.sh first? (y/n): " run_gpu_setup
        if [[ "$run_gpu_setup" == "y" || "$run_gpu_setup" == "Y" ]]; then
            chmod +x "${SCRIPT_DIR}/gpu_setup_server.sh"
            "${SCRIPT_DIR}/gpu_setup_server.sh"
            
            # Re-check GPU after setup
            get_amd_gpu_info
        fi
    else
        echo "Run gpu_setup_server.sh to configure GPU support."
    fi
    
    if [ "$AMD_GPU_AVAILABLE" = false ]; then
        read -p "Continue with CPU-only installation? (y/n): " continue_cpu
        if [[ "$continue_cpu" != "y" && "$continue_cpu" != "Y" ]]; then
            echo "Setup cancelled."
            exit 0
        fi
        PYTORCH_ROCM_VERSION="cpu"
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

# Configure conda for faster metadata fetching
print_info "Configuring conda for faster operations..."
conda config --set channel_priority strict
conda config --set remote_read_timeout_secs 120
conda config --set remote_connect_timeout_secs 30
conda config --set remote_max_retries 3

# Clean conda cache if it might be corrupted (optional but helps with hangs)
echo "Cleaning conda cache (this helps prevent metadata hangs)..."
conda clean --index-cache --quiet 2>/dev/null || true

# Check for mamba (faster alternative to conda)
if command -v mamba &> /dev/null; then
    PKG_MANAGER="mamba"
    print_success "Using mamba (faster package manager)"
else
    PKG_MANAGER="conda"
    print_warning "Using conda - this can be slow. Consider installing mamba:"
    echo "    conda install -n base -c conda-forge mamba"
    echo ""
    print_info "Tip: If conda hangs on 'Collecting package metadata', try:"
    echo "    1. Run: conda clean --all"
    echo "    2. Check network connectivity"
    echo "    3. Install mamba for faster operations"
    echo ""
fi

# Helper function: run command with mamba, fallback to conda if mamba fails
run_pkg_cmd() {
    local cmd="$1"
    shift
    if [ "$PKG_MANAGER" = "mamba" ]; then
        mamba $cmd "$@" || { print_warning "mamba failed, falling back to conda..."; conda $cmd "$@"; }
    else
        conda $cmd "$@"
    fi
}

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
            run_pkg_cmd update --all -y
            
            # Update pip packages - PyTorch ROCm
            echo "Updating PyTorch ROCm packages..."
            if [ "$PYTORCH_ROCM_VERSION" != "cpu" ]; then
                pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_ROCM_VERSION}
            else
                pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            fi
            
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
                onnxruntime \
                pyyaml
            
            # Reinstall local modules
            if [ -d "${SCRIPT_DIR}/modules" ]; then
                pip install -e "${SCRIPT_DIR}/modules" --quiet --upgrade
            fi
            
            print_success "Environment updated successfully!"
            
            # Jump to verification
            SKIP_TO_VERIFY=true
            ;;
        2|recreate|Recreate|RECREATE)
            echo "Removing existing environment..."
            run_pkg_cmd env remove -n ${ENV_NAME} -y
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

# Create the environment with explicit channels (prevents slow default channel queries)
# Using --override-channels to avoid metadata fetching from unnecessary sources
print_info "Creating environment (this may take a few minutes)..."
run_pkg_cmd create -n ${ENV_NAME} python=${PYTHON_VERSION} -c conda-forge --override-channels -y

# Activate the environment
conda activate ${ENV_NAME}

print_success "Conda environment created and activated"
echo ""

#===============================================================================
# SETUP ROCM ENVIRONMENT VARIABLES
#===============================================================================

print_header "Setting up ROCm Environment"

ROCM_PATH="/opt/rocm"

if [ -d "$ROCM_PATH" ] && [ -n "$CONDA_PREFIX" ]; then
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
    
    # Copy activation script for ROCm from modules
    ROCM_ACTIVATE_SRC="${SCRIPT_DIR}/modules/setup/rocm_env_activate.sh"
    ROCM_DEACTIVATE_SRC="${SCRIPT_DIR}/modules/setup/rocm_env_deactivate.sh"
    
    if [ -f "$ROCM_ACTIVATE_SRC" ]; then
        cp "$ROCM_ACTIVATE_SRC" "$CONDA_PREFIX/etc/conda/activate.d/rocm_env.sh"
        chmod +x "$CONDA_PREFIX/etc/conda/activate.d/rocm_env.sh"
    else
        print_warning "ROCm activation script not found: $ROCM_ACTIVATE_SRC"
    fi
    
    # Copy deactivation script for ROCm from modules
    if [ -f "$ROCM_DEACTIVATE_SRC" ]; then
        cp "$ROCM_DEACTIVATE_SRC" "$CONDA_PREFIX/etc/conda/deactivate.d/rocm_env.sh"
        chmod +x "$CONDA_PREFIX/etc/conda/deactivate.d/rocm_env.sh"
    else
        print_warning "ROCm deactivation script not found: $ROCM_DEACTIVATE_SRC"
    fi
    
    # Source the ROCm environment now
    source "$CONDA_PREFIX/etc/conda/activate.d/rocm_env.sh"
    
    print_success "ROCm environment configured"
else
    print_warning "ROCm path not found at ${ROCM_PATH} or CONDA_PREFIX not set"
fi

echo ""

#===============================================================================
# INSTALL SYSTEM UTILITIES
#===============================================================================

print_header "Installing system utilities (ncurses, pigz)"

run_pkg_cmd install -c conda-forge --override-channels ncurses pigz -y

print_success "ncurses and pigz installed"
echo ""

#===============================================================================
# INSTALL PYTORCH (ROCm VERSION)
#===============================================================================

print_header "Installing PyTorch (ROCm Version)"

echo "Installing PyTorch with ROCm support for AMD Instinct MI210..."
echo ""

# Install PyTorch with ROCm support
if [ "$PYTORCH_ROCM_VERSION" = "cpu" ]; then
    print_warning "Installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    print_info "Installing PyTorch with ${PYTORCH_ROCM_VERSION} support"
    echo ""
    echo "This may take a few minutes as ROCm PyTorch wheels are large..."
    echo ""
    
    # Install PyTorch for ROCm
    # ROCm 6.2.4 is the latest stable as of early 2026
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_ROCM_VERSION}
    
    # Alternative: nightly builds for latest features
    # pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
fi

print_success "PyTorch ROCm installed"
echo ""

#===============================================================================
# INSTALL ULTRALYTICS YOLO
#===============================================================================

print_header "Installing Ultralytics YOLO"

# Install ultralytics (includes YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11)
# NOTE: YOLOv4 is NOT supported by Ultralytics - use run_train_yolov4_server.sh instead
pip install ultralytics

print_success "Ultralytics YOLO installed (v5, v8, v9, v10, v11)"
echo ""

#===============================================================================
# INSTALL ADDITIONAL DEPENDENCIES
#===============================================================================

print_header "Installing Additional Dependencies"

# Additional useful packages
# Note: onnxruntime-gpu is NVIDIA-specific, use onnxruntime for AMD
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
    onnxruntime \
    pyyaml

print_success "Additional dependencies installed"
echo ""

#===============================================================================
# INSTALL LOCAL MODULES
#===============================================================================

print_header "Setting up Local Modules"

# Install the local modules package in development mode
if [ -d "${SCRIPT_DIR}/modules" ]; then
    # Copy setup.py from template if it doesn't exist
    if [ ! -f "${SCRIPT_DIR}/modules/setup.py" ]; then
        SETUP_TEMPLATE="${SCRIPT_DIR}/modules/setup/setup_template.py"
        if [ -f "$SETUP_TEMPLATE" ]; then
            cp "$SETUP_TEMPLATE" "${SCRIPT_DIR}/modules/setup.py"
            print_info "Created setup.py from template"
        else
            print_warning "setup_template.py not found, cannot create setup.py"
        fi
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

# Verify ROCm/HIP support
ROCM_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$ROCM_AVAIL" = "True" ]; then
    HIP_VER=$(python -c "import torch; print(torch.version.hip)" 2>/dev/null || echo "Unknown")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    print_success "ROCm/HIP available: $HIP_VER"
    print_success "GPU: $GPU_NAME"
    print_success "GPU count: $GPU_COUNT"
    
    # Test basic tensor operation on GPU
    echo ""
    echo "Testing GPU tensor operation..."
    python -c "
import torch
x = torch.randn(1000, 1000, device='cuda')
y = torch.matmul(x, x)
print(f'  Tensor operation successful on {torch.cuda.get_device_name(0)}')
print(f'  VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB')
" 2>/dev/null && print_success "GPU tensor test passed" || print_warning "GPU tensor test failed"
else
    if [ "$AMD_GPU_AVAILABLE" = true ]; then
        print_warning "ROCm not available in PyTorch (check installation)"
        echo ""
        echo "Troubleshooting tips:"
        echo "  1. Ensure ROCm is properly installed: rocm-smi"
        echo "  2. Check if user is in render group: groups"
        echo "  3. Set environment variable: export HSA_OVERRIDE_GFX_VERSION=9.0.10"
        echo "  4. Reinstall PyTorch ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2.4"
    else
        print_info "ROCm not available (CPU-only installation)"
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
# AMD MI210 OPTIMIZATION TIPS
#===============================================================================

print_header "AMD Instinct MI210 Optimization Tips"

echo "Your server has 64GB HBM2e VRAM - here are recommended settings:"
echo ""
echo "For maximum performance:"
echo "  BATCH_SIZE=64       # Leverage the large VRAM"
echo "  IMG_SIZE=1024       # High resolution (MI210 can handle it)"
echo "  WORKERS=16          # ~half of 32 threads"
echo "  CACHE=ram           # Cache images in RAM"
echo "  AMP=true            # Mixed precision training"
echo ""
echo "For training large models:"
echo "  YOLO_MODEL=\"yolo11x.pt\"  # Extra-large model"
echo "  BATCH_SIZE=32             # Still plenty of headroom"
echo ""
echo "Environment variables for MI210:"
echo "  export HSA_OVERRIDE_GFX_VERSION=9.0.10"
echo "  export HIP_VISIBLE_DEVICES=0"
echo ""

#===============================================================================
# SETUP COMPLETE
#===============================================================================

print_header "Environment Setup Complete!"

echo "To activate the environment, run:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "To verify PyTorch ROCm availability, run:"
echo "    python -c \"import torch; print(f'ROCm available: {torch.cuda.is_available()}')\""
echo ""
echo "To download YOLO models (Ultralytics), run:"
echo "    chmod +x yolo_models_setup.sh"
echo "    ./yolo_models_setup.sh"
echo ""
echo "Supported Ultralytics models: YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11"
echo ""
echo "For YOLOv4 (requires Darknet pipeline):"
echo "    chmod +x run_train_yolov4_server.sh"
echo "    ./run_train_yolov4_server.sh"
echo ""
echo "To start Ultralytics training, run:"
echo "    chmod +x run_train_server.sh"
echo "    ./run_train_server.sh"
echo ""

if [ "$ROCM_AVAIL" = "True" ]; then
    print_success "GPU training is ready!"
    echo ""
    echo "GPU: AMD Instinct MI210 (64GB HBM2e)"
    echo "Architecture: gfx90a (CDNA2)"
else
    print_warning "Training will run on CPU (slower)"
fi
echo ""
