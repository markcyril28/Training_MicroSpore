#!/bin/bash
#===============================================================================
# Microspore Phenotyping - GPU Setup Script for WSL/Ubuntu (Conda-based)
# Detects, installs, and configures NVIDIA GPU drivers and CUDA for training.
# Uses common_functions.sh for shared utilities (DRY principle).
#
# WORKSTATION SPECS (Target Configuration):
#   GPU: NVIDIA RTX 4000 Ada Generation
#   VRAM: 20GB (20475 MiB)
#   Driver: 573.44
#   CUDA: 12.8
#===============================================================================

: << 'WORKSTATION_SPECS'
nvidia-smi
Sat Jan  3 11:45:58 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.170                Driver Version: 573.44         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 4000 Ada Gene...    On  |   00000000:01:00.0  On |                  Off |
| 30%   32C    P8              9W /  130W |     861MiB /  20475MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

WORKSTATION_SPECS

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

#===============================================================================
# CONDA CHECK AND INITIALIZATION
#===============================================================================

init_conda_env() {
    print_header "Conda Environment Check"
    
    # Use common function
    if ! init_conda_shell; then
        exit 1
    fi
    
    # Check if training environment exists
    if check_conda_env_exists "${ENV_NAME}"; then
        print_success "Conda environment '${ENV_NAME}' exists"
        CONDA_ENV_EXISTS=true
    else
        print_info "Conda environment '${ENV_NAME}' not found"
        print_info "Will be created by setup_conda_training.sh"
        CONDA_ENV_EXISTS=false
    fi
    
    echo ""
}

#===============================================================================
# GPU DETECTION
#===============================================================================

detect_gpu() {
    print_header "GPU Detection"
    
    # Check if running in WSL using common function
    if is_wsl; then
        print_info "Running in WSL (Windows Subsystem for Linux)"
        IS_WSL=true
    else
        print_info "Running in native Linux"
        IS_WSL=false
    fi
    
    # Check for NVIDIA GPU via lspci (may not work in WSL)
    if command -v lspci &> /dev/null; then
        GPU_INFO=$(lspci | grep -i nvidia 2>/dev/null || echo "")
        if [ -n "$GPU_INFO" ]; then
            print_success "NVIDIA GPU detected via lspci:"
            echo "    $GPU_INFO"
        fi
    fi
    
    # Use common function to get and display GPU info
    if check_nvidia_gpu; then
        print_success "nvidia-smi is available"
        echo ""
        print_gpu_info
        GPU_READY=true
    else
        print_warning "nvidia-smi not found"
        GPU_READY=false
    fi
    
    echo ""
}

#===============================================================================
# WSL GPU SETUP INSTRUCTIONS
#===============================================================================

wsl_gpu_instructions() {
    print_header "WSL GPU Setup Instructions"
    
    echo "For WSL, GPU drivers are managed through Windows."
    echo ""
    echo "Required steps on WINDOWS (not in WSL):"
    echo ""
    echo "1. Install NVIDIA GPU Driver for Windows:"
    echo "   https://www.nvidia.com/download/index.aspx"
    echo "   (Must be version 470.76 or later for WSL support)"
    echo ""
    echo "2. Enable WSL GPU support (Windows 11 or Windows 10 21H2+):"
    echo "   - Open PowerShell as Administrator"
    echo "   - Run: wsl --update"
    echo ""
    echo "3. Restart WSL:"
    echo "   - Run in PowerShell: wsl --shutdown"
    echo "   - Reopen your WSL terminal"
    echo ""
    echo "After completing these steps, run this script again."
    echo ""
}

#===============================================================================
# CUDA TOOLKIT INSTALLATION (Conda-based)
#===============================================================================

install_cuda_conda() {
    print_header "CUDA Toolkit Installation (Conda)"
    
    # Initialize conda
    eval "$(conda shell.bash hook)"
    
    # Create or activate the training environment
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        print_info "Creating conda environment '${ENV_NAME}'..."
        conda create -n ${ENV_NAME} python=3.10 -y
    fi
    
    conda activate ${ENV_NAME}
    print_success "Activated conda environment: ${ENV_NAME}"
    
    # Check if CUDA toolkit is already installed in conda
    if conda list cuda-toolkit 2>/dev/null | grep -q cuda-toolkit; then
        CUDA_VER=$(conda list cuda-toolkit | grep cuda-toolkit | awk '{print $2}')
        print_success "CUDA Toolkit already installed in conda: $CUDA_VER"
        return 0
    fi
    
    print_info "Installing CUDA Toolkit via conda..."
    echo ""
    
    # Install CUDA toolkit from conda-forge or nvidia channel
    # This keeps everything within conda environment
    conda install -c conda-forge cudatoolkit=12.1 -y || \
    conda install -c nvidia cuda-toolkit -y || \
    print_warning "Could not install CUDA via conda, will use system CUDA"
    
    print_success "CUDA Toolkit installation complete"
}

#===============================================================================
# CUDNN INSTALLATION (Conda-based)
#===============================================================================

install_cudnn_conda() {
    print_header "cuDNN Installation (Conda)"
    
    # Ensure conda environment is active
    eval "$(conda shell.bash hook)"
    
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "Conda environment '${ENV_NAME}' not found. Skipping cuDNN."
        return 0
    fi
    
    conda activate ${ENV_NAME}
    
    # Check if cuDNN is already installed in conda
    if conda list cudnn 2>/dev/null | grep -q cudnn; then
        CUDNN_VER=$(conda list cudnn | grep cudnn | awk '{print $2}')
        print_success "cuDNN already installed in conda: $CUDNN_VER"
        return 0
    fi
    
    print_info "Installing cuDNN via conda..."
    
    # Install cuDNN from conda-forge or nvidia channel
    conda install -c conda-forge cudnn -y 2>/dev/null || \
    conda install -c nvidia cudnn -y 2>/dev/null || \
    print_info "cuDNN not installed via conda (PyTorch bundles cuDNN, so this is OK)"
    
    echo ""
    print_info "Note: PyTorch includes cuDNN, so separate installation is optional."
    echo ""
}

#===============================================================================
# CUDA ENVIRONMENT SETUP (Conda-based)
#===============================================================================

setup_cuda_env() {
    print_header "Setting up CUDA Environment (Conda)"
    
    # Initialize conda
    eval "$(conda shell.bash hook)"
    
    # Conda handles most environment variables automatically
    # We just need to set a few for compatibility
    
    # Check for conda CUDA path first
    CONDA_CUDA_PATH=""
    if [ -n "$CONDA_PREFIX" ]; then
        if [ -d "$CONDA_PREFIX/lib" ]; then
            CONDA_CUDA_PATH="$CONDA_PREFIX"
        fi
    fi
    
    # Fallback to system CUDA
    SYSTEM_CUDA_PATH=""
    if [ -d "/usr/local/cuda-12.8" ]; then
        SYSTEM_CUDA_PATH="/usr/local/cuda-12.8"
    elif [ -d "/usr/local/cuda-12.4" ]; then
        SYSTEM_CUDA_PATH="/usr/local/cuda-12.4"
    elif [ -d "/usr/local/cuda-12.1" ]; then
        SYSTEM_CUDA_PATH="/usr/local/cuda-12.1"
    elif [ -d "/usr/local/cuda-12.0" ]; then
        SYSTEM_CUDA_PATH="/usr/local/cuda-12.0"
    elif [ -d "/usr/local/cuda-11.8" ]; then
        SYSTEM_CUDA_PATH="/usr/local/cuda-11.8"
    elif [ -d "/usr/local/cuda" ]; then
        SYSTEM_CUDA_PATH="/usr/local/cuda"
    fi
    
    # Add conda activation script for CUDA environment variables
    if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/etc/conda/activate.d" ] 2>/dev/null; then
        ACTIVATE_SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh"
        
        if [ ! -f "$ACTIVATE_SCRIPT" ]; then
            mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
            cat > "$ACTIVATE_SCRIPT" << 'CUDA_ENV'
#!/bin/bash
# CUDA environment variables for training conda environment
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi
# Use system CUDA if available
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
CUDA_ENV
            chmod +x "$ACTIVATE_SCRIPT"
            print_success "Created conda activation script for CUDA"
        else
            print_info "CUDA activation script already exists"
        fi
    fi
    
    # Display current configuration
    if [ -n "$CONDA_CUDA_PATH" ]; then
        print_info "Conda CUDA path: $CONDA_CUDA_PATH"
    fi
    if [ -n "$SYSTEM_CUDA_PATH" ]; then
        print_info "System CUDA path: $SYSTEM_CUDA_PATH"
    fi
    
    echo ""
}

#===============================================================================
# CUDNN INSTALLATION
#===============================================================================

install_cudnn() {
    # Redirect to conda-based installation
    install_cudnn_conda
}

#===============================================================================
# VERIFY GPU SETUP
#===============================================================================

verify_gpu_setup() {
    print_header "Verifying GPU Setup"
    
    ERRORS=0
    
    # Check nvidia-smi
    echo "1. Checking nvidia-smi..."
    if nvidia-smi &> /dev/null; then
        print_success "nvidia-smi works"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    else
        print_error "nvidia-smi failed"
        ERRORS=$((ERRORS + 1))
    fi
    echo ""
    
    # Check CUDA
    echo "2. Checking CUDA..."
    if command -v nvcc &> /dev/null; then
        print_success "CUDA compiler (nvcc) found"
        nvcc --version | grep "release"
    else
        print_warning "nvcc not found (optional for PyTorch)"
    fi
    echo ""
    
    # Check PyTorch CUDA (if installed in conda environment)
    echo "3. Checking PyTorch CUDA support (Conda)..."
    
    # Initialize conda and check in training environment
    eval "$(conda shell.bash hook)"
    
    if conda env list | grep -q "^${ENV_NAME} "; then
        conda activate ${ENV_NAME}
        
        if python -c "import torch" 2>/dev/null; then
            PYTORCH_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
            if [ "$PYTORCH_CUDA" = "True" ]; then
                print_success "PyTorch CUDA is available in '${ENV_NAME}' environment"
                python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
                python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')"
                python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
            else
                print_warning "PyTorch installed but CUDA not available"
                print_info "Run setup_conda_training.sh to reinstall PyTorch with CUDA"
            fi
        else
            print_info "PyTorch not installed in '${ENV_NAME}' environment"
            print_info "Run setup_conda_training.sh to install PyTorch with CUDA"
        fi
    else
        print_info "Conda environment '${ENV_NAME}' not found"
        print_info "Run setup_conda_training.sh to create the environment"
    fi
    echo ""
    
    # Summary
    if [ $ERRORS -eq 0 ]; then
        print_success "GPU setup verification complete!"
    else
        print_error "Some checks failed. Please review the errors above."
    fi
    
    return $ERRORS
}

#===============================================================================
# GPU MEMORY OPTIMIZATION TIPS
#===============================================================================

show_optimization_tips() {
    print_header "GPU Memory Optimization Tips"
    
    echo "If you encounter OOM (Out of Memory) errors during training:"
    echo ""
    echo "1. Reduce batch size in run_train.sh:"
    echo "   BATCH_SIZE=8  (or even lower)"
    echo ""
    echo "2. Use smaller image size:"
    echo "   IMG_SIZE=480  (instead of 640)"
    echo ""
    echo "3. Use a smaller model:"
    echo "   YOLO_MODEL=\"yolov8n.pt\"  (nano version)"
    echo ""
    echo "4. Enable gradient checkpointing (automatic in newer YOLO)"
    echo ""
    echo "5. Use mixed precision (enabled by default):"
    echo "   AMP=true"
    echo ""
    echo "6. Disable image caching:"
    echo "   CACHE=false"
    echo ""
    
    # Show current GPU memory
    if command -v nvidia-smi &> /dev/null; then
        echo "Current GPU Memory Status:"
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
    fi
    echo ""
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================

main() {
    print_header "Microspore Phenotyping - GPU Setup (Conda)"
    
    # Check conda first
    init_conda_env
    
    # Detect GPU
    detect_gpu
    
    if [ "$GPU_READY" = true ]; then
        # GPU is accessible, proceed with setup
        
        # Check if we're in WSL
        if [ "$IS_WSL" = true ]; then
            print_info "WSL detected - using Windows GPU drivers"
            
            # Install CUDA via conda (optional)
            read -p "Install CUDA Toolkit via Conda? (recommended) [Y/n]: " install_cuda
            if [[ "$install_cuda" != "n" && "$install_cuda" != "N" ]]; then
                install_cuda_conda
                install_cudnn_conda
            fi
        else
            # Native Linux - conda-based CUDA installation
            install_cuda_conda
            install_cudnn_conda
        fi
        
        # Setup environment
        setup_cuda_env
        
        # Verify setup
        verify_gpu_setup
        
        # Show optimization tips
        show_optimization_tips
        
    else
        # GPU not accessible
        if [ "$IS_WSL" = true ]; then
            wsl_gpu_instructions
        else
            print_error "No NVIDIA GPU detected or drivers not installed."
            echo ""
            echo "For native Linux, install NVIDIA drivers:"
            echo "  sudo apt-get install nvidia-driver-535"
            echo "  sudo reboot"
        fi
        exit 1
    fi
    
    print_header "GPU Setup Complete"
    echo "Conda environment: ${ENV_NAME}"
    echo ""
    echo "Next step: Run ./setup_conda_training.sh to complete the training environment"
    echo ""
}

# Run main function
main "$@"
