#!/bin/bash
#===============================================================================
# Microspore Phenotyping - GPU Setup Core Functions
# Shared functions for GPU setup across local and workstation versions.
# This file is sourced by gpu_setup_local.sh and gpu_setup_workstation.sh
#
# Location: modules/setup/gpu_setup_core.sh
# Requires: source "${BASE_DIR}/modules/config/common_functions.sh" first
#===============================================================================

# Ensure common_functions.sh is sourced first
if [ -z "$COMMON_ENV_NAME" ]; then
    echo "ERROR: common_functions.sh must be sourced before gpu_setup_core.sh"
    exit 1
fi

# Use shared configuration
ENV_NAME="${COMMON_ENV_NAME}"

#===============================================================================
# CONDA CHECK AND INITIALIZATION
#===============================================================================

init_conda_env() {
    print_header "Conda Environment Check"
    
    if ! init_conda_shell; then
        exit 1
    fi
    
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
    
    if is_wsl; then
        print_info "Running in WSL (Windows Subsystem for Linux)"
        IS_WSL=true
    else
        print_info "Running in native Linux"
        IS_WSL=false
    fi
    
    if command -v lspci &> /dev/null; then
        GPU_INFO=$(lspci | grep -i nvidia 2>/dev/null || echo "")
        if [ -n "$GPU_INFO" ]; then
            print_success "NVIDIA GPU detected via lspci:"
            echo "    $GPU_INFO"
        fi
    fi
    
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
    
    eval "$(conda shell.bash hook)"
    
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        print_info "Creating conda environment '${ENV_NAME}'..."
        conda create -n ${ENV_NAME} python=3.10 -y
    fi
    
    conda activate ${ENV_NAME}
    print_success "Activated conda environment: ${ENV_NAME}"
    
    if conda list cuda-toolkit 2>/dev/null | grep -q cuda-toolkit; then
        CUDA_VER=$(conda list cuda-toolkit | grep cuda-toolkit | awk '{print $2}')
        print_success "CUDA Toolkit already installed in conda: $CUDA_VER"
        return 0
    fi
    
    print_info "Installing CUDA Toolkit via conda..."
    echo ""
    
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
    
    eval "$(conda shell.bash hook)"
    
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "Conda environment '${ENV_NAME}' not found. Skipping cuDNN."
        return 0
    fi
    
    conda activate ${ENV_NAME}
    
    if conda list cudnn 2>/dev/null | grep -q cudnn; then
        CUDNN_VER=$(conda list cudnn | grep cudnn | awk '{print $2}')
        print_success "cuDNN already installed in conda: $CUDNN_VER"
        return 0
    fi
    
    print_info "Installing cuDNN via conda..."
    
    conda install -c conda-forge cudnn -y 2>/dev/null || \
    conda install -c nvidia cudnn -y 2>/dev/null || \
    print_info "cuDNN not installed via conda (PyTorch bundles cuDNN, so this is OK)"
    
    echo ""
    print_info "Note: PyTorch includes cuDNN, so separate installation is optional."
    echo ""
}

#===============================================================================
# CUDA ENVIRONMENT SETUP (Conda-based)
# Note: CUDA_PATHS_PRIORITY should be set by the calling script (local/workstation)
#===============================================================================

setup_cuda_env() {
    print_header "Setting up CUDA Environment (Conda)"
    
    eval "$(conda shell.bash hook)"
    
    CONDA_CUDA_PATH=""
    if [ -n "$CONDA_PREFIX" ]; then
        if [ -d "$CONDA_PREFIX/lib" ]; then
            CONDA_CUDA_PATH="$CONDA_PREFIX"
        fi
    fi
    
    # Use CUDA_PATHS_PRIORITY if set, otherwise use default order
    SYSTEM_CUDA_PATH=""
    if [ -n "$CUDA_PATHS_PRIORITY" ]; then
        # Use custom priority order from caller
        for cuda_path in $CUDA_PATHS_PRIORITY; do
            if [ -d "$cuda_path" ]; then
                SYSTEM_CUDA_PATH="$cuda_path"
                break
            fi
        done
    else
        # Default fallback order
        for cuda_path in /usr/local/cuda-12.8 /usr/local/cuda-12.4 /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-11.8 /usr/local/cuda; do
            if [ -d "$cuda_path" ]; then
                SYSTEM_CUDA_PATH="$cuda_path"
                break
            fi
        done
    fi
    
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
    
    if [ -n "$CONDA_CUDA_PATH" ]; then
        print_info "Conda CUDA path: $CONDA_CUDA_PATH"
    fi
    if [ -n "$SYSTEM_CUDA_PATH" ]; then
        print_info "System CUDA path: $SYSTEM_CUDA_PATH"
    fi
    
    echo ""
}

#===============================================================================
# VERIFY GPU SETUP
#===============================================================================

verify_gpu_setup() {
    print_header "Verifying GPU Setup"
    
    ERRORS=0
    
    echo "1. Checking nvidia-smi..."
    if nvidia-smi &> /dev/null; then
        print_success "nvidia-smi works"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    else
        print_error "nvidia-smi failed"
        ERRORS=$((ERRORS + 1))
    fi
    echo ""
    
    echo "2. Checking CUDA..."
    if command -v nvcc &> /dev/null; then
        print_success "CUDA compiler (nvcc) found"
        nvcc --version | grep "release"
    else
        print_warning "nvcc not found (optional for PyTorch)"
    fi
    echo ""
    
    echo "3. Checking PyTorch CUDA support (Conda)..."
    
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
    
    if command -v nvidia-smi &> /dev/null; then
        echo "Current GPU Memory Status:"
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
    fi
    echo ""
}

#===============================================================================
# MAIN GPU SETUP EXECUTION
#===============================================================================

run_gpu_setup() {
    local setup_script_name="${1:-setup_conda_training.sh}"
    
    print_header "Microspore Phenotyping - GPU Setup (Conda)"
    
    init_conda_env
    detect_gpu
    
    if [ "$GPU_READY" = true ]; then
        if [ "$IS_WSL" = true ]; then
            print_info "WSL detected - using Windows GPU drivers"
            
            read -p "Install CUDA Toolkit via Conda? (recommended) [Y/n]: " install_cuda
            if [[ "$install_cuda" != "n" && "$install_cuda" != "N" ]]; then
                install_cuda_conda
                install_cudnn_conda
            fi
        else
            install_cuda_conda
            install_cudnn_conda
        fi
        
        setup_cuda_env
        verify_gpu_setup
        show_optimization_tips
        
    else
        if [ "$IS_WSL" = true ]; then
            wsl_gpu_instructions
        else
            print_error "No NVIDIA GPU detected or drivers not installed."
            echo ""
            echo "For native Linux, install NVIDIA drivers:"
            echo "  apt-get install nvidia-driver-535"
            echo "  reboot"
        fi
        exit 1
    fi
    
    print_header "GPU Setup Complete"
    echo "Conda environment: ${ENV_NAME}"
    echo ""
    echo "Next step: Run ./${setup_script_name} to complete the training environment"
    echo ""
}
