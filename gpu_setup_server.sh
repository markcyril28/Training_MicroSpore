#!/bin/bash 
#===============================================================================
# Microspore Phenotyping - GPU Setup Script (SERVER VERSION)
# Thin wrapper for AMD Instinct MI210 GPU setup with ROCm support
#
# SERVER SPECS (Dell Server with AMD ROCm):
#   GPU: AMD Instinct MI210 (Aldebaran/MI200)
#   VRAM: 64GB HBM2e
#   Architecture: gfx90a (CDNA2)
#   Driver: amdgpu
#   Compute Platform: ROCm 6.x
#===============================================================================

: << 'SERVER_SPECS'
=================================== Product Info ======================================
GPU[0]          : Card Model:           0x740f
GPU[0]          : Card Vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]          : Card SKU:             D67301V
GPU[0]          : GFX Version:          gfx90a
================================== Memory Info ========================================
GPU[0]          : VRAM Total Memory (B): 68702699520 (~64GB)
GPU[0]          : Temperature (Sensor junction) (C): 35.0
GPU[0]          : Average Graphics Package Power (W): 42.0
==========================================================================================
SERVER_SPECS

set -e  # Exit on error

# Get script directory (modules are in the same directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common functions (required first)
if [ -f "${SCRIPT_DIR}/modules/config/common_functions.sh" ]; then
    source "${SCRIPT_DIR}/modules/config/common_functions.sh"
else
    echo "ERROR: modules/config/common_functions.sh not found."
    exit 1
fi

# Use shared configuration
ENV_NAME="${COMMON_ENV_NAME}"

#===============================================================================
# AMD GPU UTILITIES (SERVER-SPECIFIC)
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
        AMD_ROCM_VERSION=$(rocm-smi --version 2>/dev/null | head -1 || echo "Unknown")
        AMD_GPU_AVAILABLE=true
        
        # Get temperature if available
        AMD_GPU_TEMP=$(rocm-smi --showtemp 2>/dev/null | grep "junction" | awk '{print $NF}' || echo "N/A")
        
        # Get power usage
        AMD_GPU_POWER=$(rocm-smi --showpower 2>/dev/null | grep "Average" | awk '{print $(NF-1)}' || echo "N/A")
        
        # Get VRAM usage
        AMD_VRAM_TOTAL=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total" | awk '{print $NF}' || echo "68702699520")
        AMD_VRAM_USED=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Used" | awk '{print $NF}' || echo "0")
    else
        AMD_GPU_NAME=""
        AMD_GPU_MEMORY=""
        AMD_GPU_ARCH=""
        AMD_ROCM_VERSION=""
        AMD_GPU_AVAILABLE=false
    fi
}

# Print AMD GPU information
print_amd_gpu_info() {
    get_amd_gpu_info
    if [ "$AMD_GPU_AVAILABLE" = true ]; then
        print_success "AMD GPU detected"
        echo "    GPU: $AMD_GPU_NAME"
        echo "    Memory: $AMD_GPU_MEMORY"
        echo "    Architecture: $AMD_GPU_ARCH"
        echo "    ROCm Version: $AMD_ROCM_VERSION"
        echo "    Temperature: ${AMD_GPU_TEMP}"
        echo "    Power: ${AMD_GPU_POWER}W"
    else
        print_warning "No AMD GPU detected via rocm-smi"
    fi
}

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
        print_info "Will be created by setup_conda_training_server.sh"
        CONDA_ENV_EXISTS=false
    fi
    
    echo ""
}

#===============================================================================
# GPU DETECTION
#===============================================================================

detect_gpu() {
    print_header "GPU Detection (AMD ROCm)"
    
    print_info "Running on Linux server"
    IS_WSL=false
    
    # Check for AMD GPU via lspci
    if command -v lspci &> /dev/null; then
        GPU_INFO=$(lspci | grep -i "display\|vga" | grep -i "amd\|ati" 2>/dev/null || echo "")
        if [ -n "$GPU_INFO" ]; then
            print_success "AMD GPU detected via lspci:"
            echo "    $GPU_INFO"
        fi
    fi
    
    # Check rocm-smi availability
    if check_amd_gpu; then
        print_success "rocm-smi is available"
        echo ""
        print_amd_gpu_info
        GPU_READY=true
    else
        print_warning "rocm-smi not found or AMD GPU not accessible"
        GPU_READY=false
    fi
    
    echo ""
}

#===============================================================================
# ROCm INSTALLATION CHECK
#===============================================================================

check_rocm_installation() {
    print_header "ROCm Installation Check"
    
    local rocm_ok=true
    
    # Check for rocm-smi
    echo "1. Checking rocm-smi..."
    if command -v rocm-smi &> /dev/null; then
        print_success "rocm-smi found"
    else
        print_warning "rocm-smi not found"
        rocm_ok=false
    fi
    
    # Check for rocminfo
    echo "2. Checking rocminfo..."
    if command -v rocminfo &> /dev/null; then
        print_success "rocminfo found"
        rocminfo 2>/dev/null | grep -E "Name:|gfx" | head -5 || true
    else
        print_warning "rocminfo not found"
    fi
    
    # Check for hipcc (HIP compiler)
    echo "3. Checking HIP compiler..."
    if command -v hipcc &> /dev/null; then
        print_success "hipcc found"
        hipcc --version 2>/dev/null | head -2 || true
    else
        print_info "hipcc not in PATH (may be in /opt/rocm/bin)"
        if [ -f "/opt/rocm/bin/hipcc" ]; then
            print_success "hipcc found at /opt/rocm/bin/hipcc"
        fi
    fi
    
    # Check amdgpu driver
    echo "4. Checking amdgpu kernel driver..."
    if lsmod | grep -q amdgpu; then
        print_success "amdgpu kernel module loaded"
    else
        print_warning "amdgpu kernel module not loaded"
        rocm_ok=false
    fi
    
    # Check ROCm installation path
    echo "5. Checking ROCm installation..."
    if [ -d "/opt/rocm" ]; then
        print_success "ROCm found at /opt/rocm"
        ROCM_PATH="/opt/rocm"
        
        # Get ROCm version
        if [ -f "/opt/rocm/.info/version" ]; then
            ROCM_VERSION=$(cat /opt/rocm/.info/version)
            echo "    ROCm Version: $ROCM_VERSION"
        elif [ -f "/opt/rocm/include/rocm-core/rocm_version.h" ]; then
            echo "    ROCm installation detected"
        fi
    else
        print_warning "ROCm not found at /opt/rocm"
        rocm_ok=false
    fi
    
    echo ""
    
    if [ "$rocm_ok" = true ]; then
        return 0
    else
        return 1
    fi
}

#===============================================================================
# ROCm INSTALLATION INSTRUCTIONS
#===============================================================================

rocm_install_instructions() {
    print_header "ROCm Installation Instructions"
    
    echo "AMD Instinct MI210 requires ROCm 5.4+ (recommended: ROCm 6.x)"
    echo ""
    echo "For Ubuntu 22.04 (recommended):"
    echo ""
    echo "1. Add AMD ROCm repository:"
    echo "   sudo apt update"
    echo "   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*.deb"
    echo "   sudo apt install ./amdgpu-install_*.deb"
    echo ""
    echo "2. Install ROCm with HIP support:"
    echo "   sudo amdgpu-install --usecase=rocm,hip"
    echo ""
    echo "3. Add user to render group:"
    echo "   sudo usermod -a -G render,video \$USER"
    echo ""
    echo "4. Reboot and verify:"
    echo "   sudo reboot"
    echo "   rocm-smi"
    echo ""
    echo "After completing these steps, run this script again."
    echo ""
}

#===============================================================================
# ROCm ENVIRONMENT SETUP
#===============================================================================

setup_rocm_env() {
    print_header "Setting up ROCm Environment"
    
    # Set environment variables
    local ROCM_PATH="/opt/rocm"
    
    if [ -d "$ROCM_PATH" ]; then
        print_info "Configuring ROCm environment..."
        
        # Create environment setup script for the conda environment
        eval "$(conda shell.bash hook)"
        
        if conda env list | grep -q "^${ENV_NAME} "; then
            conda activate ${ENV_NAME}
            
            if [ -n "$CONDA_PREFIX" ]; then
                mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
                mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
                
                # Create activation script
                cat > "$CONDA_PREFIX/etc/conda/activate.d/rocm_env.sh" << 'ROCM_ENV'
#!/bin/bash
# ROCm environment variables for AMD Instinct MI210
export ROCM_PATH="/opt/rocm"
export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"

# PyTorch ROCm settings
export HSA_OVERRIDE_GFX_VERSION=9.0.10  # For gfx90a (MI210)

# HIP visibility for single GPU training
export HIP_VISIBLE_DEVICES=0

# Disable MIOpen cache warnings
export MIOPEN_LOG_LEVEL=1
ROCM_ENV
                chmod +x "$CONDA_PREFIX/etc/conda/activate.d/rocm_env.sh"
                print_success "Created ROCm activation script"
                
                # Create deactivation script
                cat > "$CONDA_PREFIX/etc/conda/deactivate.d/rocm_env.sh" << 'ROCM_DEACTIVATE'
#!/bin/bash
unset ROCM_PATH
unset HIP_PATH
unset HSA_OVERRIDE_GFX_VERSION
unset HIP_VISIBLE_DEVICES
unset MIOPEN_LOG_LEVEL
ROCM_DEACTIVATE
                chmod +x "$CONDA_PREFIX/etc/conda/deactivate.d/rocm_env.sh"
                print_success "Created ROCm deactivation script"
            fi
        else
            print_info "Conda environment not created yet - ROCm env will be configured in setup_conda_training_server.sh"
        fi
    else
        print_warning "ROCm path not found at ${ROCM_PATH}"
    fi
    
    echo ""
}

#===============================================================================
# VERIFY GPU SETUP
#===============================================================================

verify_gpu_setup() {
    print_header "Verifying GPU Setup"
    
    ERRORS=0
    
    echo "1. Checking rocm-smi..."
    if rocm-smi --showproductname &> /dev/null; then
        print_success "rocm-smi works"
        rocm-smi --showproductname --showmeminfo vram 2>/dev/null | grep -E "GPU|VRAM" | head -6 || true
    else
        print_error "rocm-smi failed"
        ERRORS=$((ERRORS + 1))
    fi
    echo ""
    
    echo "2. Checking AMD GPU architecture..."
    if rocm-smi 2>/dev/null | grep -q "gfx90a"; then
        print_success "AMD Instinct MI210 (gfx90a) detected"
    elif lshw -C display 2>/dev/null | grep -q "MI210"; then
        print_success "AMD Instinct MI210 detected via lshw"
    else
        print_info "GPU architecture check inconclusive (may still work)"
    fi
    echo ""
    
    echo "3. Checking PyTorch ROCm support..."
    
    eval "$(conda shell.bash hook)"
    
    if conda env list | grep -q "^${ENV_NAME} "; then
        conda activate ${ENV_NAME}
        
        if python -c "import torch" 2>/dev/null; then
            PYTORCH_ROCM=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
            if [ "$PYTORCH_ROCM" = "True" ]; then
                print_success "PyTorch ROCm is available in '${ENV_NAME}' environment"
                python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
                python -c "import torch; print(f'  ROCm version: {torch.version.hip}')" 2>/dev/null || true
                python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || true
            else
                print_warning "PyTorch installed but ROCm/HIP not available"
                print_info "Run setup_conda_training_server.sh to install PyTorch ROCm"
            fi
        else
            print_info "PyTorch not installed in '${ENV_NAME}' environment"
            print_info "Run setup_conda_training_server.sh to install PyTorch ROCm"
        fi
    else
        print_info "Conda environment '${ENV_NAME}' not found"
        print_info "Run setup_conda_training_server.sh to create the environment"
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
# GPU MEMORY OPTIMIZATION TIPS (AMD MI210)
#===============================================================================

show_optimization_tips() {
    print_header "GPU Memory Optimization Tips (AMD MI210)"
    
    echo "The AMD Instinct MI210 has 64GB HBM2e VRAM - suitable for large models!"
    echo ""
    echo "Recommended settings for MI210:"
    echo ""
    echo "1. Large batch sizes (leverage the 64GB VRAM):"
    echo "   BATCH_SIZE=64  (or even 128 for smaller models)"
    echo ""
    echo "2. Use full image size:"
    echo "   IMG_SIZE=640  (default, can go higher)"
    echo ""
    echo "3. Use larger models for better accuracy:"
    echo "   YOLO_MODEL=\"yolo11x.pt\"  (extra-large version)"
    echo ""
    echo "4. Enable mixed precision for faster training:"
    echo "   AMP=true"
    echo ""
    echo "5. Enable image caching (you have the VRAM):"
    echo "   CACHE=true"
    echo ""
    echo "6. Use multiple workers for data loading:"
    echo "   WORKERS=8"
    echo ""
    
    if command -v rocm-smi &> /dev/null; then
        echo "Current GPU Memory Status:"
        rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU|VRAM" | head -4 || true
        echo ""
        echo "Current GPU Temperature:"
        rocm-smi --showtemp 2>/dev/null | grep -E "GPU|junction" | head -2 || true
    fi
    echo ""
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================

main() {
    print_header "Microspore Phenotyping - GPU Setup (Server AMD ROCm)"
    
    init_conda_env
    detect_gpu
    
    if [ "$GPU_READY" = true ]; then
        check_rocm_installation
        setup_rocm_env
        verify_gpu_setup
        show_optimization_tips
        
        print_header "GPU Setup Complete"
        echo "Conda environment: ${ENV_NAME}"
        echo "GPU: AMD Instinct MI210 (64GB HBM2e)"
        echo ""
        echo "Next step: Run ./setup_conda_training_server.sh to complete the training environment"
        echo ""
    else
        check_rocm_installation || rocm_install_instructions
        exit 1
    fi
}

# Run main
main
