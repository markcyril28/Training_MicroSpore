#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SETUP CONFIGURATION
# =============================================================================
# SERVER SPECS (Dell Server with AMD ROCm):
#   GPU: AMD Instinct MI210 (Aldebaran/MI200)
#   VRAM: 64GB HBM2e
#   Architecture: gfx90a (CDNA2)
#   Driver: amdgpu
#   Compute Platform: ROCm 6.x
#   CPU Threads: 64
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

# Environment settings
ENV_NAME="micro"                  # Name of the conda environment
PYTHON_VERSION="3.11"            # Python version (must match environment.yml)

# PyTorch settings
FORCE_PYTORCH_NIGHTLY=false      # Force nightly PyTorch even on older GPUs
CUDA_VERSION_STABLE="cu121"      # CUDA version for stable PyTorch (cu118, cu121)
CUDA_VERSION_NIGHTLY="cu128"     # CUDA version for nightly PyTorch

# Optional components
INSTALL_QT_DEPS=true             # Install Qt/X11 system dependencies
INSTALL_MATPLOTLIB=true          # Install matplotlib for plotting

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

echo "=== Filipino Micro - Conda Environment Setup ==="
echo ""

# Determine which package manager to use (mamba > conda)
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
    echo "Using mamba"
elif command -v conda &> /dev/null; then
    PKG_MGR="conda"
    echo "Using conda"
else
    echo "ERROR: No package manager found. Please install mamba or conda first."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "Project directory: $PROJECT_DIR"
echo "Environment name: $ENV_NAME"
echo ""

# Remove existing environment if it exists
echo "Checking for existing '$ENV_NAME' environment..."
if $PKG_MGR env list 2>/dev/null | grep -q "^$ENV_NAME "; then
    echo "Removing existing '$ENV_NAME' environment..."
    $PKG_MGR env remove -n "$ENV_NAME" -y
fi

# Create conda environment
echo "Creating conda environment '$ENV_NAME'..."
$PKG_MGR env create -f "$PROJECT_DIR/environment.yml" -n "$ENV_NAME" -y

echo ""
echo "Activating environment..."
if [[ "$PKG_MGR" == "mamba" ]]; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate "$ENV_NAME"
else
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
fi

# Verify we're in the right environment
echo "Active environment: ${CONDA_PREFIX:-unknown}"

echo ""
echo "Installing PyTorch with CUDA support..."
echo "Note: RTX 50-series GPUs require PyTorch nightly builds for full support."

# Try to detect GPU compute capability
COMPUTE_CAP=$(python -c "import subprocess; result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True, text=True); print(result.stdout.strip().split('.')[0] if result.returncode == 0 else '0')" 2>/dev/null || echo "0")

if [[ "$COMPUTE_CAP" -ge "10" ]] || [[ "$FORCE_PYTORCH_NIGHTLY" = true ]]; then
    # RTX 50-series (Blackwell) or newer - use nightly builds
    if [[ "$COMPUTE_CAP" -ge "10" ]]; then
        echo "Detected RTX 50-series or newer GPU (compute capability $COMPUTE_CAP.x)"
    else
        echo "Forcing PyTorch nightly build as requested"
    fi
    echo "Installing PyTorch nightly with CUDA ${CUDA_VERSION_NIGHTLY} support..."
    pip install --pre torch torchvision --index-url "https://download.pytorch.org/whl/nightly/${CUDA_VERSION_NIGHTLY}"
else
    # Older GPUs - use stable release
    echo "Installing PyTorch stable with CUDA ${CUDA_VERSION_STABLE} support..."
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION_STABLE}"
fi

if [[ "$INSTALL_MATPLOTLIB" = true ]]; then
    echo ""
    echo "Installing plotting dependencies..."
    pip install matplotlib
fi

# Install Qt GUI dependencies (required for PyQt6 on Linux/WSL)
if [[ "$INSTALL_QT_DEPS" = true ]]; then
    echo ""
    echo "Installing Qt GUI dependencies..."
    if [[ "$(uname)" == "Linux" ]]; then
        echo "Detected Linux - installing system Qt dependencies..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq \
                libxcb-xinerama0 \
                libxcb-cursor0 \
                libxcb-xkb1 \
                libxkbcommon-x11-0 \
                libxcb-render-util0 \
                libxcb-icccm4 \
                libxcb-image0 \
                libxcb-keysyms1 \
                libxcb-randr0 \
                libxcb-shape0 \
                libxcb-sync1 \
                libxcb-xfixes0 \
                libegl1 \
                libgl1
            echo "Qt system dependencies installed."
        else
            echo "Warning: apt-get not found. Please install Qt xcb dependencies manually."
        fi
    fi
fi

echo ""
echo "=== Verification ==="
echo "Python: $(python --version)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# WSL detection and hints
if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    echo ""
    echo "=== WSL2 Detected ==="
    echo "Distribution: $WSL_DISTRO_NAME"
    echo ""
    echo "GUI Support:"
    echo "  - Windows 11 / recent Windows 10: WSLg should work automatically."
    echo "  - Older Windows 10: Install VcXsrv, run it, then 'export DISPLAY=:0'"
    echo ""
    echo "Qt Troubleshooting:"
    echo "  - If Qt plugin errors occur, the run_game.sh script auto-configures Qt."
    echo "  - The xcb platform is used for X11 display compatibility."
    echo "  - Verify WSLg is working: echo \$WAYLAND_DISPLAY"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: $PKG_MGR activate $ENV_NAME"
echo "  2. Run the game: bash run_game.sh"
echo "  3. Train the ML model: bash train.sh"
