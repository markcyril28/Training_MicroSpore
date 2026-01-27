#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SETUP CONFIGURATION
# =============================================================================
# SERVER SPECS (Dell Server with AMD ROCm):
#   GPU: AMD Instinct MI210 (Aldebaran/MI200)
#   VRAM: 64GB HBM2e
#   RAM: 1TB DDR4
#   Architecture: gfx90a (CDNA2)
#   Driver: amdgpu
#   Compute Platform: ROCm 6.x
#   CPU Threads: 128
#===============================================================================

# =============================================================================
# TRAINING PARAMETERS - Optimized for MI210 64GB + 128 CPU threads + 1TB RAM
# =============================================================================
# All training parameters are now defined in the YAML config file.
# Edit the config file to change training settings.
# =============================================================================

# Config file to use (comment/uncomment to switch)
CONFIG_FILE="config/training_config_server.yaml"
# CONFIG_FILE="config/training_config_local.yaml"
# CONFIG_FILE="config/training_config.yaml"

# Optional: Profile to apply from the config file
# CONFIG_PROFILE="server"
# CONFIG_PROFILE="local"
# CONFIG_PROFILE="cpu"
CONFIG_PROFILE="server"

# -----------------------------------------------------------------------------
# Resume Settings (can override config file)
# -----------------------------------------------------------------------------
RESUME=""                        # Path to checkpoint to resume from
RESUME_LATEST=true               # Resume from latest checkpoint in models/checkpoints/

# -----------------------------------------------------------------------------
# Time-based Stopping (can override config file)
# -----------------------------------------------------------------------------
TRAIN_DURATION=""                # Override config duration (e.g., "2h", "1d", "30m")
                                 # Leave empty to use duration from config file

# =============================================================================
# END OF PARAMETERS - Do not edit below this line
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Add src to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# =============================================================================
# ROCm/MI210 Settings - MINIMAL (diagnostic shows defaults work fine)
# =============================================================================
# Only suppress logging - no other overrides needed
export MIOPEN_ENABLE_LOGGING=0              # Disable MIOpen logging
export MIOPEN_ENABLE_LOGGING_CMD=0          # Disable command logging
export AMD_LOG_LEVEL=0                      # Disable AMD driver logging
export ROCBLAS_LAYER=0                      # Disable rocBLAS logging

# =============================================================================
# torch.compile optimization - cache compiled models for faster subsequent runs
# =============================================================================
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.torch_cache"
export TORCH_COMPILE_CACHE_DIR="${PROJECT_DIR}/.torch_cache"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Skip CUDAGraph for dynamic shapes - prevents overhead from recording many graphs
export TORCHINDUCTOR_CUDAGRAPH_SKIP_DYNAMIC=1

# Enable Parallel Compilation (use remaining threads during compile phase)
export MAX_JOBS=32
export TORCHINDUCTOR_MAX_AUTOTUNE_PROCESSES=24

# Caching improvements
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1

# =============================================================================
# CPU/Memory Optimizations (128 threads: 48 self-play + 16 dataloader + 32 OMP/MKL + 32 system)
# =============================================================================
export OMP_NUM_THREADS=32                   # OpenMP threads for CPU tensor ops
export MKL_NUM_THREADS=32                   # MKL threads if using Intel MKL
export NUMEXPR_MAX_THREADS=32               # NumExpr parallelism

# Increase file descriptor limit for many workers
ulimit -n 65536 2>/dev/null || true

# Console logging
LOG_DIR="${PROJECT_DIR}/logs/console"
mkdir -p "$LOG_DIR"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/console_${LOG_TIMESTAMP}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Micro - ML Training ==="
echo ""

# Verify ROCm/HIP is available
python -c "import torch; assert torch.cuda.is_available(), 'ROCm/HIP not available. Training requires GPU.'" || {
    echo ""
    echo "ERROR: ROCm/HIP is not available."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure ROCm 6.x driver is installed"
    echo "  2. Run 'rocm-smi' to verify GPU access"
    echo "  3. Check GPU visibility: export HIP_VISIBLE_DEVICES=0"
    echo "  4. Reinstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"
    exit 1
}

echo "ROCm/HIP verified (MI210 gfx90a). Starting training..."
echo ""

# =============================================================================
# Auto-detect valid checkpoint (skip corrupted ones with NaN/Inf)
# =============================================================================
if [ "$RESUME_LATEST" = true ] && [ -z "$RESUME" ]; then
    echo "Checking checkpoints for corruption..."
    
    VALID_CHECKPOINT=$(python3 -W ignore << 'PYEOF' 2>/dev/null | head -1
import sys
from pathlib import Path
import torch

checkpoint_dir = Path("models/checkpoints")
if not checkpoint_dir.exists():
    sys.exit(0)

checkpoints = sorted(checkpoint_dir.glob("model_step_*.pt"), 
                     key=lambda p: int(p.stem.split('_')[-1]), 
                     reverse=True)

for ckpt in checkpoints:
    try:
        c = torch.load(ckpt, map_location='cpu', weights_only=False)
        if 'model_state_dict' not in c:
            continue
        is_valid = all(torch.isfinite(p).all() for p in c['model_state_dict'].values())
        if is_valid:
            print(str(ckpt))
            sys.exit(0)
    except:
        continue
PYEOF
)
    
    if [ -n "$VALID_CHECKPOINT" ]; then
        echo "  Found valid checkpoint: $(basename "$VALID_CHECKPOINT")"
        RESUME="$VALID_CHECKPOINT"
        RESUME_LATEST=false
    else
        echo "  No valid checkpoints found. Starting fresh."
        RESUME_LATEST=false
    fi
    echo ""
fi

# Build command arguments
ARGS=""

# Config file (required)
ARGS+=" --config ${CONFIG_FILE}"

# Profile (optional)
if [ -n "$CONFIG_PROFILE" ]; then
    ARGS+=" --profile ${CONFIG_PROFILE}"
fi

# Resume settings (override config if specified)
if [ -n "$RESUME" ]; then
    ARGS+=" --resume ${RESUME}"
elif [ "$RESUME_LATEST" = true ]; then
    ARGS+=" --resume-latest"
fi

# Time-based stopping (override config if specified)
if [ -n "$TRAIN_DURATION" ]; then
    ARGS+=" --train-duration ${TRAIN_DURATION}"
fi

echo "Training Configuration:"
echo ""
echo "  Config File:     ${CONFIG_FILE}"
if [ -n "$CONFIG_PROFILE" ]; then
    echo "  Profile:         ${CONFIG_PROFILE}"
fi
echo ""
echo "  [Session Overrides]"
if [ -n "$TRAIN_DURATION" ]; then
    echo "    Train Duration:    ${TRAIN_DURATION}"
else
    echo "    Train Duration:    (from config file)"
fi
if [ -n "$RESUME" ]; then
    echo "    Resume from:       ${RESUME}"
elif [ "$RESUME_LATEST" = true ]; then
    echo "    Resume from:       latest checkpoint"
else
    echo "    Resume from:       (from config file)"
fi
echo ""
echo "  See ${CONFIG_FILE} for all training parameters."
echo ""

exec python -m micro.ai.ml.trainer ${ARGS}
