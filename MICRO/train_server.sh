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
#   CPU Threads: 72
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

# =============================================================================
# TRAINING PARAMETERS - Optimized for MI210 64GB + 72 CPU + 1TB RAM
# =============================================================================

# -----------------------------------------------------------------------------
# Device Settings
# -----------------------------------------------------------------------------
DEVICE="cuda"                    # Device to train on: "cuda" (ROCm/HIP) or "cpu"
NO_AMP=true                      # DISABLED: MI210 can have gradient issues with AMP/GradScaler
COMPILE_MODEL=false              # torch.compile with reduce-overhead mode for faster training

# -----------------------------------------------------------------------------
# Self-play Settings (Optimized for 72 CPU threads)
# -----------------------------------------------------------------------------
CPU_WORKERS=32                   # Use 56 of 72 threads for self-play (leave 16 for system/dataloader)
SELFPLAY_GAMES=2048              # More games for larger replay buffer
FOCUS_SIDE="both"                # Focus side: "white", "black", or "both"
OPPONENT_FOCUS="algorithm"            # Opponent focus: "ml", "algorithm", or "both"
SELFPLAY_DIFFICULTIES="easy,medium,hard"  # Comma-separated difficulties to cycle through
NOISE_PROB=0.10                  # Lower noise for faster convergence with large batch
MAX_MOVES_PER_GAME=200           # Max moves per game

# -----------------------------------------------------------------------------
# Training Settings (Optimized for 64GB HBM2e VRAM)
# -----------------------------------------------------------------------------
BATCH_SIZE=512                   # Small batch to avoid HIPBLAS errors on MI210
LEARNING_RATE=2e-4               # Scaled with batch size (linear scaling rule)
WEIGHT_DECAY=1e-5                # Weight decay for regularization
GRAD_CLIP_NORM=1.0               # Gradient clipping for stability
TRAIN_STEPS=100000000000         # Total training steps
CHECKPOINT_EVERY=5000            # More frequent checkpoints with faster training

# -----------------------------------------------------------------------------
# DataLoader Settings (Optimized for 72 CPU + 1TB RAM)
# -----------------------------------------------------------------------------
DATALOADER_WORKERS=12            # 12 workers is optimal for this batch size
PIN_MEMORY=true                  # Pin memory for faster GPU transfer

# -----------------------------------------------------------------------------
# Model Testing Settings (ML vs Algorithm)
# -----------------------------------------------------------------------------
TEST_VS_ALGO=true                # Enable periodic testing against algorithm
TEST_EVERY=25000                 # Test more frequently to track progress
TEST_GAMES=50                    # More test games for reliable metrics
TEST_DIFFICULTY="medium"         # Algorithm difficulty for testing

# -----------------------------------------------------------------------------
# Resume Settings
# -----------------------------------------------------------------------------
RESUME=""                        # Path to checkpoint to resume from
RESUME_LATEST=true               # Resume from latest checkpoint in models/checkpoints/

# -----------------------------------------------------------------------------
# Time-based Stopping
# -----------------------------------------------------------------------------
TRAIN_DURATION="4h"              # Train for this duration

# =============================================================================
# END OF PARAMETERS - Do not edit below this line
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Add src to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# =============================================================================
# ROCm/MI210 Optimizations
# =============================================================================
# CRITICAL: Override GFX version to fix HIPBLAS issues on MI210
export HSA_OVERRIDE_GFX_VERSION=9.0.0

# Stability settings for MI210
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SDMA=0                    # Disable SDMA for stability
export GPU_MAX_HW_QUEUES=4                  # Reduce hardware queues
export HIP_LAUNCH_BLOCKING=0                # Async launches (set to 1 to debug)
export AMD_SERIALIZE_KERNEL=0               # Don't serialize (set to 3 to debug)

# MIOpen settings - use most conservative/stable options
export MIOPEN_FIND_MODE=1                   # Normal find mode
export MIOPEN_DEBUG_CONV_GEMM=0             # Disable GEMM convolution
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1    # Use implicit GEMM instead
export MIOPEN_DEBUG_CONV_DIRECT=1           # Enable direct convolution as fallback
export MIOPEN_DEBUG_CONV_FFT=0              # Disable FFT convolution
export MIOPEN_DEBUG_CONV_WINOGRAD=0         # Disable Winograd
export MIOPEN_DISABLE_CACHE=1               # Disable cache to avoid file issues
export MIOPEN_ENABLE_LOGGING=0              # Disable logging
export MIOPEN_ENABLE_LOGGING_CMD=0          # Disable command logging
export AMD_LOG_LEVEL=0                      # Disable AMD driver logging

# rocBLAS/hipBLAS settings
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
export ROCBLAS_LAYER=0                      # Disable logging
export TORCH_BLAS_PREFER_HIPBLASLT=0        # Disable hipBLASLt
export HIPBLASLT_LOG_MASK=0                 # Disable hipBLASLt logging

# PyTorch HIP settings
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256
export TORCH_USE_HIP_DSA=0                  # Disable device-side assertions
export HIP_VISIBLE_DEVICES=0                # Use only first GPU

# =============================================================================
# torch.compile optimization - cache compiled models for faster subsequent runs
# =============================================================================
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.torch_cache"
export TORCH_COMPILE_CACHE_DIR="${PROJECT_DIR}/.torch_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# Use more CPU threads for compilation (72 cores available)
export TORCHINDUCTOR_MAX_AUTOTUNE_PROCESSES=16

# =============================================================================
# CPU/Memory Optimizations
# =============================================================================
# Use all NUMA nodes efficiently
export OMP_NUM_THREADS=16                   # OpenMP threads for CPU ops
export MKL_NUM_THREADS=16                   # MKL threads if using Intel MKL
export NUMEXPR_MAX_THREADS=16               # NumExpr parallelism

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

# Device settings
ARGS+=" --device ${DEVICE}"
if [ "$NO_AMP" = true ]; then
    ARGS+=" --no-amp"
fi
if [ "$COMPILE_MODEL" = true ]; then
    ARGS+=" --compile-model"
fi

# Self-play settings
ARGS+=" --cpu-workers ${CPU_WORKERS}"
ARGS+=" --selfplay-games ${SELFPLAY_GAMES}"
ARGS+=" --focus-side ${FOCUS_SIDE}"
ARGS+=" --opponent-focus ${OPPONENT_FOCUS}"
ARGS+=" --selfplay-difficulties ${SELFPLAY_DIFFICULTIES}"
ARGS+=" --noise-prob ${NOISE_PROB}"
ARGS+=" --max-moves ${MAX_MOVES_PER_GAME}"

# Training settings
ARGS+=" --batch-size ${BATCH_SIZE}"
ARGS+=" --learning-rate ${LEARNING_RATE}"
ARGS+=" --weight-decay ${WEIGHT_DECAY}"
if [ -n "$GRAD_CLIP_NORM" ]; then
    ARGS+=" --grad-clip-norm ${GRAD_CLIP_NORM}"
fi
if [ -n "$TRAIN_STEPS" ] && [ "$TRAIN_STEPS" -gt 0 ] 2>/dev/null; then
    ARGS+=" --train-steps ${TRAIN_STEPS}"
fi
ARGS+=" --checkpoint-every ${CHECKPOINT_EVERY}"

# DataLoader settings
ARGS+=" --dataloader-workers ${DATALOADER_WORKERS}"
if [ "$PIN_MEMORY" = true ]; then
    ARGS+=" --pin-memory"
fi

# Model testing settings
if [ "$TEST_VS_ALGO" = true ]; then
    ARGS+=" --test-vs-algo"
    ARGS+=" --test-every ${TEST_EVERY}"
    ARGS+=" --test-games ${TEST_GAMES}"
    ARGS+=" --test-difficulty ${TEST_DIFFICULTY}"
fi

# Resume settings
if [ -n "$RESUME" ]; then
    ARGS+=" --resume ${RESUME}"
elif [ "$RESUME_LATEST" = true ]; then
    ARGS+=" --resume-latest"
fi

# Time-based stopping
if [ -n "$TRAIN_DURATION" ]; then
    ARGS+=" --train-duration ${TRAIN_DURATION}"
fi

echo "Training parameters:"
echo ""
echo "  [Device Settings]"
echo "    Device:            ${DEVICE}"
echo "    Mixed Precision:   $([ "$NO_AMP" = true ] && echo "disabled" || echo "enabled")"
echo "    Model Compile:     $([ "$COMPILE_MODEL" = true ] && echo "enabled" || echo "disabled")"
echo ""
echo "  [Self-play Settings]"
echo "    CPU Workers:       ${CPU_WORKERS}"
echo "    Self-play Games:   ${SELFPLAY_GAMES}"
echo "    Focus Side:        ${FOCUS_SIDE}"
echo "    Opponent Focus:    ${OPPONENT_FOCUS}"
echo "    Difficulties:      ${SELFPLAY_DIFFICULTIES} (cycling)"
echo "    Noise Probability: ${NOISE_PROB}"
echo "    Max Moves/Game:    ${MAX_MOVES_PER_GAME}"
echo ""
echo "  [Training Settings]"
echo "    Batch Size:        ${BATCH_SIZE}"
echo "    Learning Rate:     ${LEARNING_RATE}"
echo "    Weight Decay:      ${WEIGHT_DECAY}"
if [ -n "$GRAD_CLIP_NORM" ]; then
    echo "    Gradient Clip:     ${GRAD_CLIP_NORM}"
else
    echo "    Gradient Clip:     (disabled)"
fi
if [ -n "$TRAIN_STEPS" ] && [ "$TRAIN_STEPS" -gt 0 ] 2>/dev/null; then
    echo "    Train Steps:       ${TRAIN_STEPS}"
else
    echo "    Train Steps:       (indefinite)"
fi
echo "    Checkpoint:        every ${CHECKPOINT_EVERY} steps"
echo ""
echo "  [DataLoader Settings]"
echo "    Workers:           ${DATALOADER_WORKERS}"
echo "    Pin Memory:        $([ "$PIN_MEMORY" = true ] && echo "enabled" || echo "disabled")"
echo ""
echo "  [Model Testing]"
if [ "$TEST_VS_ALGO" = true ]; then
    echo "    Test vs Algorithm: enabled"
    echo "    Test Frequency:    every ${TEST_EVERY} steps"
    echo "    Test Games:        ${TEST_GAMES}"
    echo "    Test Difficulty:   ${TEST_DIFFICULTY}"
else
    echo "    Test vs Algorithm: disabled"
fi
echo ""
echo "  [Session]"
if [ -n "$TRAIN_DURATION" ]; then
    echo "    Train Duration:    ${TRAIN_DURATION}"
else
    echo "    Train Duration:    (no limit)"
fi
if [ -n "$RESUME" ]; then
    echo "    Resume from:       ${RESUME}"
elif [ "$RESUME_LATEST" = true ]; then
    echo "    Resume from:       latest checkpoint"
else
    echo "    Resume from:       (starting fresh)"
fi
echo ""

exec python -m micro.ai.ml.trainer ${ARGS}
