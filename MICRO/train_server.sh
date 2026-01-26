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
#   CPU Threads: 48
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
# TRAINING PARAMETERS - Optimized for MI210 64GB + 48 CPU + 1TB RAM
# =============================================================================

# -----------------------------------------------------------------------------
# Device Settings
# -----------------------------------------------------------------------------
DEVICE="cuda"                    # Device to train on: "cuda" (ROCm/HIP) or "cpu"
NO_AMP=true                      # DISABLED: Testing if AMP causes gradient issues on ROCm
AMP_DTYPE="bfloat16"             # OPTIONS: "bfloat16" (recommended for MI210), "float16" (may cause issues)
COMPILE_MODEL=false              # DISABLED: torch.compile may break gradients on ROCm
COMPILE_MODE="default"          # OPTIONS: "default" (fast compile), "reduce-overhead" (slow compile, fast run)

# -----------------------------------------------------------------------------
# Self-play Settings (Optimized for 48 CPU threads)
# -----------------------------------------------------------------------------
CPU_WORKERS=24                   # Half of 48 threads for self-play (rest for dataloader/system)
SELFPLAY_GAMES=2048              # Balanced: enough data without long epoch times
FOCUS_SIDE="both"                # Focus side: "white", "black", or "both"
OPPONENT_FOCUS="both"            # Opponent focus: "ml", "algorithm", or "both"
SELFPLAY_DIFFICULTIES="easy,medium,hard"  # Comma-separated difficulties to cycle through
NOISE_PROB=0.40                  # Slightly more exploration for diversity
MAX_MOVES_PER_GAME=200           # Max moves per game

# -----------------------------------------------------------------------------
# Training Settings (Optimized for 64GB HBM2e VRAM - FP32 mode)
# -----------------------------------------------------------------------------
BATCH_SIZE=1024                  # FP32 uses 2x memory vs FP16, keep moderate
LEARNING_RATE=1e-4               # Lower LR for FP32 stability (was 3e-4)
WEIGHT_DECAY=1e-5                # Weight decay for regularization
GRAD_CLIP_NORM=1.0               # Gradient clipping for stability
TRAIN_STEPS=100000000000         # Total training steps
CHECKPOINT_EVERY=2000            # Checkpoint every 2000 steps

# -----------------------------------------------------------------------------
# DataLoader Settings (Optimized for 48 CPU + 1TB RAM)
# -----------------------------------------------------------------------------
DATALOADER_WORKERS=8             # 8 workers optimal for 48 threads (leaves room for self-play)
PIN_MEMORY=true                  # Pin memory for faster GPU transfer

# -----------------------------------------------------------------------------
# Model Testing Settings (ML vs Algorithm)
# -----------------------------------------------------------------------------
TEST_VS_ALGO=false                # Enable periodic testing against algorithm
TEST_EVERY=5000                  # Test every 5000 steps (testing is expensive)
TEST_GAMES=50                    # Test games for reliable metrics
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

# Enable Parallel Compilation (use remaining threads during compile phase)
export MAX_JOBS=16
export TORCHINDUCTOR_MAX_AUTOTUNE_PROCESSES=15

# Caching improvements
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1

# =============================================================================
# CPU/Memory Optimizations (48 threads total)
# =============================================================================
# Balance threads: 24 self-play + 8 dataloader + 12 OMP = 44 (4 for system)
export OMP_NUM_THREADS=12                   # OpenMP threads for CPU ops
export MKL_NUM_THREADS=12                   # MKL threads if using Intel MKL
export NUMEXPR_MAX_THREADS=12               # NumExpr parallelism

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
else
    ARGS+=" --amp-dtype ${AMP_DTYPE}"
fi
if [ "$COMPILE_MODEL" = true ]; then
    ARGS+=" --compile-model"
    ARGS+=" --compile-mode ${COMPILE_MODE}"
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
echo "    Mixed Precision:   $([ "$NO_AMP" = true ] && echo "disabled" || echo "enabled (${AMP_DTYPE})")"
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
