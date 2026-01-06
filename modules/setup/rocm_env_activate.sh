#!/bin/bash
#===============================================================================
# ROCm Environment Activation Script
# Configures environment variables for AMD Instinct MI210 (gfx90a/CDNA2)
# This script is sourced when the conda environment is activated.
#===============================================================================

# ROCm paths
export ROCM_PATH="/opt/rocm"
export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"

# PyTorch ROCm settings for gfx90a (MI210)
export HSA_OVERRIDE_GFX_VERSION=9.0.10

# HIP visibility for single GPU training
export HIP_VISIBLE_DEVICES=0

# Reduce MIOpen cache warnings
export MIOPEN_LOG_LEVEL=1

# MIOpen cache directory (for faster subsequent runs)
export MIOPEN_USER_DB_PATH="${HOME}/.config/miopen"
export MIOPEN_SYSTEM_DB_PATH="${ROCM_PATH}/share/miopen/db"
