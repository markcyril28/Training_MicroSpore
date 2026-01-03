#!/bin/bash
#===============================================================================
# Setup Module for Microspore Phenotyping
# Environment setup, GPU configuration, and conda management.
#
# Files:
#   gpu_setup_core.sh     - GPU detection and CUDA/cuDNN installation
#   setup_conda_core.sh   - Conda environment creation and package installation
#
# Usage:
#   source "${MODULES_DIR}/setup/gpu_setup_core.sh"
#   source "${MODULES_DIR}/setup/setup_conda_core.sh"
#
# Note: Source config/common_functions.sh before using these modules.
#===============================================================================

echo "[Setup Module] Source individual files as needed:"
echo "  - gpu_setup_core.sh     (GPU detection, CUDA, cuDNN)"
echo "  - setup_conda_core.sh   (Conda environment, packages)"
