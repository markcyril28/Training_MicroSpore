#!/bin/bash
#===============================================================================
# Microspore Phenotyping - GPU Setup Script (LOCAL VERSION)
# Thin wrapper that sources core functions and runs with local configuration.
#
# LOCAL VERSION (Typical laptop/desktop):
#   GPU: Consumer GPU (GTX 1060-1080, RTX 2060-3080, etc.)
#   VRAM: 6-8GB
#   Driver: 525+
#   CUDA: 11.8 / 12.1
#===============================================================================

: << 'LOCAL_SPECS'
Sat Jan  3 11:51:43 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.274.02             Driver Version: 573.13       CUDA Version: 12.8     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 5050 ...    On  | 00000000:01:00.0 Off |                  N/A |
| N/A   50C    P8               3W /  56W |      0MiB /  8151MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
LOCAL_SPECS

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

# Source GPU setup core functions
if [ -f "${SCRIPT_DIR}/modules/setup/gpu_setup_core.sh" ]; then
    source "${SCRIPT_DIR}/modules/setup/gpu_setup_core.sh"
else
    echo "ERROR: modules/setup/gpu_setup_core.sh not found."
    exit 1
fi

#===============================================================================
# LOCAL-SPECIFIC CONFIGURATION
#===============================================================================

# CUDA path priority for local (older CUDA versions first for compatibility)
CUDA_PATHS_PRIORITY="/usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-11.8 /usr/local/cuda"

#===============================================================================
# RUN GPU SETUP
#===============================================================================

run_gpu_setup "setup_conda_training_local.sh"
