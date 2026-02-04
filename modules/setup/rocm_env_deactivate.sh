#!/bin/bash
#===============================================================================
# ROCm Environment Deactivation Script
# Cleans up environment variables when the conda environment is deactivated.
#===============================================================================

unset ROCM_PATH
unset HIP_PATH
unset HSA_OVERRIDE_GFX_VERSION
unset HIP_VISIBLE_DEVICES
unset MIOPEN_LOG_LEVEL
unset MIOPEN_USER_DB_PATH
unset MIOPEN_SYSTEM_DB_PATH
