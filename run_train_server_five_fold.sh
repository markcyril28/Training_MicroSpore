#!/bin/bash
#===============================================================================
# Microspore Phenotyping - Five-Fold YOLO Training Script (SERVER VERSION)
#===============================================================================
# Runs the five-fold continuation strategy using the shared server training runner.
#
# This entrypoint trains folds 1-5 from the checkpoint phase at:
#   trained_models_output/server/04_class_balancing_combination_continue_3
#
# Output:
#   trained_models_output/server/05_five_fold_continue_from_continue_3/
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_SCRIPTS_OVERRIDE="05_five_fold_continue_from_continue_3.sh"

exec "${SCRIPT_DIR}/run_train_server.sh" "$@"
