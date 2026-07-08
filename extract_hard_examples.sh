#!/bin/bash
#===============================================================================
# Extract Hard Examples from YOLO Validation Predictions
#
# This script runs the hard example extraction tool to identify images where
# the model made mistakes (false negatives and class confusions).
# These can be reviewed and added back to training for improvement.
#
# Usage:
#   ./run_extract_hard_examples.sh                          # Use default model
#   ./run_extract_hard_examples.sh /path/to/model_dir       # Specify model directory
#   ./run_extract_hard_examples.sh /path/to/model hard_examples_output  # Custom output
#===============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#===============================================================================
# CONFIGURATION - Modify these as needed
#===============================================================================

#-------------------------------------------------------------------------------
# OPERATION MODE - Controls what action to perform on hard examples
#-------------------------------------------------------------------------------
# Options:
#   "extract"  - Only extract/copy hard examples to output directory (non-destructive)
#   "delete"   - Only delete hard examples from the dataset (WARNING: destructive!)
#   "both"     - Extract hard examples first, then delete them from the dataset
#-------------------------------------------------------------------------------
OPERATION_MODE="both"

DEFAULT_MODEL_DIR="${SCRIPT_DIR}/trained_models_output/server/04_class_balancing_combination_continue_4_5.2_phase_3"

# Default output directory
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/TRAINING_WD/hard_examples_gallery_phase_3"

# Confusion pairs to track (format: true_class:predicted_class)
# These are the most common confusions based on training analysis
CONFUSION_PAIRS="mature_pollen:midlate_pollen,young_pollen:late_microspore,mid_microspore:late_microspore,mid_microspore:young_microspore"

# Thresholds
IOU_THRESHOLD=0.5
CONF_THRESHOLD=0.25

# Maximum examples per category (set high to capture all)
MAX_PER_CATEGORY=999999

# Dataset split to extract from: 'train', 'val', or 'both'
SPLIT="both"

#===============================================================================
# PARSE ARGUMENTS
#===============================================================================

MODEL_DIR="${1:-$DEFAULT_MODEL_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

#===============================================================================
# VALIDATE MODEL DIRECTORY
#===============================================================================

# If a base directory is given, try to find the actual model subdirectory
if [ -d "$MODEL_DIR" ]; then
    # Check if weights directory exists directly
    if [ -d "${MODEL_DIR}/weights" ]; then
        FINAL_MODEL_DIR="$MODEL_DIR"
    else
        # Look for subdirectory with weights
        SUBDIRS=$(find "$MODEL_DIR" -maxdepth 1 -type d -name "Dataset_*" 2>/dev/null | head -1)
        if [ -n "$SUBDIRS" ] && [ -d "${SUBDIRS}/weights" ]; then
            FINAL_MODEL_DIR="$SUBDIRS"
        else
            echo "ERROR: Could not find weights directory in $MODEL_DIR"
            echo "Please specify the full path to a trained model directory containing 'weights/best.pt'"
            exit 1
        fi
    fi
else
    echo "ERROR: Model directory does not exist: $MODEL_DIR"
    exit 1
fi

echo "=========================================="
echo "  Extract Hard Examples"
echo "=========================================="
echo ""
echo "Operation mode: $OPERATION_MODE"
echo "Model directory: $FINAL_MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Dataset split: $SPLIT"
echo "Confusion pairs: $CONFUSION_PAIRS"
echo "IoU threshold: $IOU_THRESHOLD"
echo "Confidence threshold: $CONF_THRESHOLD"
echo "Max per category: $MAX_PER_CATEGORY"
echo ""

#===============================================================================
# VALIDATE OPERATION MODE
#===============================================================================

if [[ "$OPERATION_MODE" != "extract" && "$OPERATION_MODE" != "delete" && "$OPERATION_MODE" != "both" ]]; then
    echo "ERROR: Invalid OPERATION_MODE '$OPERATION_MODE'"
    echo "Valid options are: 'extract', 'delete', or 'both'"
    exit 1
fi

# Safety warning for destructive operations
if [[ "$OPERATION_MODE" == "delete" || "$OPERATION_MODE" == "both" ]]; then
    echo "=========================================="
    echo "  ⚠️  WARNING: DESTRUCTIVE OPERATION  ⚠️"
    echo "=========================================="
    echo ""
    echo "Mode '$OPERATION_MODE' will DELETE files from the dataset!"
    echo "This operation cannot be undone."
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " CONFIRM
    if [[ "$CONFIRM" != "yes" ]]; then
        echo "Operation cancelled by user."
        exit 0
    fi
    echo ""
fi

#===============================================================================
# ACTIVATE CONDA ENVIRONMENT (if available)
#===============================================================================

# Try to activate conda environment
if command -v conda &> /dev/null; then
    # Check if train environment exists (preferred for YOLO training)
    if conda env list | grep -q "train"; then
        echo "Activating train conda environment..."
        eval "$(conda shell.bash hook)"
        conda activate train
    # Fallback to MICRO if it exists
    elif conda env list | grep -q "MICRO"; then
        echo "Activating MICRO conda environment..."
        eval "$(conda shell.bash hook)"
        conda activate MICRO
    fi
fi

#===============================================================================
# RUN EXTRACTION/DELETION BASED ON MODE
#===============================================================================

EXIT_CODE=0

# Run extraction if mode is "extract" or "both"
if [[ "$OPERATION_MODE" == "extract" || "$OPERATION_MODE" == "both" ]]; then
    echo "Starting hard example extraction..."
    echo ""

    python "${SCRIPT_DIR}/modules/training/extract_hard_examples.py" \
        --model-dir "$FINAL_MODEL_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --confusion-pairs "$CONFUSION_PAIRS" \
        --iou-threshold "$IOU_THRESHOLD" \
        --conf-threshold "$CONF_THRESHOLD" \
        --max-per-category "$MAX_PER_CATEGORY" \
        --split "$SPLIT"

    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "  Extraction Failed (exit code: $EXIT_CODE)"
        echo "=========================================="
        exit $EXIT_CODE
    fi
    
    echo ""
    echo "Extraction completed successfully."
    echo ""
fi

# Run deletion if mode is "delete" or "both"
if [[ "$OPERATION_MODE" == "delete" || "$OPERATION_MODE" == "both" ]]; then
    echo "Starting hard example deletion..."
    echo ""

    python "${SCRIPT_DIR}/modules/training/extract_hard_examples.py" \
        --model-dir "$FINAL_MODEL_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --confusion-pairs "$CONFUSION_PAIRS" \
        --iou-threshold "$IOU_THRESHOLD" \
        --conf-threshold "$CONF_THRESHOLD" \
        --max-per-category "$MAX_PER_CATEGORY" \
        --split "$SPLIT" \
        --delete-from-dataset

    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "  Deletion Failed (exit code: $EXIT_CODE)"
        echo "=========================================="
        exit $EXIT_CODE
    fi
    
    echo ""
    echo "Deletion completed successfully."
    echo ""
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "  Operation Complete! (Mode: $OPERATION_MODE)"
    echo "=========================================="
    echo ""
    if [[ "$OPERATION_MODE" == "extract" || "$OPERATION_MODE" == "both" ]]; then
        echo "Results saved to: $OUTPUT_DIR"
        echo "Review the hard_examples_summary.json for details."
    fi
    if [[ "$OPERATION_MODE" == "delete" || "$OPERATION_MODE" == "both" ]]; then
        echo "Hard examples have been deleted from the dataset."
    fi
else
    echo "=========================================="
    echo "  Operation Failed (exit code: $EXIT_CODE)"
    echo "=========================================="
fi

exit $EXIT_CODE
