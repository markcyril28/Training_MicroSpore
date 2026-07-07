#!/usr/bin/env bash
#===============================================================================
# Microspore Phenotyping - Interpretability Pipeline Runner
# Generates YOLO detection-confidence heatmaps and optional occlusion maps.
# Uses WSL/Linux bash + Python only; no PowerShell/cmd dependencies.
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL=""
DATA_YAML=""
IMAGES_DIR=""
OUTPUT_DIR=""
SPLIT="val"
LIMIT="12"
CONF="0.25"
IOU="0.70"
IMGSZ=""
DEVICE=""
OCCLUSION="false"
GRID_SIZE="8"
TARGET_CLASS=""
IMAGES=()

usage() {
    cat <<'USAGE'
Microspore YOLO Interpretability Pipeline

Required:
  --model PATH                Trained YOLO .pt model, e.g. weights/<exp>_best.pt

Image source (choose one):
  --data-yaml PATH            YOLO data.yaml; uses --split images and class names
  --images-dir PATH           Directory of images to interpret
  --image PATH                Individual image; repeatable

Options:
  --output-dir PATH           Output folder (default: interpretability_output/<model>_<timestamp>)
  --split NAME                data.yaml split to use (default: val)
  --limit N                   Max images to process (default: 12)
  --conf FLOAT                YOLO confidence threshold (default: 0.25)
  --iou FLOAT                 YOLO IoU/NMS threshold (default: 0.70)
  --imgsz N                   Optional inference image size
  --device DEVICE             Optional inference device, e.g. 0 or cpu
  --occlusion                 Also run slower model-agnostic occlusion sensitivity
  --grid-size N               Occlusion grid size per dimension (default: 8)
  --target-class ID_OR_NAME   Focus heatmaps/occlusion scoring on a class
  -h, --help                  Show this help

Examples:
  bash run_interpretability.sh \
    --model trained_models_output/server/run/weights/run_best.pt \
    --data-yaml TRAINING_WD/Dataset_2_OPTIMIZATION/data.yaml \
    --limit 12

  bash run_interpretability.sh \
    --model trained_models_output/local/run/weights/run_best.pt \
    --images-dir TRAINING_WD/Dataset_2_OPTIMIZATION/Test \
    --output-dir interpretability_output/review_run \
    --occlusion --limit 3
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --data-yaml) DATA_YAML="$2"; shift 2 ;;
        --images-dir) IMAGES_DIR="$2"; shift 2 ;;
        --image) IMAGES+=("$2"); shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --conf) CONF="$2"; shift 2 ;;
        --iou) IOU="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --occlusion) OCCLUSION="true"; shift ;;
        --grid-size) GRID_SIZE="$2"; shift 2 ;;
        --target-class) TARGET_CLASS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "ERROR: --model is required." >&2
    usage
    exit 2
fi

if [[ -z "${DATA_YAML}" && -z "${IMAGES_DIR}" && ${#IMAGES[@]} -eq 0 ]]; then
    echo "ERROR: provide --data-yaml, --images-dir, or at least one --image." >&2
    usage
    exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    MODEL_STEM="$(basename "${MODEL%.*}")"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${SCRIPT_DIR}/interpretability_output/${MODEL_STEM}_${TIMESTAMP}"
fi

CMD=(
    "${PYTHON_BIN}" -m modules.interpretability.pipeline
    --model "${MODEL}"
    --output-dir "${OUTPUT_DIR}"
    --split "${SPLIT}"
    --limit "${LIMIT}"
    --conf "${CONF}"
    --iou "${IOU}"
    --grid-size "${GRID_SIZE}"
)

[[ -n "${DATA_YAML}" ]] && CMD+=(--data-yaml "${DATA_YAML}")
[[ -n "${IMAGES_DIR}" ]] && CMD+=(--images-dir "${IMAGES_DIR}")
[[ -n "${IMGSZ}" ]] && CMD+=(--imgsz "${IMGSZ}")
[[ -n "${DEVICE}" ]] && CMD+=(--device "${DEVICE}")
[[ -n "${TARGET_CLASS}" ]] && CMD+=(--target-class "${TARGET_CLASS}")
[[ "${OCCLUSION}" == "true" ]] && CMD+=(--occlusion)

for image in "${IMAGES[@]}"; do
    CMD+=(--image "${image}")
done

cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "[Interpretability] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[Interpretability] Done. Output: ${OUTPUT_DIR}"
