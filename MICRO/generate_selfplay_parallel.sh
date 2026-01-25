#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# GNU Parallel Self-Play Generator
# =============================================================================
# This script uses GNU Parallel to generate self-play games much faster than
# Python multiprocessing. It's ideal for servers with many CPU cores.
#
# Requirements:
#   - GNU Parallel: sudo apt install parallel (or brew install parallel on macOS)
#
# Usage:
#   ./generate_selfplay_parallel.sh [num_games] [num_workers]
#
# Examples:
#   ./generate_selfplay_parallel.sh 1000 32   # Generate 1000 games with 32 workers
#   ./generate_selfplay_parallel.sh           # Use defaults (512 games, auto workers)
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Configuration
TOTAL_GAMES="${1:-512}"
NUM_WORKERS="${2:-$(nproc 2>/dev/null || echo 16)}"  # Auto-detect cores
DIFFICULTY="${3:-medium}"
NOISE_PROB="${4:-0.1}"
MAX_MOVES="${5:-200}"

# Calculate games per worker
GAMES_PER_WORKER=$(( (TOTAL_GAMES + NUM_WORKERS - 1) / NUM_WORKERS ))

# Output directory
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="${PROJECT_DIR}/data/replay/parallel_${TIMESTAMP}"
FINAL_OUTPUT="${PROJECT_DIR}/data/replay/replay_${TIMESTAMP}.jsonl"

echo "=============================================="
echo "GNU Parallel Self-Play Generator"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Total Games:      ${TOTAL_GAMES}"
echo "  Workers:          ${NUM_WORKERS}"
echo "  Games/Worker:     ${GAMES_PER_WORKER}"
echo "  Difficulty:       ${DIFFICULTY}"
echo "  Noise Prob:       ${NOISE_PROB}"
echo "  Max Moves:        ${MAX_MOVES}"
echo "  Output Dir:       ${OUTPUT_DIR}"
echo ""

# Check for GNU Parallel
if ! command -v parallel &> /dev/null; then
    echo "ERROR: GNU Parallel is not installed."
    echo ""
    echo "Install it with:"
    echo "  Ubuntu/Debian: sudo apt install parallel"
    echo "  macOS:         brew install parallel"
    echo "  RHEL/CentOS:   sudo yum install parallel"
    exit 1
fi

# Silence parallel citation notice
mkdir -p ~/.parallel && touch ~/.parallel/will-cite 2>/dev/null || true

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Add src to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "Starting parallel self-play generation..."
echo ""

START_TIME=$(date +%s)

# Run parallel self-play
# Each worker gets a unique seed based on its job number
seq 1 "$NUM_WORKERS" | parallel -j "$NUM_WORKERS" --progress --bar \
    "python -m micro.ai.ml.selfplay_worker \
        --games ${GAMES_PER_WORKER} \
        --output '${OUTPUT_DIR}/games_{}.jsonl' \
        --difficulty ${DIFFICULTY} \
        --noise-prob ${NOISE_PROB} \
        --max-moves ${MAX_MOVES} \
        --seed {}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Merging output files..."

# Merge all output files into a single JSONL
cat "${OUTPUT_DIR}"/games_*.jsonl > "$FINAL_OUTPUT"

# Count entries
ENTRY_COUNT=$(wc -l < "$FINAL_OUTPUT")

# Clean up temp files
rm -rf "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Self-play generation complete!"
echo "=============================================="
echo "  Time:            ${ELAPSED}s"
echo "  Games:           ${TOTAL_GAMES}"
echo "  Entries:         ${ENTRY_COUNT}"
echo "  Output:          ${FINAL_OUTPUT}"
echo "  Speed:           $(echo "scale=1; ${TOTAL_GAMES} / ${ELAPSED}" | bc) games/sec"
echo ""
echo "The replay file is ready for training."
