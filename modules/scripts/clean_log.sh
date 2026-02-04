#!/bin/bash
#===============================================================================
# Clean Log File - Remove ANSI Escape Codes
# Utility script to clean up log files containing terminal color/control codes
#
# Usage:
#   ./clean_log.sh logfile.log                    # Clean in-place
#   ./clean_log.sh logfile.log -o clean.log       # Write to new file
#   ./clean_log.sh logfile.log --stdout           # Output to stdout
#   cat logfile.log | ./clean_log.sh              # Read from stdin
#
# Location: modules/scripts/clean_log.sh
#===============================================================================

set -e

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [LOGFILE]

Remove ANSI escape codes (colors, cursor control) from log files.

Arguments:
  LOGFILE           Input log file to clean (or use stdin via pipe)

Options:
  -o, --output FILE   Write cleaned output to FILE instead of in-place
  -s, --stdout        Output to stdout (no file modification)
  -b, --backup        Create .bak backup before in-place modification
  -r, --recursive DIR Clean all .log files in directory recursively
  -h, --help          Show this help message

Examples:
  $(basename "$0") training.log                    # Clean training.log in-place
  $(basename "$0") training.log -o clean.log       # Save cleaned to clean.log
  $(basename "$0") training.log --stdout | less    # View cleaned in less
  $(basename "$0") -r logs/                        # Clean all logs in directory
  cat training.log | $(basename "$0") --stdout     # Clean from stdin

EOF
    exit 0
}

# Strip ANSI codes using perl (comprehensive) or sed (fallback)
strip_ansi() {
    if command -v perl >/dev/null 2>&1; then
        perl -pe '
            # Remove all ANSI escape sequences
            s/\e\[[0-9;]*[a-zA-Z]//g;
            # Remove OSC sequences (title bar, etc.)
            s/\e\][^\a]*\a//g;
            # Remove carriage return progress bar overwrites (keep last line)
            s/^.*\r(?!\n)//gm;
            # Clean up any remaining escape characters
            s/\e\[\?[0-9;]*[a-zA-Z]//g;
        '
    else
        # Fallback to sed
        sed -E '
            s/\x1b\[[0-9;]*[a-zA-Z]//g
            s/\x1b\][^\x07]*\x07//g
            s/\x1b\[\?[0-9;]*[a-zA-Z]//g
            s/\r[^\n]*\r/\r/g
        '
    fi
}

# Parse arguments
INPUT_FILE=""
OUTPUT_FILE=""
USE_STDOUT=false
CREATE_BACKUP=false
RECURSIVE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -s|--stdout)
            USE_STDOUT=true
            shift
            ;;
        -b|--backup)
            CREATE_BACKUP=true
            shift
            ;;
        -r|--recursive)
            RECURSIVE_DIR="$2"
            shift 2
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            usage
            ;;
        *)
            INPUT_FILE="$1"
            shift
            ;;
    esac
done

# Handle recursive directory cleaning
if [[ -n "$RECURSIVE_DIR" ]]; then
    if [[ ! -d "$RECURSIVE_DIR" ]]; then
        echo "Error: Directory not found: $RECURSIVE_DIR" >&2
        exit 1
    fi
    
    echo "Cleaning all .log files in: $RECURSIVE_DIR"
    count=0
    while IFS= read -r -d '' file; do
        echo "  Cleaning: $file"
        if [[ "$CREATE_BACKUP" == true ]]; then
            cp "$file" "${file}.bak"
        fi
        tmpfile=$(mktemp)
        strip_ansi < "$file" > "$tmpfile"
        mv "$tmpfile" "$file"
        ((count++))
    done < <(find "$RECURSIVE_DIR" -type f -name "*.log" -print0)
    
    echo "Cleaned $count log file(s)."
    exit 0
fi

# Handle stdin input
if [[ -z "$INPUT_FILE" ]]; then
    if [[ -t 0 ]]; then
        # No file and no stdin - show usage
        echo "Error: No input file specified and no stdin detected." >&2
        usage
    fi
    # Read from stdin, output to stdout
    strip_ansi
    exit 0
fi

# Validate input file
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: File not found: $INPUT_FILE" >&2
    exit 1
fi

# Handle stdout output
if [[ "$USE_STDOUT" == true ]]; then
    strip_ansi < "$INPUT_FILE"
    exit 0
fi

# Handle output to different file
if [[ -n "$OUTPUT_FILE" ]]; then
    strip_ansi < "$INPUT_FILE" > "$OUTPUT_FILE"
    echo "Cleaned log saved to: $OUTPUT_FILE"
    exit 0
fi

# In-place modification
if [[ "$CREATE_BACKUP" == true ]]; then
    cp "$INPUT_FILE" "${INPUT_FILE}.bak"
    echo "Backup created: ${INPUT_FILE}.bak"
fi

tmpfile=$(mktemp)
strip_ansi < "$INPUT_FILE" > "$tmpfile"
mv "$tmpfile" "$INPUT_FILE"
echo "Cleaned: $INPUT_FILE"
