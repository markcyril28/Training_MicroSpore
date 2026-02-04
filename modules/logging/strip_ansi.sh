#!/bin/bash
#===============================================================================
# Microspore Phenotyping - ANSI Escape Code Stripper
# Removes ANSI color codes and control characters from log files
#
# Usage:
#   ./strip_ansi.sh input.log              # Outputs to stdout
#   ./strip_ansi.sh input.log output.log   # Saves to file
#   ./strip_ansi.sh input.log --inplace    # Overwrites original file
#
# Or with pipes:
#   cat log.txt | ./strip_ansi.sh > clean.txt
#===============================================================================

strip_ansi() {
    # Remove ANSI escape sequences:
    # - Color codes: \e[...m (e.g., [34m, [1m, [0m)
    # - Cursor control: \e[...K, \e[...H, etc.
    # - Other escape sequences: \e[?...h, \e[?...l
    sed -E 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\x1b\][^]*\x07//g; s/\x1b\[[?]?[0-9;]*[a-zA-Z]//g'
}

if [ $# -eq 0 ]; then
    # Read from stdin
    strip_ansi
elif [ $# -eq 1 ]; then
    # Input file only - output to stdout
    if [ -f "$1" ]; then
        strip_ansi < "$1"
    else
        echo "Error: File not found: $1" >&2
        exit 1
    fi
elif [ $# -eq 2 ]; then
    if [ "$2" = "--inplace" ] || [ "$2" = "-i" ]; then
        # In-place modification
        if [ -f "$1" ]; then
            temp_file=$(mktemp)
            strip_ansi < "$1" > "$temp_file"
            mv "$temp_file" "$1"
            echo "Cleaned: $1"
        else
            echo "Error: File not found: $1" >&2
            exit 1
        fi
    else
        # Input and output files
        if [ -f "$1" ]; then
            strip_ansi < "$1" > "$2"
            echo "Created clean log: $2"
        else
            echo "Error: File not found: $1" >&2
            exit 1
        fi
    fi
else
    echo "Usage: $0 [input.log] [output.log|--inplace]" >&2
    exit 1
fi
