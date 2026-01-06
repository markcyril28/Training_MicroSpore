#!/bin/bash

# Chmod and Converts the file to Unix line endings
chmod +x ./*.sh
dos2unix ./*.sh
find . -type f \( -name "*.txt" -o -name "*.sh" -o -name "*.py" -o -name "*.R" \) -exec dos2unix {} + 2>/dev/null || true
find . -type f \( -name "*.sh" -o -name "*.py" -o -name "*.pl" -o -name "*.R" \) -exec chmod +x {} + 2>/dev/null || true
