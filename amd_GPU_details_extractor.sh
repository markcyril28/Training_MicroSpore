#!/usr/bin/env bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTFILE="amd_gpu_yolo_specs_${TIMESTAMP}.txt"

exec > >(tee "${OUTFILE}") 2>&1

echo "===================================================="
echo " AMD GPU SPECIFICATION REPORT FOR YOLO TRAINING"
echo " Generated on: ${TIMESTAMP}"
echo "===================================================="
echo

echo "=== 1. PCI DEVICE IDENTIFICATION ==="
lspci -nnk | grep -A3 -Ei 'vga|display|amd' || echo "lspci not available"
echo

echo "=== 2. KERNEL DRIVER INFORMATION ==="
if [ -f /proc/driver/amdgpu/version ]; then
    cat /proc/driver/amdgpu/version
else
    echo "amdgpu kernel driver info not found"
fi
echo

echo "=== 3. DISPLAY HARDWARE SUMMARY (lshw) ==="
if command -v lshw >/dev/null 2>&1; then
    sudo lshw -C display 2>/dev/null || lshw -C display
else
    echo "lshw not installed"
fi
echo

echo "=== 4. DRM / VRAM INFORMATION ==="
for CARD in /sys/class/drm/card*/device; do
    echo "Device: ${CARD}"
    [ -f ${CARD}/vendor ] && echo -n "  Vendor ID: " && cat ${CARD}/vendor
    [ -f ${CARD}/device ] && echo -n "  Device ID: " && cat ${CARD}/device
    [ -f ${CARD}/mem_info_vram_total ] && \
        echo -n "  VRAM Total (MB): " && awk '{printf "%.0f\n",$1/1024/1024}' ${CARD}/mem_info_vram_total
    [ -f ${CARD}/mem_info_vram_used ] && \
        echo -n "  VRAM Used  (MB): " && awk '{printf "%.0f\n",$1/1024/1024}' ${CARD}/mem_info_vram_used
    echo
done

echo "=== 5. ROCm ARCHITECTURE & COMPUTE UNITS ==="
if command -v rocminfo >/dev/null 2>&1; then
    rocminfo | grep -E "Name:|gfx|Compute Unit|Wavefront" || true
else
    echo "rocminfo not available (ROCm not installed)"
fi
echo

echo "=== 6. ROCm SYSTEM MANAGEMENT (rocm-smi) ==="
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname \
             --showmeminfo vram \
             --showclocks \
             --showpower \
             --showtemp
else
    echo "rocm-smi not available"
fi
echo

echo "=== 7. PCIe LINK SPEED & WIDTH ==="
lspci -vv | grep -A15 -Ei 'vga|display' | grep -E 'LnkCap|LnkSta' || echo "PCIe info unavailable"
echo

echo "=== 8. OPENCL (OPTIONAL, IF INSTALLED) ==="
if command -v clinfo >/dev/null 2>&1; then
    clinfo | grep -A20 "Device Name" || true
else
    echo "clinfo not installed"
fi
echo

echo "===================================================="
echo " END OF REPORT"
echo " Output saved to: ${OUTFILE}"
echo "===================================================="
