#!/bin/bash
cd "$(dirname "$0")"

# === OPERATION: Comment/uncomment ONE section below ===

# --- COMPRESS (xz extreme - smallest size, slower) ---
#tar -I "xz -9e -T0" -cvf "Dataset_1_TEST.tar.xz" Dataset_1_TEST
tar -I "xz -9e -T0" -cvf "Dataset_2_OPTIMIZATION.tar.xz" Dataset_2_OPTIMIZATION
#tar -I "xz -9e -T0" -cvf "Dataset_3_FINAL_RUN.tar.xz" Dataset_3_FINAL_RUN

# --- EXTRACT (xz multithreaded) ---
#tar -I "xz -T0" -xvf "Dataset_1_TEST.tar.xz"
#tar -I "xz -T0" -xvf "Dataset_2_OPTIMIZATION.tar.xz"
#tar -I "xz -T0" -xvf "Dataset_3_FINAL_RUN.tar.xz"