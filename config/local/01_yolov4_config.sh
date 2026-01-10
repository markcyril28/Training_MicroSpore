#!/bin/bash

#===============================================================================
# YOLOv4 CONFIGURATION - LOCAL (Darknet-compatible Pipeline)
#===============================================================================
# This configuration is for YOLOv4 training using the Darknet framework.
# YOLOv4 is NOT supported by Ultralytics, so it requires a separate pipeline.
#
# Supported models: yolov4, yolov4-tiny, yolov4-csp, yolov4x-mish
#===============================================================================

#===============================================================================
# YOLOv4 MODEL SELECTION
#===============================================================================
# Choose ONE model variant for training
YOLOV4_MODEL="yolov4-tiny"    # OPTIMAL for local 8GB VRAM

# Available YOLOv4 variants:
# "yolov4-tiny"     # tiny     - fastest, lowest accuracy (~6GB VRAM)
# "yolov4"          # standard - original YOLOv4 (~10GB VRAM)
# "yolov4-csp"      # CSP      - cross-stage partial (~12GB VRAM)
# "yolov4x-mish"    # xlarge   - best accuracy (~16GB+ VRAM)

#===============================================================================
# DATASET CONFIGURATION
#===============================================================================
# Parent directory containing datasets
DATASETS_DIR="TRAINING_WD"

# Dataset name (must contain data.yaml, Train/, Test/)
DATASET_NAME="Dataset_2_OPTIMIZATION"

# Class names file (auto-generated from data.yaml if not provided)
# CLASSES_FILE=""  # Leave empty to auto-generate

#===============================================================================
# TRAINING PARAMETERS
#===============================================================================
# These parameters are compatible with Darknet/YOLOv4

# Core Training Parameters
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE=16               # Lower for 8GB VRAM (subdivisions will be used)
SUBDIVISIONS=4              # Effective mini-batch = BATCH_SIZE / SUBDIVISIONS
MAX_BATCHES=6000            # Total training iterations (classes * 2000 recommended)
IMG_SIZE=416                # Input resolution (416 or 608 typical for YOLOv4)

# Learning Rate Configuration (from microspores.cfg)
# ─────────────────────────────────────────────────────────────────────────────
LEARNING_RATE=0.001         # Initial learning rate
BURN_IN=1000                # Warmup iterations (gradual LR increase)
POLICY="steps"              # LR decay policy: steps, exp, poly, sig
STEPS="4800,5400"           # When to reduce LR (80% and 90% of max_batches)
SCALES="0.1,0.1"            # LR multipliers at each step

# Momentum and Regularization
# ─────────────────────────────────────────────────────────────────────────────
MOMENTUM=0.949              # SGD momentum (from microspores.cfg)
DECAY=0.0005                # Weight decay (L2 regularization)

# Augmentation Parameters (Darknet style)
# ─────────────────────────────────────────────────────────────────────────────
ANGLE=0                     # Rotation angle range (0 = disabled for speed)
SATURATION=1.5              # Saturation augmentation
EXPOSURE=1.5                # Exposure (brightness) augmentation
HUE=0.1                     # Hue shift (from microspores.cfg)
MOSAIC=1                    # Mosaic augmentation (1=enabled, 0=disabled)
FLIP=1                      # Horizontal flip (1=enabled)

# Anchor Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Default anchors for YOLOv4 (can be recalculated with darknet calc_anchors)
# Leave empty to use default anchors from cfg file
ANCHORS=""

# IoU and NMS Configuration (from microspores.cfg)
# ─────────────────────────────────────────────────────────────────────────────
IOU_THRESH=0.213            # IoU threshold for positive samples
IOU_NORMALIZER=0.07         # IoU loss weight
CLS_NORMALIZER=1.0          # Classification loss weight
NMS_KIND="greedynms"        # NMS algorithm: greedynms, diounms, cornersnms

#===============================================================================
# OUTPUT CONFIGURATION
#===============================================================================
# Where to save trained weights
OUTPUT_DIR="trained_models_output"

# Save checkpoint every N iterations
SAVE_INTERVAL=1000

# Enable GPU (0 = first GPU, -1 = CPU)
GPU_ID=0

#===============================================================================
# ADVANCED OPTIONS
#===============================================================================
# Use pretrained weights (recommended)
PRETRAINED=true
PRETRAINED_WEIGHTS_URL="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4-tiny.conv.29"

# Random training (multi-scale)
RANDOM=1                    # 1 = multi-scale training

# Letter box (preserve aspect ratio)
LETTER_BOX=1

# Mixed precision (if supported)
MIXED_PRECISION=false       # Set true if GPU supports FP16
