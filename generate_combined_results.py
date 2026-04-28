#!/usr/bin/env python3
"""
Generate combined training results across all training phases.
Combines epoch metrics, generates graphs, and creates summary reports.
Excludes confusion matrices as requested.
"""

import os
import csv
import json
import math
from datetime import datetime
from pathlib import Path

# Try to import matplotlib - if not available, skip graph generation
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import FancyBboxPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available. Graphs will not be generated.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not available. Some calculations may be limited.")

# ============================================================================
# CONFIGURATION
# ============================================================================

import platform
import sys

# Auto-detect path format (WSL vs native Windows)
if platform.system() == "Linux" and os.path.exists("/mnt/c"):
    # Running under WSL
    BASE_DIR = Path("/mnt/c/_MicroSpore/C_TRAINING/trained_models_output/server")
else:
    # Native Windows
    BASE_DIR = Path(r"c:\_MicroSpore\C_TRAINING\trained_models_output\server")

OUTPUT_DIR = BASE_DIR / "COMBINED_ALL_PHASES"

# Define all training phases in chronological order
PHASES = [
    {
        "name": "Phase 1 (Original)",
        "short": "Phase_1",
        "dir": BASE_DIR / "04_class_balancing_combination" / "Dataset_2_OPTIMIZATION_yolov8x_gray_img1280_bal-manual_auto_e500_b8_lr0_001_20260121_045400",
        "metrics_file": "stats/epoch_metrics.csv",
        "summary_file": "stats/training_summary.json",
        "lr0": 0.001,
        "batch": 8,
        "epochs_planned": 500,
        "patience": 300,
        "source_weights": "yolov8x.pt (pretrained)",
        "color": "#2196F3",  # Blue
        "date": "2026-01-21",
    },
    {
        "name": "Phase 2 (continue_a)",
        "short": "continue_a",
        "dir": BASE_DIR / "04_class_balancing_combination_continue" / "continue_a",
        "metrics_file": "stats/epoch_metrics.csv",
        "summary_file": "stats/training_summary.json",
        "lr0": 0.0001,
        "batch": 8,
        "epochs_planned": 500,
        "patience": 100,
        "source_weights": "Phase 1 last.pt",
        "color": "#FF9800",  # Orange
        "date": "2026-01-23",
    },
    {
        "name": "Phase 3 (continue_b)",
        "short": "continue_b",
        "dir": BASE_DIR / "04_class_balancing_combination_continue" / "continue_b",
        "metrics_file": "results.csv",  # Different filename!
        "summary_file": "stats/training_summary.json",
        "lr0": 5e-05,
        "batch": 4,
        "epochs_planned": 500,
        "patience": 500,
        "source_weights": "Phase 1 last.pt",
        "color": "#4CAF50",  # Green
        "date": "2026-01-28",
    },
    {
        "name": "Phase 4 (continue_2)",
        "short": "continue_2",
        "dir": BASE_DIR / "04_class_balancing_combination_continue_2" / "continue_1",
        "metrics_file": "stats/epoch_metrics.csv",
        "summary_file": "stats/training_summary.json",
        "lr0": 1e-05,
        "batch": 8,
        "epochs_planned": 300,
        "patience": 100,
        "source_weights": "continue_b best.pt",
        "color": "#9C27B0",  # Purple
        "date": "2026-02-01",
    },
    {
        "name": "Phase 5 (continue_3)",
        "short": "continue_3",
        "dir": BASE_DIR / "04_class_balancing_combination_continue_3" / "continue_3",
        "metrics_file": "stats/epoch_metrics.csv",
        "summary_file": "stats/training_summary.json",
        "lr0": 5e-05,
        "batch": 8,
        "epochs_planned": 400,
        "patience": 150,
        "source_weights": "continue_2 best.pt",
        "color": "#F44336",  # Red
        "date": "2026-02-01",
    },
    {
        "name": "Phase 6 (continue_4_Phase_1)",
        "short": "continue_4",
        "dir": BASE_DIR / "04_class_balancing_combination_continue_4_5.2_phase_1" / "continue_4_Phase_1",
        "metrics_file": "stats/epoch_metrics.csv",
        "summary_file": "stats/training_summary.json",
        "lr0": 1e-05,
        "batch": 8,
        "epochs_planned": 300,
        "patience": 100,
        "source_weights": "continue_2 best.pt",
        "color": "#795548",  # Brown
        "date": "2026-02-02",
    },
]

# Standard columns (first 15)
STANDARD_COLUMNS = [
    "epoch", "time", "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "lr/pg0", "lr/pg1", "lr/pg2"
]

CLASS_NAMES = [
    "tetrad", "young_microspore", "mid_microspore", "late_microspore",
    "young_pollen", "midlate_pollen", "mature_pollen", "others"
]


# ============================================================================
# DATA LOADING
# ============================================================================

def read_csv_data(filepath, standardize=True):
    """Read epoch metrics CSV and standardize to 15 columns."""
    rows = []
    if not os.path.exists(filepath):
        print(f"  WARNING: File not found: {filepath}")
        return [], []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Strip whitespace from header
        header = [h.strip() for h in header]
        
        for row in reader:
            if len(row) < 12:  # Skip incomplete rows
                continue
            # Strip whitespace from values
            row = [v.strip() for v in row]
            
            if standardize and len(row) > 15:
                # Take only first 15 columns (drop extra lr columns)
                row = row[:15]
            
            # Convert to float where possible
            converted = []
            for v in row:
                try:
                    converted.append(float(v))
                except ValueError:
                    converted.append(v)
            rows.append(converted)
    
    if standardize:
        header = STANDARD_COLUMNS
    
    return header, rows


def load_all_phases():
    """Load epoch metrics from all phases."""
    all_data = []
    
    for phase in PHASES:
        filepath = phase["dir"] / phase["metrics_file"]
        print(f"Loading {phase['name']} from {filepath}")
        header, rows = read_csv_data(filepath)
        
        if rows:
            print(f"  Loaded {len(rows)} epochs")
            all_data.append({
                "phase": phase,
                "header": header,
                "rows": rows,
            })
        else:
            print(f"  WARNING: No data loaded for {phase['name']}")
    
    return all_data


def combine_phases(all_data):
    """Combine all phases into a single dataset with sequential epoch numbering."""
    combined_rows = []
    phase_boundaries = []
    cumulative_epoch = 0
    
    for data in all_data:
        phase = data["phase"]
        rows = data["rows"]
        start_epoch = cumulative_epoch + 1
        
        for row in rows:
            cumulative_epoch += 1
            # Create new row with sequential epoch number and phase info
            new_row = [cumulative_epoch] + list(row[1:])  # Replace original epoch with cumulative
            combined_rows.append({
                "data": new_row,
                "phase": phase["name"],
                "phase_short": phase["short"],
                "original_epoch": int(row[0]),
            })
        
        end_epoch = cumulative_epoch
        phase_boundaries.append({
            "phase": phase,
            "start": start_epoch,
            "end": end_epoch,
            "count": end_epoch - start_epoch + 1,
        })
        print(f"  {phase['name']}: combined epochs {start_epoch}-{end_epoch} ({len(rows)} epochs)")
    
    return combined_rows, phase_boundaries


# ============================================================================
# FILE GENERATION
# ============================================================================

def create_output_dirs():
    """Create the output directory structure."""
    dirs = [
        OUTPUT_DIR,
        OUTPUT_DIR / "stats",
        OUTPUT_DIR / "stats" / "optimization",
        OUTPUT_DIR / "visualizations",
        OUTPUT_DIR / "visualizations" / "curves",
        OUTPUT_DIR / "visualizations" / "overviews",
        OUTPUT_DIR / "logs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")


def write_combined_epoch_metrics(combined_rows, phase_boundaries):
    """Write combined epoch_metrics.csv."""
    filepath = OUTPUT_DIR / "stats" / "epoch_metrics.csv"
    
    # Header with phase column
    header = ["epoch", "phase"] + STANDARD_COLUMNS[1:]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for row_info in combined_rows:
            data = row_info["data"]
            phase = row_info["phase_short"]
            # epoch, phase, time, losses..., metrics..., lr...
            out_row = [int(data[0]), phase] + [data[i] if i < len(data) else "" for i in range(1, 15)]
            writer.writerow(out_row)
    
    print(f"  Written combined epoch_metrics.csv ({len(combined_rows)} rows)")
    return filepath


def find_best_metrics(combined_rows):
    """Find the best metrics across all combined epochs."""
    best_mAP50 = 0.0
    best_mAP50_epoch = 0
    best_mAP50_phase = ""
    best_mAP50_95 = 0.0
    best_mAP50_95_epoch = 0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    
    for row_info in combined_rows:
        data = row_info["data"]
        epoch = int(data[0])
        phase = row_info["phase"]
        
        # Indices in standardized data: 
        # 0:epoch, 1:time, 2:train/box, 3:train/cls, 4:train/dfl,
        # 5:precision, 6:recall, 7:mAP50, 8:mAP50-95,
        # 9:val/box, 10:val/cls, 11:val/dfl, 12:lr/pg0, 13:lr/pg1, 14:lr/pg2
        
        try:
            precision = float(data[5])
            recall = float(data[6])
            mAP50 = float(data[7])
            mAP50_95 = float(data[8])
        except (IndexError, ValueError, TypeError):
            continue
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if mAP50 > best_mAP50:
            best_mAP50 = mAP50
            best_mAP50_epoch = epoch
            best_mAP50_phase = phase
        
        if mAP50_95 > best_mAP50_95:
            best_mAP50_95 = mAP50_95
            best_mAP50_95_epoch = epoch
        
        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if f1 > best_f1:
            best_f1 = f1
    
    # Get final metrics from last row
    last = combined_rows[-1]["data"]
    try:
        final_precision = float(last[5])
        final_recall = float(last[6])
        final_mAP50 = float(last[7])
        final_mAP50_95 = float(last[8])
        final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
    except (IndexError, ValueError, TypeError):
        final_precision = final_recall = final_mAP50 = final_mAP50_95 = final_f1 = 0.0
    
    return {
        "best_mAP50": best_mAP50,
        "best_mAP50_epoch": best_mAP50_epoch,
        "best_mAP50_phase": best_mAP50_phase,
        "best_mAP50_95": best_mAP50_95,
        "best_mAP50_95_epoch": best_mAP50_95_epoch,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_f1": best_f1,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_mAP50": final_mAP50,
        "final_mAP50_95": final_mAP50_95,
        "final_f1": final_f1,
        "total_epochs": len(combined_rows),
    }


def write_general_results(combined_rows, phase_boundaries, metrics):
    """Write combined GENERAL_RESULTS markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = OUTPUT_DIR / f"GENERAL_RESULTS_COMBINED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    total_time_seconds = 0
    for row_info in combined_rows:
        try:
            total_time_seconds = max(total_time_seconds, float(row_info["data"][1]))
        except (ValueError, TypeError):
            pass
    
    # Compute total time across all phases
    total_time_all = 0
    for pb in phase_boundaries:
        phase_data = [r for r in combined_rows if r["phase_short"] == pb["phase"]["short"]]
        if phase_data:
            try:
                phase_time = max(float(r["data"][1]) for r in phase_data)
                total_time_all += phase_time
            except (ValueError, TypeError):
                pass
    
    total_hours = total_time_all / 3600
    total_days = total_hours / 24
    
    content = f"""# Combined Training Results Summary
**Experiment:** YOLOv8x - MicroSpore Detection (All Phases Combined)  
**Generated:** {now}  
**Total Combined Epochs:** {metrics['total_epochs']}  
**Total Training Time:** {total_hours:.1f} hours ({total_days:.1f} days)

---

## Overall Best Performance Metrics

| Metric | Value | At Epoch | Phase |
|--------|-------|----------|-------|
| **Best mAP@50** | {metrics['best_mAP50']:.4f} ({metrics['best_mAP50']*100:.1f}%) | {metrics['best_mAP50_epoch']} | {metrics['best_mAP50_phase']} |
| **Best mAP@50-95** | {metrics['best_mAP50_95']:.4f} ({metrics['best_mAP50_95']*100:.1f}%) | {metrics['best_mAP50_95_epoch']} | - |
| **Best Precision** | {metrics['best_precision']:.4f} ({metrics['best_precision']*100:.1f}%) | - | - |
| **Best Recall** | {metrics['best_recall']:.4f} ({metrics['best_recall']*100:.1f}%) | - | - |
| **Best F1 Score** | {metrics['best_f1']:.4f} ({metrics['best_f1']*100:.1f}%) | - | - |

---

## Final Metrics (Last Epoch of Last Phase)

| Metric | Value |
|--------|-------|
| **Final Precision** | {metrics['final_precision']:.4f} ({metrics['final_precision']*100:.1f}%) |
| **Final Recall** | {metrics['final_recall']:.4f} ({metrics['final_recall']*100:.1f}%) |
| **Final F1 Score** | {metrics['final_f1']:.4f} ({metrics['final_f1']*100:.1f}%) |
| **Final mAP@50** | {metrics['final_mAP50']:.4f} ({metrics['final_mAP50']*100:.1f}%) |
| **Final mAP@50-95** | {metrics['final_mAP50_95']:.4f} ({metrics['final_mAP50_95']*100:.1f}%) |

---

## Phase-by-Phase Summary

| Phase | Epochs | LR | Batch | Patience | Best mAP@50 | Best Epoch | Source Weights |
|-------|--------|----|-------|----------|-------------|------------|----------------|
"""
    
    for pb in phase_boundaries:
        p = pb["phase"]
        # Find best mAP50 in this phase
        phase_data = [r for r in combined_rows if r["phase_short"] == p["short"]]
        phase_best_mAP50 = 0
        phase_best_epoch = 0
        for r in phase_data:
            try:
                m = float(r["data"][7])
                if m > phase_best_mAP50:
                    phase_best_mAP50 = m
                    phase_best_epoch = r["original_epoch"]
            except (ValueError, TypeError, IndexError):
                pass
        
        content += f"| {p['name']} | {pb['count']} ({pb['start']}-{pb['end']}) | {p['lr0']} | {p['batch']} | {p['patience']} | {phase_best_mAP50:.4f} ({phase_best_mAP50*100:.1f}%) | {phase_best_epoch} | {p['source_weights']} |\n"
    
    content += """
---

## Training Chain / Model Lineage

```
Phase 1 (Original, 500 ep)
├── continue_a (102 ep) - [ABANDONED: performance degraded]
└── continue_b (500 ep)
    └── continue_2 (31 ep)
        ├── continue_3 (34 ep) - [higher LR, strong augmentation]
        └── continue_4_Phase_1 (41 ep) - [conservative fine-tuning]
```

### Key Training Decisions:
1. **Phase 1 → continue_a**: Reduced LR to 0.0001, but mAP degraded (0.714 → 0.693). Abandoned.
2. **Phase 1 → continue_b**: Used Phase 1 last.pt, LR=5e-05, batch=4, added mixup=0.1, label_smoothing=0.1. Maintained ~0.71 mAP.
3. **continue_b → continue_2**: Further reduced LR to 1e-05, batch=8. Best mAP50=0.713 at epoch 2.
4. **continue_2 → continue_3**: Increased LR to 5e-05, mixup=0.3, cls=1.5. Performance unstable (mAP~0.70).
5. **continue_2 → continue_4_Phase_1**: Conservative LR=1e-05, iou=0.6, mixup=0.15. Best mAP50=0.713.

---

## Quick Assessment

"""
    
    if metrics['best_mAP50'] >= 0.70:
        content += "✅ **GOOD** - mAP@50 ≥ 70% indicates reliable detection\n"
    else:
        content += "⚠️ **MODERATE** - mAP@50 < 70%\n"
    
    if metrics['best_mAP50_95'] >= 0.50:
        content += "✅ **GOOD** - mAP@50-95 ≥ 50% indicates strong localization\n"
    else:
        content += "⚠️ **MODERATE** - mAP@50-95 < 50%\n"
    
    if metrics['best_f1'] >= 0.65:
        content += "✅ **GOOD** - F1 Score ≥ 65% indicates balanced precision/recall\n"
    else:
        content += "⚠️ **MODERATE** - F1 Score < 65%\n"
    
    content += f"\n**Total training investment:** {metrics['total_epochs']} epochs across {len(phase_boundaries)} phases over {total_days:.1f} days\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  Written GENERAL_RESULTS: {filepath.name}")
    return filepath


def write_training_summary_json(combined_rows, phase_boundaries, metrics):
    """Write combined training_summary.json."""
    filepath = OUTPUT_DIR / "stats" / "training_summary.json"
    
    phase_summaries = []
    for pb in phase_boundaries:
        p = pb["phase"]
        phase_data = [r for r in combined_rows if r["phase_short"] == p["short"]]
        phase_best_mAP50 = 0
        phase_best_epoch = 0
        phase_best_mAP50_95 = 0
        final_p = final_r = 0
        for r in phase_data:
            try:
                m50 = float(r["data"][7])
                m95 = float(r["data"][8])
                if m50 > phase_best_mAP50:
                    phase_best_mAP50 = m50
                    phase_best_epoch = r["original_epoch"]
                if m95 > phase_best_mAP50_95:
                    phase_best_mAP50_95 = m95
            except (ValueError, TypeError, IndexError):
                pass
        if phase_data:
            try:
                final_p = float(phase_data[-1]["data"][5])
                final_r = float(phase_data[-1]["data"][6])
            except (ValueError, TypeError, IndexError):
                pass
        
        phase_summaries.append({
            "name": p["name"],
            "short_name": p["short"],
            "date": p["date"],
            "combined_epoch_range": [pb["start"], pb["end"]],
            "original_epochs": pb["count"],
            "lr0": p["lr0"],
            "batch_size": p["batch"],
            "patience": p["patience"],
            "epochs_planned": p["epochs_planned"],
            "source_weights": p["source_weights"],
            "best_epoch_original": phase_best_epoch,
            "best_mAP50": round(phase_best_mAP50, 5),
            "best_mAP50_95": round(phase_best_mAP50_95, 5),
            "final_precision": round(final_p, 5),
            "final_recall": round(final_r, 5),
        })
    
    summary = {
        "experiment_name": "YOLOv8x_MicroSpore_Combined_All_Phases",
        "generated_at": datetime.now().isoformat(),
        "config": {
            "model": "yolov8x",
            "task": "detect",
            "img_size": 1280,
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "dataset": "Dataset_2_OPTIMIZATION",
        },
        "combined_metrics": {
            "total_epochs": metrics["total_epochs"],
            "total_phases": len(phase_boundaries),
            "best_mAP50": round(metrics["best_mAP50"], 5),
            "best_mAP50_epoch": metrics["best_mAP50_epoch"],
            "best_mAP50_phase": metrics["best_mAP50_phase"],
            "best_mAP50_95": round(metrics["best_mAP50_95"], 5),
            "best_precision": round(metrics["best_precision"], 5),
            "best_recall": round(metrics["best_recall"], 5),
            "best_f1": round(metrics["best_f1"], 5),
            "final_precision": round(metrics["final_precision"], 5),
            "final_recall": round(metrics["final_recall"], 5),
            "final_mAP50": round(metrics["final_mAP50"], 5),
            "final_mAP50_95": round(metrics["final_mAP50_95"], 5),
            "final_f1": round(metrics["final_f1"], 5),
        },
        "training_chain": "Phase1 -> continue_b -> continue_2 -> continue_3/continue_4_Phase_1",
        "phases": phase_summaries,
    }
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Written training_summary.json")
    return filepath


def write_optimization_csvs(combined_rows, phase_boundaries):
    """Write combined optimization analysis CSVs."""
    opt_dir = OUTPUT_DIR / "stats" / "optimization"
    
    # --- Loss Breakdown ---
    filepath = opt_dir / "loss_breakdown.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "train_box_loss", "train_cls_loss", "train_dfl_loss",
                         "train_total_loss", "val_box_loss", "val_cls_loss", "val_dfl_loss",
                         "val_total_loss", "train_val_gap"])
        
        for row_info in combined_rows:
            d = row_info["data"]
            try:
                train_box = float(d[2])
                train_cls = float(d[3])
                train_dfl = float(d[4])
                val_box = float(d[9])
                val_cls = float(d[10])
                val_dfl = float(d[11])
                train_total = train_box + train_cls + train_dfl
                val_total = val_box + val_cls + val_dfl
                gap = val_total - train_total
            except (ValueError, TypeError, IndexError):
                continue
            
            writer.writerow([
                int(d[0]), row_info["phase_short"],
                f"{train_box:.5f}", f"{train_cls:.5f}", f"{train_dfl:.5f}", f"{train_total:.5f}",
                f"{val_box:.5f}", f"{val_cls:.5f}", f"{val_dfl:.5f}", f"{val_total:.5f}",
                f"{gap:.5f}"
            ])
    
    # --- LR Analysis ---
    filepath = opt_dir / "lr_analysis.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "lr_pg0", "lr_pg1", "lr_pg2", "lr_phase_config"])
        
        for row_info in combined_rows:
            d = row_info["data"]
            phase = row_info["phase_short"]
            try:
                lr0 = float(d[12])
                lr1 = float(d[13])
                lr2 = float(d[14])
            except (ValueError, TypeError, IndexError):
                continue
            
            # Find phase config lr
            phase_lr = 0
            for p in PHASES:
                if p["short"] == phase:
                    phase_lr = p["lr0"]
                    break
            
            writer.writerow([int(d[0]), phase, f"{lr0:.8f}", f"{lr1:.8f}", f"{lr2:.8f}", f"{phase_lr}"])
    
    # --- Throughput Metrics ---
    filepath = opt_dir / "throughput_metrics.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "cumulative_time_s", "epoch_time_s", "epochs_per_hour"])
        
        prev_time = {}
        for row_info in combined_rows:
            d = row_info["data"]
            phase = row_info["phase_short"]
            try:
                cum_time = float(d[1])
            except (ValueError, TypeError, IndexError):
                continue
            
            if phase not in prev_time:
                epoch_time = cum_time
            else:
                epoch_time = cum_time - prev_time[phase]
            prev_time[phase] = cum_time
            
            eph = 3600 / epoch_time if epoch_time > 0 else 0
            
            writer.writerow([int(d[0]), phase, f"{cum_time:.1f}", f"{epoch_time:.1f}", f"{eph:.2f}"])
    
    # --- Convergence Indicators ---
    filepath = opt_dir / "convergence_indicators.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "mAP50", "mAP50_change", "val_loss_total",
                         "val_loss_change", "is_improvement", "epochs_since_improvement"])
        
        best_mAP50 = 0
        epochs_since_best = 0
        prev_val_loss = None
        
        for row_info in combined_rows:
            d = row_info["data"]
            try:
                mAP50 = float(d[7])
                val_total = float(d[9]) + float(d[10]) + float(d[11])
            except (ValueError, TypeError, IndexError):
                continue
            
            mAP50_change = mAP50 - best_mAP50 if best_mAP50 > 0 else 0
            val_change = val_total - prev_val_loss if prev_val_loss is not None else 0
            is_improvement = mAP50 > best_mAP50
            
            if is_improvement:
                best_mAP50 = mAP50
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            
            prev_val_loss = val_total
            
            writer.writerow([
                int(d[0]), row_info["phase_short"],
                f"{mAP50:.5f}", f"{mAP50_change:.5f}", f"{val_total:.5f}",
                f"{val_change:.5f}", is_improvement, epochs_since_best
            ])
    
    print("  Written optimization CSVs (loss_breakdown, lr_analysis, throughput, convergence)")


def write_model_comparison(phase_boundaries, combined_rows):
    """Write model_comparison.csv showing each phase's performance."""
    filepath = OUTPUT_DIR / "stats" / "model_comparison.csv"
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "epochs", "lr0", "batch", "best_mAP50", "best_mAP50_95",
                         "final_precision", "final_recall", "final_f1", "source_weights"])
        
        for pb in phase_boundaries:
            p = pb["phase"]
            phase_data = [r for r in combined_rows if r["phase_short"] == p["short"]]
            best_m50 = best_m95 = 0
            final_p = final_r = 0
            
            for r in phase_data:
                try:
                    m50 = float(r["data"][7])
                    m95 = float(r["data"][8])
                    if m50 > best_m50:
                        best_m50 = m50
                    if m95 > best_m95:
                        best_m95 = m95
                except (ValueError, TypeError, IndexError):
                    pass
            
            if phase_data:
                try:
                    final_p = float(phase_data[-1]["data"][5])
                    final_r = float(phase_data[-1]["data"][6])
                except (ValueError, TypeError, IndexError):
                    pass
            
            final_f1 = 2 * final_p * final_r / (final_p + final_r) if (final_p + final_r) > 0 else 0
            
            writer.writerow([
                p["name"], pb["count"], p["lr0"], p["batch"],
                f"{best_m50:.5f}", f"{best_m95:.5f}",
                f"{final_p:.5f}", f"{final_r:.5f}", f"{final_f1:.5f}",
                p["source_weights"]
            ])
    
    print("  Written model_comparison.csv")


# ============================================================================
# VISUALIZATION
# ============================================================================

def extract_metric_series(combined_rows, col_idx):
    """Extract a metric series for plotting."""
    epochs = []
    values = []
    phases = []
    
    for row_info in combined_rows:
        d = row_info["data"]
        try:
            epochs.append(int(d[0]))
            values.append(float(d[col_idx]))
            phases.append(row_info["phase_short"])
        except (ValueError, TypeError, IndexError):
            continue
    
    return epochs, values, phases


def add_phase_regions(ax, phase_boundaries, y_range=None):
    """Add shaded phase regions and labels to a plot."""
    colors_alpha = {
        "Phase_1": "#2196F320",
        "continue_a": "#FF980020",
        "continue_b": "#4CAF5020",
        "continue_2": "#9C27B020",
        "continue_3": "#F4433620",
        "continue_4": "#79554820",
    }
    
    for pb in phase_boundaries:
        p = pb["phase"]
        color = colors_alpha.get(p["short"], "#00000010")
        ax.axvspan(pb["start"] - 0.5, pb["end"] + 0.5, alpha=0.08,
                   color=p["color"], label=None)
        
        # Add vertical dashed line at phase boundary
        if pb["start"] > 1:
            ax.axvline(x=pb["start"] - 0.5, color='gray', linestyle='--',
                      linewidth=0.8, alpha=0.6)
        
        # Add phase label at top
        mid = (pb["start"] + pb["end"]) / 2
        if y_range:
            y_pos = y_range[1] - (y_range[1] - y_range[0]) * 0.03
        else:
            y_pos = ax.get_ylim()[1] * 0.97
        
        # Only show label if phase has enough epochs to display
        if pb["count"] >= 15:
            ax.text(mid, y_pos, p["short"], fontsize=6, ha='center', va='top',
                    color=p["color"], fontweight='bold', alpha=0.7)


def plot_loss_curves(combined_rows, phase_boundaries):
    """Generate combined loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Combined Training Loss Curves (All Phases)", fontsize=14, fontweight='bold')
    
    loss_configs = [
        (2, "Train Box Loss", "train/box_loss"),
        (3, "Train Cls Loss", "train/cls_loss"),
        (4, "Train DFL Loss", "train/dfl_loss"),
        (9, "Val Box Loss", "val/box_loss"),
    ]
    
    for idx, (col, title, label) in enumerate(loss_configs):
        ax = axes[idx // 2][idx % 2]
        epochs, values, phases_list = extract_metric_series(combined_rows, col)
        
        if not epochs:
            continue
        
        # Plot by phase with different colors
        current_phase = phases_list[0]
        phase_epochs = []
        phase_values = []
        
        for e, v, p in zip(epochs, values, phases_list):
            if p != current_phase:
                # Plot accumulated phase data
                phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
                ax.plot(phase_epochs, phase_values, color=phase_color, linewidth=0.8, alpha=0.8)
                current_phase = p
                phase_epochs = []
                phase_values = []
            phase_epochs.append(e)
            phase_values.append(v)
        
        # Plot last phase
        if phase_epochs:
            phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
            ax.plot(phase_epochs, phase_values, color=phase_color, linewidth=0.8, alpha=0.8)
        
        add_phase_regions(ax, phase_boundaries)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Combined Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=p["color"], linewidth=2, label=p["name"])
                       for p in PHASES]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    filepath = OUTPUT_DIR / "visualizations" / "curves" / "loss_curves.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Generated: loss_curves.png")


def plot_val_loss_curves(combined_rows, phase_boundaries):
    """Generate combined validation loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Combined Validation Loss Curves (All Phases)", fontsize=14, fontweight='bold')
    
    loss_configs = [
        (9, "Val Box Loss"),
        (10, "Val Cls Loss"),
        (11, "Val DFL Loss"),
    ]
    
    for idx, (col, title) in enumerate(loss_configs):
        ax = axes[idx]
        epochs, values, phases_list = extract_metric_series(combined_rows, col)
        
        if not epochs:
            continue
        
        current_phase = phases_list[0]
        phase_epochs = []
        phase_values = []
        
        for e, v, p in zip(epochs, values, phases_list):
            if p != current_phase:
                phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
                ax.plot(phase_epochs, phase_values, color=phase_color, linewidth=0.8, alpha=0.8)
                current_phase = p
                phase_epochs = []
                phase_values = []
            phase_epochs.append(e)
            phase_values.append(v)
        
        if phase_epochs:
            phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
            ax.plot(phase_epochs, phase_values, color=phase_color, linewidth=0.8, alpha=0.8)
        
        add_phase_regions(ax, phase_boundaries)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Combined Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=p["color"], linewidth=2, label=p["name"])
                       for p in PHASES]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    filepath = OUTPUT_DIR / "visualizations" / "curves" / "val_loss_curves.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Generated: val_loss_curves.png")


def plot_metrics_curves(combined_rows, phase_boundaries):
    """Generate combined detection metrics curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Combined Detection Metrics (All Phases)", fontsize=14, fontweight='bold')
    
    metric_configs = [
        (5, "Precision"),
        (6, "Recall"),
        (7, "mAP@50"),
        (8, "mAP@50-95"),
    ]
    
    for idx, (col, title) in enumerate(metric_configs):
        ax = axes[idx // 2][idx % 2]
        epochs, values, phases_list = extract_metric_series(combined_rows, col)
        
        if not epochs:
            continue
        
        # Plot by phase
        current_phase = phases_list[0]
        phase_epochs = []
        phase_values = []
        
        for e, v, p in zip(epochs, values, phases_list):
            if p != current_phase:
                phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
                ax.plot(phase_epochs, phase_values, color=phase_color, linewidth=0.8, alpha=0.8)
                current_phase = p
                phase_epochs = []
                phase_values = []
            phase_epochs.append(e)
            phase_values.append(v)
        
        if phase_epochs:
            phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
            ax.plot(phase_epochs, phase_values, color=phase_color, linewidth=0.8, alpha=0.8)
        
        add_phase_regions(ax, phase_boundaries)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Combined Epoch")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for best value
        if values:
            best_val = max(values)
            best_epoch = epochs[values.index(best_val)]
            ax.axhline(y=best_val, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
            ax.annotate(f"Best: {best_val:.4f} (ep {best_epoch})",
                       xy=(best_epoch, best_val), fontsize=7,
                       xytext=(10, 5), textcoords='offset points',
                       color='red', alpha=0.7)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=p["color"], linewidth=2, label=p["name"])
                       for p in PHASES]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    filepath = OUTPUT_DIR / "visualizations" / "curves" / "precision_recall.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Generated: precision_recall.png")


def plot_lr_schedule(combined_rows, phase_boundaries):
    """Generate combined learning rate schedule."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle("Combined Learning Rate Schedule (All Phases)", fontsize=14, fontweight='bold')
    
    epochs, values, phases_list = extract_metric_series(combined_rows, 12)  # lr/pg0
    
    if not epochs:
        plt.close()
        return
    
    current_phase = phases_list[0]
    phase_epochs = []
    phase_values = []
    
    for e, v, p in zip(epochs, values, phases_list):
        if p != current_phase:
            phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
            ax.plot(phase_epochs, phase_values, color=phase_color, linewidth=1.2, alpha=0.9)
            current_phase = p
            phase_epochs = []
            phase_values = []
        phase_epochs.append(e)
        phase_values.append(v)
    
    if phase_epochs:
        phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
        ax.plot(phase_epochs, phase_values, color=phase_color, linewidth=1.2, alpha=0.9)
    
    add_phase_regions(ax, phase_boundaries)
    ax.set_xlabel("Combined Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate (pg0)", fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=p["color"], linewidth=2, label=f"{p['name']} (lr0={p['lr0']})")
                       for p in PHASES]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "visualizations" / "curves" / "lr_schedule.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Generated: lr_schedule.png")


def plot_f1_curve(combined_rows, phase_boundaries):
    """Generate combined F1 score curve."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle("Combined F1 Score (All Phases)", fontsize=14, fontweight='bold')
    
    epochs = []
    f1_values = []
    phases_list = []
    
    for row_info in combined_rows:
        d = row_info["data"]
        try:
            epoch = int(d[0])
            precision = float(d[5])
            recall = float(d[6])
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            epochs.append(epoch)
            f1_values.append(f1)
            phases_list.append(row_info["phase_short"])
        except (ValueError, TypeError, IndexError):
            continue
    
    if not epochs:
        plt.close()
        return
    
    current_phase = phases_list[0]
    phase_epochs = []
    phase_f1 = []
    
    for e, f, p in zip(epochs, f1_values, phases_list):
        if p != current_phase:
            phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
            ax.plot(phase_epochs, phase_f1, color=phase_color, linewidth=0.8, alpha=0.8)
            current_phase = p
            phase_epochs = []
            phase_f1 = []
        phase_epochs.append(e)
        phase_f1.append(f)
    
    if phase_epochs:
        phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
        ax.plot(phase_epochs, phase_f1, color=phase_color, linewidth=0.8, alpha=0.8)
    
    # Best F1
    best_f1 = max(f1_values)
    best_ep = epochs[f1_values.index(best_f1)]
    ax.axhline(y=best_f1, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.annotate(f"Best F1: {best_f1:.4f} (epoch {best_ep})",
               xy=(best_ep, best_f1), fontsize=8,
               xytext=(10, 10), textcoords='offset points',
               color='red', fontweight='bold')
    
    add_phase_regions(ax, phase_boundaries)
    ax.set_xlabel("Combined Epoch", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=p["color"], linewidth=2, label=p["name"])
                       for p in PHASES]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "visualizations" / "curves" / "f1_curve.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Generated: f1_curve.png")


def plot_results_overview(combined_rows, phase_boundaries, metrics):
    """Generate comprehensive results overview plot."""
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Combined Training Results Overview\nYOLOv8x MicroSpore Detection - All Phases",
                 fontsize=16, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # 1. mAP@50 curve
    ax1 = fig.add_subplot(gs[0, 0])
    epochs, mAP50_vals, phases_list = extract_metric_series(combined_rows, 7)
    _plot_phased_line(ax1, epochs, mAP50_vals, phases_list, phase_boundaries)
    ax1.set_title("mAP@50", fontsize=12, fontweight='bold')
    ax1.set_ylabel("mAP@50")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    if mAP50_vals:
        best_val = max(mAP50_vals)
        ax1.axhline(y=best_val, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # 2. mAP@50-95 curve
    ax2 = fig.add_subplot(gs[0, 1])
    epochs, mAP95_vals, phases_list = extract_metric_series(combined_rows, 8)
    _plot_phased_line(ax2, epochs, mAP95_vals, phases_list, phase_boundaries)
    ax2.set_title("mAP@50-95", fontsize=12, fontweight='bold')
    ax2.set_ylabel("mAP@50-95")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision & Recall
    ax3 = fig.add_subplot(gs[1, 0])
    epochs_p, prec_vals, phases_p = extract_metric_series(combined_rows, 5)
    epochs_r, rec_vals, phases_r = extract_metric_series(combined_rows, 6)
    _plot_phased_line(ax3, epochs_p, prec_vals, phases_p, phase_boundaries, alpha=0.5)
    _plot_phased_line(ax3, epochs_r, rec_vals, phases_r, phase_boundaries, alpha=0.5, linestyle='--')
    ax3.set_title("Precision & Recall", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Value")
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 Score
    ax4 = fig.add_subplot(gs[1, 1])
    f1_epochs = []
    f1_vals = []
    f1_phases = []
    for row_info in combined_rows:
        d = row_info["data"]
        try:
            p = float(d[5])
            r = float(d[6])
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_epochs.append(int(d[0]))
            f1_vals.append(f1)
            f1_phases.append(row_info["phase_short"])
        except (ValueError, TypeError, IndexError):
            continue
    _plot_phased_line(ax4, f1_epochs, f1_vals, f1_phases, phase_boundaries)
    ax4.set_title("F1 Score", fontsize=12, fontweight='bold')
    ax4.set_ylabel("F1")
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # 5. Training Loss (all components)
    ax5 = fig.add_subplot(gs[2, 0])
    for col, label, ls in [(2, "box", '-'), (3, "cls", '--'), (4, "dfl", ':')]:
        epochs, vals, ph = extract_metric_series(combined_rows, col)
        _plot_phased_line(ax5, epochs, vals, ph, phase_boundaries, alpha=0.6, linewidth=0.7)
    ax5.set_title("Training Losses", fontsize=12, fontweight='bold')
    ax5.set_ylabel("Loss")
    ax5.grid(True, alpha=0.3)
    
    # 6. Validation Loss (all components)
    ax6 = fig.add_subplot(gs[2, 1])
    for col, label, ls in [(9, "box", '-'), (10, "cls", '--'), (11, "dfl", ':')]:
        epochs, vals, ph = extract_metric_series(combined_rows, col)
        _plot_phased_line(ax6, epochs, vals, ph, phase_boundaries, alpha=0.6, linewidth=0.7)
    ax6.set_title("Validation Losses", fontsize=12, fontweight='bold')
    ax6.set_ylabel("Loss")
    ax6.grid(True, alpha=0.3)
    
    # 7. Learning Rate
    ax7 = fig.add_subplot(gs[3, 0])
    epochs, lr_vals, phases_list = extract_metric_series(combined_rows, 12)
    _plot_phased_line(ax7, epochs, lr_vals, phases_list, phase_boundaries)
    ax7.set_title("Learning Rate Schedule", fontsize=12, fontweight='bold')
    ax7.set_ylabel("LR (pg0)")
    ax7.set_xlabel("Combined Epoch")
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3)
    
    # 8. Phase comparison bar chart
    ax8 = fig.add_subplot(gs[3, 1])
    phase_names_short = [pb["phase"]["short"] for pb in phase_boundaries]
    phase_best_mAP50s = []
    for pb in phase_boundaries:
        pdata = [r for r in combined_rows if r["phase_short"] == pb["phase"]["short"]]
        best = max((float(r["data"][7]) for r in pdata), default=0)
        phase_best_mAP50s.append(best)
    
    colors = [pb["phase"]["color"] for pb in phase_boundaries]
    bars = ax8.bar(phase_names_short, phase_best_mAP50s, color=colors, alpha=0.8, edgecolor='white')
    ax8.set_title("Best mAP@50 by Phase", fontsize=12, fontweight='bold')
    ax8.set_ylabel("mAP@50")
    ax8.set_ylim(0, 1)
    ax8.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, phase_best_mAP50s):
        ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=p["color"], linewidth=3, label=p["name"])
                       for p in PHASES]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    
    filepath = OUTPUT_DIR / "visualizations" / "overviews" / "results_overview.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Generated: results_overview.png")


def _plot_phased_line(ax, epochs, values, phases_list, phase_boundaries,
                      alpha=0.8, linewidth=0.8, linestyle='-'):
    """Helper to plot a line colored by phase."""
    if not epochs:
        return
    
    current_phase = phases_list[0]
    phase_epochs = []
    phase_values = []
    
    for e, v, p in zip(epochs, values, phases_list):
        if p != current_phase:
            phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
            ax.plot(phase_epochs, phase_values, color=phase_color,
                   linewidth=linewidth, alpha=alpha, linestyle=linestyle)
            current_phase = p
            phase_epochs = []
            phase_values = []
        phase_epochs.append(e)
        phase_values.append(v)
    
    if phase_epochs:
        phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
        ax.plot(phase_epochs, phase_values, color=phase_color,
               linewidth=linewidth, alpha=alpha, linestyle=linestyle)
    
    add_phase_regions(ax, phase_boundaries)


def plot_train_vs_val_loss(combined_rows, phase_boundaries):
    """Generate train vs validation total loss comparison."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle("Training vs Validation Total Loss (All Phases)", fontsize=14, fontweight='bold')
    
    epochs_list = []
    train_totals = []
    val_totals = []
    phases_list = []
    
    for row_info in combined_rows:
        d = row_info["data"]
        try:
            epoch = int(d[0])
            train_total = float(d[2]) + float(d[3]) + float(d[4])
            val_total = float(d[9]) + float(d[10]) + float(d[11])
            epochs_list.append(epoch)
            train_totals.append(train_total)
            val_totals.append(val_total)
            phases_list.append(row_info["phase_short"])
        except (ValueError, TypeError, IndexError):
            continue
    
    _plot_phased_line(ax, epochs_list, train_totals, phases_list, phase_boundaries, alpha=0.6)
    
    # Plot val loss with lighter version
    current_phase = phases_list[0]
    pe = []
    pv = []
    for e, v, p in zip(epochs_list, val_totals, phases_list):
        if p != current_phase:
            phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
            ax.plot(pe, pv, color=phase_color, linewidth=0.8, alpha=0.4, linestyle='--')
            current_phase = p
            pe = []
            pv = []
        pe.append(e)
        pv.append(v)
    if pe:
        phase_color = next((ph["color"] for ph in PHASES if ph["short"] == current_phase), "black")
        ax.plot(pe, pv, color=phase_color, linewidth=0.8, alpha=0.4, linestyle='--')
    
    add_phase_regions(ax, phase_boundaries)
    
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], color='gray', linewidth=2),
               Line2D([0], [0], color='gray', linewidth=2, linestyle='--')],
              ['Training Loss', 'Validation Loss'], fontsize=9)
    
    ax.set_xlabel("Combined Epoch", fontsize=11)
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "visualizations" / "curves" / "train_vs_val_loss.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Generated: train_vs_val_loss.png")


def plot_phase_comparison(phase_boundaries, combined_rows):
    """Generate phase comparison summary chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Phase Comparison Summary", fontsize=14, fontweight='bold')
    
    phase_names = [pb["phase"]["short"] for pb in phase_boundaries]
    colors = [pb["phase"]["color"] for pb in phase_boundaries]
    
    # Compute per-phase metrics
    best_mAP50s = []
    best_mAP95s = []
    best_f1s = []
    
    for pb in phase_boundaries:
        pdata = [r for r in combined_rows if r["phase_short"] == pb["phase"]["short"]]
        best_m50 = 0
        best_m95 = 0
        best_f1 = 0
        for r in pdata:
            try:
                m50 = float(r["data"][7])
                m95 = float(r["data"][8])
                p_val = float(r["data"][5])
                r_val = float(r["data"][6])
                f1 = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0
                if m50 > best_m50:
                    best_m50 = m50
                if m95 > best_m95:
                    best_m95 = m95
                if f1 > best_f1:
                    best_f1 = f1
            except (ValueError, TypeError, IndexError):
                pass
        best_mAP50s.append(best_m50)
        best_mAP95s.append(best_m95)
        best_f1s.append(best_f1)
    
    # Bar chart 1: mAP@50
    bars1 = axes[0].bar(phase_names, best_mAP50s, color=colors, alpha=0.8, edgecolor='white')
    axes[0].set_title("Best mAP@50", fontsize=11)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, best_mAP50s):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)
    
    # Bar chart 2: mAP@50-95
    bars2 = axes[1].bar(phase_names, best_mAP95s, color=colors, alpha=0.8, edgecolor='white')
    axes[1].set_title("Best mAP@50-95", fontsize=11)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, best_mAP95s):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=30)
    
    # Bar chart 3: F1
    bars3 = axes[2].bar(phase_names, best_f1s, color=colors, alpha=0.8, edgecolor='white')
    axes[2].set_title("Best F1 Score", fontsize=11)
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, best_f1s):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "visualizations" / "overviews" / "phase_comparison.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Generated: phase_comparison.png")


def write_guide_files():
    """Write interpretation guide text files."""
    
    guides = {
        "visualizations/curves/loss_curves_guide_and_interpretation.txt": """COMBINED LOSS CURVES - Guide and Interpretation
================================================

This plot shows training and validation loss components across ALL training phases.

Each phase is shown in a different color with vertical dashed lines at phase boundaries:
- Blue: Phase 1 (Original, 500 epochs, lr=0.001)
- Orange: continue_a (102 epochs, lr=0.0001) 
- Green: continue_b (500 epochs, lr=5e-05)
- Purple: continue_2 (31 epochs, lr=1e-05)
- Red: continue_3 (34 epochs, lr=5e-05)
- Brown: continue_4_Phase_1 (41 epochs, lr=1e-05)

What to look for:
- Loss should generally decrease over time within each phase
- Phase transitions may show sudden changes due to different LR/settings
- Spikes at phase boundaries are normal when hyperparameters change
- Divergence between train and val loss indicates overfitting
""",
        "visualizations/curves/precision_recall_guide_and_interpretation.txt": """COMBINED PRECISION/RECALL & mAP CURVES - Guide and Interpretation
=================================================================

These plots show detection quality metrics across all training phases.

Metrics explained:
- Precision: Of all detections, what fraction are correct
- Recall: Of all ground truth objects, what fraction are detected
- mAP@50: Mean Average Precision at IoU=0.50 threshold
- mAP@50-95: Mean AP averaged over IoU thresholds 0.50 to 0.95

Red dashed lines show the best value achieved across all phases.
Higher is better for all metrics. The best mAP@50 indicates the peak detection performance.
""",
        "visualizations/curves/f1_curve_guide_and_interpretation.txt": """COMBINED F1 SCORE CURVE - Guide and Interpretation
===================================================

F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

F1 balances precision and recall into a single metric.
- F1 > 0.70: Good balance between precision and recall
- F1 > 0.65: Acceptable performance
- F1 < 0.60: May need improvement

The red annotation shows the best F1 achieved and at which epoch.
""",
        "visualizations/curves/lr_schedule_guide_and_interpretation.txt": """COMBINED LEARNING RATE SCHEDULE - Guide and Interpretation
============================================================

This plot shows how the learning rate changed across all training phases.

Key observations:
- Phase 1 starts with warmup then peaks at lr=0.001
- Subsequent phases use progressively lower learning rates for fine-tuning
- Each phase has its own warmup period
- The LR schedule determines how aggressively the model updates weights
""",
        "visualizations/overviews/results_guide_and_interpretation.txt": """COMBINED RESULTS OVERVIEW - Guide and Interpretation
=====================================================

This comprehensive overview shows all metrics across all training phases.

The overview includes:
1. mAP@50 curve - primary detection metric
2. mAP@50-95 curve - strict localization metric  
3. Precision & Recall curves (solid/dashed)
4. F1 Score curve
5. Training losses (box, cls, dfl components)
6. Validation losses
7. Learning rate schedule (log scale)
8. Phase comparison bar chart

Phase colors are consistent across all subplots.
""",
    }
    
    for path, content in guides.items():
        filepath = OUTPUT_DIR / path
        with open(filepath, 'w') as f:
            f.write(content)
    
    print("  Written guide/interpretation text files")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("COMBINED TRAINING RESULTS GENERATOR")
    print("=" * 70)
    print()
    
    # 1. Create output directory structure
    print("[1/7] Creating output directory structure...")
    create_output_dirs()
    print()
    
    # 2. Load all phase data
    print("[2/7] Loading epoch metrics from all phases...")
    all_data = load_all_phases()
    print(f"  Total phases loaded: {len(all_data)}")
    print()
    
    # 3. Combine phases
    print("[3/7] Combining all phases into sequential timeline...")
    combined_rows, phase_boundaries = combine_phases(all_data)
    print(f"  Total combined epochs: {len(combined_rows)}")
    print()
    
    # 4. Compute overall metrics
    print("[4/7] Computing overall best metrics...")
    metrics = find_best_metrics(combined_rows)
    print(f"  Best mAP@50: {metrics['best_mAP50']:.4f} at epoch {metrics['best_mAP50_epoch']} ({metrics['best_mAP50_phase']})")
    print(f"  Best mAP@50-95: {metrics['best_mAP50_95']:.4f}")
    print(f"  Best F1: {metrics['best_f1']:.4f}")
    print()
    
    # 5. Write combined data files
    print("[5/7] Writing combined data files...")
    write_combined_epoch_metrics(combined_rows, phase_boundaries)
    write_general_results(combined_rows, phase_boundaries, metrics)
    write_training_summary_json(combined_rows, phase_boundaries, metrics)
    write_optimization_csvs(combined_rows, phase_boundaries)
    write_model_comparison(phase_boundaries, combined_rows)
    write_guide_files()
    print()
    
    # 6. Generate visualizations
    if HAS_MATPLOTLIB:
        print("[6/7] Generating combined visualizations...")
        plot_loss_curves(combined_rows, phase_boundaries)
        plot_val_loss_curves(combined_rows, phase_boundaries)
        plot_metrics_curves(combined_rows, phase_boundaries)
        plot_lr_schedule(combined_rows, phase_boundaries)
        plot_f1_curve(combined_rows, phase_boundaries)
        plot_train_vs_val_loss(combined_rows, phase_boundaries)
        plot_results_overview(combined_rows, phase_boundaries, metrics)
        plot_phase_comparison(phase_boundaries, combined_rows)
    else:
        print("[6/7] SKIPPED - matplotlib not available")
    print()
    
    # 7. Summary
    print("[7/7] COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total combined epochs: {len(combined_rows)}")
    print(f"Total phases: {len(phase_boundaries)}")
    print()
    print("Phase breakdown:")
    for pb in phase_boundaries:
        p = pb["phase"]
        print(f"  {p['name']:30s}  epochs {pb['start']:>5d}-{pb['end']:>5d}  ({pb['count']:>4d} epochs)  lr={p['lr0']}")
    print()
    print(f"Overall best mAP@50: {metrics['best_mAP50']:.4f} at epoch {metrics['best_mAP50_epoch']}")
    print(f"Overall best mAP@50-95: {metrics['best_mAP50_95']:.4f}")
    print(f"Overall best F1: {metrics['best_f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
