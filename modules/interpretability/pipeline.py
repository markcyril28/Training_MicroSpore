#!/usr/bin/env python3
"""Interpretability pipeline for trained YOLO microspore models.

The pipeline generates model-facing explanations for validation/test images:

1. Detection-confidence heatmaps: fast, model-output-based overlays showing
   which predicted boxes/classes contributed the strongest confidence.
2. Optional occlusion sensitivity: slower, model-agnostic perturbation maps that
   re-run inference after masking image tiles and measure confidence drops.

The implementation uses only Python/Pillow/NumPy for artifact generation and
imports Ultralytics lazily only when `run_interpretability_pipeline()` is called.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class InterpretabilityArtifacts:
    """Paths generated for a single interpreted image."""

    image_path: Path
    overlay_path: Path
    heatmap_path: Path
    metadata_path: Path
    method: str = "detection_confidence"

    def as_relative_row(self, output_dir: Path, detections: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Return a CSV/report-friendly row with paths relative to output_dir."""
        top_detection = "none"
        if detections:
            top = max(detections, key=lambda det: float(det.get("confidence", 0.0)))
            top_detection = f"{top.get('class_name', 'unknown')} {float(top.get('confidence', 0.0)):.2f}"
        return {
            "image": str(self.image_path),
            "method": self.method,
            "overlay": _safe_relative(self.overlay_path, output_dir),
            "heatmap": _safe_relative(self.heatmap_path, output_dir),
            "metadata": _safe_relative(self.metadata_path, output_dir),
            "num_detections": len(detections),
            "top_detection": top_detection,
        }


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return loaded


def load_class_names(data_yaml: Optional[Union[str, Path]] = None, names: Optional[Any] = None) -> List[str]:
    """Load class names from YOLO data.yaml or an explicit names object.

    YOLO accepts either:
    - names: [class_a, class_b]
    - names: {0: class_a, 1: class_b}

    Dict keys are sorted numerically when possible so YAML order does not affect
    class id alignment.
    """
    if names is None:
        if data_yaml is None:
            return []
        names = _load_yaml(Path(data_yaml)).get("names", [])

    if names is None:
        return []

    if isinstance(names, Mapping):
        def sort_key(item: Tuple[Any, Any]) -> Tuple[int, Union[int, str]]:
            key, _value = item
            try:
                return (0, int(key))
            except (TypeError, ValueError):
                return (1, str(key))

        return [str(value) for _key, value in sorted(names.items(), key=sort_key)]

    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]

    raise ValueError(f"Unsupported YOLO class names format: {type(names).__name__}")


def _resolve_path_from_data_yaml(data_yaml: Path, split_value: Union[str, Sequence[str]]) -> List[Path]:
    data = _load_yaml(data_yaml)
    base = data_yaml.parent
    yaml_root = data.get("path")
    if yaml_root:
        yaml_root_path = Path(str(yaml_root))
        if not yaml_root_path.is_absolute():
            yaml_root_path = data_yaml.parent / yaml_root_path
        base = yaml_root_path

    raw_paths: Sequence[str]
    if isinstance(split_value, (list, tuple)):
        raw_paths = [str(item) for item in split_value]
    else:
        raw_paths = [str(split_value)]

    resolved: List[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = base / path
        resolved.append(path)
    return resolved


def _image_files_from_path(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_file() and path.suffix.lower() == ".txt":
        images: List[Path] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            image_path = Path(raw)
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            images.extend(_image_files_from_path(image_path))
        return sorted(images, key=lambda p: str(p).lower())
    if not path.exists() or not path.is_dir():
        return []
    return sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def collect_image_paths(
    data_yaml: Optional[Union[str, Path]] = None,
    images_dir: Optional[Union[str, Path]] = None,
    images: Optional[Sequence[Union[str, Path]]] = None,
    split: str = "val",
    limit: Optional[int] = None,
) -> List[Path]:
    """Collect images from explicit paths, an image directory, or a YOLO data.yaml.

    Priority is explicit images, then images_dir, then the requested split from
    data_yaml. Returned paths are sorted for reproducibility and optionally
    truncated by limit.
    """
    collected: List[Path] = []

    if images:
        for image in images:
            path = Path(image)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                collected.append(path)
            elif path.is_dir():
                collected.extend(_image_files_from_path(path))
    elif images_dir:
        collected.extend(_image_files_from_path(Path(images_dir)))
    elif data_yaml:
        data_yaml_path = Path(data_yaml)
        data = _load_yaml(data_yaml_path)
        split_value = data.get(split)
        if split_value is None and split == "val":
            split_value = data.get("test")
        if split_value is None:
            raise ValueError(f"Split '{split}' not found in {data_yaml_path}")
        for source in _resolve_path_from_data_yaml(data_yaml_path, split_value):
            collected.extend(_image_files_from_path(source))
    else:
        raise ValueError("Provide at least one image source: data_yaml, images_dir, or images")

    # De-duplicate while preserving deterministic ordering.
    unique = sorted({p.resolve() if p.exists() else p for p in collected}, key=lambda p: str(p).lower())
    if limit is not None and limit > 0:
        unique = unique[:limit]
    return unique


def _coerce_image_shape(image_shape: Union[Tuple[int, int], Tuple[int, int, int], Sequence[int]]) -> Tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least height and width")
    height = int(image_shape[0])
    width = int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape height and width must be positive")
    return height, width


def _get_detection_box(detection: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    box = detection.get("box_xyxy") or detection.get("xyxy") or detection.get("bbox")
    if box is None or len(box) < 4:
        return None
    return float(box[0]), float(box[1]), float(box[2]), float(box[3])


def build_detection_heatmap(
    image_shape: Union[Tuple[int, int], Tuple[int, int, int], Sequence[int]],
    detections: Sequence[Mapping[str, Any]],
    target_class: Optional[Union[str, int]] = None,
) -> np.ndarray:
    """Build a normalized heatmap from YOLO detections.

    Each predicted bounding box contributes its confidence score to the covered
    pixels. This creates a fast "where did the detector place confident evidence?"
    map without requiring gradients or model internals.
    """
    height, width = _coerce_image_shape(image_shape)
    heatmap = np.zeros((height, width), dtype=np.float32)

    for detection in detections:
        if target_class is not None:
            class_id = detection.get("class_id")
            class_name = detection.get("class_name")
            if str(target_class) not in {str(class_id), str(class_name)}:
                continue

        box = _get_detection_box(detection)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        left = max(0, min(width - 1, int(math.floor(min(x1, x2)))))
        right = max(0, min(width, int(math.ceil(max(x1, x2)))))
        top = max(0, min(height - 1, int(math.floor(min(y1, y2)))))
        bottom = max(0, min(height, int(math.ceil(max(y1, y2)))))
        if right <= left or bottom <= top:
            continue
        confidence = max(0.0, float(detection.get("confidence", detection.get("conf", 1.0))))
        heatmap[top:bottom, left:right] += confidence

    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap /= max_value
    return np.clip(heatmap, 0.0, 1.0)


def _resize_heatmap(heatmap: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a heatmap to (width, height) using Pillow bilinear sampling."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    image = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _make_red_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.55) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    heat = _resize_heatmap(heatmap, image.size)[..., None]
    color = np.zeros_like(base)
    color[..., 0] = 255.0
    blended = base * (1.0 - alpha * heat) + color * (alpha * heat)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _draw_detection_boxes(image: Image.Image, detections: Sequence[Mapping[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for detection in detections:
        box = _get_detection_box(detection)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        class_name = str(detection.get("class_name", detection.get("class_id", "class")))
        confidence = float(detection.get("confidence", detection.get("conf", 0.0)))
        label = f"{class_name} {confidence:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0))
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    return image


def _json_ready_detection(detection: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in detection.items():
        if isinstance(value, np.generic):
            cleaned[key] = value.item()
        elif isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        elif isinstance(value, Path):
            cleaned[key] = str(value)
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [item.item() if isinstance(item, np.generic) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


def save_heatmap_artifacts(
    image_path: Union[str, Path],
    heatmap: np.ndarray,
    detections: Sequence[Mapping[str, Any]],
    output_dir: Union[str, Path],
    prefix: Optional[str] = None,
    method: str = "detection_confidence",
) -> InterpretabilityArtifacts:
    """Save heatmap PNG, overlay PNG, and metadata JSON for one image."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    prefix = prefix or image_path.stem

    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    metadata_dir = output_dir / "metadata"
    for folder in (heatmap_dir, overlay_dir, metadata_dir):
        folder.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    resized_heatmap = _resize_heatmap(heatmap, image.size)

    heatmap_path = heatmap_dir / f"{prefix}_{method}_heatmap.png"
    overlay_path = overlay_dir / f"{prefix}_{method}_overlay.png"
    metadata_path = metadata_dir / f"{prefix}_{method}.json"

    heatmap_image = Image.fromarray((resized_heatmap * 255).astype(np.uint8), mode="L")
    heatmap_image.save(heatmap_path)

    overlay = _make_red_overlay(image, resized_heatmap)
    overlay = _draw_detection_boxes(overlay, detections)
    overlay.save(overlay_path)

    cleaned_detections = [_json_ready_detection(det) for det in detections]
    metadata = {
        "image": str(image_path),
        "method": method,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_detections": len(cleaned_detections),
        "detections": cleaned_detections,
        "overlay": str(overlay_path),
        "heatmap": str(heatmap_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return InterpretabilityArtifacts(
        image_path=image_path,
        overlay_path=overlay_path,
        heatmap_path=heatmap_path,
        metadata_path=metadata_path,
        method=method,
    )


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert torch/Ultralytics tensors or arrays to numpy without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def extract_detections_from_result(result: Any, class_names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Extract serializable detections from an Ultralytics Results object."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = _tensor_to_numpy(getattr(boxes, "xyxy", []))
    conf = _tensor_to_numpy(getattr(boxes, "conf", np.zeros(len(xyxy))))
    cls = _tensor_to_numpy(getattr(boxes, "cls", np.zeros(len(xyxy))))

    names_obj = getattr(result, "names", None)
    names: Sequence[str]
    if class_names:
        names = list(class_names)
    elif isinstance(names_obj, Mapping):
        names = load_class_names(names=names_obj)
    elif isinstance(names_obj, (list, tuple)):
        names = [str(item) for item in names_obj]
    else:
        names = []

    detections: List[Dict[str, Any]] = []
    for idx, box in enumerate(xyxy):
        class_id = int(cls[idx]) if idx < len(cls) else -1
        class_name = names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(conf[idx]) if idx < len(conf) else 0.0,
                "box_xyxy": [float(coord) for coord in box[:4]],
            }
        )
    return detections


def _prediction_score(result: Any, target_class: Optional[Union[str, int]] = None) -> float:
    detections = extract_detections_from_result(result)
    scores = []
    for detection in detections:
        if target_class is not None:
            if str(target_class) not in {str(detection.get("class_id")), str(detection.get("class_name"))}:
                continue
        scores.append(float(detection.get("confidence", 0.0)))
    return max(scores) if scores else 0.0


def build_occlusion_heatmap(
    model: Any,
    image_path: Union[str, Path],
    baseline_result: Any,
    grid_size: int = 8,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: Optional[int] = None,
    device: Optional[Union[str, int]] = None,
    target_class: Optional[Union[str, int]] = None,
) -> np.ndarray:
    """Build a model-agnostic occlusion sensitivity heatmap.

    This is intentionally optional because it re-runs inference once per grid
    tile. The heatmap is high where masking that tile decreases the target score.
    """
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    baseline_score = _prediction_score(baseline_result, target_class=target_class)
    sensitivity = np.zeros((height, width), dtype=np.float32)

    if baseline_score <= 0:
        return sensitivity

    tile_h = max(1, math.ceil(height / grid_size))
    tile_w = max(1, math.ceil(width / grid_size))
    fill = np.median(array.reshape(-1, 3), axis=0).astype(np.uint8)

    predict_kwargs: Dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz
    if device is not None:
        predict_kwargs["device"] = device

    for top in range(0, height, tile_h):
        bottom = min(height, top + tile_h)
        for left in range(0, width, tile_w):
            right = min(width, left + tile_w)
            occluded = array.copy()
            occluded[top:bottom, left:right, :] = fill
            result = model.predict(source=occluded, **predict_kwargs)[0]
            occluded_score = _prediction_score(result, target_class=target_class)
            drop = max(0.0, baseline_score - occluded_score) / baseline_score
            sensitivity[top:bottom, left:right] = drop

    max_value = float(sensitivity.max())
    if max_value > 0:
        sensitivity /= max_value
    return np.clip(sensitivity, 0.0, 1.0)


def render_markdown_report(
    output_dir: Union[str, Path],
    model_path: Union[str, Path],
    data_yaml: Optional[Union[str, Path]],
    class_names: Sequence[str],
    artifact_rows: Sequence[Mapping[str, Any]],
    occlusion_enabled: bool = False,
) -> Path:
    """Render an interpretability report summarizing generated artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "INTERPRETABILITY_REPORT.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    class_list = ", ".join(class_names) if class_names else "Not available"
    rows_markdown = []
    for row in artifact_rows:
        rows_markdown.append(
            "| {image} | {method} | {num_detections} | {top_detection} | [{overlay}]({overlay}) | [{heatmap}]({heatmap}) |".format(
                image=row.get("image", ""),
                method=row.get("method", "detection_confidence"),
                num_detections=row.get("num_detections", 0),
                top_detection=row.get("top_detection", "none"),
                overlay=row.get("overlay", ""),
                heatmap=row.get("heatmap", ""),
            )
        )
    if not rows_markdown:
        rows_markdown.append("| No images processed | - | - | - | - | - |")

    occlusion_section = """
## Occlusion sensitivity

Enabled for this run. Each occlusion heatmap masks tiles of the input image and
measures how much the top/target detection confidence drops. Red regions are
more causally important for the model's current prediction, but this method is
slower because it performs repeated inference.
""" if occlusion_enabled else """
## Occlusion sensitivity

Not enabled for this run. Re-run with `--occlusion` when you need slower but
more causal model-agnostic explanations for selected images.
"""

    content = f"""# Interpretability Report

**Generated:** {timestamp}
**Model:** `{model_path}`
**Data YAML:** `{data_yaml if data_yaml else 'N/A'}`
**Classes:** {class_list}

---

## Detection-confidence heatmaps

These heatmaps are generated from YOLO detections by filling each predicted
bounding box with its confidence score and normalizing the image to 0-1. Redder
regions are where the detector placed higher-confidence predictions. Use these
for fast triage of whether the model is focusing on microspore/pollen objects or
background artifacts.

Important limitation: detection-confidence heatmaps explain the final predicted
boxes, not the internal neural-network feature maps. For a stronger perturbation
check, enable occlusion sensitivity.

{occlusion_section}

---

## Generated artifacts

| Image | Method | Detections | Top detection | Overlay | Heatmap |
|---|---:|---:|---|---|---|
{chr(10).join(rows_markdown)}

---

## How to read this output

- High heat on the object body/edges: the detector is probably using relevant
  visual evidence.
- High heat on labels, borders, background bubbles, or microscope artifacts:
  inspect the training set for shortcut learning or annotation bias.
- Correct boxes with low confidence: consider more examples of that class or
  lower-confidence review thresholds.
- Repeated heat on adjacent developmental stages: compare these images with the
  confusion matrix and hard-example galleries.
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def _write_summary_csv(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    csv_path = output_dir / "interpretability_summary.csv"
    fieldnames = ["image", "method", "num_detections", "top_detection", "overlay", "heatmap", "metadata"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return csv_path


def run_interpretability_pipeline(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    data_yaml: Optional[Union[str, Path]] = None,
    images_dir: Optional[Union[str, Path]] = None,
    images: Optional[Sequence[Union[str, Path]]] = None,
    split: str = "val",
    limit: int = 12,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: Optional[int] = None,
    device: Optional[Union[str, int]] = None,
    occlusion: bool = False,
    grid_size: int = 8,
    target_class: Optional[Union[str, int]] = None,
) -> Dict[str, Path]:
    """Run YOLO interpretability on selected images and write artifacts."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required to run model inference. Activate the training "
            "environment or run setup_conda_training_server.sh/local setup first."
        ) from exc

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(
        data_yaml=data_yaml,
        images_dir=images_dir,
        images=images,
        split=split,
        limit=limit,
    )
    if not image_paths:
        raise ValueError("No images found for interpretability. Check --data-yaml/--images-dir/--image.")

    class_names = load_class_names(data_yaml) if data_yaml else []
    model = YOLO(str(model_path))
    if not class_names and hasattr(model, "names"):
        class_names = load_class_names(names=getattr(model, "names"))

    predict_kwargs: Dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz
    if device is not None:
        predict_kwargs["device"] = device

    summary_rows: List[Dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[Interpretability] {index}/{len(image_paths)} {image_path}")
        result = model.predict(source=str(image_path), **predict_kwargs)[0]
        detections = extract_detections_from_result(result, class_names=class_names)

        image = Image.open(image_path)
        heatmap = build_detection_heatmap((image.height, image.width), detections, target_class=target_class)
        prefix = f"{index:03d}_{image_path.stem}"
        artifacts = save_heatmap_artifacts(
            image_path=image_path,
            heatmap=heatmap,
            detections=detections,
            output_dir=output_dir,
            prefix=prefix,
            method="detection_confidence",
        )
        summary_rows.append(artifacts.as_relative_row(output_dir, detections))

        if occlusion:
            occlusion_heatmap = build_occlusion_heatmap(
                model=model,
                image_path=image_path,
                baseline_result=result,
                grid_size=grid_size,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                target_class=target_class,
            )
            occlusion_artifacts = save_heatmap_artifacts(
                image_path=image_path,
                heatmap=occlusion_heatmap,
                detections=detections,
                output_dir=output_dir,
                prefix=prefix,
                method="occlusion_sensitivity",
            )
            summary_rows.append(occlusion_artifacts.as_relative_row(output_dir, detections))

    summary_csv = _write_summary_csv(output_dir, summary_rows)
    report = render_markdown_report(
        output_dir=output_dir,
        model_path=model_path,
        data_yaml=data_yaml,
        class_names=class_names,
        artifact_rows=summary_rows,
        occlusion_enabled=occlusion,
    )

    print(f"[Interpretability] Summary CSV: {summary_csv}")
    print(f"[Interpretability] Report: {report}")
    return {"output_dir": output_dir, "summary_csv": summary_csv, "report": report}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate YOLO interpretability heatmaps and reports")
    parser.add_argument("--model", required=True, help="Path to trained YOLO .pt model")
    parser.add_argument("--output-dir", required=True, help="Directory for interpretability artifacts")
    parser.add_argument("--data-yaml", default=None, help="YOLO data.yaml; used for class names and split images")
    parser.add_argument("--images-dir", default=None, help="Directory of images to interpret")
    parser.add_argument("--image", action="append", dest="images", help="Individual image path; repeatable")
    parser.add_argument("--split", default="val", help="data.yaml split to use when --data-yaml is provided (default: val)")
    parser.add_argument("--limit", type=int, default=12, help="Maximum number of images to process (default: 12)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO IoU/NMS threshold (default: 0.7)")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size")
    parser.add_argument("--device", default=None, help="Optional inference device, e.g. 0 or cpu")
    parser.add_argument("--occlusion", action="store_true", help="Also generate slower occlusion sensitivity maps")
    parser.add_argument("--grid-size", type=int, default=8, help="Occlusion grid size per image dimension (default: 8)")
    parser.add_argument("--target-class", default=None, help="Optional class id/name to focus heatmaps and occlusion scoring")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_interpretability_pipeline(
        model_path=args.model,
        output_dir=args.output_dir,
        data_yaml=args.data_yaml,
        images_dir=args.images_dir,
        images=args.images,
        split=args.split,
        limit=args.limit,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        occlusion=args.occlusion,
        grid_size=args.grid_size,
        target_class=args.target_class,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
