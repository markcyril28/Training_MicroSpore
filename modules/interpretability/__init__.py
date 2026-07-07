"""Interpretability tools for Microspore YOLO image recognition models.

The package is intentionally import-light: Ultralytics is imported only when the
runtime pipeline loads a model, so utility functions and tests can run in a
plain Python environment.
"""

_EXPORTS = {
    "InterpretabilityArtifacts",
    "build_detection_heatmap",
    "collect_image_paths",
    "extract_detections_from_result",
    "load_class_names",
    "render_markdown_report",
    "run_interpretability_pipeline",
    "save_heatmap_artifacts",
}


def __getattr__(name):
    """Load pipeline symbols lazily to keep `python -m ...pipeline` warning-free."""
    if name in _EXPORTS:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
