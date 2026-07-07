import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from modules.interpretability import (
    build_detection_heatmap,
    collect_image_paths,
    load_class_names,
    render_markdown_report,
    save_heatmap_artifacts,
)


class InterpretabilityPipelineTests(unittest.TestCase):
    def test_load_class_names_supports_yolo_dict_names_sorted_numerically(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_yaml = Path(tmp) / "data.yaml"
            data_yaml.write_text(
                "path: .\n"
                "train: images/train\n"
                "val: images/val\n"
                "names:\n"
                "  10: mature_pollen\n"
                "  2: tetrad\n"
                "  1: young_microspore\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_class_names(data_yaml),
                ["young_microspore", "tetrad", "mature_pollen"],
            )

    def test_collect_image_paths_uses_data_yaml_relative_split_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val_dir = root / "images" / "val"
            val_dir.mkdir(parents=True)
            for name in ["b.png", "a.jpg", "ignore.txt", "c.jpeg"]:
                p = val_dir / name
                if p.suffix == ".txt":
                    p.write_text("not an image", encoding="utf-8")
                else:
                    Image.new("RGB", (8, 8), "white").save(p)
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "path: .\ntrain: images/train\nval: images/val\nnames: [microspore]\n",
                encoding="utf-8",
            )

            images = collect_image_paths(data_yaml=data_yaml, split="val", limit=2)

            self.assertEqual([p.name for p in images], ["a.jpg", "b.png"])

    def test_collect_image_paths_supports_yolo_txt_image_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            first = image_dir / "first.jpg"
            second = image_dir / "second.png"
            Image.new("RGB", (8, 8), "white").save(first)
            Image.new("RGB", (8, 8), "white").save(second)
            list_file = root / "val.txt"
            list_file.write_text("images/second.png\n# comment\nimages/first.jpg\n", encoding="utf-8")
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "path: .\ntrain: images/train\nval: val.txt\nnames: [microspore]\n",
                encoding="utf-8",
            )

            images = collect_image_paths(data_yaml=data_yaml, split="val")

            self.assertEqual([p.name for p in images], ["first.jpg", "second.png"])

    def test_build_detection_heatmap_prioritizes_high_confidence_box_region(self):
        detections = [
            {"box_xyxy": [10, 10, 30, 30], "confidence": 0.90, "class_name": "tetrad"},
            {"box_xyxy": [50, 50, 70, 70], "confidence": 0.30, "class_name": "other"},
        ]

        heatmap = build_detection_heatmap((100, 100), detections)

        self.assertEqual(heatmap.shape, (100, 100))
        self.assertAlmostEqual(float(heatmap.max()), 1.0, places=6)
        self.assertGreater(float(heatmap[20, 20]), float(heatmap[60, 60]))
        self.assertEqual(float(heatmap[0, 0]), 0.0)

    def test_save_heatmap_artifacts_writes_overlay_heatmap_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (32, 32), (20, 40, 60)).save(image_path)
            heatmap = np.zeros((32, 32), dtype=np.float32)
            heatmap[8:24, 8:24] = 1.0
            detections = [{"class_name": "tetrad", "confidence": 0.88, "box_xyxy": [8, 8, 24, 24]}]

            artifacts = save_heatmap_artifacts(
                image_path=image_path,
                heatmap=heatmap,
                detections=detections,
                output_dir=tmp_path / "out",
                prefix="sample",
            )

            self.assertTrue(artifacts.overlay_path.exists())
            self.assertTrue(artifacts.heatmap_path.exists())
            self.assertTrue(artifacts.metadata_path.exists())
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["image"], str(image_path))
            self.assertEqual(metadata["num_detections"], 1)
            self.assertEqual(metadata["detections"][0]["class_name"], "tetrad")

    def test_render_markdown_report_summarizes_artifacts_and_guidance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = render_markdown_report(
                output_dir=tmp_path,
                model_path=Path("weights/best.pt"),
                data_yaml=Path("data.yaml"),
                class_names=["tetrad", "other"],
                artifact_rows=[
                    {
                        "image": "a.jpg",
                        "overlay": "visualizations/a_overlay.png",
                        "heatmap": "heatmaps/a_heatmap.png",
                        "num_detections": 2,
                        "top_detection": "tetrad 0.91",
                    }
                ],
                occlusion_enabled=True,
            )

            content = report_path.read_text(encoding="utf-8")
            self.assertIn("# Interpretability Report", content)
            self.assertIn("weights/best.pt", content)
            self.assertIn("Detection-confidence heatmaps", content)
            self.assertIn("Occlusion sensitivity", content)
            self.assertIn("tetrad 0.91", content)


if __name__ == "__main__":
    unittest.main()
