"""Run Otsu, adaptive-threshold, and K-means segmentation consistently.

Each method saves binary cell/nucleus masks as 8-bit grayscale PNGs and
instance-label masks as 16-bit grayscale PNGs. Processing settings are written
beside the outputs in settings.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import skimage
import scipy

import jh
import jh_adaptive
import jh_kmeans


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "train_image_norm"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "threshold_mask_comparison"
METHODS = {
    "otsu": jh,
    "adaptive": jh_adaptive,
    "kmeans": jh_kmeans,
}


def method_settings(name: str, module) -> dict:
    common = {
        "input_color_space": "OpenCV BGR, uint8",
        "gray_conversion": "BGR to grayscale (OpenCV)",
        "body_color_space": "BGR to RGB to CIELAB (skimage.color.rgb2lab)",
        "background_estimation": {
            "method": "median CIELAB color on image border",
            "border_ratio": module.BACKGROUND_BORDER_RATIO,
        },
        "color_distance_filter": {
            "method": "scipy.ndimage.median_filter",
            "size": module.COLOR_DISTANCE_MEDIAN_SIZE,
        },
        "gaussian_blur": {
            "source_image_blur": "none",
            "nucleus_distance_transform_sigma_px": module.NUCLEUS_DISTANCE_SIGMA,
        },
        "morphology": {
            "body_closing_radius_px": module.CELL_BODY_CLOSING_RADIUS,
            "minimum_body_area_px": module.MIN_CELL_BODY_AREA,
            "nucleus_opening_disk_radius_px": module.MORPH_DISK_RADIUS,
            "minimum_nucleus_area_px": module.MIN_NUCLEUS_AREA,
            "maximum_nucleus_area_px": module.MAX_NUCLEUS_AREA,
            "surrounding_body_ratio_minimum": module.MIN_SURROUNDING_BODY_RATIO,
        },
        "watershed": {
            "nucleus_distance_sigma_px": module.NUCLEUS_DISTANCE_SIGMA,
            "nucleus_peak_prominence": module.NUCLEUS_PEAK_PROMINENCE,
            "minimum_seed_area_px": module.MIN_NUCLEUS_SEED_AREA,
            "cell_watershed_mode": (
                "edge-guided" if module.USE_EDGE_GUIDED_CELL_WATERSHED
                else "body-distance"
            ),
        },
    }
    if name == "otsu":
        common["cell_body_threshold"] = "global Otsu on Lab background color-distance image"
        common["nucleus_threshold"] = (
            "per-cell Otsu on grayscale pixels, then a second Otsu threshold "
            "on darker pixels"
        )
    elif name == "adaptive":
        common["cell_body_threshold"] = {
            "method": "Gaussian adaptive threshold, binary",
            "configured_block_size_px": module.CELL_ADAPTIVE_BLOCK_SIZE,
            "effective_block_size": "min(configured size, input short side), then reduced to odd; minimum 3",
            "C": module.CELL_ADAPTIVE_C,
            "input": "normalized uint8 Lab background color-distance image",
        }
        common["nucleus_threshold"] = {
            "method": "Gaussian adaptive threshold, inverse binary",
            "configured_block_size_px": module.NUCLEUS_ADAPTIVE_BLOCK_SIZE,
            "effective_block_size": "min(configured size, ROI short side), then reduced to odd",
            "C": module.NUCLEUS_ADAPTIVE_C,
            "input": "grayscale crop of each cell-body component",
        }
    else:
        common["cell_body_threshold"] = {
            "method": "OpenCV K-means, 2 clusters",
            "input": "Lab background color-distance image",
            "selected_cluster": "higher-distance cluster",
            "attempts": 5,
            "max_iterations": 100,
            "epsilon": 0.2,
            "rng_seed": 0,
        }
        common["nucleus_threshold"] = {
            "method": "OpenCV K-means, 2 clusters per cell-body component",
            "input": "grayscale pixels inside each cell-body component",
            "selected_cluster": "lower-intensity cluster",
            "attempts": 5,
            "max_iterations": 100,
            "epsilon": 0.2,
            "rng_seed": 0,
        }
    return common


def binary_mask(labels: np.ndarray) -> np.ndarray:
    return np.where(labels > 0, 255, 0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in jh.IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "image_count": len(image_paths),
        "mask_formats": {
            "binary_masks": "PNG, 8-bit grayscale, background=0, foreground=255",
            "instance_label_masks": "PNG, 16-bit grayscale, background=0, labels=1..N",
        },
        "library_versions": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
        },
        "methods": {
            name: method_settings(name, module)
            for name, module in METHODS.items()
        },
    }
    with (args.output_dir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

    rows = []
    gt_count_map = jh.build_gt_count_map(BASE_DIR / "test_image_norm")
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[{index}/{len(image_paths)}] Could not read {image_path.name}")
            continue

        gt_count = gt_count_map.get(image_path.stem.casefold())
        for method_name, module in METHODS.items():
            preprocessed = module.preprocess_image(image)
            segmented = module.segment_image(preprocessed)
            nuclei = segmented["labeled_nuclei"].astype(np.uint16)
            cells = segmented["segmented_cells"].astype(np.uint16)
            method_dir = args.output_dir / image_path.stem / method_name
            method_dir.mkdir(parents=True, exist_ok=True)

            masks = {
                "cell_mask.png": binary_mask(cells),
                "nucleus_mask.png": binary_mask(nuclei),
                "cell_instances.png": cells,
                "nucleus_instances.png": nuclei,
            }
            for filename, mask in masks.items():
                if not cv2.imwrite(str(method_dir / filename), mask):
                    raise OSError(f"Could not write {method_dir / filename}")

            nucleus_count = int(nuclei.max())
            cell_count = int(cells.max())
            rows.append({
                "image": image_path.name,
                "method": method_name,
                "gt_nucleus_count_from_filename": gt_count,
                "detected_nucleus_count": nucleus_count,
                "detected_cell_count": cell_count,
                "nucleus_mask": str((method_dir / "nucleus_mask.png").relative_to(args.output_dir)),
                "cell_mask": str((method_dir / "cell_mask.png").relative_to(args.output_dir)),
                "nucleus_instances": str((method_dir / "nucleus_instances.png").relative_to(args.output_dir)),
                "cell_instances": str((method_dir / "cell_instances.png").relative_to(args.output_dir)),
            })
        print(f"[{index}/{len(image_paths)}] {image_path.name}")

    if rows:
        with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved masks, summary.csv, and settings.json under {args.output_dir}")


if __name__ == "__main__":
    main()
