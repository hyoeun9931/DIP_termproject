"""Nuclear.ipynb-inspired cell/nucleus segmentation and mask export.

Pipeline: color denoising -> light Gaussian smoothing -> unsharp mask ->
LAB CLAHE -> HSV cell candidates -> cell watershed -> black-hat/dark-pixel
nucleus candidates restricted to cells. All stage masks are saved as 8-bit
single-channel PNGs. Settings and dataset-level metrics are saved alongside.
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

import watershed_nuclei as pipeline

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "train_image_norm"
DEFAULT_GT_DIR = BASE_DIR / "test_image_norm"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "nuclear_watershed_analysis"
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def notebook_nucleus_masks(smooth, enhanced, cell_labels):
    """Apply the notebook's dark-pixel + normalized black-hat rule in cells."""
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(
        enhanced_gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    )
    # Nuclear.ipynb normalizes black-hat to [0, 255] before using > 40.
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(
        cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), (5, 5), 0
    )
    in_cells = cell_labels >= 2
    dark_cutoff = np.percentile(gray, 4)
    candidates = (
        (gray <= dark_cutoff) & (blackhat > 40) & in_cells
    ).astype(np.uint8) * 255
    before_morph = candidates.copy()
    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    after_morph = candidates.copy()
    contours, _ = cv2.findContours(
        candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    final_mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 150 or area > 6000:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < 0.60:
            continue
        cv2.drawContours(final_mask, [contour], -1, 255, cv2.FILLED)

    result = enhanced.copy()
    nucleus_contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for index, contour in enumerate(nucleus_contours, start=1):
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        x, y, _, _ = cv2.boundingRect(contour)
        cv2.putText(
            result, str(index), (x, max(y - 6, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
    for cell_id in np.unique(cell_labels):
        if cell_id < 2:
            continue
        region = (cell_labels == cell_id).astype(np.uint8) * 255
        cell_contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, cell_contours, -1, (255, 255, 0), 1)
    return final_mask, result, len(nucleus_contours), before_morph, after_morph, after_morph


def notebook_settings() -> dict:
    return {
        "reference": "Nuclear.ipynb, image processing and nucleus-mask cells",
        "deep_learning": False,
        "source_image": "OpenCV BGR uint8",
        "denoising": {
            "method": "cv2.fastNlMeansDenoisingColored",
            "h": 6,
            "hColor": 6,
            "templateWindowSize": 7,
            "searchWindowSize": 21,
        },
        "gaussian_smoothing": {
            "method": "cv2.GaussianBlur",
            "kernel": [5, 5],
            "sigma": 1.2,
        },
        "unsharp_mask": {
            "method": "cv2.addWeighted(smooth, 1.5, broad_blur, -0.5, 0)",
            "broad_blur_sigma_x": 8,
        },
        "color_space_and_contrast": {
            "conversion": "BGR -> LAB -> BGR",
            "clahe_channel": "L",
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": [8, 8],
        },
        "cell_candidate_hsv": {
            "hue_ranges": [[105, 179], [0, 12]],
            "minimum_saturation": pipeline.CELL_SATURATION_MIN,
            "morphology_close_kernel": [7, 7],
            "morphology_open_kernel": [7, 7],
            "minimum_component_area_px": pipeline.MIN_CELL_AREA,
        },
        "cell_watershed": {
            "seed_method": "dark black-hat regions; distance-transform fallback",
            "blackhat_kernel": [41, 41],
            "dark_percentile_inside_candidates": 18,
            "blackhat_minimum": 35,
            "seed_area_px": [150, 2500],
            "seed_circularity_minimum": 0.52,
            "distance_peak_relative_cutoff": 0.28,
        },
        "nucleus_mask_from_notebook": {
            "gray_source": "5x5 Gaussian blurred grayscale of denoised image",
            "dark_pixel_percentile": 4,
            "blackhat_source": "grayscale CLAHE-enhanced image",
            "blackhat_kernel": [41, 41],
            "blackhat_normalization": "cv2.normalize to [0, 255] before threshold",
            "blackhat_minimum": 40,
            "allowed_region": "watershed cell labels only",
            "morphology_open_kernel": [5, 5],
            "morphology_close_kernel": [9, 9],
            "contour_area_px": [150, 6000],
            "circularity_minimum": 0.60,
        },
        "mask_output": {
            "format": "PNG",
            "channels": 1,
            "dtype": "uint8",
            "background": 0,
            "foreground": 255,
        },
        "library_versions": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
        },
    }


def save_mask(path: Path, mask: np.ndarray) -> None:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), binary):
        raise OSError(f"Could not write mask: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"이미지가 없습니다: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(notebook_settings(), f, ensure_ascii=False, indent=2)

    rows = []
    for i, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"이미지 읽기 실패: {image_path}")
            continue

        smooth, enhanced, hsv = pipeline.preprocess(image)
        hue, saturation, _ = cv2.split(hsv)
        hsv_raw = (
            (((hue >= pipeline.CELL_HUE_MIN) | (hue <= pipeline.CELL_HUE_MAX_LOW))
             & (saturation >= pipeline.CELL_SATURATION_MIN)).astype(np.uint8) * 255
        )
        cell_candidates = pipeline.make_cell_candidates(hsv)
        cell_labels = pipeline.make_watershed_cells(enhanced, cell_candidates)
        (nuclei_mask, result, nucleus_count, dark_candidates,
         morphed_candidates, size_accepted) = notebook_nucleus_masks(
            smooth, enhanced, cell_labels
        )

        # Ground-truth masks in this dataset are rendered overlays; this loader
        # reconstructs approximate filled instances from the yellow outlines.
        gt_matches = sorted(args.ground_truth_dir.glob(f"{image_path.stem}_GT_*.png"))
        gt_count = None
        iou_matches = None
        gt_source = "not found"
        if gt_matches:
            gt_labels, gt_count, approximate = pipeline.load_ground_truth_instances(gt_matches[0])
            if gt_labels.shape == nuclei_mask.shape:
                iou_matches = pipeline.matched_iou_count(
                    nuclei_mask, gt_labels, gt_count, threshold=0.5
                )
                gt_source = "yellow-outline approximation" if approximate else "mask"
            else:
                gt_source = "size mismatch"

        image_dir = args.output_dir / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        stages = {
            "01_hsv_raw_cell_candidates.png": hsv_raw,
            "02_hsv_cleaned_cell_candidates.png": cell_candidates,
            "03_watershed_cell_mask.png": (cell_labels > 0).astype(np.uint8) * 255,
            "04_dark_pixel_and_blackhat_candidates.png": dark_candidates,
            "05_morphology_nucleus_candidates.png": morphed_candidates,
            "06_nucleus_size_accepted.png": size_accepted,
            "07_final_nucleus_mask.png": nuclei_mask,
        }
        for filename, mask in stages.items():
            save_mask(image_dir / filename, mask)
        if not cv2.imwrite(str(image_dir / "08_overlay_result.png"), result):
            raise OSError(f"결과 오버레이 저장 실패: {image_dir}")

        cell_count = len([x for x in np.unique(cell_labels) if x > 0])
        rows.append({
            "image": image_path.name,
            "gt_nucleus_count": gt_count,
            "detected_nucleus_count": nucleus_count,
            "detected_cell_count": cell_count,
            "iou_050_matched_nuclei": iou_matches,
            "gt_source": gt_source,
            "output_dir": image_dir.name,
        })
        match_text = f" / IoU≥0.50 일치 {iou_matches}개" if iou_matches is not None else ""
        print(
            f"[{i}/{len(image_paths)}] {image_path.name}: "
            f"세포 {cell_count}개, 핵 {nucleus_count}개, GT {gt_count}개{match_text}"
        )

    if rows:
        with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"마스크와 설정, 요약 저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
