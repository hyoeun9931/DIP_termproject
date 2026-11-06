"""HSV candidate regions -> watershed cell segmentation -> nucleus detection.

Uses the denoise, unsharp mask, LAB-CLAHE, and HSV steps from Nuclear.ipynb.
The heatmap visualization is omitted. Run from this folder:
    ../.venv/bin/python watershed_nuclei.py
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

# HSV hue/saturation are used only to propose foreground; watershed divides it.
CELL_HUE_MIN = 105
CELL_HUE_MAX_LOW = 12
CELL_SATURATION_MIN = 35
MIN_CELL_AREA = 1800
MIN_NUCLEUS_AREA = 180
MAX_NUCLEUS_AREA = 6000


def preprocess(image):
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
    )
    smooth = cv2.GaussianBlur(denoised, (5, 5), 1.2)
    broad_blur = cv2.GaussianBlur(smooth, (0, 0), sigmaX=8)
    sharpened = cv2.addWeighted(smooth, 1.5, broad_blur, -0.5, 0)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(
        cv2.merge([clahe.apply(lightness), a_channel, b_channel]), cv2.COLOR_LAB2BGR
    )
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    return smooth, enhanced, hsv


def make_cell_candidates(hsv):
    hue, saturation, _ = cv2.split(hsv)
    hue_is_cell_like = (hue >= CELL_HUE_MIN) | (hue <= CELL_HUE_MAX_LOW)
    mask = (hue_is_cell_like & (saturation >= CELL_SATURATION_MIN)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= MIN_CELL_AREA:
            cleaned[labels == label] = 255
    return cleaned


def make_watershed_cells(enhanced, coarse_mask):
    """Use dark, round nucleus-like markers to split HSV cell candidates."""
    if not np.any(coarse_mask):
        return np.zeros(coarse_mask.shape, dtype=np.int32)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    )
    dark_cutoff = np.percentile(gray[coarse_mask > 0], 18)
    seed_mask = (
        (gray <= dark_cutoff) & (blackhat > 35) & (coarse_mask > 0)
    ).astype(np.uint8) * 255
    seed_mask = cv2.morphologyEx(
        seed_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    contours, _ = cv2.findContours(
        seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Dark round spots inside the coarse foreground are cell-center markers.
    # Leave other pixels unknown; after watershed, keep only the HSV foreground.
    markers = np.zeros(coarse_mask.shape, dtype=np.int32)
    next_label = 2
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 150 or area > 2500:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < 0.52:
            continue
        cv2.drawContours(markers, [contour], -1, next_label, cv2.FILLED)
        next_label += 1

    if next_label == 2:
        # Fall back to distance-transform peaks for regions without dark seeds.
        distance = cv2.distanceTransform(coarse_mask, cv2.DIST_L2, 5)
        footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        local_max = cv2.dilate(distance, footprint)
        cutoff = max(8.0, float(distance.max()) * 0.28)
        peaks = ((distance >= local_max - 1e-3) & (distance >= cutoff)).astype(np.uint8)
        peaks = cv2.dilate(peaks, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        seed_count, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(
            peaks, connectivity=8
        )
        for seed in range(1, seed_count):
            if seed_stats[seed, cv2.CC_STAT_AREA] >= 3:
                markers[seed_labels == seed] = next_label
                next_label += 1

    # Inverted distance transform makes cell interiors watershed basins and
    # narrow gaps between adjacent cells become the separating ridges.
    distance = cv2.distanceTransform(coarse_mask, cv2.DIST_L2, 5)
    surface = 255 - cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    watershed_image = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_image, markers)
    labels = markers.copy()
    labels[(coarse_mask == 0) | (labels < 2)] = 0
    for label in np.unique(labels):
        if label >= 2 and np.count_nonzero(labels == label) < MIN_CELL_AREA:
            labels[labels == label] = 0
    return labels


def find_nuclei(smooth, enhanced, cell_labels):
    """Reuse the notebook's dark-pixel/black-hat nuclei criteria in cell areas."""
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(
        enhanced_gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    )
    gray = cv2.GaussianBlur(
        cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), (5, 5), 0
    )
    in_cells = cell_labels >= 2
    nuclei_mask = np.zeros(gray.shape, dtype=np.uint8)
    result = enhanced.copy()

    if not np.any(in_cells):
        return nuclei_mask, result, 0

    # Match Nuclear.ipynb's 4th-percentile threshold, then restrict candidates
    # to watershed cell regions so background specks are excluded.
    dark_cutoff = np.percentile(gray, 4)
    candidates = (
        (gray <= dark_cutoff) & (blackhat > 40) & in_cells
    ).astype(np.uint8) * 255
    candidates = cv2.morphologyEx(
        candidates,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    candidates = cv2.morphologyEx(
        candidates,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    contours, _ = cv2.findContours(
        candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    total = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_NUCLEUS_AREA or area > MAX_NUCLEUS_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < 0.60:
            continue
        cv2.drawContours(nuclei_mask, [contour], -1, 255, cv2.FILLED)
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
        x, y, _, _ = cv2.boundingRect(contour)
        total += 1
        cv2.putText(
            result, str(total), (x, max(y - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )

    # Draw cell boundaries on the result for visual inspection.
    for cell_id in np.unique(cell_labels):
        if cell_id < 2:
            continue
        region = (cell_labels == cell_id).astype(np.uint8) * 255
        cell_contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, cell_contours, -1, (255, 255, 0), 2)
    return nuclei_mask, result, total


def colorize_labels(labels):
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels):
        if label >= 2:
            hue = (int(label) * 37) % 180
            hsv_color = np.uint8([[[hue, 220, 230]]])
            colored[labels == label] = cv2.cvtColor(
                hsv_color, cv2.COLOR_HSV2BGR
            )[0, 0]
    return colored


def process_image(image_path, output_dir):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    smooth, enhanced, hsv = preprocess(image)
    coarse_mask = make_cell_candidates(hsv)
    cell_labels = make_watershed_cells(enhanced, coarse_mask)
    nuclei_mask, result, nucleus_count = find_nuclei(smooth, enhanced, cell_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = image_path.stem
    cv2.imwrite(str(output_dir / f"{name}_hsv_saturation.png"), hsv[:, :, 1])
    cv2.imwrite(str(output_dir / f"{name}_cell_candidates.png"), coarse_mask)
    cv2.imwrite(str(output_dir / f"{name}_watershed_cells.png"), colorize_labels(cell_labels))
    cv2.imwrite(str(output_dir / f"{name}_nuclei_mask.png"), nuclei_mask)
    cv2.imwrite(str(output_dir / f"{name}_result.png"), result)
    cell_count = sum(label >= 2 for label in np.unique(cell_labels))
    print(f"[{image_path.name}] watershed 세포 영역: {cell_count}개, 핵 후보: {nucleus_count}개")


def main():
    parser = argparse.ArgumentParser(description="HSV 세포 후보를 watershed로 나누고 각 영역 안에서 핵을 검출합니다.")
    parser.add_argument("images", nargs="*", default=["image.png", "image2.png"])
    parser.add_argument("--output-dir", default="output_watershed")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    for image_path in args.images:
        process_image(image_path, output_dir)


if __name__ == "__main__":
    main()
