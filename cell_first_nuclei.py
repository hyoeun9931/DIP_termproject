"""Approximate cell-region segmentation followed by nucleus detection.

Run from this folder with:
    python cell_first_nuclei.py
    python cell_first_nuclei.py image.png image2.png --output-dir output_cell_first

Cell regions are estimated from the red-magenta channel in LAB color space.
This is a color-based heuristic for the provided stained images; adjust
CELL_A_FLOOR and the nucleus thresholds for images with different staining.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


CELL_A_FLOOR = 132
MIN_NUCLEUS_AREA = 220
MAX_NUCLEUS_AREA = 6000


def make_cell_mask(image):
    """Estimate stained cell-body regions from LAB's red-green channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = cv2.GaussianBlur(lab[:, :, 1], (0, 0), 3)

    # Otsu adapts to each image; the floor keeps bluish/neutral background
    # from being classified as cells in the supplied images.
    otsu_threshold, _ = cv2.threshold(
        a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(int(otsu_threshold), CELL_A_FLOOR)
    mask = cv2.threshold(a_channel, threshold, 255, cv2.THRESH_BINARY)[1]

    # Join small gaps in cell bodies, then remove isolated staining specks.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    # Keep cell-sized connected regions and fill internal holes (including
    # dark nuclei), so the later nucleus search is not blocked by the mask.
    height, width = mask.shape
    min_cell_area = max(500, int(height * width * 0.00025))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_cell_area and area < height * width * 0.75:
            clean[labels == label] = 255

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(clean)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def find_nuclei(image, cell_mask):
    """Find dark, round nucleus candidates only inside estimated cell regions."""
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
    )
    smooth = cv2.GaussianBlur(denoised, (5, 5), 1.2)

    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(
        cv2.merge([clahe.apply(lightness), a_channel, b_channel]), cv2.COLOR_LAB2BGR
    )

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    inside_cells = cell_mask > 0
    if not np.any(inside_cells):
        return enhanced, np.zeros_like(cell_mask), enhanced.copy(), 0

    # Threshold relative to cell pixels rather than the whole image.
    dark_threshold = np.percentile(gray[inside_cells], 12)
    candidate = (
        (gray <= dark_threshold) & (blackhat > 35) & inside_cells
    ).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    nuclei_mask = np.zeros_like(cell_mask)
    result = enhanced.copy()
    count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_NUCLEUS_AREA or area > MAX_NUCLEUS_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < 0.48:
            continue

        cv2.drawContours(nuclei_mask, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
        x, y, _, _ = cv2.boundingRect(contour)
        count += 1
        cv2.putText(
            result,
            str(count),
            (x, max(y - 6, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return enhanced, nuclei_mask, result, count


def process_image(image_path, output_dir):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    cell_mask = make_cell_mask(image)
    enhanced, nuclei_mask, result, count = find_nuclei(image, cell_mask)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = image_path.stem

    cv2.imwrite(str(output_dir / f"{name}_cell_mask.png"), cell_mask)
    cv2.imwrite(str(output_dir / f"{name}_nuclei_mask.png"), nuclei_mask)
    cv2.imwrite(str(output_dir / f"{name}_enhanced.png"), enhanced)
    cv2.imwrite(str(output_dir / f"{name}_result.png"), result)
    print(f"[{image_path.name}] 세포 영역 안에서 검출된 핵: {count}개")


def main():
    parser = argparse.ArgumentParser(
        description="세포 영역을 먼저 추정한 뒤 그 안에서 핵을 검출합니다."
    )
    parser.add_argument(
        "images",
        nargs="*",
        default=["image.png", "image2.png"],
        help="입력 이미지 경로 (기본값: image.png image2.png)",
    )
    parser.add_argument(
        "--output-dir",
        default="output_cell_first",
        help="결과 저장 폴더 (기본값: output_cell_first)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for image_path in args.images:
        process_image(image_path, output_dir)


if __name__ == "__main__":
    main()
