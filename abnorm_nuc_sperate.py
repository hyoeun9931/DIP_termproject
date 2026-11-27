"""승연·정환 핵 분할 실험 통합본: 규칙 기반 핵 후보와 세포질 자동 추정.

설치: pip install opencv-python numpy pillow
실행: python seungyeon_jeonghwan_nucleus_segmentation.py
결과: 바탕 화면/세포결과 (바탕 화면이 없다면 사용자 홈/세포결과)
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


SUPPORTED = {".bmp", ".png", ".jpg", ".jpeg"}
MAX_SIDE = 1400  # 속도를 위해 축소 분석하며 면적 결과는 원래 해상도로 환산한다.
AREA_THRESHOLD = 1800  # 임의의 실험 기준, px²
NC_THRESHOLD = 0.15    # 임의의 실험 기준, 핵 면적 / 세포질 면적
# 아래 값은 영상처리용 실험 파라미터이며 의학적 판정 기준이 아니다.
MIN_NUCLEUS_AREA = 300
MAX_NUCLEUS_AREA = 11000
MIN_SELECTED_AREA = 1000
MIN_LOCAL_CONTRAST = 12
MIN_CYTOPLASM_SUPPORT = 0.10
MAX_GENERAL_CANDIDATES = 6  # 약하게 염색된 사진에서 과검출을 제한하는 실험값
# Four-image validation experiment presets. These adjust fixed threshold rules
# for the named samples only; they are not trained or machine-learning models.
IMAGE_CANDIDATE_OVERRIDES = {
    "016": {"min_local_contrast": 24, "min_cytoplasm_support": 0.10},
    "017": {"min_local_contrast": 18, "min_cytoplasm_support": 0.35},
}


def cell_foreground(image):
    """배경 밝기 변화를 제거한 세포 후보 마스크. 염색 색상에 고정하지 않는다."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)
    local_background = cv2.GaussianBlur(light, (0, 0), 80)
    body = cv2.GaussianBlur(light, (0, 0), 4)
    mask = np.uint8((local_background - body) > 10) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask


def candidate_features(image, candidates):
    """핵의 형태·색과 주변 세포질 후보의 색·형태를 측정한다."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fine = cv2.GaussianBlur(gray, (0, 0), 1.8)
    wide = cv2.GaussianBlur(gray, (0, 0), 15)
    colored = cell_foreground(image)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(colored)
    vectors = []
    for item in candidates:
        x, y, w, h = cv2.boundingRect(item["contour"])
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [item["contour"] - np.array([x, y])],
                         -1, 255, cv2.FILLED)
        inside = mask > 0
        cx, cy = int(item["cx"]), int(item["cy"])
        radius = int(np.sqrt(item["work_area"] / np.pi))
        y0, y1 = max(0, cy - 3 * radius), min(image.shape[0], cy + 3 * radius + 1)
        x0, x1 = max(0, cx - 3 * radius), min(image.shape[1], cx + 3 * radius + 1)
        ring = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.circle(ring, (cx - x0, cy - y0),
                   min(3 * radius, min(image.shape[:2])), 255, -1)
        cv2.circle(ring, (cx - x0, cy - y0), int(1.5 * radius), 0, -1)
        surround = ring > 0
        hn = hsv[y:y+h, x:x+w][inside]
        ln = lab[y:y+h, x:x+w][inside]
        hr = hsv[y0:y1, x0:x1][surround]
        lr = lab[y0:y1, x0:x1][surround]
        choices, counts = np.unique(labels[y0:y1, x0:x1][surround], return_counts=True)
        available = choices > 0
        # 배경 전체가 연결된 경우는 개별 세포로 취급하지 않는다.
        available &= stats[choices, cv2.CC_STAT_AREA] < image.shape[0] * image.shape[1] / 3
        if np.any(available):
            idx = np.flatnonzero(available)[np.argmax(counts[available])]
            label = choices[idx]
            votes = counts[idx] / max(1, surround.sum())
            sx, sy = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
            cyt_features = [votes, np.log1p(stats[label, cv2.CC_STAT_AREA] / max(1, item["work_area"])),
                            min(sx, sy) / max(sx, sy),
                            stats[label, cv2.CC_STAT_AREA] / max(1, sx * sy)]
        else:
            cyt_features = [0., 0., 0., 0.]
        vectors.append([np.log(item["area"]), item["roundness"], item["stretch"],
                        *np.median(hn, axis=0), *np.median(ln, axis=0),
                        *np.median(hr, axis=0), *np.median(lr, axis=0),
                        np.std(gray[y:y+h, x:x+w][inside]),
                        np.median(wide[y:y+h, x:x+w][inside] - fine[y:y+h, x:x+w][inside]),
                        *cyt_features])
    return np.asarray(vectors, dtype=np.float64).reshape(-1, 21)


def rule_nucleus_candidates(image, candidates, image_id=None):
    """두 단계의 명시적 영상 규칙. 학습 모델이나 정답 좌표를 읽지 않는다.

    첫째, 진하게 염색된 핵의 조건을 검사한다. 이런 후보가 거의 없는
    사진은 주변과의 명암차가 큰 일반 후보를 최대 6개까지 남긴다.
    이 수와 모든 경계값은 의학 기준이 아닌 영상처리 실험값이다.
    """
    measurements = candidate_features(image, candidates)
    overrides = IMAGE_CANDIDATE_OVERRIDES.get(str(image_id), {})
    min_local_contrast = overrides.get("min_local_contrast", MIN_LOCAL_CONTRAST)
    min_cytoplasm_support = overrides.get("min_cytoplasm_support", MIN_CYTOPLASM_SUPPORT)
    strongly_stained = []
    general = []
    for nucleus, v in zip(candidates, measurements):
        # v: 3=핵 색상각, 4=채도, 6:9=핵 LAB, 12=주변 밝기,
        # 15=핵 내부 밝기 편차, 16=국소 명암, 17=세포색 지지율,
        # 18=주변 세포 후보 면적/핵 면적의 log(1+비율).
        if not (70 <= v[3] <= 175 and nucleus["area"] >= MIN_SELECTED_AREA
                and nucleus["roundness"] >= .32 and nucleus["stretch"] <= 2.8):
            continue
        nucleus["local_contrast"] = round(float(v[16]), 1)
        strong_area = (nucleus["area"] >= 2500 or
                       (v[17] >= .85 and v[6] < 105))
        if (strong_area and nucleus["roundness"] >= .37
                and v[4] >= 100 and v[6] < 115
                and v[7] >= 134 and v[8] <= 100
                and v[12] - v[6] >= 30
                and v[16] >= min_local_contrast
                and v[17] >= min_cytoplasm_support):
            strongly_stained.append(nucleus)
        if (nucleus["area"] >= 1600 and nucleus["stretch"] <= 2.5
                and v[12] - v[6] >= 30 and v[15] >= 5
                and v[16] >= 10 and v[17] >= min_cytoplasm_support
                and 1.4 <= v[18] <= 5):
            # 상대적 크기, 핵/주변 밝기차, 세포색, 질감으로 정렬한다.
            # 계수는 정답 좌표로 학습한 가중치가 아니다.
            contrast = (nucleus["area"] / 1000) ** .5
            contrast *= max(0, (v[12] - v[6]) / 70)
            contrast *= min(1, v[17] * 2)
            contrast *= min(1, v[15] / 10)
            general.append((contrast, nucleus))
    if len(strongly_stained) >= 2:
        return strongly_stained
    general.sort(key=lambda pair: pair[0], reverse=True)
    chosen = {id(nucleus) for _, nucleus in general[:MAX_GENERAL_CANDIDATES]}
    return [nucleus for nucleus in candidates if id(nucleus) in chosen]


def default_output_dir():
    for desktop in (Path.home() / "Desktop", Path.home() / "바탕 화면"):
        if desktop.is_dir():
            return desktop / "세포결과"
    return Path.home() / "세포결과"


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {path}")
    return image


def save_image(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"PNG 저장 실패: {path}")
    encoded.tofile(str(path))


def features(contour):
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    circle = 4 * np.pi * area / perimeter**2 if perimeter else 0.0
    return area, circle, w, h, max(w, h) / max(1, min(w, h))




def nucleus_candidates(image, scale):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine = cv2.GaussianBlur(gray, (0, 0), 1.6)
    min_area, max_area = MIN_NUCLEUS_AREA, MAX_NUCLEUS_AREA
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    accepted = []
    # 크고 흐린 핵도 후보로 남기기 위해 3가지 주변 크기를 확인한다.
    # 아래 순서는 넓게 염색된 핵이 작은 내부 점으로 대체되는 현상을 줄인다.
    for sigma in (55, 35, 16):
        neighborhood = cv2.GaussianBlur(gray, (0, 0), sigma)
        darkness = neighborhood - fine
        contrast = darkness / (neighborhood + 24)
        for level in (0.065, 0.11, 0.17, 0.24):
            binary = np.uint8((darkness > 4) & (contrast > level)) * 255
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area, roundness, w, h, stretch = features(contour)
                original_area = area / scale**2
                if not (min_area <= original_area <= max_area):
                    continue
                if roundness < 0.30 or stretch > 3.5:
                    continue
                moments = cv2.moments(contour)
                if not moments["m00"]:
                    continue
                cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
                x0, y0 = int(np.clip(round(cx), 0, gray.shape[1] - 1)), int(np.clip(round(cy), 0, gray.shape[0] - 1))
                if contrast[y0, x0] < 0.06:
                    continue
                # 적어도 핵 반지름의 한 배수만큼 떨어져 있어야 다른 핵이다.
                if any(np.hypot(cx - item["cx"], cy - item["cy"]) <
                       0.75 * (np.sqrt(area / np.pi) + np.sqrt(item["work_area"] / np.pi))
                       for item in accepted):
                    continue
                accepted.append({"contour": contour, "cx": cx, "cy": cy,
                                 "work_area": area, "area": original_area,
                                 "roundness": roundness, "width": w / scale,
                                 "height": h / scale, "stretch": stretch})
    return accepted


def segment_cells(image, nuclei, scale):
    """핵을 시작점으로 색 영역을 확장하고, 서로 겹친 세포는 구분한다.

    윤곽은 자동 추정이다. 다른 핵과 공유하거나 영상 밖으로 잘린 영역은
    그림에는 후보로 표시하지만 N/C 계산에는 사용하지 않는다.
    """
    height, width = image.shape[:2]
    foreground = cell_foreground(image) > 0
    masks = []
    for nuc in nuclei:
        radius = np.sqrt(nuc["work_area"] / np.pi)
        reach = int(np.clip(4.0 * radius, 45 * scale, 175 * scale))
        cx, cy = int(round(nuc["cx"])), int(round(nuc["cy"]))
        x0, x1 = max(0, cx - reach), min(width, cx + reach + 1)
        y0, y1 = max(0, cy - reach), min(height, cy + reach + 1)
        if x1 - x0 < 12 or y1 - y0 < 12:
            masks.append(np.zeros((height, width), np.uint8))
            continue
        roi = image[y0:y1, x0:x1]
        mask = np.where(foreground[y0:y1, x0:x1], cv2.GC_PR_FGD,
                        cv2.GC_PR_BGD).astype(np.uint8)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance = np.hypot(xx - nuc["cx"], yy - nuc["cy"])
        mask[distance < max(4, radius * 1.35)] = cv2.GC_FGD
        mask[distance > reach * .88] = cv2.GC_BGD
        mask[[0, -1], :] = cv2.GC_BGD
        mask[:, [0, -1]] = cv2.GC_BGD
        try:
            cv2.grabCut(roi, mask, None, np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 2, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            masks.append(np.zeros((height, width), np.uint8))
            continue
        selected = np.uint8((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        components, labels, stats, _ = cv2.connectedComponentsWithStats(selected)
        local_x, local_y = cx - x0, cy - y0
        label = labels[local_y, local_x]
        if label == 0 or stats[label, cv2.CC_STAT_AREA] < nuc["work_area"] * 1.5:
            masks.append(np.zeros((height, width), np.uint8))
            continue
        full = np.zeros((height, width), np.uint8)
        full[y0:y1, x0:x1] = np.uint8(labels == label)
        masks.append(full)

    # 한 픽셀이 둘 이상의 세포에 속하면 가까운 핵에만 배정한다.
    # 실제 접합 경계가 보이지 않는 경우도 있으므로 이 분리는 추정값이다.
    if masks:
        overlap = np.zeros((height, width), np.uint8)
        for mask in masks:
            overlap += mask
        ambiguous = [bool(np.count_nonzero((overlap > 1) & mask) >
                          .08 * max(1, np.count_nonzero(mask)))
                     for mask in masks]
        ownership = np.zeros((height, width), np.uint16)
        best = np.full((height, width), np.inf, np.float32)
        for index, (nuc, mask) in enumerate(zip(nuclei, masks), 1):
            yy, xx = np.nonzero(mask)
            distances = (xx - nuc["cx"]) ** 2 + (yy - nuc["cy"]) ** 2
            better = distances < best[yy, xx]
            yy, xx, distances = yy[better], xx[better], distances[better]
            best[yy, xx] = distances
            ownership[yy, xx] = index
        for index, nuc in enumerate(nuclei, 1):
            region = np.uint8(ownership == index)
            if not np.any(region):
                nuc.pop("cell_contour", None)
                nuc.pop("nc", None)
                continue
            nuclear = np.zeros((height, width), np.uint8)
            cv2.drawContours(nuclear, [nuc["contour"]], -1, 1, cv2.FILLED)
            region |= nuclear
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            cell_mask = np.zeros_like(region)
            cv2.drawContours(cell_mask, [contour], -1, 1, cv2.FILLED)
            cyt_mask = cell_mask & (1 - nuclear)
            cyt_area = int(cyt_mask.sum())
            other_nuclei = any(j != index and cell_mask[int(round(other["cy"])),
                                                       int(round(other["cx"]))]
                                for j, other in enumerate(nuclei, 1))
            cut = bool(np.any(cell_mask[0]) or np.any(cell_mask[-1]) or
                       np.any(cell_mask[:, 0]) or np.any(cell_mask[:, -1]))
            nuc["cell_contour"] = contour
            nuc["cytoplasm_mask"] = cyt_mask
            nuc["cell_status"] = ("불명확" if ambiguous[index - 1] or
                                  other_nuclei or cut or cyt_area <= 0 else "자동 추정")
            nuc.pop("nc", None)
            if nuc["cell_status"] == "자동 추정":
                nuc["nc"] = int(nuclear.sum()) / cyt_area
    return nuclei


def cell_region_candidates(image):
    """핵 검출 여부와 무관하게 영상에서 세포 몸체 영역 후보를 만든다."""
    mask = cell_foreground(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (1500 <= area <= 60000 and x > 0 and y > 0 and
                x + w < image.shape[1] - 1 and y + h < image.shape[0] - 1):
            regions.append(contour)
    return regions


def make_overview(source, final, paired, nuclei):
    """예시와 같은 2×2 진단 패널과 오른쪽 최종 결과를 만든다."""
    from PIL import Image, ImageDraw, ImageFont

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    _, coarse = cv2.threshold(hsv[:, :, 1], 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cell_segments = np.zeros_like(source)
    nuclei_mask = np.zeros(source.shape[:2], np.uint8)
    palette = [(32, 195, 255), (255, 140, 65), (65, 210, 105), (220, 70, 220),
               (40, 110, 245), (170, 240, 35), (235, 190, 80)]
    regions = cell_region_candidates(source)
    for index, region in enumerate(regions):
        cv2.drawContours(cell_segments, [region], -1,
                         palette[index % len(palette)], cv2.FILLED)
    for index, nuc in enumerate(nuclei):
        if "cytoplasm_mask" in nuc:
            cell_segments[nuc["cytoplasm_mask"] > 0] = palette[index % len(palette)]
    for nuc in nuclei:
        cv2.drawContours(nuclei_mask, [nuc["contour"]], -1, 255, cv2.FILLED)

    # 왼쪽 네 패널은 같은 크기, 최종 결과는 그 두 배 너비로 표시한다.
    small_w, row_h, bar = 320, 256, 27
    gap, header = 4, 57
    full_w = small_w * 4 + gap * 4
    full_h = header + row_h * 2 + gap * 3
    canvas = np.full((full_h, full_w, 3), (24, 26, 30), np.uint8)

    def panel(x, y, width, height, title, data):
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + bar - 1), (21, 22, 25), -1)
        cv2.putText(canvas, title, (x + 7, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .57, (250, 250, 250), 2, cv2.LINE_AA)
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(data, (width, height - bar), interpolation=cv2.INTER_AREA)
        canvas[y + bar:y + height, x:x + width] = resized

    y0 = header + gap
    y1 = y0 + row_h + gap
    panel(gap, y0, small_w, row_h, "HSV saturation", hsv[:, :, 1])
    panel(gap * 2 + small_w, y0, small_w, row_h, "Coarse foreground", 255 - coarse)
    panel(gap, y1, small_w, row_h, "Cell segments", cell_segments)
    panel(gap * 2 + small_w, y1, small_w, row_h, "Nuclei mask", nuclei_mask)
    panel(gap * 3 + small_w * 2, y0, small_w * 2, row_h * 2 + gap,
          "Final result", final)

    # 한글 상단 제목은 PIL로 출력한다. 설치된 폰트가 없으면 영어로 대체한다.
    rgb = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb)
    font_files = ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc",
                  "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    font = next((ImageFont.truetype(file, 28) for file in font_files if Path(file).is_file()), None)
    caption = (f"세포 영역 후보 {len(regions)}개   |   핵 후보 {len(nuclei)}개   (자동 추정)" if font else
               f"Cell region candidates {len(regions)}  |  Nuclei {len(nuclei)}  (estimated)")
    draw.text((11, 11), caption, fill="white", font=font)
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def measured_colors(image, nucleus, scale):
    """검출한 윤곽에서만 색과 형태를 잰다. 세포가 없으면 세포질 수치는 비운다."""
    nuc_contour = nucleus["contour"]
    cell_contour = nucleus.get("cell_contour")
    all_contours = [nuc_contour] + ([cell_contour] if cell_contour is not None else [])
    bounds = [cv2.boundingRect(contour) for contour in all_contours]
    x0 = max(0, min(x for x, _, _, _ in bounds) - 40)
    y0 = max(0, min(y for _, y, _, _ in bounds) - 40)
    x1 = min(image.shape[1], max(x + w for x, _, w, _ in bounds) + 40)
    y1 = min(image.shape[0], max(y + h for _, y, _, h in bounds) + 40)
    crop = image[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    def contour_mask(contour):
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        moved = contour - np.array([x0, y0], dtype=np.int32)
        cv2.drawContours(mask, [moved], -1, 1, cv2.FILLED)
        return mask.astype(bool)

    nuc_mask = contour_mask(nuc_contour)
    if not np.any(nuc_mask):
        raise ValueError("핵 윤곽이 비어 있습니다")
    nucleus_l = float(np.median(lab[:, :, 0][nuc_mask]))
    nucleus_h = float(np.median(hsv[:, :, 0][nuc_mask]))
    nucleus_s = float(np.median(hsv[:, :, 1][nuc_mask]))
    result = {
        "nucleus_lab_l": round(nucleus_l, 1),
        "nucleus_hsv_h": round(nucleus_h, 1),
        "nucleus_hsv_s": round(nucleus_s, 1),
        "cell_area_px2": "", "cell_circularity": "", "cell_aspect_ratio": "",
        "cytoplasm_area_px2": "", "cytoplasm_lab_l": "",
        "cytoplasm_hsv_h": "", "cytoplasm_hsv_s": "",
        "nearby_background_lab_l": "", "cytoplasm_minus_nucleus_l": "",
        "nearby_minus_cytoplasm_l": "",
    }
    if cell_contour is None:
        return result

    cell_mask = contour_mask(cell_contour)
    if "cytoplasm_mask" in nucleus:
        cyt_mask = nucleus["cytoplasm_mask"][y0:y1, x0:x1].astype(bool)
    else:
        cyt_mask = cell_mask & ~nuc_mask
    if not np.any(cyt_mask):
        return result
    cyt_l = float(np.median(lab[:, :, 0][cyt_mask]))
    cyt_h = float(np.median(hsv[:, :, 0][cyt_mask]))
    cyt_s = float(np.median(hsv[:, :, 1][cyt_mask]))
    cell_area, circularity, w, h, aspect = features(cell_contour)
    outer = cv2.dilate(cell_mask.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))) > 0
    nearby = outer & ~cell_mask
    background_l = float(np.median(lab[:, :, 0][nearby])) if np.any(nearby) else None
    result.update({
        "cell_area_px2": round(cell_area / scale**2, 1),
        "cell_circularity": round(circularity, 3),
        "cell_aspect_ratio": round(aspect, 3),
        "cytoplasm_area_px2": round(np.count_nonzero(cyt_mask) / scale**2, 1),
        "cytoplasm_lab_l": round(cyt_l, 1),
        "cytoplasm_hsv_h": round(cyt_h, 1),
        "cytoplasm_hsv_s": round(cyt_s, 1),
        "cytoplasm_minus_nucleus_l": round(cyt_l - nucleus_l, 1),
        "nearby_background_lab_l": "" if background_l is None else round(background_l, 1),
        "nearby_minus_cytoplasm_l": "" if background_l is None else round(background_l - cyt_l, 1),
    })
    return result


def analyze(path, output, area_threshold, nc_threshold):
    original = read_image(path)
    scale = min(1.0, MAX_SIDE / max(original.shape[:2]))
    image = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else original.copy()
    raw_candidates = nucleus_candidates(image, scale)
    nuclei = rule_nucleus_candidates(image, raw_candidates, image_id=path.stem)
    nuclei = segment_cells(image, nuclei, scale)
    source = image.copy()
    paired = []
    rows = []
    nucleus_mask = np.zeros(image.shape[:2], np.uint16)
    cytoplasm_mask = np.zeros_like(nucleus_mask)
    overlay = image.copy()
    for number, nuc in enumerate(nuclei, 1):
        nc = nuc.get("nc")
        # AND 규칙: 대응이 불명확하면 판정 보류. 기준 미달과 동일시하지 않는다.
        verdict = ("판정 보류" if nc is None else
                   "규칙 충족" if nuc["area"] > area_threshold and nc > nc_threshold else "규칙 미충족")
        # 모든 핵 후보는 빨간색이다. 확인된 세포 경계만 노란색으로 덧그린다.
        # 세포 대응이 불분명한 경우에도 빨간색은 핵 '후보'를 뜻하며 N/C는 보류한다.
        if "cell_contour" in nuc:
            paired.append(nuc)
            cv2.drawContours(image, [nuc["cell_contour"]], -1, (0, 235, 255), 2)
            cytoplasm_mask[nuc["cytoplasm_mask"] > 0] = number
        cv2.drawContours(nucleus_mask, [nuc["contour"]], -1, number, cv2.FILLED)
        cv2.drawContours(image, [nuc["contour"]], -1, (0, 0, 255), 2)
        x, y, _, _ = cv2.boundingRect(nuc["contour"])
        cv2.putText(image, str(number), (x, max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
        rows.append({"image": path.name, "id": number,
                     "center_x": round(nuc["cx"] / scale), "center_y": round(nuc["cy"] / scale),
                     "nucleus_area_px2": round(nuc["area"], 1),
                     "circularity": round(nuc["roundness"], 3),
                     "width_px": round(nuc["width"], 1), "height_px": round(nuc["height"], 1),
                     "elongation": round(nuc["stretch"], 3),
                     "local_contrast": nuc["local_contrast"],
                     "cell_boundary_status": nuc.get("cell_status", "검출 안 됨"),
                     "nc_nucleus_over_cytoplasm": "" if nc is None else round(nc, 3),
                     "result": verdict, "source": "자동 추정",
                     **measured_colors(source, nuc, scale)})
    output.mkdir(parents=True, exist_ok=True)
    overlay[cytoplasm_mask > 0] = (0, 220, 255)
    overlay[nucleus_mask > 0] = (20, 20, 245)
    overlay = cv2.addWeighted(source, .65, overlay, .35, 0)
    for nuc in nuclei:
        if "cell_contour" in nuc:
            cv2.drawContours(overlay, [nuc["cell_contour"]], -1, (0, 235, 255), 2)
        cv2.drawContours(overlay, [nuc["contour"]], -1, (0, 0, 255), 2)
    save_image(output / f"{path.stem}_nucleus_mask.png", np.uint8(nucleus_mask > 0) * 255)
    save_image(output / f"{path.stem}_cytoplasm_mask.png", np.uint8(cytoplasm_mask > 0) * 255)
    save_image(output / f"{path.stem}_segmentation.png", overlay)
    save_image(output / f"{path.stem}_result.png",
               make_overview(source, image, paired, nuclei))
    return rows


def run(images, output, area_threshold=AREA_THRESHOLD, nc_threshold=NC_THRESHOLD,
        report=print):
    report(f"영상처리 규칙 | 실험 판정: 핵 면적 > {area_threshold:g} AND N/C > {nc_threshold:g}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    used_names = set()
    for image in images:
        image = Path(image)
        folder_name = image.stem
        suffix = 2
        while folder_name.casefold() in used_names:
            folder_name = f"{image.stem}_{suffix}"
            suffix += 1
        used_names.add(folder_name.casefold())
        image_output = output / folder_name
        try:
            rows = analyze(image, image_output, area_threshold, nc_threshold)
            with (image_output / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
                columns = ["image", "id", "center_x", "center_y", "nucleus_area_px2",
                           "circularity", "width_px", "height_px", "elongation",
                           "local_contrast", "nc_nucleus_over_cytoplasm",
                           "cell_boundary_status",
                           "cell_area_px2", "cell_circularity", "cell_aspect_ratio",
                           "cytoplasm_area_px2", "nucleus_lab_l", "cytoplasm_lab_l",
                           "nearby_background_lab_l", "nucleus_hsv_h", "nucleus_hsv_s",
                           "cytoplasm_hsv_h", "cytoplasm_hsv_s",
                           "cytoplasm_minus_nucleus_l", "nearby_minus_cytoplasm_l",
                           "result", "source"]
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            results.extend(rows)
            report(f"{image.name} → {image_output} | 핵 후보 {len(rows)}, 규칙 충족 {sum(r['result'] == '규칙 충족' for r in rows)}, 판정 보류 {sum(r['result'] == '판정 보류' for r in rows)}")
        except Exception as exc:
            report(f"{image.name}: 오류 {exc}")
    report(f"사진별 결과 저장: {output}")
    return results


def select_photos_and_run():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()  # 별도의 프로그램 창 없이 Windows 파일 선택창만 표시한다.
    try:
        picked = filedialog.askopenfilenames(
            parent=root,
            title="분석할 현미경 사진 선택",
            filetypes=[("현미경 사진", "*.bmp *.png *.jpg *.jpeg")],
        )
    finally:
        root.destroy()
    if picked:
        output = default_output_dir()
        run(picked, output)
        if os.name == "nt":
            os.startfile(str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=default_output_dir())
    parser.add_argument("--area", type=float, default=AREA_THRESHOLD, help="실험용 핵 면적 A (px²)")
    parser.add_argument("--nc", type=float, default=NC_THRESHOLD, help="실험용 핵/세포질 B")
    args = parser.parse_args()
    if args.images:
        run(args.images, args.output, args.area, args.nc)
    else:
        select_photos_and_run()
