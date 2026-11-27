# 독립 실행 버전 3: 붙은 핵을 먼저 분리한 후 세포 watershed 수행
import os

# 여러 작업자가 실행될 때 내부 스레드 증가를 제한합니다.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import time
from pathlib import Path
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

import cv2 as cv
import numpy as np
import pandas as pd

from scipy import ndimage as ndi
from skimage import (
    color,
    filters,
    measure,
    morphology,
    segmentation,
)


# ==================================================
# 1. 실행 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "test_image"
    / "im_Superficial-Intermediate"
)

REFERENCE_PATH = (
    BASE_DIR
    / "output"
    / "normal_reference_v3.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "fast_analysis"
)

# None이면 전체 이미지 처리
MAX_SAMPLES = None

# 기존 6작업자 유지. 최적값은 실측으로 비교하세요.
WORKERS = 6

SAVE_VISUALIZATION = True

# 분석은 원본 해상도로 수행합니다.
# 아래 크기는 저장용 결과 이미지에만 적용됩니다.
PANEL_WIDTH = 640
PANEL_HEIGHT = 480

PNG_COMPRESSION = 1


# ==================================================
# 2. 기존 분할 설정
# ==================================================

BACKGROUND_BORDER_RATIO = 0.05
COLOR_DISTANCE_MEDIAN_SIZE = 5

CELL_BODY_CLOSING_RADIUS = 5
MIN_CELL_BODY_AREA = 600

MIN_NUCLEUS_AREA = 80
MAX_NUCLEUS_AREA = 3000
MORPH_DISK_RADIUS = 2

NUCLEUS_RING_INNER_RADIUS = 3
NUCLEUS_RING_OUTER_RADIUS = 15

MIN_SURROUNDING_BODY_RATIO = 0.70


# ==================================================
# 3. 특징 및 결과 열 정의
# ==================================================

BASE_FEATURES = [
    "Nucleus Area (px)",
    "N/C Ratio",
    "Circularity",
]








# ==================================================
# 4. 파일 읽기 및 정상 기준 검증
# ==================================================


# 버전 3 특징/분리 설정
FEATURE_SETTINGS = {'version': 3, 'gray': 'OpenCV BGR2GRAY uint8, no filtering', 'nc': 'nucleus_area / cytoplasm_area', 'texture_levels': 32, 'texture_offsets_yx': [[0, 1], [1, 0], [1, 1], [1, -1]], 'texture_symmetric': True, 'texture_min_pairs': 32, 'perinuclear_inner_distance_px': [0, 5], 'perinuclear_outer_distance_px': [5, 15], 'perinuclear_min_pixels_each': 20, 'diameter': 'sqrt(4*nucleus_area/pi), pixels', 'border': 'mask touches first or last image row or column', 'nucleus_components': '8-connectivity', 'cell_type': 'Unknown unless verified external annotation is supplied'}
EXTRA_FEATURES = [
    'Nucleus Solidity','Nucleus Axis Ratio','Nucleus Mean Intensity',
    'Nucleus Intensity Std','Nucleus-Cytoplasm Intensity Difference',
    'Nucleus GLCM Contrast','Nucleus GLCM Homogeneity','Nucleus GLCM Energy',
    'Perinuclear Brightness Difference','Nucleus Equivalent Diameter (px)',
]
FEATURE_COLUMNS = [
    'Cell ID','Color Type','Cell Type','Cell Type Source',
    'Centroid (X, Y)','Centroid X','Centroid Y',
    'Cell Area (px)','Nucleus Area (px)','Cytoplasm Area (px)',
    'N/C Ratio','Circularity','Aspect Ratio',*EXTRA_FEATURES,
    'QC Valid','QC Reason','QC Review','QC Review Reasons',
    'Cell Touches Border','Nucleus Touches Border','Nucleus Component Count',
    'Nucleus Split From Cluster','Nucleus Seed Count In Source',
]
REFERENCE_COLUMNS = [
    'Reference Status','Reference Reasons','Reference Rule Count',
    'Observation Outliers','Observation Missing','Observation Outlier Count',
    'Segmentation QC Status','Segmentation QC Reasons',
    'Morphology Reference Status','Morphology Reference Reasons','Shape Flag Count',
]
DETAIL_COLUMNS = ['File Name','Image Width','Image Height',
                  'GT Normal Cell Count','GT Minus PASS Count',
                  *FEATURE_COLUMNS,*REFERENCE_COLUMNS]
RULES = {'TH_NUC_AREA':'>','TH_NC_RATIO':'>','TH_CIRCULARITY':'<',
         'TH_SOLIDITY':'<','TH_AXIS_RATIO':'>'}
SHAPE_FLAGS_REQUIRED = 2
COMBINATION = 'AREA_HIGH OR NC_HIGH OR SHAPE_FLAGS_AT_LEAST_2'
REFERENCE_IMAGE_DIR = BASE_DIR / 'reference_images'
NUCLEUS_DISTANCE_SIGMA = 1.0
NUCLEUS_PEAK_PROMINENCE = 1.5
MIN_NUCLEUS_SEED_AREA = 1
USE_EDGE_GUIDED_CELL_WATERSHED = False
CELL_EDGE_WEIGHT = 0.65

def read_image(path):
    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    return cv.imdecode(
        data,
        cv.IMREAD_COLOR,
    )


def write_image(path, image):
    success, encoded = cv.imencode(
        ".png",
        image,
        [
            cv.IMWRITE_PNG_COMPRESSION,
            PNG_COMPRESSION,
        ],
    )

    if not success:
        raise OSError(
            f"이미지 인코딩 실패: {path}"
        )

    encoded.tofile(str(path))




def build_gt_count_map(reference):
    """저장된 정답 목록에서 이미지별 정상 세포 수를 집계합니다."""
    cells_by_image = {}

    for item in reference.get(
        "reference_cells",
        [],
    ):
        name = os.path.basename(
            str(item["File Name"])
        ).casefold()

        cell_id = str(
            item["GT Cell ID"]
        )

        cells_by_image.setdefault(
            name,
            set(),
        ).add(cell_id)

    return {
        name: len(cell_ids)
        for name, cell_ids in cells_by_image.items()
    }


# ==================================================
# 5. 전처리
# ==================================================

def preprocess_image(image):
    gray = cv.cvtColor(
        image,
        cv.COLOR_BGR2GRAY,
    )

    rgb = cv.cvtColor(
        image,
        cv.COLOR_BGR2RGB,
    )

    lab = color.rgb2lab(rgb)

    height, width = gray.shape
    short_side = min(height, width)

    border_width = max(
        1,
        min(
            max(
                5,
                int(
                    short_side
                    * BACKGROUND_BORDER_RATIO
                ),
            ),
            max(
                1,
                short_side // 2,
            ),
        ),
    )

    border = np.zeros(
        gray.shape,
        dtype=bool,
    )

    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    background = np.median(
        lab[border],
        axis=0,
    )

    distance = np.linalg.norm(
        lab - background,
        axis=2,
    )

    distance = ndi.median_filter(
        distance,
        size=COLOR_DISTANCE_MEDIAN_SIZE,
    )

    if np.ptp(distance) <= 1e-6:
        body = np.zeros(
            gray.shape,
            dtype=bool,
        )

    else:
        threshold = filters.threshold_otsu(
            distance
        )

        body = distance > threshold

        body = morphology.binary_closing(
            body,
            morphology.disk(
                CELL_BODY_CLOSING_RADIUS
            ),
        )

        body = morphology.remove_small_objects(
            body,
            min_size=MIN_CELL_BODY_AREA,
        )

        body = ndi.binary_fill_holes(
            body
        )

    body_mask = (
        body.astype(np.uint8) * 255
    )

    hsv = cv.cvtColor(
        image,
        cv.COLOR_BGR2HSV,
    )

    pink1 = cv.inRange(
        hsv,
        np.array([0, 20, 40], np.uint8),
        np.array([20, 255, 255], np.uint8),
    )

    pink2 = cv.inRange(
        hsv,
        np.array([140, 20, 40], np.uint8),
        np.array([180, 255, 255], np.uint8),
    )

    pink = cv.bitwise_or(
        pink1,
        pink2,
    )

    blue = cv.inRange(
        hsv,
        np.array([80, 20, 40], np.uint8),
        np.array([135, 255, 255], np.uint8),
    )

    pink = cv.bitwise_and(
        pink,
        body_mask,
    )

    blue = cv.bitwise_and(
        blue,
        body_mask,
    )

    return {
        "gray": gray,
        "cell_body_mask": body_mask,
        "mask_pink": pink,
        "mask_blue": blue,
    }


# ==================================================
# 6. 핵 및 세포 분할
# ==================================================



# ==================================================
# 7. 질감 계산
# ==================================================

def calculate_texture(gray, mask):
    matrix = np.zeros(
        (32, 32),
        dtype=np.float64,
    )

    quantized = (
        gray.astype(np.int32)
        * 32
        // 256
    )

    height, width = mask.shape
    pair_count = 0

    for dy, dx in [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    ]:
        y0 = max(0, -dy)
        y1 = min(height, height - dy)

        x0 = max(0, -dx)
        x1 = min(width, width - dx)

        if y1 <= y0 or x1 <= x0:
            continue

        valid = (
            mask[y0:y1, x0:x1]
            & mask[
                y0 + dy:y1 + dy,
                x0 + dx:x1 + dx,
            ]
        )

        a = quantized[
            y0:y1,
            x0:x1,
        ][valid]

        b = quantized[
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx,
        ][valid]

        pair_count += len(a)

        np.add.at(
            matrix,
            (a, b),
            1,
        )

    if pair_count < 32:
        return [
            np.nan,
            np.nan,
            np.nan,
        ]

    matrix += matrix.T.copy()

    probability = (
        matrix / matrix.sum()
    )

    i, j = np.indices(
        probability.shape
    )

    contrast = float(
        np.sum(
            probability * (i - j) ** 2
        )
    )

    homogeneity = float(
        np.sum(
            probability
            / (1 + (i - j) ** 2)
        )
    )

    energy = float(
        np.sqrt(
            np.sum(
                probability ** 2
            )
        )
    )

    return [
        contrast,
        homogeneity,
        energy,
    ]


# ==================================================
# 8. 세포별 특징 추출
# ==================================================

def _extract_features_original(
    preprocessed,
    segmented,
):
    gray = preprocessed["gray"]

    nuclei = segmented["labeled_nuclei"]
    cells = segmented["segmented_cells"]

    cytoplasm = (
        segmented["cytoplasm_mask"] > 0
    )

    rows = []

    height, width = gray.shape

    for index, prop in enumerate(
        measure.regionprops(nuclei),
        start=1,
    ):
        minr, minc, maxr, maxc = prop.bbox

        nucleus_crop = (
            nuclei[minr:maxr, minc:maxc]
            == prop.label
        )

        overlapping_labels = cells[
            minr:maxr,
            minc:maxc,
        ][nucleus_crop]

        overlapping_labels = overlapping_labels[
            overlapping_labels > 0
        ]

        reasons = []
        nucleus_area = float(prop.area)

        if overlapping_labels.size:
            target_label = int(
                np.bincount(
                    overlapping_labels.astype(np.int64)
                ).argmax()
            )

            cell = cells == target_label

        else:
            cell = np.zeros(
                gray.shape,
                dtype=bool,
            )

            local_cell = cell[
                minr:maxr,
                minc:maxc,
            ]

            local_cell[nucleus_crop] = True

            reasons.append(
                "대응 세포 라벨 없음"
            )

        cyto = (
            cell
            & cytoplasm
            & (nuclei == 0)
        )

        cytoplasm_area = float(
            np.count_nonzero(cyto)
        )

        if cytoplasm_area <= 0:
            reasons.append(
                "세포질 영역 없음"
            )

        if not np.all(
            cell[minr:maxr, minc:maxc][nucleus_crop]
        ):
            reasons.append(
                "핵이 대응 세포 영역 밖에 존재"
            )

        perimeter = float(prop.perimeter)

        circularity = (
            float(
                4 * np.pi
                * nucleus_area
                / perimeter ** 2
            )
            if perimeter > 0
            else np.nan
        )

        axis_ratio = (
            float(
                prop.axis_major_length
                / prop.axis_minor_length
            )
            if prop.axis_minor_length > 0
            else np.nan
        )

        nucleus_gray = gray[
            minr:maxr,
            minc:maxc,
        ]

        pixels = nucleus_gray[
            nucleus_crop
        ].astype(float)

        mean_intensity = float(
            pixels.mean()
        )

        std_intensity = float(
            pixels.std(ddof=0)
        )

        intensity_difference = (
            float(
                gray[cyto].mean()
                - mean_intensity
            )
            if cytoplasm_area > 0
            else np.nan
        )

        texture = calculate_texture(
            nucleus_gray,
            nucleus_crop,
        )

        y0 = max(0, minr - 16)
        x0 = max(0, minc - 16)

        y1 = min(height, maxr + 16)
        x1 = min(width, maxc + 16)

        local_nucleus = (
            nuclei[y0:y1, x0:x1]
            == prop.label
        )

        local_cyto = cyto[
            y0:y1,
            x0:x1,
        ]

        distance = ndi.distance_transform_edt(
            ~local_nucleus
        )

        inner = (
            local_cyto
            & (distance > 0)
            & (distance <= 5)
        )

        outer = (
            local_cyto
            & (distance > 5)
            & (distance <= 15)
        )

        perinuclear_difference = np.nan

        if (
            np.count_nonzero(inner) >= 20
            and np.count_nonzero(outer) >= 20
        ):
            local_gray = gray[
                y0:y1,
                x0:x1,
            ]

            perinuclear_difference = float(
                local_gray[inner].mean()
                - local_gray[outer].mean()
            )

        pink_count = int(
            np.count_nonzero(
                preprocessed["mask_pink"][cell]
            )
        )

        blue_count = int(
            np.count_nonzero(
                preprocessed["mask_blue"][cell]
            )
        )

        if pink_count + blue_count == 0:
            color_type = "Unknown"

        elif blue_count > pink_count:
            color_type = "Blue (Intermediate)"

        else:
            color_type = "Pink (Superficial)"

        cy, cx = prop.centroid

        rows.append({
            "Cell ID": index,
            "Color Type": color_type,
            "Centroid (X, Y)": (
                int(cx),
                int(cy),
            ),
            "Centroid X": float(cx),
            "Centroid Y": float(cy),
            "Cell Area (px)": float(
                np.count_nonzero(cell)
            ),
            "Nucleus Area (px)": nucleus_area,
            "Cytoplasm Area (px)": cytoplasm_area,
            "N/C Ratio": (
                nucleus_area / cytoplasm_area
                if cytoplasm_area > 0
                else np.nan
            ),
            "Circularity": circularity,
            "Aspect Ratio": float(
                (maxc - minc)
                / (maxr - minr)
            ),
            "Nucleus Solidity": float(
                prop.solidity
            ),
            "Nucleus Axis Ratio": axis_ratio,
            "Nucleus Mean Intensity": mean_intensity,
            "Nucleus Intensity Std": std_intensity,
            "Nucleus-Cytoplasm Intensity Difference": (
                intensity_difference
            ),
            "Nucleus GLCM Contrast": texture[0],
            "Nucleus GLCM Homogeneity": texture[1],
            "Nucleus GLCM Energy": texture[2],
            "Perinuclear Brightness Difference": (
                perinuclear_difference
            ),
            "QC Valid": not reasons,
            "QC Reason": (
                "; ".join(reasons)
                if reasons
                else "None"
            ),
        })

    return rows


# ==================================================
# 9. 정상 기준 비교
# ==================================================

def finite_number(value):
    if (
        value is None
        or isinstance(
            value,
            (bool, np.bool_),
        )
    ):
        return None

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None

    if not math.isfinite(number):
        return None

    return number




def summarize_features(rows):
    result = {
        "Measured Nucleus Count": len(rows),
        "Nucleus Area CV": np.nan,
        "Mean Nearest Nucleus Distance (px)": np.nan,
    }

    if len(rows) < 2:
        return result

    areas = np.array(
        [
            row["Nucleus Area (px)"]
            for row in rows
        ],
        dtype=float,
    )

    if areas.mean() > 0:
        result["Nucleus Area CV"] = float(
            areas.std(ddof=0)
            / areas.mean()
        )

    coordinates = np.array(
        [
            [
                row["Centroid X"],
                row["Centroid Y"],
            ]
            for row in rows
        ],
        dtype=float,
    )

    differences = (
        coordinates[:, None, :]
        - coordinates[None, :, :]
    )

    distances = np.sqrt(
        np.sum(
            differences ** 2,
            axis=2,
        )
    )

    np.fill_diagonal(
        distances,
        np.inf,
    )

    result[
        "Mean Nearest Nucleus Distance (px)"
    ] = float(
        distances.min(axis=1).mean()
    )

    return result


# ==================================================
# 10. 결과 이미지 생성
# ==================================================



def make_panel(
    image,
    title,
    nearest=False,
):
    if image.ndim == 2:
        image = cv.cvtColor(
            image,
            cv.COLOR_GRAY2BGR,
        )

    height, width = image.shape[:2]

    scale = min(
        PANEL_WIDTH / width,
        (PANEL_HEIGHT - 35) / height,
    )

    resized = cv.resize(
        image,
        (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        ),
        interpolation=(
            cv.INTER_NEAREST
            if nearest
            else cv.INTER_AREA
        ),
    )

    panel = np.full(
        (
            PANEL_HEIGHT,
            PANEL_WIDTH,
            3,
        ),
        30,
        dtype=np.uint8,
    )

    resized_height, resized_width = resized.shape[:2]

    x0 = (
        PANEL_WIDTH - resized_width
    ) // 2

    y0 = 35 + (
        PANEL_HEIGHT - 35 - resized_height
    ) // 2

    panel[
        y0:y0 + resized_height,
        x0:x0 + resized_width,
    ] = resized

    draw_text(
        panel,
        title,
        (12, 24),
        scale=0.58,
    )

    return panel




# ==================================================
# 11. 이미지 한 장 처리
# ==================================================

def make_initial_summary(
    image_name,
    gt_count,
):
    return {
        "File Name": image_name,
        "Image Width": None,
        "Image Height": None,
        "GT Normal Cell Count": gt_count,
        "GT Minus PASS Count": None,
        "Nucleus Count (세포핵 개수)": None,
        "Cell Count (세포 개수)": None,
        "Normal Range Count": 0,
        "Review Required Count": 0,
        "Invalid Count": 0,
        "Cells With Observation Flags": 0,
        "Image Status": "",
        "Image Reason": "",
        "Measured Nucleus Count": None,
        "Nucleus Area CV": np.nan,
        "Mean Nearest Nucleus Distance (px)": np.nan,
        "Read Time (s)": np.nan,
        "Preprocess Time (s)": np.nan,
        "Segmentation Time (s)": np.nan,
        "Feature Time (s)": np.nan,
        "Check Time (s)": np.nan,
        "Visualization Time (s)": 0.0,
        "Total Time (s)": np.nan,
        "Visualization Error": "",
    }


def process_one(
    path_string,
    reference,
    gt_count,
):
    cv.setNumThreads(1)

    start = time.perf_counter()

    path = Path(path_string)
    image_name = path.name

    summary = make_initial_summary(
        image_name,
        gt_count,
    )

    details = []

    try:
        tick = time.perf_counter()

        image = read_image(path)

        summary["Read Time (s)"] = (
            time.perf_counter() - tick
        )

        if image is None:
            raise ValueError(
                "이미지를 읽지 못했습니다."
            )

        height, width = image.shape[:2]

        summary["Image Width"] = width
        summary["Image Height"] = height

        tick = time.perf_counter()

        preprocessed = preprocess_image(
            image
        )

        summary["Preprocess Time (s)"] = (
            time.perf_counter() - tick
        )

        tick = time.perf_counter()

        segmented = segment_image(
            preprocessed
        )

        summary["Segmentation Time (s)"] = (
            time.perf_counter() - tick
        )
        summary.update(segmented.get('diagnostics', {}))


        tick = time.perf_counter()

        rows = extract_features(
            preprocessed,
            segmented,
        )

        summary["Feature Time (s)"] = (
            time.perf_counter() - tick
        )

        tick = time.perf_counter()

        rows = [
            {
                **row,
                **check_cell(
                    row,
                    reference,
                    image.shape,
                ),
            }
            for row in rows
        ]

        summary["Check Time (s)"] = (
            time.perf_counter() - tick
        )

        detected_count = len(rows)

        summary[
            "Nucleus Count (세포핵 개수)"
        ] = detected_count

        summary[
            "Cell Count (세포 개수)"
        ] = detected_count

        for status, column in [
            (
                "정상 범위",
                "Normal Range Count",
            ),
            (
                "검토 필요",
                "Review Required Count",
            ),
            (
                "분석 불가",
                "Invalid Count",
            ),
        ]:
            summary[column] = sum(
                row["Reference Status"] == status
                for row in rows
            )

        # 변경된 차이 계산:
        # 이미지별 정답 정상 세포 수 - PASS 개수
        if gt_count is not None:
            summary["GT Minus PASS Count"] = (
                gt_count
                - summary["Normal Range Count"]
            )

        summary[
            "Cells With Observation Flags"
        ] = sum(
            (
                row["Observation Outlier Count"]
                or 0
            ) > 0
            for row in rows
        )

        summary.update(
            summarize_features(rows)
        )
        summary['QC Review Count'] = sum(row['Segmentation QC Status']=='REVIEW' for row in rows)
        summary['Morphology Outside Count'] = sum(row['Morphology Reference Status']=='Outside reference' for row in rows)


        if (
            width != reference["image_width"]
            or height != reference["image_height"]
        ):
            summary["Image Status"] = (
                "Incompatible image size"
            )
            summary["Image Reason"] = (
                "정상 기준 영상과 크기가 다릅니다."
            )

        elif detected_count == 0:
            summary["Image Status"] = (
                "No cells detected"
            )
            summary["Image Reason"] = (
                "정상 여부 판단 불가"
            )

        elif summary["Invalid Count"] > 0:
            summary["Image Status"] = (
                "Contains invalid cells"
            )

        elif summary["Review Required Count"] > 0:
            summary["Image Status"] = (
                "Review required"
            )

        else:
            summary["Image Status"] = (
                "Within reference"
            )

        for row in rows:
            details.append({
                **row,
                "File Name": image_name,
                "Image Width": width,
                "Image Height": height,
                "GT Normal Cell Count": gt_count,
                "GT Minus PASS Count": summary[
                    "GT Minus PASS Count"
                ],
            })

        if SAVE_VISUALIZATION:
            tick = time.perf_counter()

            output_path = (
                OUTPUT_DIR
                / f"multicell_result_{path.stem}.png"
            )

            try:
                metadata = save_visualization(
                    image,
                    preprocessed,
                    segmented,
                    rows,
                    summary,
                    output_path,
                )
                summary.update(metadata)

            except Exception as error:
                summary[
                    "Visualization Error"
                ] = str(error)

            summary[
                "Visualization Time (s)"
            ] = (
                time.perf_counter() - tick
            )

    except Exception as error:
        summary["Image Status"] = (
            "Processing error"
        )
        summary["Image Reason"] = str(error)

        # 처리 실패 시 차이를 유효한 결과로 표시하지 않습니다.
        summary["GT Minus PASS Count"] = None

        for item in details:
            item["GT Minus PASS Count"] = None

    summary["Total Time (s)"] = (
        time.perf_counter() - start
    )

    return summary, details


# ==================================================
# 12. 결과 출력
# ==================================================

def print_result(
    index,
    total,
    summary,
):
    gt_count = summary[
        "GT Normal Cell Count"
    ]

    difference = summary[
        "GT Minus PASS Count"
    ]

    detected = summary[
        "Cell Count (세포 개수)"
    ]

    gt_text = (
        f"{gt_count}개"
        if gt_count is not None
        else "정답 목록에 없음"
    )

    difference_text = (
        f"{difference:+d}개"
        if difference is not None
        else "비교 불가"
    )

    detected_text = (
        f"{detected}개"
        if detected is not None
        else "계산 실패"
    )

    print(
        f"\n[{index}/{total}] "
        f"{summary['File Name']}\n"
        f"  정답 정상 세포: {gt_text}\n"
        f"  자동 검출 후보: {detected_text}\n"
        f"  PASS로 판단된 세포: "
        f"{summary['Normal Range Count']}개\n"
        f"  차이(정답 정상 세포 - PASS): "
        f"{difference_text}\n"
        f"  검토 필요 "
        f"{summary['Review Required Count']}개 / "
        f"분석 불가 "
        f"{summary['Invalid Count']}개\n"
        f"  보조 특징 이탈 세포: "
        f"{summary['Cells With Observation Flags']}개\n"
        f"  처리 시간: "
        f"{summary['Total Time (s)']:.2f}초"
    )

    if summary["Image Reason"]:
        print(
            "  참고: "
            f"{summary['Image Reason']}"
        )

    if summary["Visualization Error"]:
        print(
            "  이미지 저장 오류: "
            f"{summary['Visualization Error']}"
        )


# ==================================================
# 13. CSV 저장
# ==================================================

def save_results(
    summaries,
    details,
):
    summaries = sorted(
        summaries,
        key=lambda row: (
            row["File Name"].casefold()
        ),
    )

    details = sorted(
        details,
        key=lambda row: (
            row["File Name"].casefold(),
            row["Cell ID"],
        ),
    )

    df_summary = pd.DataFrame(
        summaries
    )

    df_details = pd.DataFrame(
        details,
        columns=DETAIL_COLUMNS,
    )

    for dataframe in [
        df_summary,
        df_details,
    ]:
        for column in [
            "GT Normal Cell Count",
            "GT Minus PASS Count",
        ]:
            dataframe[column] = pd.array(
                dataframe[column],
                dtype="Int64",
            )

    summary_path = (
        OUTPUT_DIR
        / "all_images_summary.csv"
    )

    detail_path = (
        OUTPUT_DIR
        / "all_cells_detail.csv"
    )

    df_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    df_details.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    return df_summary


# ==================================================
# 14. 실행
# ==================================================

def main():
    if (
        type(WORKERS) is not int
        or WORKERS < 1
    ):
        raise ValueError(
            "WORKERS는 1 이상의 정수여야 합니다."
        )

    if MAX_SAMPLES is not None:
        if (
            type(MAX_SAMPLES) is not int
            or MAX_SAMPLES < 1
        ):
            raise ValueError(
                "MAX_SAMPLES는 None 또는 "
                "1 이상의 정수여야 합니다."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATA_DIR.is_dir():
        print(
            f"데이터 폴더가 없습니다: {DATA_DIR}"
        )
        return

    try:
        reference = load_reference(
            REFERENCE_PATH
        )

    except (OSError, ValueError) as error:
        print(
            f"[정상 기준 불러오기 실패] {error}"
        )
        return

    gt_count_map = build_gt_count_map(
        reference
    )

    paths = sorted(
        [
            path
            for path in DATA_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".bmp"
        ],
        key=lambda path: (
            path.name.casefold()
        ),
    )

    names = [
        path.name.casefold()
        for path in paths
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            "대소문자만 다른 이미지 파일명이 중복됩니다."
        )

    if MAX_SAMPLES is not None:
        paths = paths[:MAX_SAMPLES]

    if not paths:
        print(
            "분석할 BMP 이미지가 없습니다."
        )
        return

    worker_count = min(
        WORKERS,
        len(paths),
    )

    print("==================================================")
    print("          단일 파일 고속 세포 영상 분석")
    print("==================================================")
    print(
        f"입력 폴더: {DATA_DIR}"
    )
    print(
        f"출력 폴더: {OUTPUT_DIR}"
    )
    print(
        f"분석 이미지: {len(paths)}개"
    )
    print(
        f"작업자 수: {worker_count}"
    )
    print(
        f"결과 이미지 저장: {SAVE_VISUALIZATION}"
    )
    print(
        "차이 계산: 이미지별 정답 정상 세포 수 - PASS 개수"
    )
    print(
        "현재 기준은 검증 전 정상 참조 기준입니다."
    )

    nucleus_limit = reference[
        "candidate_thresholds"
    ]["TH_NUC_AREA"]

    if MAX_NUCLEUS_AREA < nucleus_limit:
        print(
            "\n[주의] 핵 검출 면적 상한이 "
            "정상 기준의 핵 면적 임계값보다 작습니다."
        )
        print(
            "기존 설정을 보존했으며 "
            "상한을 넘는 핵은 특징 검사 전에 제외됩니다."
        )

    start = time.perf_counter()

    summaries = []
    details = []

    if worker_count == 1:
        for index, path in enumerate(
            paths,
            start=1,
        ):
            gt_count = gt_count_map.get(
                path.name.casefold()
            )

            summary, cell_details = process_one(
                str(path),
                reference,
                gt_count,
            )

            summaries.append(summary)
            details.extend(cell_details)

            print_result(
                index,
                len(paths),
                summary,
            )

    else:
        with ProcessPoolExecutor(
            max_workers=worker_count
        ) as executor:
            futures = {
                executor.submit(
                    process_one,
                    str(path),
                    reference,
                    gt_count_map.get(
                        path.name.casefold()
                    ),
                ): path
                for path in paths
            }

            for index, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                path = futures[future]

                try:
                    summary, cell_details = (
                        future.result()
                    )

                except Exception as error:
                    summary = make_initial_summary(
                        path.name,
                        gt_count_map.get(
                            path.name.casefold()
                        ),
                    )

                    summary["Image Status"] = (
                        "Worker error"
                    )
                    summary["Image Reason"] = (
                        str(error)
                    )

                    cell_details = []

                summaries.append(summary)
                details.extend(cell_details)

                print_result(
                    index,
                    len(paths),
                    summary,
                )

    df_summary = save_results(
        summaries,
        details,
    )

    elapsed = (
        time.perf_counter() - start
    )

    print("\n==================================================")
    print("                  전체 처리 완료")
    print("==================================================")
    print(
        f"전체 경과 시간: {elapsed:.2f}초"
    )
    print(
        f"이미지 처리 기록: {len(df_summary)}개"
    )
    print(
        f"결과 폴더: {OUTPUT_DIR}"
    )

    for label, column in [
        (
            "정상 범위(PASS)",
            "Normal Range Count",
        ),
        (
            "검토 필요",
            "Review Required Count",
        ),
        (
            "분석 불가",
            "Invalid Count",
        ),
    ]:
        print(
            f"{label}: "
            f"{int(df_summary[column].sum())}개"
        )

    # 정답 개수와 PASS 차이가 계산된 이미지만 합산합니다.
    comparable = df_summary[
        df_summary[
            "GT Normal Cell Count"
        ].notna()
        & df_summary[
            "GT Minus PASS Count"
        ].notna()
    ]

    if not comparable.empty:
        total_gt = int(
            comparable[
                "GT Normal Cell Count"
            ].sum()
        )

        total_pass = int(
            comparable[
                "Normal Range Count"
            ].sum()
        )

        print("\n[정답과 PASS 개수 비교]")
        print(
            f"비교 가능한 이미지: "
            f"{len(comparable)}개"
        )
        print(
            f"해당 이미지의 정답 정상 세포 합계: "
            f"{total_gt}개"
        )
        print(
            f"해당 이미지의 PASS 합계: "
            f"{total_pass}개"
        )
        print(
            f"차이 합계(정답 - PASS): "
            f"{total_gt - total_pass:+d}개"
        )

    print()
    print(
        "정답 개수는 저장된 이미지별 정답 목록을 사용합니다."
    )
    print(
        "정답 목록에 없는 이미지는 개수 차이를 계산하지 않습니다."
    )
    print(
        "정답 - PASS 차이는 개수 차이이며, "
        "오분류 세포 수나 검출 정확도가 아닙니다."
    )
    print(
        "기존 버전과의 결과 일치 여부는 별도 비교가 필요합니다."
    )



# ==================================================
# 버전 3 + 붙은 핵 분리 개선
# 아래 정의가 실행 시 사용됩니다.
# ==================================================

# 픽셀 단위의 탐색 설정입니다. GT를 보고 조정해야 합니다.
# False는 기존 몸통 거리 기반 세포 watershed를 유지합니다.
# 핵 분리 개선 효과부터 비교하도록 기본값을 False로 둡니다.



def segment_image(preprocessed):
    gray=preprocessed['gray']
    body=preprocessed['cell_body_mask']>0
    body_labels=measure.label(body)
    nuclei=np.zeros(gray.shape,dtype=np.int32)
    next_label=1
    split_components=0
    rejected_small=rejected_large=rejected_ring=0
    seed_counts={}
    split_labels=set()
    selem=morphology.disk(MORPH_DISK_RADIUS)
    inner_disk=morphology.disk(NUCLEUS_RING_INNER_RADIUS)
    outer_disk=morphology.disk(NUCLEUS_RING_OUTER_RADIUS)
    height,width=gray.shape

    for bp in measure.regionprops(body_labels):
        # 몸통별 ROI에서 핵 후보를 구합니다.
        r0,c0,r1,c1=bp.bbox
        local_body=body_labels[r0:r1,c0:c1]==bp.label
        local_gray=gray[r0:r1,c0:c1]
        pixels=local_gray[local_body]
        if pixels.size<MIN_NUCLEUS_AREA or np.ptp(pixels)==0:
            continue
        threshold=filters.threshold_otsu(pixels)
        dark=pixels[pixels<threshold]
        if dark.size<2 or np.ptp(dark)==0:
            continue
        nuclear_threshold=filters.threshold_otsu(dark)
        candidates=(local_gray<nuclear_threshold)&local_body
        candidates=morphology.opening(candidates,selem)
        candidates=morphology.closing(candidates,selem)
        candidates=ndi.binary_fill_holes(candidates)&local_body
        candidate_labels=measure.label(candidates)
        for prop in measure.regionprops(candidate_labels):
            if prop.area<MIN_NUCLEUS_AREA:
                rejected_small+=1
                continue
            # 합쳐진 큰 덩어리도 분리 전에는 면적 상한으로 버리지 않습니다.
            a,b,c,d=prop.bbox
            # 거리 변환을 위해 사방에 배경 1픽셀을 둡니다.
            component=np.pad(candidate_labels[a:c,b:d]==prop.label,1)
            distance=ndi.distance_transform_edt(component)
            smooth=ndi.gaussian_filter(distance,NUCLEUS_DISTANCE_SIGMA)
            smooth[~component]=0
            maxima=morphology.h_maxima(smooth,NUCLEUS_PEAK_PROMINENCE).astype(bool)&component
            maxima=morphology.remove_small_objects(maxima,min_size=MIN_NUCLEUS_SEED_AREA)
            markers,seed_count=ndi.label(maxima,structure=np.ones((3,3),np.uint8))
            if seed_count==0:
                markers=np.zeros(component.shape,dtype=np.int32)
                markers[np.unravel_index(np.argmax(distance),distance.shape)]=1
                seed_count=1
            if seed_count>1:
                split_components+=1
            # watershed_line=False: 분리 경계 픽셀을 버리지 않습니다.
            pieces=segmentation.watershed(-smooth,markers,mask=component)
            pieces=pieces[1:-1,1:-1]
            for label in np.unique(pieces):
                if label==0:
                    continue
                local_piece=pieces==label
                area=int(local_piece.sum())
                if area<MIN_NUCLEUS_AREA:
                    rejected_small+=1
                    continue
                if area>MAX_NUCLEUS_AREA:
                    rejected_large+=1
                    continue
                yy,xx=np.nonzero(local_piece)
                yy=yy+r0+a
                xx=xx+c0+b
                y0=max(0,int(yy.min())-NUCLEUS_RING_OUTER_RADIUS)
                x0=max(0,int(xx.min())-NUCLEUS_RING_OUTER_RADIUS)
                y1=min(height,int(yy.max())+NUCLEUS_RING_OUTER_RADIUS+1)
                x1=min(width,int(xx.max())+NUCLEUS_RING_OUTER_RADIUS+1)
                patch=np.zeros((y1-y0,x1-x0),dtype=bool)
                patch[yy-y0,xx-x0]=True
                ring=(ndi.binary_dilation(patch,structure=outer_disk)
                      & ~ndi.binary_dilation(patch,structure=inner_disk))
                surrounding=body_labels[y0:y1,x0:x1]==bp.label
                if not ring.any() or np.mean(surrounding[ring])<MIN_SURROUNDING_BODY_RATIO:
                    rejected_ring+=1
                    continue
                nuclei[yy,xx]=next_label
                seed_counts[next_label]=int(seed_count)
                if seed_count>1:
                    split_labels.add(next_label)
                next_label+=1

    # 중요: 이진 핵 마스크를 다시 label()하면 붙은 핵이 합쳐질 수 있습니다.
    # 위에서 부여한 서로 다른 정수 마커를 그대로 보존합니다.
    nuclei_mask=(nuclei>0).astype(np.uint8)*255
    if next_label==1:
        cells=np.zeros(gray.shape,dtype=np.int32)
    else:
        seeded=np.unique(body_labels[nuclei>0])
        valid=body&np.isin(body_labels,seeded[seeded>0])
        distance=ndi.distance_transform_edt(valid)
        if USE_EDGE_GUIDED_CELL_WATERSHED:
            # 실험 옵션: 색·밝기 경계가 실제 세포 경계와 같은지 GT 검증 필요
            gradient=filters.sobel(ndi.gaussian_filter(gray.astype(float)/255.,1.0))
            gmax=float(gradient[valid].max()) if valid.any() else 0.
            gradient=gradient/gmax if gmax>0 else gradient
            dmax=float(distance.max())
            normalized=distance/dmax if dmax>0 else distance
            elevation=CELL_EDGE_WEIGHT*gradient-(1-CELL_EDGE_WEIGHT)*normalized
        else:
            elevation=-distance
        cells=segmentation.watershed(elevation,markers=nuclei,mask=valid).astype(np.int32)
    cytoplasm=(cells>0)&(nuclei==0)
    return {
        'nuclei_mask':nuclei_mask,'labeled_nuclei':nuclei,
        'segmented_cells':cells,'cytoplasm_mask':cytoplasm.astype(np.uint8)*255,
        'split_labels':split_labels,'seed_counts':seed_counts,
        'diagnostics':{
            'Split Source Component Count':split_components,
            'Rejected Small Nucleus Count':rejected_small,
            'Rejected Large Nucleus Count':rejected_large,
            'Rejected Ring Nucleus Count':rejected_ring,
            'Cell Watershed Mode':'edge-guided' if USE_EDGE_GUIDED_CELL_WATERSHED else 'body-distance',
        },
    }


def _touches_border(mask):
    return bool(mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any())


def extract_features(preprocessed,segmented):
    rows=_extract_features_original(preprocessed,segmented)
    nuclei,cells=segmented['labeled_nuclei'],segmented['segmented_cells']
    for row,prop in zip(rows,measure.regionprops(nuclei)):
        nucleus=nuclei==prop.label
        overlap=cells[nucleus]
        overlap=overlap[overlap>0]
        cell=cells==int(np.bincount(overlap.astype(np.int64)).argmax()) if overlap.size else nucleus.copy()
        _,components=ndi.label(nucleus,structure=np.ones((3,3),np.uint8))
        reasons=[] if row['QC Valid'] else [row['QC Reason']]
        if components!=1:
            reasons.append('핵 마스크가 여러 연결 영역으로 나뉨')
        for name in [*BASE_FEATURES,'Nucleus Solidity','Nucleus Axis Ratio']:
            if not np.isfinite(row[name]):
                reasons.append(f'형태 특징 계산 불가: {name}')
        cb,nb=_touches_border(cell),_touches_border(nucleus)
        reviews=[]
        if cb:
            reviews.append('세포가 영상 가장자리에 접촉: 잘림 여부 확인')
        if nb:
            reviews.append('핵이 영상 가장자리에 접촉: 잘림 여부 확인')
        row.update({
            'Cell Type':'Unknown','Cell Type Source':'Not annotated',
            'Nucleus Equivalent Diameter (px)':float(np.sqrt(4*row['Nucleus Area (px)']/np.pi)),
            'QC Valid':not reasons,'QC Reason':'; '.join(reasons) or 'None',
            'QC Review':bool(reviews),'QC Review Reasons':'; '.join(reviews) or 'None',
            'Cell Touches Border':cb,'Nucleus Touches Border':nb,'Nucleus Component Count':int(components),
            'Nucleus Split From Cluster':prop.label in segmented['split_labels'],
            'Nucleus Seed Count In Source':segmented['seed_counts'].get(prop.label,1),
        })
    return rows


def load_reference(path):
    with open(path,encoding='utf-8') as file:
        r=json.load(file)
    if not isinstance(r,dict) or r.get('schema_version')!=3:
        raise ValueError('버전 3 정상 참조 JSON이 필요합니다.')
    if r.get('feature_settings')!=FEATURE_SETTINGS:
        raise ValueError('정상 참조와 특징 설정이 다릅니다.')
    if r.get('rules')!=RULES or r.get('combination')!=COMBINATION or r.get('shape_flags_required')!=SHAPE_FLAGS_REQUIRED:
        raise ValueError('판정 규칙이 다릅니다.')
    if r.get('nc_ratio_definition')!='nucleus_area / cytoplasm_area':
        raise ValueError('N/C 정의가 다릅니다.')
    for key in ['image_width','image_height']:
        if type(r.get(key)) is not int or r[key]<=0:
            raise ValueError('기준 영상 크기 오류')
    for key in RULES:
        v=r.get('candidate_thresholds',{}).get(key)
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:
            raise ValueError(f'임계값 오류: {key}')
    if r['candidate_thresholds']['TH_NUC_AREA']<=0:
        raise ValueError('핵 면적 임계값 오류')
    if not isinstance(r.get('statistics'),dict):
        raise ValueError('특징 통계가 없습니다.')
    for name in [*BASE_FEATURES,*EXTRA_FEATURES]:
        st=r['statistics'].get(name)
        if not isinstance(st,dict) or type(st.get('count')) is not int or st['count']<0:
            raise ValueError(f'특징 통계 오류: {name}')
        if st['count']>=2:
            lo,hi=st.get('lower_quantile'),st.get('upper_quantile')
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in [lo,hi]) or lo>hi:
                raise ValueError(f'특징 범위 오류: {name}')
    return r


def check_cell(cell,reference,image_shape):
    invalid=[]
    h,w=image_shape[:2]
    if (w,h)!=(reference['image_width'],reference['image_height']):
        invalid.append('영상 크기 불일치')
    if not bool(cell.get('QC Valid',False)):
        invalid.append(str(cell.get('QC Reason','QC 오류')))
    values={n:finite_number(cell.get(n)) for n in [*BASE_FEATURES,'Nucleus Solidity','Nucleus Axis Ratio']}
    if any(v is None or v<0 for v in values.values()) or (values['Nucleus Area (px)'] or 0)<=0:
        invalid.append('핵심 형태 특징의 결측 또는 범위 오류')
    if invalid:
        return {'Reference Status':'분석 불가','Reference Reasons':'; '.join(invalid),
                'Reference Rule Count':None,'Observation Outliers':'None','Observation Missing':'None',
                'Observation Outlier Count':None,'Segmentation QC Status':'INVALID',
                'Segmentation QC Reasons':'; '.join(invalid),'Morphology Reference Status':'Not assessed',
                'Morphology Reference Reasons':'QC 또는 특징 오류','Shape Flag Count':None}
    t=reference['candidate_thresholds']
    shape=[]
    for name,key,op in [('Circularity','TH_CIRCULARITY','<'),('Nucleus Solidity','TH_SOLIDITY','<'),('Nucleus Axis Ratio','TH_AXIS_RATIO','>')]:
        v=values[name]
        if (v<t[key] if op=='<' else v>t[key]):
            shape.append(f'{name}: {v:.6f} {op} {t[key]:.6f}')
    morphology=[]
    for name,key in [('Nucleus Area (px)','TH_NUC_AREA'),('N/C Ratio','TH_NC_RATIO')]:
        if values[name]>t[key]:
            morphology.append(f'{name}: {values[name]:.6f} > {t[key]:.6f}')
    if len(shape)>=SHAPE_FLAGS_REQUIRED:
        morphology.append('핵 형태 복합 이탈: '+'; '.join(shape))
    outliers,missing=[],[]
    for name in EXTRA_FEATURES:
        v=finite_number(cell.get(name)); st=reference['statistics'][name]
        if v is None or st['count']<2:
            missing.append(name)
        elif v<st['lower_quantile'] or v>st['upper_quantile']:
            outliers.append(f'{name}={v:.6f} [{st["lower_quantile"]:.6f}, {st["upper_quantile"]:.6f}]')
    if len(shape)<SHAPE_FLAGS_REQUIRED:
        outliers.extend('Shape observation: '+text for text in shape)
    qc_review=bool(cell.get('QC Review',False))
    qc_reason=str(cell.get('QC Review Reasons','None')) if qc_review else 'None'
    reasons=list(morphology)
    if qc_review:
        reasons.append('분할 QC 검토: '+qc_reason)
    return {'Reference Status':'검토 필요' if reasons else '정상 범위',
            'Reference Reasons':'; '.join(reasons) or 'None','Reference Rule Count':len(morphology),
            'Observation Outliers':'; '.join(outliers) or 'None','Observation Missing':'; '.join(missing) or 'None',
            'Observation Outlier Count':len(outliers),'Segmentation QC Status':'REVIEW' if qc_review else 'PASS',
            'Segmentation QC Reasons':qc_reason,
            'Morphology Reference Status':'Outside reference' if morphology else 'Within reference',
            'Morphology Reference Reasons':'; '.join(morphology) or 'None','Shape Flag Count':len(shape)}


def draw_text(image,text,position,color=(255,255,255),scale=0.6,thickness=1):
    cv.putText(image,text,position,cv.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thickness+3,cv.LINE_AA)
    cv.putText(image,text,position,cv.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv.LINE_AA)


def save_visualization(image,preprocessed,segmented,rows,summary,output_path):
    overlay=image.copy()
    cells,nuclei=segmented['segmented_cells'],segmented['labeled_nuclei']
    props={i:p for i,p in enumerate(measure.regionprops(nuclei),1)}
    selected_labels=set(); selected_nuclei=np.zeros(nuclei.shape,np.uint8)
    for row in rows:
        if row['Reference Status']!='정상 범위':
            continue
        prop=props.get(int(row['Cell ID']))
        if prop is None:
            continue
        mask=nuclei==prop.label
        overlap=cells[mask]; overlap=overlap[overlap>0]
        if not overlap.size:
            continue
        selected_labels.add(int(np.bincount(overlap.astype(np.int64)).argmax()))
        selected_nuclei[mask]=1
    if selected_labels:
        selected=np.where(np.isin(cells,list(selected_labels)),cells,0)
        boundary=segmentation.find_boundaries(selected,connectivity=1,mode='inner',background=0)
        thick=cv.dilate(boundary.astype(np.uint8),np.ones((3,3),np.uint8),iterations=1)
        overlay[thick>0]=(0,255,255)
    contours,_=cv.findContours(selected_nuclei,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(overlay,contours,-1,(0,0,255),4,cv.LINE_AA)
    for row in rows:
        if row['Reference Status']!='정상 범위':
            continue
        x,y=row['Centroid (X, Y)']
        star='*' if (row.get('Observation Outlier Count') or 0)>0 else ''
        draw_text(overlay,f"#{row['Cell ID']} PASS{star}",(int(x)+12,int(y)-15),(0,255,0),scale=1.,thickness=2)
    colors=np.zeros_like(image)
    colors[preprocessed['mask_pink']>0]=(203,192,255)
    colors[preprocessed['mask_blue']>0]=(250,206,135)
    palette=np.random.default_rng(0).integers(40,256,size=(int(cells.max())+1,3),dtype=np.uint8)
    palette[0]=0
    gt=None; gt_path=''; gt_status='Not found'
    target=f"gt_reference_all_{Path(summary['File Name']).stem}.png".casefold()
    if REFERENCE_IMAGE_DIR.is_dir():
        matches=[p for p in REFERENCE_IMAGE_DIR.iterdir() if p.is_file() and p.name.casefold()==target]
        if len(matches)==1:
            gt_path=str(matches[0])
            try:
                gt=read_image(matches[0])
                gt_status='Read error' if gt is None else ('Size mismatch' if gt.shape[:2]!=image.shape[:2] else 'Loaded (identity not verified)')
            except Exception as error:
                gt_status=f'Read error: {error}'
        elif len(matches)>1:
            gt_status='Duplicate filenames'
    else:
        gt_status='Reference folder not found'
    blank=np.full((PANEL_HEIGHT,PANEL_WIDTH,3),30,np.uint8)
    gt_panel=make_panel(gt,'6. GT Reference: '+gt_status) if gt is not None else blank.copy()
    if gt is None:
        draw_text(gt_panel,'6. GT Reference (not available)',(12,25),scale=.52)
    info=blank.copy()
    difference=summary['GT Minus PASS Count']
    lines=[f"File: {summary['File Name']}",f"Detected: {len(rows)}",
           f"PASS: {summary['Normal Range Count']}",f"REVIEW: {summary['Review Required Count']}",
           f"INVALID: {summary['Invalid Count']}",f"GT normal: {summary['GT Normal Cell Count']}",
           f'GT minus PASS: {difference:+d}' if difference is not None else 'GT minus PASS: N/A',
           f"QC REVIEW: {summary.get('QC Review Count',0)}",f"Morphology outside: {summary.get('Morphology Outside Count',0)}",
           '', 'Version 3: exploratory reference','Not a clinical diagnosis.']
    for i,line in enumerate(lines):
        draw_text(info,line,(12,28+i*30),scale=.51)
    notes=blank.copy()
    lines=['Nucleus separation diagnostics',
           f"Split source components: {summary.get('Split Source Component Count',0)}",
           f"Rejected small pieces: {summary.get('Rejected Small Nucleus Count',0)}",
           f"Rejected large pieces: {summary.get('Rejected Large Nucleus Count',0)}",
           f"Rejected ring pieces: {summary.get('Rejected Ring Nucleus Count',0)}",'',
           'Panel 5: PASS boundaries only','Yellow: cell outline / Red: nucleus',
           'Panels 3/4 and GT: ALL objects','Splitting may also over-segment nuclei.',
           'Hidden overlapping cytoplasm is not recovered.',
           'GT filenames must refer to the SAME image.','No IoU/Dice is calculated from GT PNGs.']
    for i,line in enumerate(lines):
        draw_text(notes,line,(12,28+i*29),scale=.47)
    panels=[make_panel(image,'1. Original'),make_panel(colors,'2. Color Candidates',True),
            make_panel((nuclei>0).astype(np.uint8)*255,'3. ALL Nuclei after Splitting',True),
            make_panel(palette[cells],'4. ALL Cell Watershed',True),
            make_panel(overlay,'5. PASS Only: Cell and Nucleus'),gt_panel,info,notes]
    write_image(output_path,np.vstack([np.hstack(panels[:4]),np.hstack(panels[4:])]))
    return {'GT Reference Image':gt_path,'GT Reference Image Status':gt_status}


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()