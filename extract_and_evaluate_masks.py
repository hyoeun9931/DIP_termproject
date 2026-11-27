"""Extract reference-mask features and compare predictions with verified masks.

Run from the repository root after placing the dataset at the defaults, or pass
--normal-source and --abnormal-source explicitly. Outputs are exploratory
image-processing measurements, not diagnostic results.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import norm_nuc_sperate as norm
import abnorm_nuc_sperate as abnorm


ROOT = Path(__file__).resolve().parent
MASK_ROOT = ROOT / "masks"
IMAGE_STEMS = {"normal": ["009", "019"], "abnormal": ["016", "017"]}
LABEL_DIR = {"normal": "norm", "abnormal": "abnorm"}
AREA_THRESHOLD_PX2 = 1800.0
NC_THRESHOLD = 0.15
INSTANCE_MATCH_IOU_THRESHOLD = 0.50
RULE_SOURCE = "기존 abnorm 코드의 탐색값; 네 명의 합의/의학 기준으로 검증되지 않음"


def read_mask(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image > 0


def binary_png(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.uint8) * 255


def union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=bool)
    return np.logical_or.reduce(masks)


def extract_nucleus_features(mask: np.ndarray) -> dict:
    area = int(np.count_nonzero(mask))
    result = {
        "nucleus_area_px2": area,
        "nucleus_perimeter_px": None,
        "nucleus_circularity": None,
        "nucleus_aspect_ratio": None,
        "nucleus_width_px": None,
        "nucleus_height_px": None,
        "nucleus_calculation_unavailable_reason": "",
    }
    if area == 0:
        result["nucleus_calculation_unavailable_reason"] = "핵 마스크 전경 픽셀 없음"
        return result
    contours, _ = cv2.findContours(binary_png(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        result["nucleus_calculation_unavailable_reason"] = "핵 경계를 추출하지 못함"
        return result
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    result["nucleus_perimeter_px"] = perimeter
    result["nucleus_width_px"] = int(width)
    result["nucleus_height_px"] = int(height)
    result["nucleus_aspect_ratio"] = max(width, height) / min(width, height) if min(width, height) > 0 else None
    if perimeter <= 0:
        result["nucleus_calculation_unavailable_reason"] = "핵 둘레가 0 또는 경계가 불안정함"
    else:
        # Area is the foreground pixel count; perimeter is OpenCV arcLength on
        # the external contour with CHAIN_APPROX_NONE. Do not clamp raster effects.
        result["nucleus_circularity"] = 4.0 * math.pi * area / (perimeter * perimeter)
    return result


def feature_rows(label: str) -> list[dict]:
    rows = []
    summary_path = MASK_ROOT / LABEL_DIR[label] / "reference" / "mask_summary.csv"
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        references = list(csv.DictReader(stream))
    for ref in references:
        image_id = ref["image"]
        if image_id not in IMAGE_STEMS[label]:
            continue
        per_cell = MASK_ROOT / LABEL_DIR[label] / "reference" / image_id / "per_cell"
        nuc_id, cell_id = ref["nucleus_id"], ref["cell_id"]
        nucleus_path = per_cell / f"{image_id}_{nuc_id}_nucleus.png" if nuc_id else None
        cell_path = per_cell / f"{image_id}_{cell_id}_cell.png" if cell_id else None
        cytoplasm_path = per_cell / f"{image_id}_{cell_id}_cytoplasm.png" if cell_id else None
        nucleus = read_mask(nucleus_path) if nucleus_path else None
        cell = read_mask(cell_path) if cell_path else None
        cytoplasm = read_mask(cytoplasm_path) if cytoplasm_path else None
        reasons = []
        if nucleus is None:
            reasons.append("핵 마스크 파일 없음 또는 읽기 실패")
        if cell is None:
            reasons.append("핵-세포 대응 불명확 또는 세포 마스크 없음")
        if cytoplasm is None:
            reasons.append("세포질 마스크 파일 없음 또는 읽기 실패")
        shapes = [m.shape for m in (nucleus, cell, cytoplasm) if m is not None]
        if shapes and any(s != shapes[0] for s in shapes):
            reasons.append("핵/세포/세포질 마스크 크기 불일치")
        nuc_features = extract_nucleus_features(nucleus) if nucleus is not None else {
            "nucleus_area_px2": None, "nucleus_perimeter_px": None,
            "nucleus_circularity": None, "nucleus_aspect_ratio": None,
            "nucleus_width_px": None, "nucleus_height_px": None,
            "nucleus_calculation_unavailable_reason": "핵 마스크 파일 없음 또는 읽기 실패",
        }
        if nuc_features["nucleus_calculation_unavailable_reason"]:
            reasons.append(nuc_features["nucleus_calculation_unavailable_reason"])
        cell_area = int(np.count_nonzero(cell)) if cell is not None else None
        cyt_area = int(np.count_nonzero(cytoplasm)) if cytoplasm is not None else None
        if cell is not None and nucleus is not None and cell.shape == nucleus.shape:
            inside = int(np.count_nonzero(cell & nucleus))
            nucleus_inside_fraction = inside / max(1, int(np.count_nonzero(nucleus)))
            if nucleus_inside_fraction < 0.98:
                reasons.append("핵 픽셀의 2% 초과가 연결 세포 마스크 밖에 있음")
        else:
            nucleus_inside_fraction = None
        nc = None
        if cyt_area is None or nuc_features["nucleus_area_px2"] is None:
            reasons.append("핵 또는 세포질 면적 계산 불가")
        elif cyt_area <= 0:
            reasons.append("세포질 면적이 0")
        else:
            nc = nuc_features["nucleus_area_px2"] / cyt_area
        unavailable = "; ".join(dict.fromkeys(reasons))
        area = nuc_features["nucleus_area_px2"]
        status = "분석 불가" if unavailable else (
            "검토 필요" if area > AREA_THRESHOLD_PX2 and nc is not None and nc > NC_THRESHOLD
            else "규칙 미충족"
        )
        rows.append({
            "label": label,
            "image_id": image_id,
            "cell_id": cell_id or "",
            "nucleus_id": nuc_id or "",
            "measurement_units": "area=px²; perimeter=px; circularity=dimensionless; aspect_ratio=dimensionless; N/C=dimensionless",
            "mask_source": "DAT reference binary masks",
            **nuc_features,
            "cell_area_px2": cell_area,
            "cytoplasm_area_px2": cyt_area,
            "nc_ratio_nucleus_over_cytoplasm": nc,
            "nucleus_inside_cell_fraction": nucleus_inside_fraction,
            "calculation_unavailable_reason": unavailable,
            "exploratory_rule": f"nucleus_area_px2 > {AREA_THRESHOLD_PX2:g} AND N/C > {NC_THRESHOLD:g}",
            "exploratory_rule_source": RULE_SOURCE,
            "exploratory_status": status,
            "cytoplasm_manual_review_status": "김승연 포함 팀 육안 검토 대기",
        })
    return rows


def load_instances(directory: Path, pattern: str) -> list[tuple[str, np.ndarray]]:
    instances = []
    for path in sorted(directory.glob(pattern)):
        mask = read_mask(path)
        if mask is not None and np.any(mask):
            instances.append((path.stem, mask))
    return instances


def score_union(reference: list[np.ndarray], predicted: list[np.ndarray], shape: tuple[int, int]) -> tuple[float, float]:
    gt, pred = union_masks(reference, shape), union_masks(predicted, shape)
    intersection = int(np.count_nonzero(gt & pred))
    denominator = int(np.count_nonzero(gt | pred))
    iou = intersection / denominator if denominator else None
    dice_denominator = int(np.count_nonzero(gt)) + int(np.count_nonzero(pred))
    dice = 2 * intersection / dice_denominator if dice_denominator else None
    return dice, iou


def match_instances(reference: list[np.ndarray], predicted: list[np.ndarray],
                    minimum_iou: float = INSTANCE_MATCH_IOU_THRESHOLD) -> dict:
    matrix = np.zeros((len(reference), len(predicted)), dtype=np.float64)
    for i, gt in enumerate(reference):
        for j, pred in enumerate(predicted):
            if gt.shape != pred.shape:
                continue
            union = int(np.count_nonzero(gt | pred))
            matrix[i, j] = int(np.count_nonzero(gt & pred)) / union if union else 0.0
    pairs = []
    if matrix.size:
        gt_idx, pred_idx = linear_sum_assignment(1.0 - matrix)
        pairs = [(int(i), int(j), float(matrix[i, j]))
                 for i, j in zip(gt_idx, pred_idx)
                 if matrix[i, j] >= minimum_iou]
    matched = len(pairs)
    return {
        "matched": matched,
        "missed": max(0, len(reference) - matched),
        "false_positive": max(0, len(predicted) - matched),
        "matched_instance_mean_iou": sum(x[2] for x in pairs) / matched if matched else None,
        "matched_instance_mean_iou_per_reference": sum(x[2] for x in pairs) / max(1, len(reference)) if reference else None,
    }


def save_prediction_instances(label: str, stem: str, instances: dict[str, list[tuple[str, np.ndarray]]]) -> None:
    if label == "normal":
        folder = MASK_ROOT / "norm" / "prediction" / stem / "per_cell"
        prefixes = {"nucleus": f"{stem}_nucleus_", "cell": f"{stem}_cell_", "cytoplasm": f"{stem}_cytoplasm_"}
    else:
        folder = MASK_ROOT / "abnorm" / "train_prediction" / stem / "per_cell"
        prefixes = {"nucleus": f"{stem}_nucleus_", "cell": f"{stem}_cell_", "cytoplasm": f"{stem}_cytoplasm_"}
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    for kind, items in instances.items():
        for identity, mask in items:
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", identity)
            cv2.imwrite(str(folder / f"{prefixes[kind]}{safe}.png"), binary_png(mask))
        if items:
            union = union_masks([mask for _, mask in items], items[0][1].shape)
        else:
            union = np.zeros(instances["nucleus"][0][1].shape, dtype=bool) if instances["nucleus"] else None
        if union is not None:
            suffix = "union" if label == "normal" else "mask"
            cv2.imwrite(str(folder.parent / f"{stem}_{kind}_{suffix}.png"), binary_png(union))


def predict_normal(path: Path) -> tuple[dict, tuple[int, int]]:
    image = norm.read_image(path)
    segmented = norm.segment_image(norm.preprocess_image(image), image_id=path.stem)
    nuclei = segmented["labeled_nuclei"]
    cells = segmented["segmented_cells"]
    output = {"nucleus": [], "cell": [], "cytoplasm": []}
    for identity in sorted(int(x) for x in np.unique(nuclei) if x > 0):
        nuc = nuclei == identity
        cell = cells == identity
        cyt = cell & ~nuc
        output["nucleus"].append((f"{identity:03d}", nuc))
        if np.any(cell):
            output["cell"].append((f"{identity:03d}", cell))
        if np.any(cyt):
            output["cytoplasm"].append((f"{identity:03d}", cyt))
    return output, image.shape[:2]


def predict_abnormal(path: Path) -> tuple[dict, tuple[int, int]]:
    original = abnorm.read_image(path)
    source_shape = original.shape[:2]
    scale = min(1.0, abnorm.MAX_SIDE / max(source_shape))
    work = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else original.copy()
    candidates = abnorm.rule_nucleus_candidates(
        work, abnorm.nucleus_candidates(work, scale), image_id=path.stem
    )
    nuclei = abnorm.segment_cells(work, candidates, scale, image_id=path.stem)
    output = {"nucleus": [], "cell": [], "cytoplasm": []}
    target_size = (source_shape[1], source_shape[0])
    for identity, item in enumerate(nuclei, start=1):
        nm = np.zeros(work.shape[:2], np.uint8)
        cv2.drawContours(nm, [item["contour"]], -1, 255, cv2.FILLED)
        nm = cv2.resize(nm, target_size, interpolation=cv2.INTER_NEAREST) > 0 if scale < 1.0 else nm > 0
        output["nucleus"].append((f"{identity:03d}", nm))
        if "cell_contour" in item:
            cm = np.zeros(work.shape[:2], np.uint8)
            cv2.drawContours(cm, [item["cell_contour"]], -1, 255, cv2.FILLED)
            cy = item.get("cytoplasm_mask", np.zeros(work.shape[:2], np.uint8))
            if scale < 1.0:
                cm = cv2.resize(cm, target_size, interpolation=cv2.INTER_NEAREST) > 0
                cy = cv2.resize(np.asarray(cy, dtype=np.uint8), target_size, interpolation=cv2.INTER_NEAREST) > 0
            else:
                cm, cy = cm > 0, np.asarray(cy) > 0
            output["cell"].append((f"{identity:03d}", cm))
            if np.any(cy):
                output["cytoplasm"].append((f"{identity:03d}", cy))
    return output, source_shape


def feature_summary(rows: list[dict], path: Path) -> None:
    columns = [
        "nucleus_area_px2", "nucleus_perimeter_px", "nucleus_circularity",
        "nucleus_aspect_ratio", "cell_area_px2", "cytoplasm_area_px2",
        "nc_ratio_nucleus_over_cytoplasm",
    ]
    output = []
    for label in ("normal", "abnormal"):
        subset = [r for r in rows if r["label"] == label]
        image_count = len({r["image_id"] for r in subset})
        for col in columns:
            values = np.array([float(r[col]) for r in subset if r[col] not in (None, "")], dtype=float)
            if not len(values):
                continue
            output.append({
                "label": label, "image_count": image_count, "cell_count": len(values),
                "feature": col, "unit": "px²" if col.endswith("px2") else "px" if col.endswith("px") else "dimensionless",
                "mean": float(np.mean(values)), "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "min": float(np.min(values)), "q1": float(np.quantile(values, .25)),
                "median": float(np.median(values)), "q3": float(np.quantile(values, .75)), "max": float(np.max(values)),
            })
    write_csv(path, output)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def overlay_example(image_path: Path, ref_masks: list[np.ndarray], pred_masks: list[np.ndarray], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    gt = union_masks(ref_masks, image.shape[:2]).astype(np.uint8) * 255
    pred = union_masks(pred_masks, image.shape[:2]).astype(np.uint8) * 255
    for mask, color in ((gt, (0, 255, 0)), (pred, (255, 0, 255))):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def run(args) -> None:
    out = ROOT / "reports"
    feature_rows_all = []
    eval_rows = []
    for label, stems in IMAGE_STEMS.items():
        source = args.normal_source if label == "normal" else args.abnormal_source
        ref_root = MASK_ROOT / LABEL_DIR[label] / "reference"
        for stem in stems:
            summary_path = ref_root / "mask_summary.csv"
            with summary_path.open(encoding="utf-8-sig", newline="") as stream:
                summary_rows = [r for r in csv.DictReader(stream) if r["image"] == stem]
            ref_per = ref_root / stem / "per_cell"
            refs = {"nucleus": [], "cell": [], "cytoplasm": []}
            for row in summary_rows:
                nuc = read_mask(ref_per / f"{stem}_{row['nucleus_id']}_nucleus.png") if row["nucleus_id"] else None
                cell = read_mask(ref_per / f"{stem}_{row['cell_id']}_cell.png") if row["cell_id"] else None
                cyt = read_mask(ref_per / f"{stem}_{row['cell_id']}_cytoplasm.png") if row["cell_id"] else None
                if nuc is not None: refs["nucleus"].append(nuc)
                if cell is not None: refs["cell"].append(cell)
                if cyt is not None: refs["cytoplasm"].append(cyt)
            image_path = source / f"{stem}.bmp"
            if not image_path.is_file():
                raise FileNotFoundError(f"원본 이미지 없음: {image_path}")
            preds, shape = predict_normal(image_path) if label == "normal" else predict_abnormal(image_path)
            save_prediction_instances(label, stem, preds)
            reference_counts = {k: len(v) for k, v in refs.items()}
            prediction_counts = {k: len(v) for k, v in preds.items()}
            params = (norm.IMAGE_SEGMENTATION_OVERRIDES.get(stem, {}) if label == "normal"
                      else abnorm.IMAGE_CANDIDATE_OVERRIDES.get(stem, {}))
            row_out = {"label": label, "image_id": stem,
                       "instance_match_iou_threshold": INSTANCE_MATCH_IOU_THRESHOLD,
                       "image_specific_parameters": "; ".join(f"{k}={v}" for k, v in params.items())}
            for kind in ("nucleus", "cell", "cytoplasm"):
                ref_masks, pred_masks = refs[kind], [m for _, m in preds[kind]]
                if any(m.shape != shape for m in ref_masks + pred_masks):
                    raise ValueError(f"마스크 크기 불일치: {label}/{stem}/{kind}")
                dice, iou = score_union(ref_masks, pred_masks, shape)
                matched = match_instances(ref_masks, pred_masks)
                row_out.update({
                    f"reference_{kind}_count": reference_counts[kind],
                    f"predicted_{kind}_count": prediction_counts[kind],
                    f"matched_{kind}_count": matched["matched"],
                    f"missed_{kind}_count": matched["missed"],
                    f"false_positive_{kind}_count": matched["false_positive"],
                    f"{kind}_union_dice": dice,
                    f"{kind}_union_iou": iou,
                    f"{kind}_instance_mean_iou": matched["matched_instance_mean_iou"],
                    f"{kind}_instance_mean_iou_per_reference": matched["matched_instance_mean_iou_per_reference"],
                })
            row_out["cytoplasm_review_status"] = "김승연 포함 팀 육안 검토 대기"
            eval_rows.append(row_out)
            overlay_example(image_path, refs["cell"], [m for _, m in preds["cell"]], out / "mask_overlays" / f"{label}_{stem}_cell.png")
            overlay_example(image_path, refs["nucleus"], [m for _, m in preds["nucleus"]], out / "mask_overlays" / f"{label}_{stem}_nucleus.png")
    # Reference features only: predictions are never used to set the exploratory rule.
    for label in ("normal", "abnormal"):
        feature_rows_all.extend(feature_rows(label))
    write_csv(out / "reference_cell_features.csv", feature_rows_all)
    feature_summary(feature_rows_all, out / "feature_distribution_summary.csv")
    write_csv(out / "mask_quality_metrics.csv", eval_rows)
    write_csv(out / "exploratory_rule_results.csv", feature_rows_all)
    # Compact shareable examples and limitations for team review.
    examples = []
    for label in ("normal", "abnormal"):
        subset = [r for r in feature_rows_all if r["label"] == label and r["calculation_unavailable_reason"] == ""]
        if not subset:
            continue
        ordered = sorted(subset, key=lambda r: float(r["nc_ratio_nucleus_over_cytoplasm"]))
        examples.extend([ordered[0], ordered[len(ordered)//2], ordered[-1]])
    lines = [
        "# 마스크 특징 및 탐색 규칙 요약", "",
        "## 범위", "",
        "정상 009·019와 abnormal 016·017의 DAT reference 마스크를 사용했습니다. 세포 수가 아니라 이미지 수 기준으로 클래스당 2개 이미지뿐이며, 세포는 이미지 안에 군집되어 있습니다. 아래 수치는 예시 관찰이며 일반화된 기준이 아닙니다.", "",
        "## 계산 정의", "",
        "- 면적: 이진 마스크의 흰색 전경 픽셀 수(px²).", 
        "- 둘레: 외곽 윤곽에 `cv2.arcLength`와 `CHAIN_APPROX_NONE`을 적용한 픽셀 길이(px).",
        "- 원형도: `4π × 전경 픽셀 수 / 둘레²`.",
        "- 종횡비: 외곽 윤곽의 bounding box에서 긴 변/짧은 변.",
        "- 세포질 면적: reference 세포질 마스크 전경 픽셀 수(px²). N/C는 핵 면적/세포질 면적.",
        "- 실제 단위 환산은 픽셀 크기 보정값이 없어 수행하지 않았습니다.", "",
        "## 임시 탐색 규칙", "",
        f"`핵 면적 > {AREA_THRESHOLD_PX2:g} px² AND N/C > {NC_THRESHOLD:g}`이면 `검토 필요`, 그 외 계산 가능한 행은 `규칙 미충족`, 계산에 필요한 마스크/대응이 없으면 `분석 불가`로 표시합니다. 임계값은 {RULE_SOURCE}.",
        "이 규칙은 세포 라벨을 진단하지 않으며, 네 명의 팀 합의가 확인되기 전까지 임시 예시입니다.", "",
        "## 예시 행", "",
        "| 그룹 | 이미지 | 세포 | 핵 면적 (px²) | 둘레 (px) | 원형도 | 종횡비 | 세포질 (px²) | N/C | 상태 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in examples:
        lines.append(f"| {r['label']} | {r['image_id']} | {r['cell_id']} | {r['nucleus_area_px2']} | {r['nucleus_perimeter_px']:.2f} | {r['nucleus_circularity']:.3f} | {r['nucleus_aspect_ratio']:.3f} | {r['cytoplasm_area_px2']} | {r['nc_ratio_nucleus_over_cytoplasm']:.3f} | {r['exploratory_status']} |")
    lines += ["", "## 세포질 품질 확인", "", "`mask_quality_metrics.csv`의 Dice/IoU와 `mask_overlays/`를 이용해 김승연을 포함한 팀이 세포질 경계를 수동 검토해야 합니다. 이 파일을 생성하는 과정에서는 팀원의 합의를 대신 표시하지 않았습니다.", ""]
    (out / "team_feature_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"특징 행: {len(feature_rows_all)} → {out / 'reference_cell_features.csv'}")
    print(f"이미지 평가: {len(eval_rows)} → {out / 'mask_quality_metrics.csv'}")
    print(f"요약/예시: {out / 'team_feature_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-source", type=Path, default=ROOT / "test_image" / "im_Superficial-Intermediate")
    parser.add_argument("--abnormal-source", type=Path, default=ROOT / "test_image" / "im_Koilocytotic")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
