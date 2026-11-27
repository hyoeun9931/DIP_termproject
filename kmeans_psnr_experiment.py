"""Compare OpenCV K-means masks and PSNR against four DAT references.

K-means is unsupervised clustering (often categorized as machine learning),
so this script is a separate experiment and does not change the project
segmentation pipelines. The best cluster subset in the report is selected
after seeing reference masks, so it is an optimistic diagnostic, not deployable
performance. Run with --normal-source and --abnormal-source dataset folders.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
STEMS = {"normal": ["009", "019"], "abnormal": ["016", "017"]}
FOLDERS = {"normal": "norm", "abnormal": "abnorm"}
MAX_SIDE = 900
K_VALUES = (3, 4, 5, 6)
COLOR_SPACES = ("LAB", "HSV", "LAB_XY")


def read_mask(path: Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise FileNotFoundError(path)
    return data > 0


def union_reference(label: str, stem: str, kind: str) -> np.ndarray:
    folder = ROOT / "masks" / FOLDERS[label] / "reference" / stem / "per_cell"
    masks = [read_mask(path) for path in sorted(folder.glob(f"{stem}_*_{kind}.png"))]
    if not masks:
        raise FileNotFoundError(f"정답 {kind} 마스크 없음: {label}/{stem}")
    return np.logical_or.reduce(masks)


def union_prediction(label: str, stem: str, kind: str) -> np.ndarray:
    category = "prediction" if label == "normal" else "train_prediction"
    folder = ROOT / "masks" / FOLDERS[label] / category / stem / "per_cell"
    masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
             for path in sorted(folder.glob(f"{stem}_{kind}_*.png"))]
    masks = [mask for mask in masks if mask is not None and mask.any()]
    if not masks:
        raise FileNotFoundError(f"예측 {kind} 마스크 없음: {label}/{stem}")
    return np.logical_or.reduce(masks)


def metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    intersection = int(np.count_nonzero(gt & pred))
    union = int(np.count_nonzero(gt | pred))
    denom = int(gt.sum() + pred.sum())
    dice = 2 * intersection / denom if denom else 1.0
    iou = intersection / union if union else 1.0
    # Binary masks are compared as 8-bit 0/255 images.
    mse = 255.0**2 * float(np.count_nonzero(gt ^ pred)) / gt.size
    psnr = math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)
    return dice, iou, psnr


def kmeans_labels(image: np.ndarray, color_space: str, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color_space == "LAB":
        features = lab.astype(np.float32)
    elif color_space == "HSV":
        features = hsv.astype(np.float32)
    else:
        xy = np.indices((h, w), dtype=np.float32)
        # Spatial channels are deliberately weak relative to 8-bit color.
        features = np.dstack((lab.astype(np.float32), xy[1] * (12.0 / max(1, w)),
                              xy[0] * (12.0 / max(1, h))))
    flat = np.ascontiguousarray(features.reshape((-1, features.shape[2])))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    cv2.setRNGSeed(1701 + k + sum(map(ord, color_space)))
    _, raw_labels, centers = cv2.kmeans(flat, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    labels = raw_labels.reshape((h, w))
    return labels, centers, lab


def best_subsets(labels: np.ndarray, gt: np.ndarray, k: int) -> tuple[tuple[int, ...], float, float, float]:
    counts = np.bincount(labels.ravel(), minlength=k)
    intersections = np.bincount(labels[gt].ravel(), minlength=k)
    best = ((), -1.0, -1.0, -1.0)
    for size in range(1, k + 1):
        for subset in itertools.combinations(range(k), size):
            area = int(counts[list(subset)].sum())
            inter = int(intersections[list(subset)].sum())
            union = int(gt.sum()) + area - inter
            iou = inter / union if union else 1.0
            dice = 2 * inter / (int(gt.sum()) + area) if int(gt.sum()) + area else 1.0
            if iou > best[1]:
                best = (subset, iou, dice, area)
    subset = best[0]
    pred = np.isin(labels, subset)
    _, _, psnr = metrics(gt, pred)
    return subset, best[1], best[2], psnr


def heuristic_subsets(labels: np.ndarray, centers: np.ndarray, lab: np.ndarray,
                      color_space: str, k: int) -> dict[str, tuple[int, ...]]:
    # Centers are projected to Lab; cluster identity is chosen by explicit
    # color rules, never by reference masks.
    if color_space == "LAB":
        center_lab = centers[:, :3]
    elif color_space == "HSV":
        center_hsv = np.uint8(np.clip(centers[:, :3].reshape((-1, 1, 3)), 0, 255))
        center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)
        center_lab = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2LAB).reshape((-1, 3))
    else:
        center_lab = centers[:, :3]
    # Dark purple proxy: low L with higher red/blue Lab channels.
    nuc_score = (255 - center_lab[:, 0]) + 0.35 * center_lab[:, 1] + 0.25 * center_lab[:, 2]
    nucleus = (int(np.argmax(nuc_score)),)
    # Cell/stain proxy: reject the lightest cluster (usually slide background).
    light_order = np.argsort(center_lab[:, 0])
    tissue = tuple(int(x) for x in light_order[:-1]) if k > 1 else (int(light_order[0]),)
    return {"dark_purple_one_cluster": nucleus, "all_but_lightest_cluster": tissue}


def run(args) -> None:
    all_rows = []
    best_masks = {}
    for label, stems in STEMS.items():
        source = args.normal_source if label == "normal" else args.abnormal_source
        for stem in stems:
            image_path = source / f"{stem}.bmp"
            original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if original is None:
                raise FileNotFoundError(image_path)
            scale = min(1.0, MAX_SIDE / max(original.shape[:2]))
            small = cv2.resize(original, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA) if scale < 1 else original
            gts = {kind: union_reference(label, stem, kind)
                   for kind in ("nucleus", "cell", "cytoplasm")}
            baselines = {kind: union_prediction(label, stem, kind)
                         for kind in ("nucleus", "cell", "cytoplasm")}
            if scale < 1:
                gt_small = {kind: cv2.resize(mask.astype(np.uint8), (small.shape[1], small.shape[0]),
                                              interpolation=cv2.INTER_NEAREST) > 0
                            for kind, mask in gts.items()}
                base_small = {kind: cv2.resize(mask.astype(np.uint8), (small.shape[1], small.shape[0]),
                                                interpolation=cv2.INTER_NEAREST) > 0
                              for kind, mask in baselines.items()}
            else:
                gt_small, base_small = gts, baselines

            candidates = []
            for color_space in COLOR_SPACES:
                for k in K_VALUES:
                    labels, centers, lab = kmeans_labels(small, color_space, k)
                    heuristic = heuristic_subsets(labels, centers, lab, color_space, k)
                    for kind in ("nucleus", "cell", "cytoplasm"):
                        gt = gt_small[kind]
                        base_dice, base_iou, base_psnr = metrics(gt, base_small[kind])
                        subset, best_iou, best_dice, best_psnr = best_subsets(labels, gt, k)
                        oracle = np.isin(labels, subset)
                        all_rows.append({
                            "label": label, "image_id": stem, "mask_type": kind,
                            "method": "kmeans_best_cluster_subset_GT_selected",
                            "color_space": color_space, "k": k,
                            "clusters": ";".join(map(str, subset)),
                            "dice": best_dice, "iou": best_iou, "psnr_db": best_psnr,
                            "baseline_dice": base_dice, "baseline_iou": base_iou,
                            "baseline_psnr_db": base_psnr,
                            "selection_note": "정답 마스크를 보고 군집 조합 선택; 낙관적 진단값, 배포 성능 아님",
                        })
                        candidates.append((best_iou, kind, color_space, k, subset, oracle))
                        for heuristic_name, heuristic_subset in heuristic.items():
                            pred = np.isin(labels, heuristic_subset)
                            d, i, p = metrics(gt, pred)
                            all_rows.append({
                                "label": label, "image_id": stem, "mask_type": kind,
                                "method": f"kmeans_heuristic_{heuristic_name}",
                                "color_space": color_space, "k": k,
                                "clusters": ";".join(map(str, heuristic_subset)),
                                "dice": d, "iou": i, "psnr_db": p,
                                "baseline_dice": base_dice, "baseline_iou": base_iou,
                                "baseline_psnr_db": base_psnr,
                                "selection_note": "색상 중심의 고정 규칙으로 군집 선택; 학습 없음",
                            })
            # Keep the oracle-best mask for visual inspection only.
            for kind in ("nucleus", "cell", "cytoplasm"):
                options = [x for x in candidates if x[1] == kind]
                best = max(options, key=lambda item: item[0])
                _, _, space, k, subset, _ = best
                labels, _, _ = kmeans_labels(small, space, k)
                mask = np.isin(labels, subset).astype(np.uint8) * 255
                if scale < 1:
                    mask = cv2.resize(mask, (original.shape[1], original.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                best_masks[(label, stem, kind)] = (mask, space, k, subset)

    report_dir = ROOT / "reports" / "kmeans_experiment"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    summaries = []
    for label, stems in STEMS.items():
        for stem in stems:
            for kind in ("nucleus", "cell", "cytoplasm"):
                eligible = [r for r in all_rows if r["label"] == label and r["image_id"] == stem and r["mask_type"] == kind]
                baseline = eligible[0]
                oracle = max((r for r in eligible if r["method"] == "kmeans_best_cluster_subset_GT_selected"),
                             key=lambda r: float(r["iou"]))
                heuristic = max((r for r in eligible if r["method"].startswith("kmeans_heuristic")),
                                key=lambda r: float(r["iou"]))
                summaries.append({
                    "label": label, "image_id": stem, "mask_type": kind,
                    "baseline_iou": baseline["baseline_iou"],
                    "baseline_psnr_db": baseline["baseline_psnr_db"],
                    "best_heuristic_iou": heuristic["iou"],
                    "best_heuristic_psnr_db": heuristic["psnr_db"],
                    "best_heuristic_method": heuristic["method"],
                    "best_heuristic_space": heuristic["color_space"],
                    "best_heuristic_k": heuristic["k"],
                    "best_oracle_iou": oracle["iou"],
                    "best_oracle_psnr_db": oracle["psnr_db"],
                    "best_oracle_space": oracle["color_space"],
                    "best_oracle_k": oracle["k"],
                    "best_oracle_clusters": oracle["clusters"],
                })
    with (report_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# OpenCV K-means / PSNR 실험", "",
        "주의: K-means는 비지도 군집화이며 머신러닝으로 분류되는 방법입니다. 사용자가 허용해 실험용으로만 분리했으며, 정상·abnormal 기존 분할 코드에는 넣지 않았습니다. 기존 분할 코드는 계속 고전 영상처리만 사용합니다.", "",
        f"원본 이미지를 긴 변 {MAX_SIDE}px로 축소한 좌표계에서 LAB, HSV, LAB+XY 특징과 K={', '.join(map(str, K_VALUES))}를 비교했습니다. 비교 대상은 현재 고전 영상처리 예측 마스크이며, Dice·IoU·8-bit 이진 마스크 PSNR을 계산했습니다.", "",
        "‘고정 색 규칙’은 가장 어두운 보라색 중심 또는 가장 밝은 배경 중심을 제외하는 식으로 군집을 선택합니다. 설정 탐색 중 가장 나은 조합을 표에 넣었으므로, 이 선택 또한 네 정답을 확인한 사후 비교입니다. ‘GT-selected oracle’은 더 직접적으로 정답 마스크에 가장 잘 맞는 군집 조합을 골랐으며 실제 적용 가능한 성능으로 보면 안 됩니다.", "",
        "| 그룹 | 이미지 | 마스크 | 기존 IoU | 기존 PSNR (dB) | 색 규칙 K-means 최고 IoU | GT 선택 상한 IoU |", "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['label']} | {row['image_id']} | {row['mask_type']} | {float(row['baseline_iou']):.3f} | {float(row['baseline_psnr_db']):.2f} | {float(row['best_heuristic_iou']):.3f} | {float(row['best_oracle_iou']):.3f} |")
    lines += [
        "", "## 해석", "",
        "네 장에서 nucleus는 K-means 색 군집만으로 현재 마스크보다 나아지지 않았습니다. Abnormal 세포·세포질도 정답으로 군집을 고른 낙관적 상한조차 현 코드보다 낮았습니다. Normal의 세포·세포질은 일부 K-means 설정에서 개선됐지만 이미지마다 가장 좋은 색 공간/군집 조합이 달라 일관된 규칙으로 확인되진 않았습니다.", "",
        "PSNR은 참고 수치로만 보세요. 예를 들어 abnormal 016 nucleus는 PSNR이 높아도 IoU가 낮습니다. 정답 foreground가 전체 픽셀에서 차지하는 비율이 작으면 배경이 맞는 것만으로 PSNR이 올라갈 수 있습니다. 분할 판단은 IoU/Dice 및 누락·오검출과 오버레이를 우선합니다.", "",
        "이 네 장을 이용해 설정을 선택했으므로 결과는 탐색용입니다. 새 이미지 성능을 뜻하지 않습니다.", "",
    ]
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    for (label, stem, kind), (mask, space, k, subset) in best_masks.items():
        cv2.imwrite(str(report_dir / f"{label}_{stem}_{kind}_best_oracle.png"), mask)
    print(f"Saved {len(all_rows)} metric rows to {report_dir / 'metrics.csv'}")
    for row in summaries:
        print(f"{row['label']} {row['image_id']} {row['mask_type']}: "
              f"baseline IoU={float(row['baseline_iou']):.3f}, PSNR={float(row['baseline_psnr_db']):.2f} dB; "
              f"heuristic best={float(row['best_heuristic_iou']):.3f}; "
              f"GT-selected oracle best={float(row['best_oracle_iou']):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-source", type=Path, required=True)
    parser.add_argument("--abnormal-source", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
