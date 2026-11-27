# DIP Term Project

세포와 핵 분할 코드는 아래 두 파일을 참고하세요.

- [`blackhat_nucleus_segmentation.ipynb`](blackhat_nucleus_segmentation.ipynb): 노이즈 제거, 언샤프 마스킹, LAB-CLAHE, black-hat 기반 핵 마스크 생성
- [`hsv_watershed_cell_nucleus_segmentation.py`](hsv_watershed_cell_nucleus_segmentation.py): HSV로 세포 후보를 찾고 watershed로 세포 영역을 나눈 뒤, 영역 안에서 핵 검출

## 실행

노트북:

```bash
jupyter notebook blackhat_nucleus_segmentation.ipynb
```

watershed 스크립트:

```bash
../.venv/bin/python hsv_watershed_cell_nucleus_segmentation.py
```

## 최종 첨부 자료

- `norm_nuc_sperate.py`: 기존 `jh.py`를 정상세포 실험용 이름으로 정리한 코드
- `abnorm_nuc_sperate.py`: abnormal 이미지에서 핵 후보와 세포질을 분리하는 규칙 기반 코드

`masks/` 아래에는 최종 선택한 정상·비정상 이미지의 reference 마스크와 코드 예측 마스크를 따로 저장했습니다.

- `masks/norm/reference/`: DAT 좌표 기반 정답 마스크 중 009와 019만 보관합니다.
- `masks/norm/prediction/`: `norm_nuc_sperate.py`로 같은 원본 009·019 이미지를 분할한 예측 마스크입니다.
- `masks/norm/ranking_metrics.csv`: 003·009·011·019의 정답 대비 union IoU 비교 결과입니다. 핵 마스크 IoU와 세포 마스크 IoU의 평균이 높은 019(0.5468), 009(0.4471)를 선택했습니다.
- `masks/abnorm/reference/`: abnormal 정답 마스크 중 016과 017만 보관합니다.
- `masks/abnorm/train_prediction/`: abnormal 예측 마스크 중 016과 017만 보관합니다. 예측 마스크는 정답과 같은 2048×1536 원본 해상도로 맞췄습니다.

각 세포별 `cell` 마스크는 핵을 포함한 세포 외곽이고, `cytoplasm` 마스크는 해당 외곽에서 핵 영역을 뺀 영역입니다. 선택한 정상 reference와 예측은 동일한 원본 이미지 좌표에서 평가했습니다. abnormal 예측은 1400×1050 분석 마스크를 최근접 보간으로 2048×1536 원본 크기에 맞춰 reference와 비교할 수 있게 했습니다.

## 특징 추출과 분할 평가

- `extract_and_evaluate_masks.py`: 선택한 reference 마스크의 핵 면적·둘레·원형도·종횡비, 세포질 면적·N/C 비율을 계산하고 예측 분할의 Dice·IoU, 매칭·누락·오검출 수를 저장합니다.
- `reports/reference_cell_features.csv`: 세포별 특징값, 단위, 계산 불가 사유 및 임시 규칙 결과입니다.
- `reports/feature_distribution_summary.csv`와 `reports/team_feature_summary.md`: 정상·abnormal 특징 분포와 예시입니다.
- `reports/mask_quality_metrics.csv`: 이미지별 분할 품질 평가입니다.
- `reports/tuning_comparison.csv`: 기존 코드와 네 이미지에 조정한 코드의 union IoU 비교입니다.
- `kmeans_psnr_experiment.py`: 요청한 OpenCV K-means 및 PSNR 비교 실험입니다. 기존 분할 코드를 대체하지 않는 별도 실험입니다.
- `reports/kmeans_experiment/summary.md`, `summary.csv`, `metrics.csv`: 네 이미지에서 K-means와 기존 마스크를 비교한 결과입니다. 정답으로 선택한 군집 조합은 참고 상한일 뿐 적용 가능한 성능이 아닙니다.
- `reports/mask_overlays/`: reference 경계(초록)와 예측 경계(자홍)를 원본 이미지에 겹친 확인용 이미지입니다.
- `docs/feature_calculation.md`: 계산 방법, 임시 규칙의 한계와 검토 상태를 기록합니다.

세포질 경계는 김승연을 포함한 팀의 육안 검토가 아직 완료되지 않았으므로 보고서에서 검토 대기로 표시했습니다. 규칙은 `핵 면적 > 1800 px² AND N/C > 0.15`인 탐색용 예시일 뿐이며, 네 명의 합의나 의학적 기준으로 확인되지 않았습니다. 결과 상태는 진단명이 아닙니다.

정상 009·019와 abnormal 016·017에 대해서는 정답 마스크를 이용해 기존 고전 영상처리 임계값을 이미지별로 조정했습니다. 본 분할 코드에는 머신러닝을 넣지 않았습니다. 별도 K-means 실험 코드는 비지도 머신러닝에 해당하는 색 군집화를 수행하며 요청에 따라 비교용으로만 분리했습니다. 해당 네 장에 대한 튜닝이므로 다른 이미지에 대한 성능을 보장하지 않습니다. 자세한 설정과 제한은 `docs/feature_calculation.md`를 참고하세요.
