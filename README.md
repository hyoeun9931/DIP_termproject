# DIP Term Project

## 프로젝트 역할

이 프로젝트는 **향후 AI 모델 학습에 사용할 세포 영상과 마스크를 준비·검토하는 전처리 도구**입니다. 현재 코드는 학습된 AI 모델로 세포를 판정하지 않습니다. OpenCV·scikit-image 기반 영상처리 규칙으로 세포·핵 후보 마스크를 만들고, 정답 마스크와 비교해 사람이 확인할 자료를 저장합니다.

```text
원본 염색 영상
  → 색상·명암 보정 및 세포 영역 후보화
  → 핵 후보 분할과 세포 경계 추정
  → 예측 마스크 / 특징 CSV / 정답 비교 자료 저장
  → 사람 검토·수정 후 학습 데이터로 사용
```

예측 마스크는 정답 라벨이 아닙니다. 현재 정답이 있는 영상은 정상 009·019, abnormal 016·017 각 2장뿐이므로, 다른 영상에 그대로 적용하기 전에 경계를 확인하고 필요하면 수정해야 합니다. 이 코드는 의료 진단이나 학습 데이터의 자동 승인에 사용하지 않습니다.

## 현재 분할 방식

- 정상 코드: LAB 배경 색 차이로 세포 후보를 만들고, HSV 염색 색 정보와 LAB 국소 명암 차이를 이용해 약한 핵 후보를 거릅니다. 핵 후보는 Otsu 임계값과 거리 변환 watershed로 나눕니다.
- Abnormal 코드: HSV·LAB 색상/밝기와 주변 명암을 조합해 핵 후보를 선별하고, OpenCV K-means 색 군집과 형태학 연산·watershed로 마스크를 다듬습니다.
- 두 코드는 분할 임계값과 모폴로지 설정이 다른 실험용 파이프라인입니다. 두 방식 모두 학습된 분류 모델이나 딥러닝 추론을 포함하지 않습니다.

K-means는 픽셀 색 군집화에만 사용하며 학습된 모델을 저장하거나 이미지 종류를 예측하지 않습니다. 다만 K-means는 넓은 의미에서 비지도 군집화 알고리즘으로 분류될 수 있으므로, 과제 기준이 모든 머신러닝 알고리즘을 금지한다면 abnormal 코드에서 이 단계를 제외해야 합니다.

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
- `reports/mask_overlays/`: reference 경계(초록)와 예측 경계(자홍)를 원본 이미지에 겹친 확인용 이미지입니다.
- `docs/feature_calculation.md`: 계산 방법, 임시 규칙의 한계와 검토 상태를 기록합니다.

세포질 경계는 김승연을 포함한 팀의 육안 검토가 아직 완료되지 않았으므로 보고서에서 검토 대기로 표시했습니다. 규칙은 `핵 면적 > 1800 px² AND N/C > 0.15`인 탐색용 예시일 뿐이며, 네 명의 합의나 의학적 기준으로 확인되지 않았습니다. 결과 상태는 진단명이 아닙니다.

정상 009·019와 abnormal 016·017에 대해서는 정답 마스크를 이용해 고전 영상처리 임계값을 이미지별로 조정했습니다. 학습이나 머신러닝은 사용하지 않았습니다. 해당 네 장에 대한 튜닝이므로 다른 이미지에 대한 성능을 보장하지 않습니다. 자세한 설정과 제한은 `docs/feature_calculation.md`를 참고하세요.
