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
