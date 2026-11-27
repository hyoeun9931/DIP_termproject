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

`masks/` 아래에는 정상·비정상 reference 마스크와 코드 예측 마스크를 따로 저장했습니다.

- `masks/norm/reference/`: 정상 데이터의 DAT 좌표 기반 정답 마스크. 원본 이미지 기준 `003`, `009`, `011`, `019`가 있으며 이미지별 폴더에 전체 마스크와 세포별 마스크가 있습니다.
- `masks/norm/train_roi_prediction/`: `train_image_norm`의 512×512 ROI에 코드를 적용한 핵·세포질 예측 마스크와 CSV 결과
- `masks/abnorm/reference/`: `train_image_abnorm`과 같은 원본 이미지의 DAT 좌표 기반 정답 마스크 `001`–`020`
- `masks/abnorm/train_prediction/`: abnormal 이미지에 코드를 적용한 핵·세포질 예측 마스크와 CSV 결과

각 세포별 `cell` 마스크는 핵을 포함한 세포 외곽이고, `cytoplasm` 마스크는 해당 외곽에서 핵 영역을 뺀 영역입니다. 정상 reference 마스크는 원본 2048×1536 좌표이고 정상 train ROI 예측 마스크는 512×512이므로 두 세트를 픽셀 단위로 직접 비교할 수 없습니다. abnormal reference와 예측은 같은 원본 이미지 좌표를 사용합니다.
