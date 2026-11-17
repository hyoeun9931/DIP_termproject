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
