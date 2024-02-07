# level-2 대회 :글자 검출 프로젝트 대회

## 팀 소개
| 이름 | 역할 |
| ---- | --- |
| [박상언](https://github.com/PSangEon) | 라벨링 가이드 작성, 라벨링 툴 실험(CVAT), 라벨링, 외부 데이터 가중치를 사용한 미세 조정, 깃헙 템플릿 작성 |
| [지현동](https://github.com/tolfromj) | 라벨링 가이드 작성, 라벨링 툴 실험(LabelMe), 라벨링, 데이터 증강 기법 실험(dilation), 외부 데이터를 포함한 모형 학습 |
| [오왕택](https://github.com/ohkingtaek) | 라벨링 가이드 작성, 라벨링 툴 실험(CVAT), 라벨링 검수, 데이터 시각화, 베이스라인 코드 실험, </br> 데이터 증강 기법 실험(Curving), 깃헙 템플릿 작성, UFO to COCO format 코드 작성 |
| [이동호](https://github.com/as9786) | 라벨링 가이드 작성, 라벨링 툴 실험(CVAT), 라벨링, EDA, 정규화 코드, </br>데이터 증강 기법 실험(Distortion, Gaussian Noise, CLAHE), 외부 데이터 수집,</br>외부 데이터 사전 학습, 깃헙 템플릿 작성, AiHub format to UFO format, COCO format to UFO format |
| [송지민](https://github.com/Remiing) | 라벨링 가이드 작성, 라벨링 툴 실험(CVAT), 라벨링, 데이터 증강 기법 실험(noise, CLAHE), 깃헙 템플릿 작성 |
| [이주헌](https://github.com/LeeJuheonT6138) | 라벨링 가이드 작성, 라벨링 툴 실험(CVAT), 라벨링, 데이터 증강 기법 실험(blur), 깃헙 템플릿 작성, 깃헙 관리 |

## 프로젝트 소개
![](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-09/assets/49676680/5eb9615b-2b7e-4a06-bbb7-2a60b86e32b7)

우리는 진료 확인서, 진단서와 같은 의학 데이터에서 글자가 존재하는 영역을 추출하는 프로젝트를 진행하였다. OCR 과정 중에서 글자의 내용을 확인하기 전에 영역을 잘 구하는 것이 주 목표였다.


## 프로젝트 일정
프로젝트 전체 일정
- 01/24 10:00 ~ 02/01 19:00

프로젝트 세부 일정
- 01/22 ~ 01/23 강의 수강, 제공 데이터 및 코드 확인
- 01/24 ~ 01/26 Baseline code 작성, Git branch 생성, EDA, 라벨링 가이드 정의, 라벨링 및 검수, 정답 형식 변환 코드 작성, 외부 데이터 수집
- 01/29 ~ 02/01 정규화 수행, 다양한 증강 기법 수행, 모형 학습

## 프로젝트 수행
- EDA : 클래스 불균형 확인, 경계박스 비율 확인
- Baseline code, 사용할 라벨링 툴 선택 : LabelMe, CVAT, Supervisely 비교하고 실험
- 라벨링 가이드 정의, 라벨링 및 검수 : 가이드 라인을 정의하고 라벨링 검수자를 정하여 기준에 맞게 라벨링했는지 검수
- 라벨링 형식 변환 코드 : 라벨링을 고쳐 적재적소에 활용하기 위한 COCO to UFO, UFO to COCO, AiHub to UFO 변환
- 다양한 증강 기법 실험 : Albumentation뿐만 아니라 Geometric한 변환도 수행하려 했으나 시간의 한계에 부딪힘

## 프로젝트 결과
- 프로젝트 결과는 Public 9등, Private 9등이라는 결과를 얻었습니다.
![](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-09/assets/49676680/c0bb7e5e-3a9e-4e76-b3fe-630f32ae3ddc)
![](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-09/assets/49676680/8c6800ab-8dd9-48d5-a71d-77e280dbba8a)

## Wrap-Up Report

- [Wrap-Up Report](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-09/blob/main/docs/SCV_Lv2_Wrap_up_report_OCR.pdf)

## File Tree

```bash
├── code
│   ├── configs
│   │   ├── train_0.yaml
│   │   └── train.yaml
│   ├── dataset.py
│   ├── detect.py
│   ├── deteval.py
│   ├── east_dataset.py
│   ├── inference.py
│   ├── loss.py
│   ├── model.py
│   ├── preprocess.py
│   ├── requirements.txt
│   ├── train.py
│   ├── train_refactor.py
│   ├── train_valid.py
│   └── utils.py
├── docs
│   └── SCV_Lv2_Wrap_up_report_OCR.pdf
└── notebooks
    ├── AIhub_to_UFO.ipynb
    ├── EDA.ipynb
    ├── clahe.ipynb
    ├── coco_to_ufo.ipynb
    ├── concatenate_AIHub.ipynb
    ├── data_split.ipynb
    └── ufo_to_coco.ipynb
```

| File(.py) | Description |
| --- | --- |
| dataset.py | dataset정의 및 관련 코드 |
| preprocess.py | 빠른 학습을 위한 데이터 전처리 코드 |
| train.py | 학습 코드 |
| train_refactor.py | 학습 wandb 추가 및 리팩토링 코드 |
| train_valid.py | 학습과 검증 진행 코드 |
| utils.py | 학습에 필요한 나머지 코드 |
| AIhub_to_UFO.ipynb | AiHub to UFO 변환 |
| EDA.ipynb | EDA 코드 |
| clahe.ipynb | clahe augmentation 코드 |
| coco_to_ufo.ipynb | COCO to UFO 변환 |
| concatenate_AIHub.ipynb | aihub와 데이터 병합 코드 |
| data_split.ipynb | 데이터 train 과 validation 분할 코드 |
| ufo_to_coco.ipynb | UFO to AiHub 변환 |

## License
네이버 부스트캠프 AI Tech 교육용 데이터로 대회용 데이터임을 알려드립니다.
