# ovarian-cancer-ai
난소암 조기진단 AI (CNN+Transformer + Grad-CAM)

# 난소암 조기진단 AI
CNN + Transformer 하이브리드 모델을 활용한 난소 초음파 영상 분류 시스템

## 프로젝트 소개
- MMOTU 난소 초음파 데이터셋으로 양성/악성 분류
- Grad-CAM으로 AI 판단 근거 시각화
- IOTA 기준 기반 임상 평가지표 (민감도/특이도/F1) 적용

## 모델 구조
- CNN (ResNet34) → 미세 종양 결절 감지
- Transformer → 전체 난소 구조 파악
- Early-fusion 하이브리드 방식

## 데이터셋
MMOTU (Multi-Modality Ovarian Tumor Ultrasound) Dataset
- 다운로드: https://www.kaggle.com/datasets/orvile/mmotu-ovarian-ultrasound-images-dataset
- OTU_2d 폴더를 프로젝트 루트에 넣으세요

## 실행 방법
```bash
# 패키지 설치
pip install -r requirements.txt

# 데이터 변환
python mmotu_loader.py

# 악성 데이터 증강
python augment_malignant.py

# 학습
python train.py

# 추론 + Grad-CAM
python inference.py
```

## 파일 구조
ovarian_project/
├── model.py            # CNN+Transformer 하이브리드 모델
├── dataset.py          # 데이터셋 & 전처리
├── train.py            # 학습 + 임상 평가지표
├── inference.py        # 추론 + Grad-CAM 시각화
├── mmotu_loader.py     # MMOTU 데이터 변환
├── augment_malignant.py # 악성 데이터 증강
└── requirements.txt    # 패키지 목록

## 평가지표
- 민감도 (Sensitivity): 실제 암 환자를 놓치지 않는 능력
- 특이도 (Specificity): 정상인을 정상으로 보는 능력
- F1-Score / AUC-ROC

## 개발 환경
- Python 3.10
- PyTorch 2.11.0 (CPU)
- Windows 11 / PyCharm / Anaconda