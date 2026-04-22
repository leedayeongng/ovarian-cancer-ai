"""
난소 초음파 데이터셋 (MMOTU / AI-Hub 호환)
IOTA 기준 기반 데이터 구조 설계
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 클래스 레이블 정의 (IOTA 표준 기준)
# ─────────────────────────────────────────────
CLASS_NAMES = {
    0: "Benign (양성)",
    1: "Borderline (경계성)",
    2: "Malignant (악성/난소암)"
}

IOTA_FEATURES = {
    "B-features (양성 특징)": [
        "단방성 낭종 (Unilocular cyst)",
        "고형 성분 없음 (No solid components)",
        "음향 음영 (Acoustic shadows)",
        "매끄러운 내벽 (Smooth inner wall)",
        "낮은 혈류 (No flow)"
    ],
    "M-features (악성 특징)": [
        "불규칙한 고형 성분 (Irregular solid component)",
        "복수 (Ascites)",
        "복막 전이 (Peritoneal metastases)",
        "불규칙한 다방성 낭종 (Irregular multilocular solid tumor)",
        "높은 혈류 (Strong flow, Color score 4)"
    ]
}


# ─────────────────────────────────────────────
# 데이터 전처리 & 증강
# ─────────────────────────────────────────────
def get_transforms(mode='train', image_size=224):
    """
    초음파 영상 특성에 맞는 전처리
    - 가우시안 블러: 초음파 노이즈(스펙클) 완화
    - 수평 플립만: 수직 플립은 해부학적으로 부자연스러움
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet 평균 (전이학습)
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ─────────────────────────────────────────────
# 데이터셋 클래스
# ─────────────────────────────────────────────
class OvarianUltrasoundDataset(Dataset):
    """
    난소 초음파 데이터셋
    
    디렉토리 구조:
        data/
        ├── train/
        │   ├── benign/        (0)
        │   ├── borderline/    (1)
        │   └── malignant/     (2)
        ├── val/
        └── test/
    
    또는 CSV 파일:
        image_path, label (0/1/2)
    """
    def __init__(self, root_dir=None, csv_file=None, mode='train', image_size=224):
        self.transform = get_transforms(mode, image_size)
        self.samples = []  # (image_path, label) 리스트
        self.class_counts = [0, 0, 0]

        if csv_file:
            self._load_from_csv(csv_file)
        elif root_dir:
            self._load_from_dir(root_dir)
        else:
            raise ValueError("root_dir 또는 csv_file 중 하나를 지정하세요.")

        print(f"[{mode.upper()}] 데이터셋 로드 완료")
        for i, name in CLASS_NAMES.items():
            print(f"  {name}: {self.class_counts[i]}장")

    def _load_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            label = int(row['label'])
            self.samples.append((row['image_path'], label))
            self.class_counts[label] += 1

    def _load_from_dir(self, root_dir):
        class_dirs = {
            'benign': 0,
            'borderline': 1,
            'malignant': 2
        }
        for class_name, label in class_dirs.items():
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️  폴더 없음: {class_path}")
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(class_path, fname), label))
                    self.class_counts[label] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

    def get_sampler(self):
        """클래스 불균형 해소: WeightedRandomSampler"""
        weights = []
        total = sum(self.class_counts)
        class_weights = [total / (c + 1e-6) for c in self.class_counts]
        for _, label in self.samples:
            weights.append(class_weights[label])
        return WeightedRandomSampler(weights, len(weights))


# ─────────────────────────────────────────────
# DataLoader 생성 헬퍼
# ─────────────────────────────────────────────
def get_dataloaders(data_root, batch_size=32, image_size=224, num_workers=4):
    """
    train/val/test DataLoader 반환
    train은 WeightedRandomSampler로 클래스 불균형 처리
    """
    loaders = {}
    for mode in ['train', 'val', 'test']:
        path = os.path.join(data_root, mode)
        if not os.path.exists(path):
            print(f"⚠️  {mode} 폴더 없음, 스킵")
            continue
        dataset = OvarianUltrasoundDataset(
            root_dir=path, mode=mode, image_size=image_size
        )
        sampler = dataset.get_sampler() if mode == 'train' else None
        loaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(mode == 'train' and sampler is None),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    return loaders
