"""
악성(Malignant) 클래스 데이터 증강
38장 → 400장으로 늘리기
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def augment_image(img):
    """
    초음파 이미지에 적합한 증강 방법들
    무작위로 조합해서 적용
    """
    augmentations = []

    # 1. 수평 뒤집기
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        augmentations.append("flip")

    # 2. 회전 (-20 ~ +20도)
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, fillcolor=(0, 0, 0))
    augmentations.append(f"rot{int(angle)}")

    # 3. 밝기 조절 (초음파 밝기 차이 반영)
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 4. 대비 조절
    factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 5. 가우시안 블러 (초음파 노이즈 시뮬레이션)
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # 6. 크롭 후 리사이즈 (줌인 효과)
    if random.random() > 0.5:
        w, h = img.size
        crop_ratio = random.uniform(0.8, 0.95)
        left   = random.randint(0, int(w * (1 - crop_ratio)))
        top    = random.randint(0, int(h * (1 - crop_ratio)))
        right  = left + int(w * crop_ratio)
        bottom = top  + int(h * crop_ratio)
        img = img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)

    return img


def augment_malignant(
    malignant_dir: str = "./data/train/malignant",
    target_count: int = 400,
    seed: int = 42
):
    """
    악성 폴더 이미지를 증강해서 target_count장까지 늘리기
    
    Args:
        malignant_dir: 악성 이미지 폴더 경로
        target_count:  목표 이미지 수 (기본 400장)
    """
    random.seed(seed)
    src_dir = Path(malignant_dir)

    if not src_dir.exists():
        print(f"❌ 폴더 없음: {malignant_dir}")
        return

    # 기존 원본 이미지만 가져오기 (aug_ 접두사 제외)
    original_files = [
        f for f in src_dir.glob("*.JPG")
        if not f.stem.startswith("aug_")
    ] + [
        f for f in src_dir.glob("*.jpg")
        if not f.stem.startswith("aug_")
    ] + [
        f for f in src_dir.glob("*.png")
        if not f.stem.startswith("aug_")
    ]

    current_count = len(list(src_dir.glob("*.*")))
    need_count = target_count - current_count

    print(f"📂 악성 폴더: {src_dir}")
    print(f"📊 현재 이미지 수: {current_count}장")
    print(f"🎯 목표 이미지 수: {target_count}장")
    print(f"➕ 생성할 증강 이미지: {need_count}장\n")

    if need_count <= 0:
        print("✅ 이미 목표 수량 이상입니다!")
        return

    if not original_files:
        print("❌ 원본 이미지가 없습니다.")
        return

    generated = 0
    for i in range(need_count):
        # 원본 중 랜덤 선택
        src_file = random.choice(original_files)
        img = Image.open(src_file).convert('RGB')

        # 증강 적용
        aug_img = augment_image(img)

        # 저장
        save_name = f"aug_{i:04d}_{src_file.stem}.jpg"
        save_path = src_dir / save_name
        aug_img.save(save_path, quality=95)
        generated += 1

        if (i + 1) % 50 == 0:
            print(f"  진행중... {i+1}/{need_count}장 생성")

    final_count = len(list(src_dir.glob("*.*")))
    print(f"\n✅ 증강 완료!")
    print(f"  악성 이미지: {current_count}장 → {final_count}장")
    print(f"\n▶  이제 train.py를 다시 실행하세요!")


if __name__ == '__main__':
    augment_malignant(
        malignant_dir="./data/train/malignant",
        target_count=400
    )
