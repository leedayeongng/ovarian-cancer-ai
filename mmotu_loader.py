"""
MMOTU 데이터 로더 (txt 파일 기반)
파일명 + 클래스ID로 데이터 구성
"""

import os
import shutil
import random
from pathlib import Path


# MMOTU 8개 클래스 → 임상 3클래스 매핑
MMOTU_TO_CLINICAL = {
    0: (0, "benign"),      # Chocolate cyst
    1: (0, "benign"),      # Serous cystadenoma
    2: (0, "benign"),      # Teratoma
    3: (0, "benign"),      # Theca cell tumor (여기선 benign으로)
    4: (0, "benign"),      # Simple cyst
    5: (0, "benign"),      # Normal ovary
    6: (0, "benign"),      # Mucinous cystadenoma
    7: (2, "malignant"),   # High grade serous ← 핵심 타겟
}

MMOTU_CLASS_NAMES = {
    0: "Chocolate cyst",
    1: "Serous cystadenoma",
    2: "Teratoma",
    3: "Theca cell tumor",
    4: "Simple cyst",
    5: "Normal ovary",
    6: "Mucinous cystadenoma",
    7: "High grade serous (악성)"
}

CLINICAL_NAMES = {
    0: "benign",
    1: "borderline",
    2: "malignant"
}


def read_cls_txt(txt_path):
    """txt 파일 읽어서 {파일명: 클래스ID} 딕셔너리 반환"""
    data = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            filename = parts[0]
            class_id = int(parts[1])
            data[filename] = class_id
    return data


def prepare_mmotu_dataset(
    mmotu_root: str = r"C:\dev\ovarian_project\OTU_2d",
    output_root: str = "./data"
):
    mmotu_path = Path(mmotu_root)
    images_path = mmotu_path / "images"
    train_txt = mmotu_path / "train_cls.txt"
    val_txt   = mmotu_path / "val_cls.txt"

    print(f"📂 MMOTU 경로: {mmotu_path}")
    print(f"📂 출력 경로: {output_root}\n")

    # txt 파일 읽기
    train_data = read_cls_txt(train_txt)
    val_data   = read_cls_txt(val_txt)

    # val을 val/test로 반반 나누기
    val_items  = list(val_data.items())
    random.seed(42)
    random.shuffle(val_items)
    half = len(val_items) // 2
    val_final  = dict(val_items[:half])
    test_final = dict(val_items[half:])

    splits = {
        "train": train_data,
        "val":   val_final,
        "test":  test_final
    }

    # 클래스별 통계
    stats = {s: {0: 0, 1: 0, 2: 0} for s in splits}

    for split_name, split_data in splits.items():
        for filename, class_id in split_data.items():
            clinical_id, clinical_name = MMOTU_TO_CLINICAL[class_id]
            src = images_path / filename
            if not src.exists():
                print(f"  ⚠️  파일 없음: {filename}")
                continue

            dest_dir = Path(output_root) / split_name / clinical_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / filename)
            stats[split_name][clinical_id] += 1

    # 결과 출력
    print("=" * 50)
    print("✅ 변환 완료!")
    for split_name, class_counts in stats.items():
        total = sum(class_counts.values())
        print(f"\n  [{split_name.upper()}] 총 {total}장")
        for cid, cname in CLINICAL_NAMES.items():
            print(f"    {cname:10s}: {class_counts[cid]}장")
    print("=" * 50)


def visualize_distribution(data_root: str = "./data"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False

    data_root = Path(data_root)
    colors = {"benign": "#4CAF50", "borderline": "#FF9800", "malignant": "#F44336"}

    split_counts = {}
    for split in ["train", "val", "test"]:
        split_path = data_root / split
        if not split_path.exists():
            continue
        counts = {}
        for cls_dir in sorted(split_path.iterdir()):
            if cls_dir.is_dir():
                counts[cls_dir.name] = len(list(cls_dir.glob("*.*")))
        split_counts[split] = counts

    fig, axes = plt.subplots(1, len(split_counts), figsize=(14, 5))
    if len(split_counts) == 1:
        axes = [axes]

    for ax, (split, counts) in zip(axes, split_counts.items()):
        bar_colors = [colors.get(k, "gray") for k in counts.keys()]
        bars = ax.bar(counts.keys(), counts.values(),
                      color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f"{split.upper()}", fontsize=13, fontweight='bold')
        ax.set_ylabel("이미지 수")
        for bar, val in zip(bars, counts.values()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, str(val),
                    ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("MMOTU 클래스 분포", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = str(data_root / "distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 분포 그래프 저장: {save_path}")
    plt.show()


if __name__ == '__main__':
    prepare_mmotu_dataset(
        mmotu_root=r"C:\dev\ovarian_project\OTU_2d",
        output_root="./data"
    )
    visualize_distribution("./data")
    print("\n▶  이제 train.py 실행하세요!")