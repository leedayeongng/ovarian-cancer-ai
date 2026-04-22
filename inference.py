"""
추론 + Grad-CAM 시각화
"AI가 어느 부분을 보고 암이라고 판단했는지" 히트맵으로 시각화
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
from torchvision import transforms
from PIL import Image
import cv2

from model import HybridOvarianNet
from dataset import CLASS_NAMES


def load_model(checkpoint_path, device='cpu'):
    """저장된 모델 로드"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', {'num_classes': 3, 'embed_dim': 256})
    model = HybridOvarianNet(
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        pretrained=False
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"✅ 모델 로드 완료 (Epoch: {ckpt.get('epoch', '?')})")
    return model


def preprocess_image(image_path, image_size=224):
    """초음파 이미지 전처리"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0), img   # (1, C, H, W), PIL원본


def predict_with_gradcam(model, image_path, device='cpu'):
    """
    단일 이미지 예측 + Grad-CAM 시각화
    Returns:
        prediction: 예측 클래스 이름
        confidence: 신뢰도 (%)
        cam: 히트맵 배열
    """
    img_tensor, img_pil = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    # 예측
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_class = probs.argmax().item()
    confidence = probs[pred_class].item() * 100

    # Grad-CAM (악성 클래스 기준 히트맵)
    malignant_idx = 2
    cam, _ = model.get_gradcam(img_tensor, class_idx=malignant_idx)

    return {
        'class': CLASS_NAMES[pred_class],
        'class_idx': pred_class,
        'confidence': confidence,
        'probs': {CLASS_NAMES[i]: round(probs[i].item() * 100, 1) for i in range(3)},
        'cam': cam,
        'img_pil': img_pil
    }


def visualize_result(result, save_path=None):
    """진단 결과 + Grad-CAM 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('🔬 난소암 AI 진단 결과 (Grad-CAM 분석)', fontsize=14, fontweight='bold')

    img_np = np.array(result['img_pil'])

    # 1) 원본 초음파
    axes[0].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
    axes[0].set_title('원본 초음파 영상')
    axes[0].axis('off')

    # 2) Grad-CAM 히트맵 오버레이
    cam_resized = cv2.resize(result['cam'], (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    axes[1].imshow(overlay)
    axes[1].set_title('Grad-CAM (악성 의심 부위)')
    axes[1].axis('off')

    # 3) 확률 막대그래프
    classes = list(result['probs'].keys())
    probs = list(result['probs'].values())
    colors = ['#4CAF50', '#FF9800', '#F44336']  # 양성/경계성/악성
    bars = axes[2].barh(classes, probs, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel('확률 (%)')
    axes[2].set_title('클래스별 예측 확률')
    for bar, prob in zip(bars, probs):
        axes[2].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{prob:.1f}%', va='center', fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)

    # 최종 판정 텍스트
    verdict_color = ['green', 'orange', 'red'][result['class_idx']]
    fig.text(0.5, 0.01,
             f"판정: {result['class']}  |  신뢰도: {result['confidence']:.1f}%",
             ha='center', fontsize=13, color=verdict_color, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 결과 저장: {save_path}")
    plt.show()

    # 임상 조언 출력
    print(f"\n━━ 🏥 임상 해석 ━━")
    print(f"  예측: {result['class']}")
    print(f"  신뢰도: {result['confidence']:.1f}%")
    if result['class_idx'] == 2:
        print("  ⚠️  악성 의심 → IOTA M-feature 재확인 및 즉시 전문의 협진 권장")
    elif result['class_idx'] == 1:
        print("  🟡 경계성 → 추적 관찰 또는 추가 영상 검사 권장 (6개월 f/u)")
    else:
        print("  ✅ 양성 소견 → IOTA B-feature 확인 후 주기적 추적 관찰")


# ─────────────────────────────────────────────
# 실행 예시
# ─────────────────────────────────────────────
if __name__ == '__main__':
    CHECKPOINT = './checkpoints/best_model.pth'
    IMAGE_PATH = './sample_ultrasound.jpg'   # 테스트할 초음파 이미지

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(CHECKPOINT, device)

    result = predict_with_gradcam(model, IMAGE_PATH, device)
    visualize_result(result, save_path='./result_gradcam.png')
