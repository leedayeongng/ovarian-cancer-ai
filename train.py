"""
학습 스크립트
핵심 평가지표: 민감도(Sensitivity), 특이도(Specificity), F1-Score, AUC-ROC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, roc_curve
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False
import os, json
from datetime import datetime

from model import HybridOvarianNet
from dataset import get_dataloaders, CLASS_NAMES


# ─────────────────────────────────────────────
# 임상 평가지표 계산
# ─────────────────────────────────────────────
def compute_clinical_metrics(y_true, y_pred, y_prob, class_names=CLASS_NAMES):
    """
    IOTA 기준에 맞는 임상 지표 계산
    
    핵심:
    - 민감도(Sensitivity/Recall): 실제 암 환자를 놓치지 않는 능력 → 암 진단 최우선
    - 특이도(Specificity): 정상인을 정상으로 맞추는 능력 → 불필요한 수술 방지
    - F1-Score: 민감도와 정밀도의 균형 지표
    - AUC-ROC: 전체적인 모델 판별력
    """
    results = {}

    # --- 전체 정확도 ---
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    results['accuracy'] = round(acc * 100, 2)

    # --- 클래스별 민감도 / 특이도 ---
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    sensitivities = {}
    specificities = {}

    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        sensitivities[class_names[i]] = round(sens * 100, 2)
        specificities[class_names[i]] = round(spec * 100, 2)

    results['sensitivity'] = sensitivities
    results['specificity'] = specificities

    # --- F1-Score (macro: 클래스 불균형 고려) ---
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    results['f1_macro'] = round(f1_macro * 100, 2)
    results['f1_per_class'] = {
        class_names[i]: round(float(f1_per_class[i]) * 100, 2)
        if i < len(f1_per_class) else 0.0
        for i in range(n_classes)
    }

    # --- AUC-ROC (다중 클래스: OvR 방식) ---
    try:
        y_prob_arr = np.array(y_prob)
        auc = roc_auc_score(y_true, y_prob_arr, multi_class='ovr', average='macro')
        results['auc_roc'] = round(auc * 100, 2)
    except Exception:
        results['auc_roc'] = None

    results['confusion_matrix'] = cm.tolist()

    return results


def print_clinical_report(metrics):
    """임상적 관점에서 결과 출력"""
    print("\n" + "=" * 60)
    print("  🏥 난소암 조기진단 AI - 임상 평가 리포트")
    print("=" * 60)
    print(f"\n  📊 전체 정확도    : {metrics['accuracy']}%")
    print(f"  📈 AUC-ROC       : {metrics['auc_roc']}%")
    print(f"  ⚖️  F1-Score(매크로): {metrics['f1_macro']}%")

    print("\n  ━━ 민감도 (Sensitivity) - 암 환자를 놓치지 않는 능력 ━━")
    for cls, val in metrics['sensitivity'].items():
        bar = "█" * int(val / 5)
        flag = "⚠️ 낮음" if val < 85 and "악성" in cls else ""
        print(f"  {cls[:10]:12s}: {val:5.1f}% {bar} {flag}")

    print("\n  ━━ 특이도 (Specificity) - 정상인을 정상으로 보는 능력 ━━")
    for cls, val in metrics['specificity'].items():
        bar = "█" * int(val / 5)
        print(f"  {cls[:10]:12s}: {val:5.1f}% {bar}")

    print("\n  ━━ F1-Score (클래스별) ━━")
    for cls, val in metrics['f1_per_class'].items():
        print(f"  {cls[:10]:12s}: {val:5.1f}%")

    # IOTA 기준 임상 조언
    malignant_sens = metrics['sensitivity'].get("Malignant (악성/난소암)", 0)
    print("\n  ━━ 임상 판정 (IOTA 기준) ━━")
    if malignant_sens >= 95:
        print("  ✅ 악성 민감도 95% 이상 → 임상 적용 검토 가능")
    elif malignant_sens >= 85:
        print("  🟡 악성 민감도 85~95% → 추가 학습 또는 앙상블 권장")
    else:
        print("  🔴 악성 민감도 85% 미만 → 실제 운용 불가, 모델 재설계 필요")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"💻 디바이스: {self.device}")

        # 모델
        self.model = HybridOvarianNet(
            num_classes=config['num_classes'],
            embed_dim=config['embed_dim'],
            pretrained=True
        ).to(self.device)

        # 손실함수: 클래스 가중치 적용 (악성 클래스 가중치 높임)
        # 악성(Malignant)을 놓쳤을 때 패널티를 2배로
        class_weights = torch.tensor([1.0, 1.5, 2.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 옵티마이저: CNN과 Transformer를 다른 학습률로
        self.optimizer = optim.AdamW([
            {'params': self.model.cnn.parameters(), 'lr': config['lr'] * 0.1},
            {'params': self.model.patch_embed.parameters(), 'lr': config['lr']},
            {'params': self.model.transformer.parameters(), 'lr': config['lr']},
            {'params': self.model.classifier.parameters(), 'lr': config['lr']}
        ], weight_decay=1e-4)

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'], eta_min=1e-6
        )

        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        self.best_val_f1 = 0.0
        os.makedirs(config['save_dir'], exist_ok=True)

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient Clipping: 불안정한 학습 방지
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(loader), correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []
        total_loss = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        metrics = compute_clinical_metrics(all_labels, all_preds, all_probs)
        return total_loss / len(loader), metrics

    def train(self, loaders):
        print(f"\n🚀 학습 시작 (총 {self.config['epochs']} 에폭)\n")

        for epoch in range(1, self.config['epochs'] + 1):
            train_loss, train_acc = self.train_epoch(loaders['train'])
            val_loss, val_metrics = self.evaluate(loaders['val'])
            self.scheduler.step()

            val_f1 = val_metrics['f1_macro']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_f1)

            print(f"Epoch {epoch:3d}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                  f"Val Loss: {val_loss:.4f} F1: {val_f1:.1f}%")

            # 최고 F1 모델 저장
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                ckpt_path = os.path.join(self.config['save_dir'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_metrics': val_metrics,
                    'config': self.config
                }, ckpt_path)
                print(f"  💾 최고 모델 저장 (F1: {val_f1:.1f}%)")

        # 최종 테스트 평가
        if 'test' in loaders:
            print("\n📋 최종 테스트 평가 중...")
            _, test_metrics = self.evaluate(loaders['test'])
            print_clinical_report(test_metrics)
            self.plot_results()

    def plot_results(self):
        """학습 곡선 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(self.history['train_loss']) + 1)

        axes[0].plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', ms=3)
        axes[0].plot(epochs, self.history['val_loss'], 'r-o', label='Val Loss', ms=3)
        axes[0].set_title('학습/검증 손실')
        axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, self.history['val_f1'], 'g-o', label='Val F1', ms=3)
        axes[1].plot(epochs, self.history['val_acc'], 'm-o', label='Val Accuracy', ms=3)
        axes[1].set_title('검증 성능 (F1 / Accuracy)')
        axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.config['save_dir'], 'training_curve.png')
        plt.savefig(save_path, dpi=150)
        print(f"📊 학습 곡선 저장: {save_path}")
        plt.show()


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
if __name__ == '__main__':
    CONFIG = {
        # 데이터
        'data_root': './data',          # train/val/test 폴더 위치
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,

        # 모델
        'num_classes': 3,              # 양성/경계성/악성
        'embed_dim': 256,

        # 학습
        'epochs': 50,
        'lr': 1e-4,

        # 저장
        'save_dir': './checkpoints',
    }

    # DataLoader 생성
    loaders = get_dataloaders(
        data_root=CONFIG['data_root'],
        batch_size=CONFIG['batch_size'],
        image_size=CONFIG['image_size'],
        num_workers=CONFIG['num_workers']
    )

    # 학습 시작
    trainer = Trainer(CONFIG)
    trainer.train(loaders)
