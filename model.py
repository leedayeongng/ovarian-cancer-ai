"""
난소암 조기진단 AI - Hybrid CNN + Transformer (Early-fusion)
평가지표: 민감도(Sensitivity), 특이도(Specificity), F1-Score, AUC
XAI: Grad-CAM 히트맵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────
# 1. CNN 백본 (ResNet34 기반 특징 추출)
# ─────────────────────────────────────────────
class CNNBackbone(nn.Module):
    """ResNet34 기반 로컬 특징 추출 (미세 질감, 결절 탐지)"""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)
        # 마지막 FC 레이어 제거 → feature map 추출
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = 512

    def forward(self, x):
        return self.feature_extractor(x)  # (B, 512, H', W')


# ─────────────────────────────────────────────
# 2. Patch Embedding (CNN feature → Transformer 입력)
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """CNN feature map을 패치 시퀀스로 변환"""
    def __init__(self, in_channels=512, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.proj(x)           # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        return x


# ─────────────────────────────────────────────
# 3. Transformer Encoder (전역 구조 파악)
# ─────────────────────────────────────────────
class TransformerEncoder(nn.Module):
    """멀티헤드 어텐션으로 전역 난소 구조 파악"""
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True   # Pre-LN: 학습 안정성↑
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)    # CLS 토큰 앞에 추가
        x = self.transformer(x)
        return x[:, 0]                     # CLS 토큰만 반환 → 전역 표현


# ─────────────────────────────────────────────
# 4. Hybrid Model (Early-Fusion CNN + Transformer)
# ─────────────────────────────────────────────
class HybridOvarianNet(nn.Module):
    """
    난소암 진단 하이브리드 모델
    - CNN: 미세 종양 결절, 불규칙한 고형 성분 감지 (IOTA 기준)
    - Transformer: 전체 난소 구조, 복수, 혈류 분포 파악
    - Early-fusion: CNN 특징 추출 직후 Transformer에 입력
    """
    def __init__(self, num_classes=3, embed_dim=256, pretrained=True):
        """
        num_classes:
            0 = Benign (양성)
            1 = Borderline (경계성 종양)
            2 = Malignant (악성/난소암)
        """
        super().__init__()
        self.cnn = CNNBackbone(pretrained=pretrained)
        self.patch_embed = PatchEmbedding(
            in_channels=self.cnn.out_channels,
            embed_dim=embed_dim
        )
        self.transformer = TransformerEncoder(embed_dim=embed_dim)

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Grad-CAM을 위한 훅 저장소
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Grad-CAM을 위한 forward/backward 훅 등록"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # CNN 마지막 레이어에 훅 등록
        target_layer = list(self.cnn.feature_extractor.children())[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x):
        # 1) CNN: 로컬 특징 추출
        cnn_feat = self.cnn(x)             # (B, 512, H', W')

        # 2) Patch Embedding: CNN feature → 패치 시퀀스
        patches = self.patch_embed(cnn_feat)  # (B, N, embed_dim)

        # 3) Transformer: 전역 컨텍스트 파악
        global_feat = self.transformer(patches)  # (B, embed_dim)

        # 4) 분류
        logits = self.classifier(global_feat)
        return logits

    def get_gradcam(self, x, class_idx=None):
        """
        Grad-CAM 히트맵 생성
        Args:
            x: 입력 이미지 텐서 (1, C, H, W)
            class_idx: 시각화할 클래스 (None=예측 클래스)
        Returns:
            cam: 정규화된 히트맵 (H, W)
        """
        self.eval()
        x.requires_grad_(True)
        logits = self.forward(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.zero_grad()
        logits[0, class_idx].backward()

        # GAP (Global Average Pooling) of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # 정규화 [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx
