"""
utils/losses.py
───────────────
Loss functions cho VM-UNetV2:

  MultiClassDiceLoss   — Dice loss chuẩn cho segmentation đa lớp
  CombinedLoss         — CE + Dice (dùng khi KHÔNG có deep supervision)
  DeepSupervisionLoss  — Wrapper bọc bất kỳ loss nào, cộng có trọng số
                         4 đầu ra của VM-UNetV2 ở chế độ training
"""

from typing import List

import torch
import torch.nn as nn
from fastai.losses import CrossEntropyLossFlat

from configs.config import TrainConfig


# ──────────────────────────────────────────────────────────────────────────────
# Dice Loss đa lớp
# ──────────────────────────────────────────────────────────────────────────────

class MultiClassDiceLoss(nn.Module):
    """
    Dice loss cho bài toán segmentation đa lớp (N class).

    Input
    -----
    pred   : (B, C, H, W)  raw logits hoặc probabilities
    target : (B, H, W)     long tensor, giá trị ∈ [0, C-1]

    Công thức
    ---------
    Dice_c = (2 * |P_c ∩ T_c| + ε) / (|P_c| + |T_c| + ε)
    Loss   = 1 - mean(Dice_c)  trung bình trên tất cả class
    """

    def __init__(self, axis: int = 1, smooth: float = 1e-5):
        super().__init__()
        self.axis   = axis
        self.smooth = smooth

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pred         = torch.softmax(pred, dim=self.axis)
        num_classes  = pred.shape[1]

        # One-hot encode target: (B,H,W) → (B,C,H,W)
        eye         = torch.eye(num_classes, device=pred.device)
        target_1hot = eye[target.squeeze(1).long()].permute(0, 3, 1, 2)

        dims         = (2, 3)
        intersection = (pred * target_1hot).sum(dim=dims)
        union        = pred.sum(dim=dims) + target_1hot.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Loss kết hợp CE + Dice  (dùng cho single-output, không deep supervision)
# ──────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    CrossEntropy + Dice với hệ số tùy chỉnh.

    Dùng khi model ở chế độ INFERENCE (trả về 1 tensor)
    hoặc khi không cần deep supervision.
    """

    def __init__(self, ce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 smooth: float = 1e-5):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.ce   = CrossEntropyLossFlat(axis=1)
        self.dice = MultiClassDiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return (self.ce_weight   * self.ce(pred, target) +
                self.dice_weight * self.dice(pred, target))


# ──────────────────────────────────────────────────────────────────────────────
# Deep Supervision Loss  (dùng cho VM-UNetV2 ở chế độ training)
# ──────────────────────────────────────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Bọc CombinedLoss và áp dụng cho 4 đầu ra của VM-UNetV2.

    Trong training, model trả về list [aux1, aux2, aux3, final].
    Trong inference, model trả về tensor duy nhất → fallback về CombinedLoss.

    Trọng số mặc định: [0.2, 0.3, 0.4, 1.0]
    → final prediction được weight mạnh hơn 5× so với aux1.
    """

    def __init__(self, train_cfg: TrainConfig):
        super().__init__()
        self.weights  = train_cfg.ds_weights
        self.base_loss = CombinedLoss(
            ce_weight=train_cfg.ce_weight,
            dice_weight=train_cfg.dice_weight,
            smooth=train_cfg.dice_smooth,
        )

    def forward(self, preds, target: torch.Tensor) -> torch.Tensor:
        # inference mode → model trả về tensor, không phải list
        if isinstance(preds, torch.Tensor):
            return self.base_loss(preds, target)

        # training mode → list của 4 tensors
        assert len(preds) == len(self.weights), (
            f"Expected {len(self.weights)} predictions, got {len(preds)}"
        )
        total = preds[0].new_zeros(1)
        for pred, w in zip(preds, self.weights):
            total = total + w * self.base_loss(pred, target)
        return total
