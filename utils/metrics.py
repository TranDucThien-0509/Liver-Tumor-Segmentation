"""
utils/metrics.py
────────────────
FastAI-compatible metrics cho segmentation đa lớp.

  foreground_acc      — accuracy chỉ tính trên vùng non-background
  cust_foreground_acc — wrapper bỏ qua idx=3 (tính cả 3 class)
  DSAwareDiceMulti    — DiceMulti xử lý list output của deep supervision
"""

import torch
from fastai.metrics import DiceMulti
from configs.config import DataConfig


# ──────────────────────────────────────────────────────────────────────────────
# Accuracy
# ──────────────────────────────────────────────────────────────────────────────

def foreground_acc(inp: torch.Tensor,
                   targ: torch.Tensor,
                   bkg_idx: int = 0,
                   axis: int = 1) -> torch.Tensor:
    """
    Accuracy bỏ qua background class.
    inp  : (B, C, H, W) logits
    targ : (B, H, W) hoặc (B, 1, H, W) labels
    """
    targ = targ.squeeze(1) if targ.ndim == 4 else targ
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


def cust_foreground_acc(inp, targ: torch.Tensor) -> torch.Tensor:
    """
    Accuracy toàn bộ 3 class — trick bkg_idx=3 để mask luôn True.
    Tự động lấy final pred khi inp là list (deep supervision).
    """
    if isinstance(inp, (list, tuple)):
        inp = inp[-1]
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# DiceMulti wrapper cho deep supervision
# ──────────────────────────────────────────────────────────────────────────────

class DSAwareDiceMulti(DiceMulti):
    """
    DiceMulti của FastAI nhưng xử lý đầu ra list của VM-UNetV2.

    Khi training: model trả về [aux1, aux2, aux3, final]
    Khi eval    : model trả về tensor (B, C, H, W)

    Cách hoạt động: override accumulate, tạm thời swap learn.pred
    sang final tensor trước khi gọi logic Dice của class cha.
    """

    def accumulate(self, learn):
        pred_orig = learn.pred
        # Nếu là list deep supervision, chỉ lấy final output
        if isinstance(pred_orig, (list, tuple)):
            learn.pred = pred_orig[-1]
        try:
            super().accumulate(learn)
        finally:
            learn.pred = pred_orig   # khôi phục để không ảnh hưởng gì khác


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_metrics(data_cfg: DataConfig = None):
    """
    Trả về list metrics cho Learner.
    data_cfg tùy chọn, hiện chưa dùng nhưng để mở rộng sau.
    """
    return [cust_foreground_acc, DSAwareDiceMulti()]
