"""
vmunet_v2.py  —  VM-UNetV2 wrapper
====================================
Nâng cấp so với V1:
  • Deep supervision: trả về 4 logit maps trong training (coarse → fine),
    chỉ trả về map cuối trong inference.
  • Upsample tất cả aux outputs về resolution gốc (H×W) trước khi trả về.
  • Thêm DeepSupervisionLoss helper để tính weighted loss trên 4 outputs.
  • Logic load_from() giữ nguyên từ V1, chỉ refactor code cho gọn hơn.
"""

from .vmamba_v2 import VSSM
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class VMUNet(nn.Module):
    """
    VM-UNetV2: wrapper bọc VSSM V2 backbone cho bài toán segmentation.

    Điểm khác biệt so với VMUNet V1:
    ─────────────────────────────────
    • VSSM V2 trả về LIST 4 logit tensors (aux1, aux2, aux3, final).
    • Wrapper này upsample tất cả về (H, W) gốc và áp sigmoid (nếu binary).
    • Training mode: trả về list 4 predictions → dùng cho deep supervision loss.
    • Inference mode: chỉ trả về prediction cuối cùng (fine nhất).

    Args:
        input_channels (int):   Số channel đầu vào (1 hoặc 3). Default: 3.
        num_classes    (int):   Số class. Default: 1.
        depths         (list):  Số block mỗi encoder stage.
        depths_decoder (list):  Số block mỗi decoder stage.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        load_ckpt_path (str):   Đường dẫn checkpoint pretrained. Default: None.
    """

    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 1,
                 depths: List[int] = None,
                 depths_decoder: List[int] = None,
                 drop_path_rate: float = 0.2,
                 load_ckpt_path: str = None):
        super().__init__()

        if depths is None:
            depths = [2, 2, 9, 2]
        if depths_decoder is None:
            depths_decoder = [2, 9, 2, 2]

        self.load_ckpt_path = load_ckpt_path
        self.num_classes    = num_classes

        self.vmunet = VSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) \
            -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass với deep supervision.

        Pipeline:
            1. Grayscale → RGB nếu cần (repeat channel)
            2. Ghi nhớ H, W gốc để upsample về sau
            3. VSSM trả về [aux1, aux2, aux3, final] — 4 logit tensors NCHW
               tại các resolution khác nhau (H/16, H/8, H/4, H/4)
            4. Upsample tất cả về (H, W) bằng bilinear interpolation
            5. Áp sigmoid cho binary segmentation

        Training:
            Trả về list 4 predictions [aux1, aux2, aux3, final]
            → dùng với DeepSupervisionLoss để tính weighted loss

        Inference:
            Chỉ trả về final prediction (tensor duy nhất)

        Args:
            x: (B, C, H, W)
        Returns:
            Training: List[Tensor(B, num_classes, H, W)] — 4 phần tử
            Inference: Tensor(B, num_classes, H, W)
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)   # grayscale → RGB giả

        H, W = x.shape[2], x.shape[3]   # lưu resolution gốc

        # VSSM V2 trả về list 4 logit tensors (raw, chưa sigmoid)
        all_logits: List[torch.Tensor] = self.vmunet(x)

        # Upsample tất cả về resolution gốc (H×W)
        # Ép về float32 trước khi nội suy để tránh lỗi với bf16/fp16
        all_preds = [
            F.interpolate(logit.to(torch.float32), size=(H, W),
                          mode='bilinear', align_corners=False).to(logit.dtype)
            for logit in all_logits
        ]

        # Áp activation
        if self.num_classes == 1:
            all_preds = [torch.sigmoid(p) for p in all_preds]
        # Multi-class: giữ raw logits, áp softmax trong loss function

        if self.training:
            return all_preds          # [aux1, aux2, aux3, final] — deep supervision
        else:
            return all_preds[-1]      # chỉ final map cho inference

    def load_from(self):
        """
        Load pretrained VMamba checkpoint, khởi tạo cả encoder và decoder.

        Bước 1 — Encoder:
            Load trực tiếp các key khớp từ checkpoint vào model.

        Bước 2 — Decoder (remap từ encoder):
            Không có decoder trong checkpoint pretrained → dùng encoder weights.
            Lật ngược thứ tự stage: layers.i → layers_up.(3-i)
        """
        if self.load_ckpt_path is None:
            return

        # ── Bước 1: Load encoder ──────────────────────────────────────────────
        model_dict      = self.vmunet.state_dict()
        checkpoint_data = torch.load(self.load_ckpt_path)
        pretrained_dict = checkpoint_data['model']

        # Lọc: chỉ giữ key có trong model hiện tại
        new_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict}
        model_dict.update(new_dict)
        print('Encoder loading — '
              f'model keys: {len(model_dict)}, '
              f'pretrained keys: {len(pretrained_dict)}, '
              f'matched: {len(new_dict)}')
        self.vmunet.load_state_dict(model_dict)

        not_loaded = [k for k in pretrained_dict if k not in new_dict]
        print('Not loaded (encoder):', not_loaded)
        print("Encoder loaded finished!")

        # ── Bước 2: Remap encoder → decoder ──────────────────────────────────
        # Mapping: layers.i → layers_up.(3-i) (lật thứ tự encoder sang decoder)
        remap = {'layers.0': 'layers_up.3',
                 'layers.1': 'layers_up.2',
                 'layers.2': 'layers_up.1',
                 'layers.3': 'layers_up.0'}

        model_dict      = self.vmunet.state_dict()
        pretrained_dict = checkpoint_data['model']
        remapped = {}
        for k, v in pretrained_dict.items():
            for src, dst in remap.items():
                if src in k:
                    remapped[k.replace(src, dst)] = v
                    break

        new_dict = {k: v for k, v in remapped.items() if k in model_dict}
        model_dict.update(new_dict)
        print('Decoder loading — '
              f'model keys: {len(model_dict)}, '
              f'remapped keys: {len(remapped)}, '
              f'matched: {len(new_dict)}')
        self.vmunet.load_state_dict(model_dict)

        not_loaded = [k for k in remapped if k not in new_dict]
        print('Not loaded (decoder):', not_loaded)
        print("Decoder loaded finished!")


# =============================================================================
# DeepSupervisionLoss — Helper tính weighted loss trên 4 predictions
# =============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    Wrapper tính deep supervision loss từ 4 prediction levels.

    Công thức:
        total_loss = Σ (weights[i] × base_loss(preds[i], target))

    Default weights: [0.2, 0.3, 0.4, 1.0] (coarse → fine)
        → final prediction được weight 5× so với coarsest auxiliary

    Lý do dùng deep supervision:
        - Gradient lan truyền trực tiếp đến các tầng decoder trung gian
        - Tránh vanishing gradient ở các tầng sâu
        - Ép các tầng trung gian học feature có ý nghĩa

    Cách dùng:
        criterion = DeepSupervisionLoss(base_loss=nn.BCELoss(),
                                        weights=[0.2, 0.3, 0.4, 1.0])
        preds = model(images)          # list 4 tensors (training mode)
        loss  = criterion(preds, mask) # scalar

    Args:
        base_loss (nn.Module): Loss function cơ sở (BCELoss, DiceLoss, v.v.).
        weights   (list):      Trọng số cho mỗi level, từ coarse đến fine.
    """
    def __init__(self, base_loss: nn.Module,
                 weights: List[float] = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights   = weights or [0.2, 0.3, 0.4, 1.0]

    def forward(self,
                preds: List[torch.Tensor],
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds : list 4 tensors (B, num_classes, H, W) — từ VMUNet training mode
            target: (B, num_classes, H, W) — ground truth mask
        Returns:
            scalar loss
        """
        assert len(preds) == len(self.weights), \
            f"Expected {len(self.weights)} predictions, got {len(preds)}"
        total = preds[0].new_zeros(1)
        for pred, w in zip(preds, self.weights):
            total = total + w * self.base_loss(pred, target)
        return total
