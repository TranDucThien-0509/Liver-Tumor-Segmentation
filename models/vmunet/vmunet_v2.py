"""
vmunet_v2.py  —  VM-UNetV2 wrapper
Wraps the upgraded VSSM backbone and handles:
  • Greyscale → RGB channel repeat
  • Deep supervision: returns all 4 logit maps during training,
    only the final full-resolution mask during inference
  • Checkpoint loading with encoder→decoder key remapping (unchanged from V1)
"""

from .vmamba_v2 import VSSM
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class VMUNet(nn.Module):
    """
    VM-UNetV2 segmentation model.

    Parameters
    ----------
    input_channels   : int   — number of input image channels (1 or 3)
    num_classes      : int   — number of output segmentation classes
    depths           : list  — depth of each encoder VSSLayer
    depths_decoder   : list  — depth of each decoder VSSLayer_up
    drop_path_rate   : float — stochastic depth rate
    load_ckpt_path   : str | None — path to a pre-trained VMamba checkpoint

    Forward inputs / outputs
    ─────────────────────────
    Input  : (B, C, H, W)  image tensor

    Training (self.training == True):
        Returns a list of 4 sigmoid-activated masks, coarsest → finest:
            [pred_aux1, pred_aux2, pred_aux3, pred_final]
        All tensors are upsampled to the full input resolution (H×W).
        Use pred_final as the main loss target; the aux masks provide
        deep supervision at intermediate scales.

    Inference (self.training == False):
        Returns only pred_final — a single (B, num_classes, H, W) mask
        with sigmoid applied (binary) or raw logits (multi-class).
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

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) \
            -> Union[torch.Tensor, List[torch.Tensor]]:
        # Handle single-channel inputs
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        H, W = x.shape[2], x.shape[3]

        # VSSM returns [aux1, aux2, aux3, final]  — all raw logits, NCHW
        all_logits: List[torch.Tensor] = self.vmunet(x)

        # Upsample every output to full input resolution
        all_preds = [
            # Ép logit sang float32 để nội suy, sau đó trả về kiểu ban đầu
            F.interpolate(logit.to(torch.float32), size=(H, W), mode='bilinear',
                          align_corners=False).to(logit.dtype)
            for logit in all_logits
        ]

        # Apply activation
        if self.num_classes == 1:
            all_preds = [torch.sigmoid(p) for p in all_preds]
        # (for multi-class, raw logits are returned; apply softmax in the loss)

        if self.training:
            # Return all 4 predictions for deep-supervision loss computation
            return all_preds          # [aux1, aux2, aux3, final]
        else:
            # Inference: return only the main (finest) prediction
            return all_preds[-1]      # final full-resolution mask

    # ── Checkpoint loading (unchanged logic from V1, extended for decoder) ────
    def load_from(self):
        """
        Load a pre-trained VMamba encoder checkpoint and remap its weights
        to initialise both the encoder and decoder of VM-UNetV2.
        """
        if self.load_ckpt_path is None:
            return

        # ── Step 1: load encoder weights ─────────────────────────────────────
        model_dict       = self.vmunet.state_dict()
        checkpoint_data  = torch.load(self.load_ckpt_path)
        pretrained_dict  = checkpoint_data['model']

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

        # ── Step 2: remap encoder weights to decoder (mirrored stages) ───────
        # layers.0 → layers_up.3, .1 → .2, .2 → .1, .3 → .0
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


# Deep Supervision Loss helper
class DeepSupervisionLoss(nn.Module):
    """
    Wraps any base loss (e.g. BCE, DiceLoss) and combines it across the
    four prediction levels returned during training.

    Weights default to [0.2, 0.3, 0.4, 1.0] (coarse → fine), so the main
    prediction is weighted 5× more than the coarsest auxiliary output.

    Usage
    -----
        criterion = DeepSupervisionLoss(base_loss=nn.BCELoss(),
                                        weights=[0.2, 0.3, 0.4, 1.0])
        preds = model(images)          # list of 4 tensors (training mode)
        loss  = criterion(preds, mask) # scalar
    """
    def __init__(self, base_loss: nn.Module,
                 weights: List[float] = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights   = weights or [0.2, 0.3, 0.4, 1.0]

    def forward(self,
                preds: List[torch.Tensor],
                target: torch.Tensor) -> torch.Tensor:
        assert len(preds) == len(self.weights), \
            f"Expected {len(self.weights)} predictions, got {len(preds)}"
        total = preds[0].new_zeros(1)
        for pred, w in zip(preds, self.weights):
            total = total + w * self.base_loss(pred, target)
        return total
