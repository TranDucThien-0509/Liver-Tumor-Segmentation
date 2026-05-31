"""
utils/checkpoint.py
───────────────────
Load weights từ checkpoint VMamba pretrained vào VM-UNetV2.

Hai bước:
  1. Nạp trực tiếp các key khớp → khởi tạo encoder
  2. Remap layers.X → layers_up.Y → khởi tạo decoder (mirror)

Hàm load_pretrained_weights() kiểm tra shape trước khi copy
để tránh lỗi khi checkpoint có kiến trúc khác nhẹ.
"""

from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn


# Map encoder stage → decoder stage (mirrored)
_ENCODER_TO_DECODER = {
    "layers.0": "layers_up.3",
    "layers.1": "layers_up.2",
    "layers.2": "layers_up.1",
    "layers.3": "layers_up.0",
}


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load file .pth, hỗ trợ cả dict có key 'model' lẫn raw state_dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def _filter_by_shape(pretrained: Dict, model_dict: Dict) -> Dict:
    """Chỉ giữ lại key có cả tên lẫn shape khớp với model hiện tại."""
    matched = {
        k: v for k, v in pretrained.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    return matched


def load_pretrained_weights(model: nn.Module,
                             checkpoint_path: Path,
                             verbose: bool = True) -> nn.Module:
    """
    Nạp weights pretrained vào VMUNet (V2).

    Parameters
    ----------
    model           : VMUNet instance (chưa chạy load_from)
    checkpoint_path : đường dẫn tới file .pth của VMamba
    verbose         : in thống kê số key được load

    Returns
    -------
    model với weights đã được khởi tạo từ pretrained checkpoint
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Checkpoint không tồn tại: {ckpt_path}")
        print("   Model sẽ dùng random initialization.")
        return model

    print(f"Loading checkpoint: {ckpt_path.name}")
    pretrained = _load_checkpoint(ckpt_path)

    # ── Bước 1: Nạp encoder ──────────────────────────────────────────────────
    model_dict  = model.vmunet.state_dict()
    enc_matched = _filter_by_shape(pretrained, model_dict)
    model_dict.update(enc_matched)

    if verbose:
        print(f"   Encoder — tổng key model: {len(model_dict)}, "
              f"pretrained: {len(pretrained)}, "
              f"khớp: {len(enc_matched)}")

    not_loaded_enc = [k for k in pretrained if k not in enc_matched]
    if verbose and not_loaded_enc:
        print(f"   Không load (encoder): {not_loaded_enc[:5]}"
              f"{'...' if len(not_loaded_enc) > 5 else ''}")

    # ── Bước 2: Remap và nạp decoder ─────────────────────────────────────────
    remapped: Dict = {}
    for k, v in pretrained.items():
        for enc_key, dec_key in _ENCODER_TO_DECODER.items():
            if enc_key in k:
                remapped[k.replace(enc_key, dec_key)] = v
                break

    dec_matched = _filter_by_shape(remapped, model_dict)
    model_dict.update(dec_matched)

    if verbose:
        print(f"   Decoder — remapped: {len(remapped)}, "
              f"khớp: {len(dec_matched)}")

    not_loaded_dec = [k for k in remapped if k not in dec_matched]
    if verbose and not_loaded_dec:
        print(f"   Không load (decoder): {not_loaded_dec[:5]}"
              f"{'...' if len(not_loaded_dec) > 5 else ''}")

    # ── Bước 3: Load vào model ────────────────────────────────────────────────
    msg = model.vmunet.load_state_dict(model_dict, strict=False)
    if verbose:
        missing  = [k for k in msg.missing_keys
                    if "sdi" not in k and "aux_head" not in k]
        unexpected = msg.unexpected_keys
        print(f"   Missing (ngoài SDI/aux): {len(missing)}")
        if unexpected:
            print(f"   Unexpected keys       : {len(unexpected)}")
        print("Pretrained weights loaded.")

    return model
