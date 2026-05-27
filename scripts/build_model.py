"""
scripts/build_model.py
──────────────────────
Factory function: tạo VMUNet V2 từ Config, optionally load pretrained weights.
Import ở bất kỳ script/notebook nào cần model.
"""

import sys
import torch
from pathlib import Path

# Đảm bảo project root trong sys.path khi chạy script trực tiếp
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import Config

def load_pretrained_weights(model: torch.nn.Module, ckpt_path: Path, verbose: bool = True):
    """
    Hàm phụ trợ để load trọng số (weights) một cách an toàn.
    """
    if verbose:
        print(f"Loading pretrained weights from: {ckpt_path.name}...")
        
    # Load file .pth vào RAM (map_location='cpu' để an toàn trước khi đẩy lên GPU)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Các file .pth thường cất trọng số trong key 'model' hoặc 'state_dict'
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    # Nạp trọng số vào model (strict=False để bỏ qua các lớp không khớp giữa model gốc và model finetune)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if verbose:
        print(f"  ➜ Missing keys: {len(missing_keys)}")
        print(f"  ➜ Unexpected keys: {len(unexpected_keys)}")
        print("Pretrained weights loaded successfully!")
        
    return model


def build_model(cfg: Config,
                load_pretrained: bool = True,
                verbose: bool = True):
    """
    Khởi tạo VMUNet V2 từ Config.

    Parameters
    ----------
    cfg              : Config object tổng hợp
    load_pretrained  : có load checkpoint VMamba không
    verbose          : in thông tin khi load weights

    Returns
    -------
    VMUNet instance (chưa đưa lên GPU — Learner sẽ tự xử lý)
    """
    
    # Đưa logic import vào trong hàm để có thể dùng được biến cfg
    # (Dùng getattr để đề phòng trường hợp cfg.model không có thuộc tính version)
    if getattr(cfg.model, "version", "v2") == "v2":
        from models.vmunet.vmunet_v2 import VMUNet
    else:
        from models.vmunet.vmunet import VMUNet

    model = VMUNet(
        input_channels=cfg.model.input_channels,
        num_classes=cfg.model.num_classes,
        depths=cfg.model.depths,
        depths_decoder=cfg.model.depths_decoder,
        drop_path_rate=cfg.model.drop_path_rate,
        load_ckpt_path=None,    # Tắt load trong __init__, dùng hàm riêng ở dưới
    )

    if verbose:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable    = sum(p.numel() for p in model.parameters()
                           if p.requires_grad) / 1e6
        print(f"VM-UNet built:")
        print(f"   Total params     : {total_params:.1f}M")
        print(f"   Trainable params : {trainable:.1f}M")
        print(f"   Num classes      : {cfg.model.num_classes}")
        print(f"   Encoder depths   : {cfg.model.depths}")
        print(f"   Decoder depths   : {cfg.model.depths_decoder}")

    if load_pretrained and cfg.paths.pretrained_weights.exists():
        model = load_pretrained_weights(
            model,
            cfg.paths.pretrained_weights,
            verbose=verbose,
        )
    elif load_pretrained:
        print(f"⚠️ Pretrained weights not found at: {cfg.paths.pretrained_weights}")

    return model