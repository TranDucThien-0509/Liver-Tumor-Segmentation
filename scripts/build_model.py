"""
scripts/build_model.py
──────────────────────
Factory function: tạo VMUNet V2 từ Config, optionally load pretrained weights.
Import ở bất kỳ script/notebook nào cần model.
"""

import sys
import torch
from pathlib import Path
import segmentation_models_pytorch as smp

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
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Các file .pth thường cất trọng số trong key 'model' hoặc 'state_dict'
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    # Nạp trọng số vào model (strict=False để bỏ qua các lớp không khớp giữa model gốc và model finetune)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if verbose:
        print(f"   ➜ Missing keys: {len(missing_keys)}")
        print(f"   ➜ Unexpected keys: {len(unexpected_keys)}")
        print("Pretrained weights loaded successfully!")
        
    return model


def build_model(cfg, load_pretrained=True, verbose=True):
    """
    Factory function to build models based on configuration.
    """
    import segmentation_models_pytorch as smp
    
    # Lấy tên model từ config, nếu không có thì mặc định là vmunet_v2
    model_name = getattr(cfg.model, "name", "vmunet_v2")
    
    # =========================================================
    # 1. XỬ LÝ CÁC MÔ HÌNH BASELINE (Tải thông qua thư viện SMP)
    # =========================================================
    
    # A. Standard U-Net chuẩn kinh điển (Đã đồng bộ in_channels)
    if model_name == "unet":
        if verbose: print("[Baseline] Initializing Standard U-Net...")
        return smp.Unet(
            encoder_name="vgg16",
            encoder_weights=None,                  # ← tắt imagenet
            in_channels=cfg.data.input_channels,   # = 1
            classes=cfg.data.num_classes
        )

    # B. ResNet-50 U-Net (ĐÃ BỔ SUNG LẠI ĐẦY ĐỦ)
    elif model_name == "resnet50_unet":
        if verbose: print("[Baseline] Initializing ResNet-50 U-Net...")
        return smp.Unet(
            encoder_name="resnet50", 
            encoder_weights="imagenet" if load_pretrained else None, 
            in_channels=cfg.data.input_channels, # <-- ĐÃ THÊM: Đồng bộ động theo config
            classes=cfg.data.num_classes
        )

    # C. ViT-based U-Net (Vision Transformer Encoder)
    elif model_name == "vit_unet":
        if verbose: print("[Baseline] Initializing Vision Transformer (ViT) U-Net...")
        return smp.Unet(
            encoder_name    = "tu-swinv2_small_window8_256",
            encoder_weights = "imagenet" if load_pretrained else None,
            in_channels     = cfg.data.input_channels,
            classes         = cfg.data.num_classes,
        )

    # =========================================================
    # 2. XỬ LÝ DÒNG VM-UNet CHÍNH CHỦ (V1, V2, V3)
    # =========================================================
    elif model_name in ["vmunet_v1", "vmunet_v2", "vmunet_v3"]:
        if model_name == "vmunet_v1":
            from models.vmunet.vmunet_v1 import VMUNet
        elif model_name == "vmunet_v2":
            from models.vmunet.vmunet_v2 import VMUNet
        elif model_name == "vmunet_v3":
            from models.vmunet.vmunet_v3 import VMUNet
            
    else:
        raise ValueError(f"Unknown model name configuration: {model_name}")

    # Khởi tạo kiến trúc cụ thể cho dòng VM-UNet dựa trên class đã import ở trên
    model = VMUNet(
        input_channels=cfg.model.input_channels,
        num_classes=cfg.model.num_classes,
        depths=cfg.model.depths,
        depths_decoder=cfg.model.depths_decoder,
        drop_path_rate=cfg.model.drop_path_rate,
        load_ckpt_path=None,    # Tắt chế độ tự load trong __init__ gốc
    )

    # In thông số cấu hình nếu yêu cầu verbose=True
    if verbose:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"VM-UNet ({model_name}) built successfully:")
        print(f"   Total params     : {total_params:.1f}M")
        print(f"   Trainable params : {trainable:.1f}M")
        print(f"   Num classes      : {cfg.model.num_classes}")
        print(f"   Encoder depths   : {cfg.model.depths}")
        print(f"   Decoder depths   : {cfg.model.depths_decoder}")

    # Nạp trọng số pretrained checkpoint riêng của dòng Mamba
    if load_pretrained and cfg.paths.pretrained_weights.exists():
        model = load_pretrained_weights(
            model,
            cfg.paths.pretrained_weights,
            verbose=verbose,
        )
    elif load_pretrained:
        print(f"⚠️ Pretrained weights not found at target: {cfg.paths.pretrained_weights}")

    return model