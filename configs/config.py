"""
configs/config.py
─────────────────
Toàn bộ hyperparameter và đường dẫn được quản lý tập trung tại đây.
Thay đổi thông số → chỉ cần sửa file này, không cần đụng vào code.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


#  Đường dẫn dữ liệu & weights
@dataclass
class PathConfig:
    # Root workspace
    root: Path = Path("/workspace")

    # Dữ liệu
    data_dir: Path = Path("/workspace/dataset")
    train_images: Path = Path("/workspace/dataset/train/images")
    train_masks: Path = Path("/workspace/dataset/train/masks")
    
    test_images: Optional[Path] = Path("/workspace/dataset/test/images")
    test_masks:  Optional[Path] = Path("/workspace/dataset/test/masks")

    # Weights pretrained VMamba
    pretrained_weights: Path = Path(
        "/workspace/VM-UNet/pre_trained_weights/vmamba_small_e238_ema.pth"
    )

    # Nơi lưu model sau khi train
    output_dir: Path = Path("/workspace/outputs")
    model_name: str  = "vmunetv2_liver"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def export_path(self) -> Path:
        return self.output_dir / f"{self.model_name}.pkl"

    @property
    def best_model_path(self) -> Path:
        return self.output_dir / f"{self.model_name}_best"


#  Dữ liệu
@dataclass
class DataConfig:
    # Kích thước ảnh sau resize
    image_size: int = 256

    # Batch size
    batch_size: int = 8

    # Số class (gồm background)
    num_classes: int = 3

    # Tên các class (background phải ở index 0)
    class_names: List[str] = field(
        default_factory=lambda: ["background", "liver", "tumor"]
    )

    # Tỷ lệ train/valid split (theo patient-level)
    train_ratio: float = 0.8

    # Random seed cho reproducibility
    seed: int = 42

    # Ảnh input: 1 (grayscale BW) hoặc 3 (RGB)
    input_channels: int = 1   # CT scan grayscale

    # Augmentation parameters
    do_flip: bool      = True
    flip_vert: bool    = True
    max_rotate: float  = 10.0
    max_zoom: float    = 1.05
    max_lighting: float = 0.02
    max_warp: float    = 0.0

    # Số workers DataLoader
    num_workers: int = 4

    @property
    def background_idx(self) -> int:
        return 0



#  Kiến trúc model VM-UNetV2
@dataclass
class ModelConfig:
    version: str = "v1"   # "v1" hoặc "v2"
    
    # Số channel input (phải khớp DataConfig.input_channels sau repeat)
    input_channels: int = 3

    # Số class output
    num_classes: int = 3

    # Độ sâu mỗi stage của encoder / decoder
    depths: List[int] = field(default_factory=lambda: [2, 2, 9, 2])
    depths_decoder: List[int] = field(default_factory=lambda: [2, 9, 2, 2])

    # Channel của encoder / decoder
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    dims_decoder: List[int] = field(default_factory=lambda: [768, 384, 192, 96])

    # Stochastic depth
    drop_path_rate: float = 0.2

    # SSM state size
    d_state: int = 16

    # Gradient checkpointing (tiết kiệm VRAM, chậm hơn ~15%)
    use_checkpoint: bool = False


#  Huấn luyện
@dataclass
class TrainConfig:
    # Số epochs fine-tune
    num_epochs: int = 10

    # Learning rate (None → dùng lr_find)
    learning_rate: Optional[float] = 1e-3

    # Weight decay
    weight_decay: float = 0.1

    # Mixed precision: 'bf16' | 'fp16' | None
    mixed_precision: str = "fp16"

    # Deep supervision loss weights [aux1, aux2, aux3, final]
    # Coarse → fine, final được weight cao nhất
    ds_weights: List[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.4, 1.0]
    )

    # Hệ số kết hợp CE + Dice
    ce_weight: float   = 1.0
    dice_weight: float = 1.0

    # Dice loss smooth
    dice_smooth: float = 1e-5

    # Có dùng SaveModelCallback không
    save_best: bool = True

    # freeze_epochs: freeze encoder trong N epoch đầu (0 = không freeze)
    freeze_epochs: int = 0

    # Gradient clipping
    grad_clip: float = 1.0


#  Tổng hợp: một object duy nhất import ở mọi nơi
@dataclass
class Config:
    paths:  PathConfig  = field(default_factory=PathConfig)
    data:   DataConfig  = field(default_factory=DataConfig)
    model:  ModelConfig = field(default_factory=ModelConfig)
    train:  TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        # Đảm bảo num_classes đồng bộ giữa data và model
        self.model.num_classes = self.data.num_classes

    def summary(self):
        print("=" * 55)
        print("  VM-UNetV2  —  Config Summary")
        print("=" * 55)
        print(f"  Data dir      : {self.paths.data_dir}")
        print(f"  Image size    : {self.data.image_size}×{self.data.image_size}")
        print(f"  Batch size    : {self.data.batch_size}")
        print(f"  Num classes   : {self.data.num_classes}  {self.data.class_names}")
        print(f"  Model depths  : enc={self.model.depths}  dec={self.model.depths_decoder}")
        print(f"  Drop path     : {self.model.drop_path_rate}")
        print(f"  Epochs        : {self.train.num_epochs}")
        print(f"  LR            : {self.train.learning_rate}")
        print(f"  Weight decay  : {self.train.weight_decay}")
        print(f"  Mixed prec.   : {self.train.mixed_precision}")
        print(f"  DS weights    : {self.train.ds_weights}")
        print(f"  Output dir    : {self.paths.output_dir}")
        print("=" * 55)


# Instance mặc định — import trực tiếp khi chạy nhanh
cfg = Config()
