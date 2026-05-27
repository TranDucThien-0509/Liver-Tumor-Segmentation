"""
data/dataset.py
───────────────
Xây dựng FastAI DataBlock và DataLoaders cho bài toán
segmentation CT gan/khối u (3 class: background / liver / tumor).

Hỗ trợ cả ảnh grayscale (PILImageBW) — phổ biến với CT scan.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from fastai.vision.all import (
    DataBlock, ImageBlock, MaskBlock, PILImageBW,
    get_image_files, FuncSplitter, Resize, aug_transforms,
)

from configs.config import DataConfig, PathConfig


# Helpers
def get_volume_id(fname: Path) -> str:
    """Lấy patient/volume ID từ tên file (phần trước dấu '_' đầu tiên)."""
    return fname.name.split("_")[0]


# ---------------------------------------------------------
# FIX PICKLE ERROR 1: Chuyển hàm lồng thành Class Splitter
# ---------------------------------------------------------
class PatientSplitter:
    """
    Class hỗ trợ tách train/valid theo patient-level.
    Dùng class thay vì hàm lồng (closure) để có thể export/pickle model.
    """
    def __init__(self, fnames, data_cfg: DataConfig):
        volume_ids = list(set(get_volume_id(f) for f in fnames))
        rng = np.random.RandomState(data_cfg.seed)
        rng.shuffle(volume_ids)
        
        split_idx = int(len(volume_ids) * data_cfg.train_ratio)
        self.valid_vols = set(volume_ids[split_idx:])

    def __call__(self, fname: Path) -> bool:
        return get_volume_id(fname) in self.valid_vols


# ---------------------------------------------------------
# FIX PICKLE ERROR 2: Chuyển hàm lồng thành Class Label
# ---------------------------------------------------------
class GetMaskPath:
    """
    Class map đường dẫn ảnh -> mask.
    Dùng class để tránh lỗi "Can't pickle local object" khi export.
    """
    def __init__(self, masks_path: Path):
        self.masks_path = Path(masks_path)
        
    def __call__(self, x: Path) -> Path:
        return self.masks_path / f"{x.stem}_mask.png"


# DataBlock builder
def build_datablock(data_cfg: DataConfig, paths_cfg: PathConfig) -> DataBlock:
    """
    Xây dựng FastAI DataBlock cho segmentation đa lớp.
    """
    fnames = get_image_files(paths_cfg.train_images)
    
    # Khởi tạo 2 class vừa viết
    patient_splitter = PatientSplitter(fnames, data_cfg)
    label_func       = GetMaskPath(paths_cfg.train_masks)

    db = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), MaskBlock(data_cfg.class_names)),
        get_items=get_image_files,
        get_y=label_func,
        splitter=FuncSplitter(patient_splitter),
        item_tfms=Resize(data_cfg.image_size),
        batch_tfms=[
            *aug_transforms(
                do_flip=data_cfg.do_flip,
                flip_vert=data_cfg.flip_vert,
                max_rotate=data_cfg.max_rotate,
                max_zoom=data_cfg.max_zoom,
                max_lighting=data_cfg.max_lighting,
                max_warp=data_cfg.max_warp,
            )
        ],
    )
    return db


def build_dataloaders(data_cfg: DataConfig, paths_cfg: PathConfig):
    """
    Convenience wrapper: trả về cả (db, dls, ds) cùng lúc.
    """
    db  = build_datablock(data_cfg, paths_cfg)
    dls = db.dataloaders(
        paths_cfg.train_images,
        bs=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
    )
    ds  = db.datasets(source=paths_cfg.train_images)
    return db, dls, ds


# Sanity check
def verify_batch(dls, data_cfg: DataConfig) -> Tuple:
    """In shape của một batch và trả về (xb, yb)."""
    xb, yb = dls.one_batch()
    print(f"Input  batch : {tuple(xb.shape)}  "
          f"dtype={xb.dtype}  min={xb.min():.2f}  max={xb.max():.2f}")
    print(f"Target batch : {tuple(yb.shape)}  "
          f"dtype={yb.dtype}  unique={yb.unique().tolist()}")
    assert xb.shape[1] in (1, 3), \
        f"Unexpected channel count: {xb.shape[1]}"
    assert yb.max() < data_cfg.num_classes, \
        f"Mask value {yb.max()} >= num_classes {data_cfg.num_classes}"
    print("Batch sanity check passed.")
    return xb, yb