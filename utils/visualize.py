"""
utils/visualize.py
──────────────────
Visualization helpers:
  • show_sample()      — hiển thị 1 cặp ảnh / mask từ dataset
  • show_predictions() — loop qua ảnh có tumor và so sánh GT vs Pred
  • find_tumor_images() — lọc ảnh chứa class tumor (idx=2)
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

from PIL import Image
from fastai.vision.all import PILImage, PILMask

from configs.config import DataConfig, PathConfig


_COLORS_LIST = ["gray", "orange", "red"]
_CMAP   = ListedColormap(_COLORS_LIST)
_VMIN   = 0
_VMAX   = len(_COLORS_LIST) - 1


# Cấu hình màu sắc chuẩn cho 3 class: 0 (Background), 1 (Liver), 2 (Tumor)
_COLORS_LIST = ["gray", "orange", "red"]
_CMAP   = ListedColormap(_COLORS_LIST)
_VMIN   = 0
_VMAX   = len(_COLORS_LIST) - 1


def show_sample(ds, idx: int = 0, data_cfg=None):
    """
    Hiển thị ảnh CT trắng đen và mask với bảng màu tương phản cao siêu nổi bật.
    """
    img, mask = ds[idx]
    vocab = data_cfg.class_names if data_cfg else ["Background", "Liver", "Tumor"]

    # ---------------------------------------------------------
    # BẢNG MÀU CUSTOM TƯƠNG PHẢN CAO
    # 0: Đen (Nền), 1: Xanh lá chói (Gan), 2: Vàng chóe (U)
    # ---------------------------------------------------------
    custom_colors = ["black", "limegreen", "red"]
    cmap = ListedColormap(custom_colors[:len(vocab)])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # 1. Vẽ ảnh CT gốc (Trắng đen chuẩn y tế)
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title(f"CT Image (Sample #{idx})")
    axs[0].axis("off")

    # 2. Vẽ Ground Truth Mask với màu Custom
    axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=len(vocab)-1)
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis("off")

    # 3. Gắn Legend (Chú thích)
    patches = [
        mpatches.Patch(color=custom_colors[i], label=name)
        for i, name in enumerate(vocab)
    ]
    axs[1].legend(handles=patches, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.show()


def find_tumor_images(paths_cfg: PathConfig,
                      tumor_class_idx: int = 2) -> List[Path]:
    """
    Lọc ra những ảnh có ít nhất 1 pixel thuộc class tumor trong mask.

    Returns
    -------
    List[Path] các đường dẫn ảnh CT tương ứng
    """
    tumor_images = []

    for mask_path in paths_cfg.train_masks.iterdir():
        if not mask_path.suffix.lower() == ".png":
            continue

        mask = np.array(Image.open(mask_path))
        if (mask == tumor_class_idx).any():
            img_name = (mask_path.name
                        .replace("_mask", "")
                        .replace(".png", ".jpg"))
            img_path = paths_cfg.train_images / img_name
            if img_path.exists():
                tumor_images.append(img_path)

    print(f"🔍 Tìm thấy {len(tumor_images)} ảnh có tumor.")
    return tumor_images


def show_predictions(learn,
                     image_paths: List[Path],
                     paths_cfg: PathConfig,
                     n: int = 10,
                     data_cfg: Optional[DataConfig] = None):
    """
    Hiển thị ảnh CT, GT mask, và predicted mask cạnh nhau.

    Parameters
    ----------
    learn       : FastAI Learner đã train xong
    image_paths : list đường dẫn ảnh CT (từ find_tumor_images)
    paths_cfg   : PathConfig (để tìm mask GT)
    n           : số ảnh hiển thị (mặc định 10)
    data_cfg    : DataConfig (tùy chọn, để hiển thị legend)
    """
    # Đảm bảo model ở đúng device
    device = next(learn.model.parameters()).device

    for img_path in image_paths[:n]:
        img = PILImage.create(img_path)

        mask_name = f"{img_path.stem}_mask.png"
        mask_path = paths_cfg.train_masks / mask_name
        gt_mask   = PILMask.create(mask_path)

        # Dự đoán
        _, _, probs = learn.predict(img)
        pred_mask   = probs.argmax(dim=0).cpu().numpy()

        # Vẽ
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].imshow(np.array(img), cmap="gray")
        axs[0].set_title("CT Image")
        axs[0].axis("off")

        axs[1].imshow(np.array(gt_mask), cmap=_CMAP, vmin=_VMIN, vmax=_VMAX)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(pred_mask, cmap=_CMAP, vmin=_VMIN, vmax=_VMAX)
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")

        # Đã fix lỗi Legend ở đây (Dùng _COLORS_LIST thay vì _COLORS)
        if data_cfg is not None:
            patches = [
                mpatches.Patch(color=_COLORS_LIST[i] if i < len(_COLORS_LIST) else "gray", label=name)
                for i, name in enumerate(data_cfg.class_names)
            ]
            axs[2].legend(handles=patches, loc="lower right", fontsize=7)

        plt.suptitle(img_path.name, fontsize=9, color="gray")
        plt.tight_layout()
        plt.show()