"""
scripts/predict.py
──────────────────
Inference với model đã export.

Chạy:
    python scripts/predict.py

Hoặc trong notebook:
    from scripts.predict import run_inference
    run_inference(cfg, n=20)
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastai.vision.all import load_learner

from configs import Config, cfg as default_cfg
from utils   import find_tumor_images, show_predictions


def run_inference(cfg: Config = None, n: int = 20):
    """
    Load model đã export và chạy prediction trên ảnh có tumor.

    Parameters
    ----------
    cfg : Config (dùng default nếu None)
    n   : số ảnh chạy inference (mặc định 20)
    """
    if cfg is None:
        cfg = default_cfg

    export_path = cfg.paths.export_path
    if not export_path.exists():
        raise FileNotFoundError(
            f"Model chưa được export tại: {export_path}\n"
            "Chạy scripts/train.py trước."
        )

    print(f"Loading exported model: {export_path}")
    learn = load_learner(export_path)

    # Đưa model lên GPU nếu có
    learn.model.cuda()
    learn.dls.to("cuda")

    # Tìm ảnh có tumor
    tumor_images = find_tumor_images(cfg.paths, tumor_class_idx=2)

    if not tumor_images:
        print("Không tìm thấy ảnh có tumor trong tập dữ liệu.")
        return

    # Visualize predictions
    show_predictions(
        learn=learn,
        image_paths=tumor_images,
        paths_cfg=cfg.paths,
        n=n,
        data_cfg=cfg.data,
    )


if __name__ == "__main__":
    run_inference(default_cfg, n=20)
