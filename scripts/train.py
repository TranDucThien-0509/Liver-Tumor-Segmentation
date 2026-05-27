"""
scripts/train.py
────────────────
Script huấn luyện VM-UNetV2 end-to-end.

Chạy:
    python scripts/train.py

Hoặc override config trực tiếp trong notebook:
    from scripts.train import train
    train(cfg)           # dùng config mặc định
    cfg.train.num_epochs = 20
    train(cfg)           # override
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from fastai.learner import Learner
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.schedule import fit_one_cycle

from configs.config import Config, cfg as default_cfg
from data.dataset     import build_dataloaders, verify_batch
from utils.losses    import DeepSupervisionLoss
from utils.metrics   import build_metrics
from scripts.build_model import build_model


# Learner builder
def build_learner(cfg: Config, dls, model) -> Learner:
    """
    Tạo FastAI Learner từ model và DataLoaders đã build sẵn.
    Áp dụng mixed precision, loss, metrics theo config.
    """
    loss_func = DeepSupervisionLoss(cfg.train)
    metrics   = build_metrics(cfg.data)

    learn = Learner(
        dls,
        model,
        loss_func=loss_func,
        metrics=metrics,
        path=cfg.paths.output_dir,
        model_dir=".",
    )

    # Mixed precision
    if cfg.train.mixed_precision == "bf16":
        learn = learn.to_bf16()
    elif cfg.train.mixed_precision == "fp16":
        learn = learn.to_fp16()

    return learn


# Tìm learning rate
def find_lr(cfg: Config = None) -> float:
    """
    Chạy lr_find và trả về learning rate được đề xuất.
    Dùng để calibrate trước khi train thật.
    """
    if cfg is None:
        cfg = default_cfg

    cfg.summary()
    _, dls, _ = build_dataloaders(cfg.data, cfg.paths)
    model     = build_model(cfg, load_pretrained=True)
    learn     = build_learner(cfg, dls, model)

    print("\nRunning lr_find ...")
    result = learn.lr_find()
    print(f"   Suggested LR: {result.valley:.2e}")
    return result.valley


# Training
def train(cfg: Config = None) -> Learner:
    """
    Pipeline huấn luyện đầy đủ.

    1. Build DataLoaders
    2. Build model + load pretrained
    3. Build Learner
    4. (Optional) Freeze encoder N epoch đầu
    5. fine_tune với SaveModelCallback
    6. Export model

    Returns
    -------
    learn : Learner đã huấn luyện xong
    """
    if cfg is None:
        cfg = default_cfg

    cfg.summary()

    # ── Dữ liệu ─────────────────────────────────────────────────────────────
    print("\nBuilding DataLoaders ...")
    _, dls, _ = build_dataloaders(cfg.data, cfg.paths)
    verify_batch(dls, cfg.data)

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nBuilding model ...")
    model = build_model(cfg, load_pretrained=True)

    # ── Learner ──────────────────────────────────────────────────────────────
    learn = build_learner(cfg, dls, model)

    # ── Callbacks ────────────────────────────────────────────────────────────
    cbs = []
    if cfg.train.save_best:
        cbs.append(SaveModelCallback(
            monitor="valid_loss",
            fname=cfg.paths.best_model_path.name,
        ))

    # ── Optional freeze encoder ──────────────────────────────────────────────
    if cfg.train.freeze_epochs > 0:
        print(f"\nFreezing encoder for {cfg.train.freeze_epochs} epochs ...")
        learn.freeze()
        learn.fit_one_cycle(
            cfg.train.freeze_epochs,
            lr_max=cfg.train.learning_rate or 1e-3,
            wd=cfg.train.weight_decay,
        )
        learn.unfreeze()

    # ── Fine-tune ────────────────────────────────────────────────────────────
    lr = cfg.train.learning_rate or 1e-3
    print(f"\nFine-tuning for {cfg.train.num_epochs} epochs, LR={lr:.2e} ...")

    learn.fine_tune(
        cfg.train.num_epochs,
        base_lr=lr,
        wd=cfg.train.weight_decay,
        cbs=cbs,
    )

    # ── Export ───────────────────────────────────────────────────────────────
    print(f"\nExporting model to {cfg.paths.export_path} ...")
    learn.export(cfg.paths.export_path)
    print("Training complete.")

    return learn


if __name__ == "__main__":
    train(default_cfg)
