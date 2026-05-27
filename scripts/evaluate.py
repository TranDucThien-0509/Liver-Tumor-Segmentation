"""
scripts/evaluate.py
────────────────────
Đánh giá VM-UNet (V1 hoặc V2) trên tập TEST (hoặc fallback valid set).

Nguồn ảnh đầu vào — ưu tiên theo thứ tự:
  1. cfg.paths.test_images + cfg.paths.test_masks   ← tập test độc lập
  2. valid set được lưu bên trong learner             ← fallback nếu không có test

Metrics:
  Per-class  : Dice, IoU, Precision, Recall, F1, Support
  Overall    : mDice, mIoU, Pixel Accuracy
  Foreground : Dice/IoU bỏ background (quan trọng cho CT)
  Thống kê  : mean ± std trên toàn bộ tập đánh giá

Outputs:
  - Bảng kết quả in terminal
  - <model>_eval_per_image.csv       — metrics từng ảnh
  - <model>_eval_summary.json        — tổng hợp mean±std
  - <model>_confusion_matrix.png     — confusion matrix chuẩn hóa
  - <model>_dice_distribution.png    — box plot + histogram Dice
  - <model>_cases_worst/best.png     — worst/best cases visualization

Chạy:
    python scripts/evaluate.py

Hoặc trong notebook:
    from scripts.evaluate import run_evaluation
    results = run_evaluation(cfg)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from fastai.vision.all import load_learner, PILImage

from configs.config import Config, cfg as default_cfg


#  Data structures
@dataclass
class PerClassMetrics:
    """Metrics cho một class cụ thể."""
    class_name : str
    dice       : float = 0.0
    iou        : float = 0.0
    precision  : float = 0.0
    recall     : float = 0.0
    f1         : float = 0.0
    support    : int   = 0       # tổng số pixel GT của class này


@dataclass
class ImageMetrics:
    """Metrics cho một ảnh đơn lẻ."""
    filename     : str
    per_class    : List[PerClassMetrics] = field(default_factory=list)
    pixel_acc    : float = 0.0
    mean_dice    : float = 0.0
    mean_iou     : float = 0.0
    inference_ms : float = 0.0


@dataclass
class EvalResults:
    """Kết quả tổng hợp toàn bộ tập đánh giá."""
    model_path       : str
    eval_set         : str            # "test" hoặc "valid"
    num_classes      : int
    class_names      : List[str]
    num_images       : int
    per_class_mean   : List[PerClassMetrics] = field(default_factory=list)
    per_class_std    : List[PerClassMetrics] = field(default_factory=list)
    pixel_acc        : float = 0.0
    pixel_acc_std    : float = 0.0
    mean_dice        : float = 0.0
    mean_dice_std    : float = 0.0
    mean_iou         : float = 0.0
    mean_iou_std     : float = 0.0
    fg_dice          : float = 0.0   # Dice foreground (không tính background)
    fg_iou           : float = 0.0
    avg_inference_ms : float = 0.0
    total_time_s     : float = 0.0



#  Lấy danh sách ảnh để evaluate
def _resolve_eval_set(
    cfg   : Config,
    learn,
) -> Tuple[List[Path], List[Path], str]:
    """
    Xác định nguồn ảnh đánh giá theo thứ tự ưu tiên:
      1. cfg.paths.test_images / test_masks  — tập test riêng
      2. valid set bên trong learner         — fallback

    Returns
    -------
    img_paths  : list Path ảnh
    mask_paths : list Path mask tương ứng (cùng thứ tự)
    eval_label : "test" hoặc "valid"
    """
    # ── Ưu tiên 1: tập test riêng ─────────────────────────────────────────
    test_imgs  = cfg.paths.test_images
    test_masks = cfg.paths.test_masks

    if (test_imgs is not None and test_imgs.exists() and
            test_masks is not None and test_masks.exists()):

        img_paths  = sorted(test_imgs.glob("*"))
        img_paths  = [p for p in img_paths
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")]

        # Ghép với mask tương ứng
        paired_imgs, paired_masks = [], []
        missing = 0
        for img_p in img_paths:
            mask_p = _find_mask_path(img_p, test_masks)
            if mask_p is not None:
                paired_imgs.append(img_p)
                paired_masks.append(mask_p)
            else:
                missing += 1

        if missing:
            print(f"Bỏ qua {missing} ảnh test không tìm được mask.")

        if paired_imgs:
            print(f"Dùng tập TEST: {len(paired_imgs)} ảnh "
                  f"({test_imgs})")
            return paired_imgs, paired_masks, "test"
        else:
            print(f"Thư mục test tồn tại nhưng không có ảnh/mask hợp lệ.")

    # ── Fallback: valid set từ learner ────────────────────────────────────
    print("Không có tập test riêng → dùng VALID set từ learner.")
    try:
        valid_ds  = learn.dls.valid_ds
        img_paths = [Path(str(s[0])) for s in valid_ds.items]
    except Exception:
        from data.dataset import build_dataloaders
        _, dls, _ = build_dataloaders(cfg.data, cfg.paths)
        img_paths = [Path(str(s[0])) for s in dls.valid_ds.items]

    paired_imgs, paired_masks = [], []
    for img_p in img_paths:
        mask_p = _find_mask_path(img_p, cfg.paths.train_masks)
        if mask_p is not None:
            paired_imgs.append(img_p)
            paired_masks.append(mask_p)

    print(f"Dùng tập VALID: {len(paired_imgs)} ảnh")
    return paired_imgs, paired_masks, "valid"


#  Helpers
def _find_mask_path(img_path: Path, masks_dir: Path) -> Optional[Path]:
    """
    Tìm mask tương ứng với ảnh.
    Convention thử theo thứ tự:
      <stem>_mask.png  →  <stem>.png  →  <stem>_mask.jpg
    """
    for candidate in [
        masks_dir / f"{img_path.stem}_mask.png",
        masks_dir / f"{img_path.stem}.png",
        masks_dir / f"{img_path.stem}_mask.jpg",
        masks_dir / f"{img_path.stem}.jpg",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_gt_mask(mask_path: Path) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(mask_path)).astype(np.int32)


def _predict_single(learn, img_path: Path) -> np.ndarray:
    """Inference 1 ảnh → pred mask (H, W) int numpy."""
    img = PILImage.create(img_path)
    with torch.no_grad():
        _, _, probs = learn.predict(img)
    return probs.argmax(dim=0).cpu().numpy().astype(np.int32)


#  Metric computation
def _compute_per_class(
    pred_mask  : np.ndarray,
    gt_mask    : np.ndarray,
    num_classes: int,
    smooth     : float = 1e-7,
) -> List[Dict]:
    results = []
    for c in range(num_classes):
        pred_c = (pred_mask == c).astype(np.float64)
        gt_c   = (gt_mask   == c).astype(np.float64)

        tp = (pred_c * gt_c).sum()
        fp = (pred_c * (1 - gt_c)).sum()
        fn = ((1 - pred_c) * gt_c).sum()

        precision = (tp + smooth) / (tp + fp + smooth)
        recall    = (tp + smooth) / (tp + fn + smooth)
        f1        = 2 * precision * recall / (precision + recall + smooth)
        iou       = (tp + smooth) / (tp + fp + fn + smooth)

        results.append(dict(
            dice=float(f1), iou=float(iou),
            precision=float(precision), recall=float(recall),
            f1=float(f1), support=int(gt_c.sum()),
        ))
    return results


def _pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred == gt).mean())


#  Aggregation
def _aggregate(
    all_metrics : List[ImageMetrics],
    class_names : List[str],
) -> Tuple[List[PerClassMetrics], List[PerClassMetrics]]:
    C = len(class_names)
    N = len(all_metrics)
    dice_a = np.zeros((N, C)); iou_a  = np.zeros((N, C))
    prec_a = np.zeros((N, C)); rec_a  = np.zeros((N, C))
    f1_a   = np.zeros((N, C)); sup_a  = np.zeros((N, C), dtype=np.int64)

    for i, m in enumerate(all_metrics):
        for c, pc in enumerate(m.per_class):
            dice_a[i,c] = pc.dice;      iou_a[i,c]  = pc.iou
            prec_a[i,c] = pc.precision; rec_a[i,c]  = pc.recall
            f1_a[i,c]   = pc.f1;        sup_a[i,c]  = pc.support

    means, stds = [], []
    for c, name in enumerate(class_names):
        means.append(PerClassMetrics(
            class_name=name,
            dice=float(dice_a[:,c].mean()), iou=float(iou_a[:,c].mean()),
            precision=float(prec_a[:,c].mean()), recall=float(rec_a[:,c].mean()),
            f1=float(f1_a[:,c].mean()), support=int(sup_a[:,c].sum()),
        ))
        stds.append(PerClassMetrics(
            class_name=name,
            dice=float(dice_a[:,c].std()), iou=float(iou_a[:,c].std()),
            precision=float(prec_a[:,c].std()), recall=float(rec_a[:,c].std()),
            f1=float(f1_a[:,c].std()), support=0,
        ))
    return means, stds


#  Report
def print_report(results: EvalResults):
    W = 74
    set_label = results.eval_set.upper()
    print()
    print("═" * W)
    print(f"  Evaluation Report [{set_label} SET]  —  {Path(results.model_path).name}")
    print(f"  Images: {results.num_images}  |  Classes: {results.num_classes}  "
          f"|  {results.class_names}")
    print("═" * W)

    hdr = (f"  {'Class':<14} {'Dice':>8} {'±std':>6}  "
           f"{'IoU':>8} {'±std':>6}  "
           f"{'Precision':>9} {'Recall':>8}  {'Support':>10}")
    print(hdr)
    print("  " + "─" * (W - 2))

    for m, s in zip(results.per_class_mean, results.per_class_std):
        print(
            f"  {m.class_name:<14} "
            f"{m.dice:>8.4f} {'±'+f'{s.dice:.4f}':>6}  "
            f"{m.iou:>8.4f} {'±'+f'{s.iou:.4f}':>6}  "
            f"{m.precision:>9.4f} {m.recall:>8.4f}  "
            f"{m.support:>10,}"
        )

    print("  " + "─" * (W - 2))
    bkg = results.class_names[0]
    print(f"  {'Foreground mDice':<28} (excl. {bkg}): {results.fg_dice:.4f}")
    print(f"  {'Foreground mIoU':<28} (excl. {bkg}): {results.fg_iou:.4f}")
    print()
    print(f"  {'Overall mDice':<36}: {results.mean_dice:.4f}  ±{results.mean_dice_std:.4f}")
    print(f"  {'Overall mIoU':<36}: {results.mean_iou:.4f}  ±{results.mean_iou_std:.4f}")
    print(f"  {'Pixel Accuracy':<36}: {results.pixel_acc:.4f}  ±{results.pixel_acc_std:.4f}")
    print()
    print(f"  {'Avg inference time':<36}: {results.avg_inference_ms:.1f} ms/image")
    print(f"  {'Total eval time':<36}: {results.total_time_s:.1f} s")
    print("═" * W)


#  Save
def save_csv(all_metrics : List[ImageMetrics],
             class_names : List[str],
             out_path    : Path) -> pd.DataFrame:
    rows = []
    for m in all_metrics:
        row = {"filename": m.filename, "pixel_acc": m.pixel_acc,
               "mean_dice": m.mean_dice, "mean_iou": m.mean_iou,
               "inference_ms": m.inference_ms}
        for pc in m.per_class:
            row[f"dice_{pc.class_name}"]      = pc.dice
            row[f"iou_{pc.class_name}"]       = pc.iou
            row[f"precision_{pc.class_name}"] = pc.precision
            row[f"recall_{pc.class_name}"]    = pc.recall
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Per-image CSV      → {out_path}")
    return df


def save_json(results: EvalResults, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(results), f, indent=2, ensure_ascii=False)
    print(f"Summary JSON       → {out_path}")


#  Plots
def plot_confusion_matrix(all_preds  : List[np.ndarray],
                          all_gts    : List[np.ndarray],
                          class_names: List[str],
                          out_path   : Optional[Path] = None):
    C  = len(class_names)
    cm = np.zeros((C, C), dtype=np.int64)
    for pred, gt in zip(all_preds, all_gts):
        for t in range(C):
            for p in range(C):
                cm[t, p] += int(((gt == t) & (pred == p)).sum())

    row_sum  = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm.astype(np.float64) / row_sum

    fig, ax = plt.subplots(figsize=(max(5, C * 2.2), max(4, C * 1.8)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(C)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(C)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    for t in range(C):
        for p in range(C):
            color = "white" if cm_norm[t,p] > 0.5 else "black"
            ax.text(p, t, f"{cm_norm[t,p]:.2f}\n({cm[t,p]:,})",
                    ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix   → {out_path}")
    plt.show()
    return cm, cm_norm


def plot_dice_distribution(df          : pd.DataFrame,
                           class_names : List[str],
                           out_path    : Optional[Path] = None):
    dice_cols = [f"dice_{n}" for n in class_names if f"dice_{n}" in df.columns]
    if not dice_cols:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(class_names)))

    data_box = [df[c].dropna().values for c in dice_cols]
    bp = axes[0].boxplot(data_box, labels=class_names, patch_artist=True,
                          medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    axes[0].set_title("Dice Distribution (box plot)")
    axes[0].set_ylabel("Dice Score"); axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)

    for col, name, color in zip(dice_cols, class_names, colors):
        axes[1].hist(df[col].dropna(), bins=20, alpha=0.55,
                     label=name, color=color, density=True)
    axes[1].set_title("Dice Distribution (histogram)")
    axes[1].set_xlabel("Dice Score"); axes[1].set_ylabel("Density")
    axes[1].legend(); axes[1].set_xlim(0, 1); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Dice distribution  → {out_path}")
    plt.show()


def plot_worst_best(all_metrics  : List[ImageMetrics],
                   all_preds     : List[np.ndarray],
                   all_gts       : List[np.ndarray],
                   img_paths     : List[Path],
                   class_names   : List[str],
                   n             : int = 3,
                   sort_by_class : Optional[str] = None,
                   out_path      : Optional[Path] = None):
    """Hiển thị n ảnh Dice thấp nhất (worst) và cao nhất (best)."""
    from PIL import Image as PILImg

    def _score(m: ImageMetrics) -> float:
        if sort_by_class is None:
            return m.mean_dice
        for pc in m.per_class:
            if pc.class_name == sort_by_class:
                return pc.dice
        return m.mean_dice

    scored = sorted(zip(all_metrics, all_preds, all_gts, img_paths),
                    key=lambda x: _score(x[0]))
    worst = scored[:n]
    best  = scored[-n:][::-1]

    _CMAP = "tab10"; _VMAX = len(class_names) - 1
    patches = [mpatches.Patch(color=plt.cm.tab10(i/10), label=n)
               for i, n in enumerate(class_names)]

    for label, cases in [("WORST", worst), ("BEST", best)]:
        fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        sort_label = sort_by_class or "mDice"
        fig.suptitle(f"{label} {n} — sorted by {sort_label}",
                     fontsize=13, fontweight="bold")

        for row, (m, pred, gt, img_p) in enumerate(cases):
            ct = np.array(PILImg.open(img_p).convert("L"))
            dice_str = "  ".join(f"{pc.class_name}={pc.dice:.3f}"
                                 for pc in m.per_class)
            score = _score(m)

            axes[row,0].imshow(ct, cmap="gray")
            axes[row,0].set_title(img_p.name, fontsize=8); axes[row,0].axis("off")

            axes[row,1].imshow(gt, cmap=_CMAP, vmin=0, vmax=_VMAX)
            axes[row,1].set_title("Ground Truth", fontsize=8); axes[row,1].axis("off")

            axes[row,2].imshow(pred, cmap=_CMAP, vmin=0, vmax=_VMAX)
            axes[row,2].set_title(f"Pred  {sort_label}={score:.3f}\n{dice_str}",
                                  fontsize=7)
            axes[row,2].axis("off")
            axes[row,2].legend(handles=patches, loc="lower right",
                               fontsize=6, framealpha=0.7)

        plt.tight_layout()
        if out_path:
            p = out_path.parent / f"{out_path.stem}_{label.lower()}{out_path.suffix}"
            plt.savefig(p, dpi=130, bbox_inches="tight")
            print(f"{label} cases       → {p}")
        plt.show()



#  Main evaluation loop
def run_evaluation(
    cfg            : Config = None,
    learn          = None,               # <--- ĐÃ THÊM THAM SỐ NÀY ĐỂ NHẬN MODEL TỪ RAM
    save_dir       : Optional[Path] = None,
    plot_cm        : bool = True,
    plot_cases     : bool = True,
    n_cases        : int  = 3,
    sort_by_class  : Optional[str] = "tumor",
    smooth         : float = 1e-7,
) -> EvalResults:
    """
    Đánh giá model trên tập TEST (hoặc valid nếu không có test).

    Parameters
    ----------
    cfg           : Config — dùng default_cfg nếu None
    learn         : FastAI Learner (truyền vào nếu vừa train xong, khỏi cần load file)
    save_dir      : thư mục lưu output (mặc định cfg.paths.output_dir)
    plot_cm       : vẽ confusion matrix
    plot_cases    : vẽ worst/best cases
    n_cases       : số ảnh mỗi loại worst/best
    sort_by_class : tên class để sort worst/best (None → mDice)
    smooth        : epsilon tránh chia 0

    Returns
    -------
    EvalResults
    """
    if cfg is None:
        cfg = default_cfg
    if save_dir is None:
        save_dir = cfg.paths.output_dir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    class_names = cfg.data.class_names
    num_classes = cfg.data.num_classes

    # ── Load model ────────────────────────────────────────────────────────────
    if learn is None:
        export_path = cfg.paths.export_path
        if not export_path.exists():
            raise FileNotFoundError(
                f"Model chưa export tại: {export_path}\n"
                "Chạy scripts/train.py trước hoặc truyền thẳng biến `learn` vào."
            )
        print(f"\nLoading model: {export_path.name}")
        from fastai.learner import load_learner
        learn = load_learner(export_path)
    else:
        print("\n🚀 Đang sử dụng mô hình Learner đã có sẵn trong RAM.")

    learn.model.eval()
    device = next(learn.model.parameters()).device
    print(f"   Device: {device}")

    # ── Xác định tập đánh giá (test hoặc valid) ───────────────────────────────
    print("\nXác định tập đánh giá...")
    img_paths, mask_paths, eval_label = _resolve_eval_set(cfg, learn)

    if not img_paths:
        raise RuntimeError("Không có ảnh nào để đánh giá. "
                           "Kiểm tra cfg.paths.test_images / test_masks.")

    print(f"\nBắt đầu inference {len(img_paths)} ảnh [{eval_label.upper()} SET]...\n")

    # ── Inference loop ────────────────────────────────────────────────────────
    all_img_metrics : List[ImageMetrics] = []
    all_preds       : List[np.ndarray]   = []
    all_gts         : List[np.ndarray]   = []

    t_start = time.time()

    for img_path, mask_path in zip(img_paths, mask_paths):

        gt_mask = _load_gt_mask(mask_path)

        t0       = time.perf_counter()
        pred     = _predict_single(learn, img_path)
        infer_ms = (time.perf_counter() - t0) * 1000

        # Resize pred về đúng kích thước GT nếu cần
        if pred.shape != gt_mask.shape:
            pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
            pred   = (F.interpolate(pred_t, size=gt_mask.shape, mode="nearest")
                      .squeeze().numpy().astype(np.int32))

        # Tính metrics
        pc_list   = _compute_per_class(pred, gt_mask, num_classes, smooth)
        pixel_acc = _pixel_accuracy(pred, gt_mask)

        img_m = ImageMetrics(
            filename     = img_path.name,
            pixel_acc    = pixel_acc,
            mean_dice    = float(np.mean([pc["dice"] for pc in pc_list])),
            mean_iou     = float(np.mean([pc["iou"]  for pc in pc_list])),
            inference_ms = infer_ms,
            per_class    = [
                PerClassMetrics(class_name=class_names[c], **pc_list[c])
                for c in range(num_classes)
            ],
        )

        all_img_metrics.append(img_m)
        all_preds.append(pred)
        all_gts.append(gt_mask)

    t_total = time.time() - t_start

    # ── Aggregate ─────────────────────────────────────────────────────────────
    per_class_mean, per_class_std = _aggregate(all_img_metrics, class_names)

    pixel_accs  = [m.pixel_acc   for m in all_img_metrics]
    mean_dices  = [m.mean_dice   for m in all_img_metrics]
    mean_ious   = [m.mean_iou    for m in all_img_metrics]
    infer_times = [m.inference_ms for m in all_img_metrics]

    bkg_name = class_names[0]
    fg_dice  = float(np.mean([pc.dice for pc in per_class_mean
                               if pc.class_name != bkg_name]))
    fg_iou   = float(np.mean([pc.iou  for pc in per_class_mean
                               if pc.class_name != bkg_name]))

    results = EvalResults(
        model_path       = "In-Memory Model" if learn is not None else str(export_path),
        eval_set         = eval_label,
        num_classes      = num_classes,
        class_names      = class_names,
        num_images       = len(all_img_metrics),
        per_class_mean   = per_class_mean,
        per_class_std    = per_class_std,
        pixel_acc        = float(np.mean(pixel_accs)),
        pixel_acc_std    = float(np.std (pixel_accs)),
        mean_dice        = float(np.mean(mean_dices)),
        mean_dice_std    = float(np.std (mean_dices)),
        mean_iou         = float(np.mean(mean_ious)),
        mean_iou_std     = float(np.std (mean_ious)),
        fg_dice          = fg_dice,
        fg_iou           = fg_iou,
        avg_inference_ms = float(np.mean(infer_times)),
        total_time_s     = float(t_total),
    )

    # ── Print + Save ──────────────────────────────────────────────────────────
    print_report(results)

    tag = f"{cfg.paths.model_name}_{eval_label}"
    print()
    df = save_csv(all_img_metrics, class_names,
                  save_dir / f"{tag}_per_image.csv")
    save_json(results, save_dir / f"{tag}_summary.json")

    if plot_cm:
        print()
        plot_confusion_matrix(all_preds, all_gts, class_names,
                              out_path=save_dir / f"{tag}_confusion_matrix.png")

    print()
    plot_dice_distribution(df, class_names,
                           out_path=save_dir / f"{tag}_dice_distribution.png")

    if plot_cases:
        print()
        plot_worst_best(all_img_metrics, all_preds, all_gts, img_paths,
                        class_names, n=n_cases, sort_by_class=sort_by_class,
                        out_path=save_dir / f"{tag}_cases.png")

    return results


#  So sánh nhiều model
def compare_models(results_list : List[EvalResults],
                   model_labels : List[str],
                   out_path     : Optional[Path] = None):
    """
    Grouped bar chart so sánh nhiều model.

    Ví dụ:
        r1 = run_evaluation(cfg_v1)
        r2 = run_evaluation(cfg_v2)
        compare_models([r1, r2], ['VM-UNet V1', 'VM-UNet V2'])
    """
    assert len(results_list) == len(model_labels)
    class_names     = results_list[0].class_names
    metrics_to_plot = ["dice", "iou", "precision", "recall"]
    n_models        = len(results_list)
    C               = len(class_names)

    fig, axes = plt.subplots(1, len(metrics_to_plot),
                              figsize=(5 * len(metrics_to_plot),
                                       max(4, C * 0.9 + 2)))
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_models))
    x      = np.arange(C)
    width  = 0.8 / n_models

    for ax_i, metric in enumerate(metrics_to_plot):
        ax = axes[ax_i]
        for m_i, (res, lbl) in enumerate(zip(results_list, model_labels)):
            vals = [getattr(pc, metric) for pc in res.per_class_mean]
            errs = [getattr(pc, metric) for pc in res.per_class_std]
            offset = (m_i - n_models / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width, label=lbl,
                            color=colors[m_i], alpha=0.85,
                            yerr=errs, capsize=3,
                            error_kw=dict(elinewidth=1))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.set_title(metric.capitalize()); ax.set_ylim(0, 1.12)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    set_labels = " vs ".join(f"{l}[{r.eval_set}]"
                              for l, r in zip(model_labels, results_list))
    plt.suptitle(f"Model Comparison — {set_labels}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Model comparison   → {out_path}")
    plt.show()


if __name__ == "__main__":
    run_evaluation(
        cfg           = default_cfg,
        plot_cm       = True,
        plot_cases    = True,
        n_cases       = 3,
        sort_by_class = "tumor",
    )