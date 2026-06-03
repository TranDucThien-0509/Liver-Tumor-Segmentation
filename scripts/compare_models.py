"""
compare_models.py
=================
So sánh nhiều file CSV kết quả evaluation segmentation model.
Mỗi file CSV cần có các cột: filename, pixel_acc, mean_dice, mean_iou,
dice_liver, iou_liver, precision_liver, recall_liver,
dice_tumor, iou_tumor, precision_tumor, recall_tumor

Usage:
    python compare_models.py vmunetv1_liver_test_per_image.csv vmunetv2_liver_test_per_image.csv vmunetv3_liver_test_per_image.csv
    python compare_models.py file1.csv file2.csv --names "V1 Baseline" "V2 SDI+CBAM" --output report.png
    python compare_models.py *.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


# ── Màu mặc định cho từng model ──────────────────────────────────────────────
DEFAULT_COLORS = [
    "#6b7280",  # gray   — V1
    "#3b82f6",  # blue   — V2
    "#f43f5e",  # rose   — V3
    "#10b981",  # emerald — V4
    "#f59e0b",  # amber  — V5
    "#8b5cf6",  # violet — V6
]

NEAR_FAIL_THRESH = 1e-6
EXCELLENT_THRESH = 0.9


# ── Tính toán metrics từ DataFrame ───────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    stats = {
        "total": total,
        "pixel_acc":    df["pixel_acc"].mean(),
        "overall_mdice": df["mean_dice"].mean(),
        "overall_mdice_std": df["mean_dice"].std(),
        "overall_miou": df["mean_iou"].mean(),
    }

    for cls in ["liver", "tumor"]:
        col = f"dice_{cls}"
        stats[f"dice_{cls}"]      = df[col].mean()
        stats[f"dice_{cls}_std"]  = df[col].std()
        stats[f"dice_{cls}_med"]  = df[col].median()
        stats[f"iou_{cls}"]       = df[f"iou_{cls}"].mean()
        stats[f"precision_{cls}"] = df[f"precision_{cls}"].mean()
        stats[f"recall_{cls}"]    = df[f"recall_{cls}"].mean()

        nf    = (df[col] < NEAR_FAIL_THRESH).sum()
        exc   = (df[col] > EXCELLENT_THRESH).sum()
        nf_df = df[df[col] < NEAR_FAIL_THRESH]
        prec_col = f"precision_{cls}"
        rec_col  = f"recall_{cls}"
        overseg  = ((nf_df[rec_col] > 0.5) & (nf_df[prec_col] < 0.01)).sum()

        stats[f"near_fail_{cls}"]      = nf
        stats[f"near_fail_{cls}_pct"]  = nf / total * 100
        stats[f"excellent_{cls}"]      = exc
        stats[f"excellent_{cls}_pct"]  = exc / total * 100
        stats[f"overseg_{cls}_pct"]    = overseg / max(nf, 1) * 100

    stats["fg_mdice"] = (stats["dice_liver"] + stats["dice_tumor"]) / 2
    stats["fg_miou"]  = (stats["iou_liver"]  + stats["iou_tumor"])  / 2
    return stats


# ── Vẽ bảng tổng hợp ─────────────────────────────────────────────────────────
def plot_summary_table(ax, names, stats_list, colors):
    ax.axis("off")
    rows = [
        ("Foreground mDice",  "fg_mdice",        "{:.4f}"),
        ("Foreground mIoU",   "fg_miou",          "{:.4f}"),
        ("Overall mDice",     "overall_mdice",    "{:.4f} ±{overall_mdice_std:.3f}"),
        ("Pixel accuracy",    "pixel_acc",        "{:.4f}"),
        ("Dice liver",        "dice_liver",       "{:.4f} ±{dice_liver_std:.3f}"),
        ("IoU liver",         "iou_liver",        "{:.4f}"),
        ("Precision liver",   "precision_liver",  "{:.4f}"),
        ("Recall liver",      "recall_liver",     "{:.4f}"),
        ("Dice tumor",        "dice_tumor",       "{:.4f} ±{dice_tumor_std:.3f}"),
        ("IoU tumor",         "iou_tumor",        "{:.4f}"),
        ("Precision tumor",   "precision_tumor",  "{:.4f}"),
        ("Recall tumor",      "recall_tumor",     "{:.4f}"),
        ("Near-fail liver %", "near_fail_liver_pct",  "{:.1f}%"),
        ("Near-fail tumor %", "near_fail_tumor_pct",  "{:.1f}%"),
        ("Over-seg liver %",  "overseg_liver_pct",    "{:.1f}%"),
        ("Over-seg tumor %",  "overseg_tumor_pct",    "{:.1f}%"),
    ]

    col_labels = ["Metric"] + names
    table_data = []
    best_col   = []

    for label, key, fmt in rows:
        row = [label]
        vals = []
        for s in stats_list:
            try:
                val_str = fmt.format(s[key], **s)
            except KeyError:
                val_str = "—"
            row.append(val_str)
            vals.append(s.get(key, np.nan))

        # tìm best (nhỏ hơn tốt hơn với near-fail / over-seg)
        lower_better = "near_fail" in key or "overseg" in key
        if not all(np.isnan(v) for v in vals):
            best_idx = int(np.nanargmin(vals) if lower_better else np.nanargmax(vals))
        else:
            best_idx = -1

        table_data.append(row)
        best_col.append(best_idx + 1)   # +1 vì cột 0 là Metric

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)

    n_cols = len(col_labels)
    n_rows = len(table_data)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        cell.set_linewidth(0.5)

        if r == 0:                              # header
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        elif c == 0:                            # metric name column
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(color="#374151", ha="left")
            # left-align bằng cách pad text
        else:                                   # data cells
            model_idx = c - 1
            base_color = colors[model_idx % len(colors)]
            cell.set_facecolor("#ffffff")
            cell.set_text_props(color="#111827")

            # tô màu cột best
            data_r = r - 1
            if 0 <= data_r < n_rows and best_col[data_r] == c:
                cell.set_facecolor(base_color + "22")
                cell.set_text_props(color=base_color, fontweight="bold")

        # stripe rows
        if r > 0 and r % 2 == 0 and c > 0 and best_col[r-1] != c:
            cell.set_facecolor("#f9fafb")

    # header màu model
    for ci, color in enumerate(colors[:len(names)]):
        tbl[0, ci + 1].set_facecolor(color)


# ── Vẽ bar chart metric ───────────────────────────────────────────────────────
def plot_bar_group(ax, names, stats_list, colors, metrics, title, ylim=(0.6, 1.0)):
    x        = np.arange(len(metrics))
    n        = len(names)
    width    = 0.8 / n
    offsets  = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    for i, (name, s, color) in enumerate(zip(names, stats_list, colors)):
        vals = [s.get(m, 0) for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width * 0.9,
                      color=color, alpha=0.85, label=name,
                      zorder=3, linewidth=0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=color, fontweight="bold")

    metric_labels = [m.replace("_", "\n").replace("liver", "🫀").replace("tumor", "🔴") for m in metrics]
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("dice_", "Dice ").replace("iou_", "IoU ")
                         .replace("precision_", "Prec ").replace("recall_", "Rec ")
                         .replace("liver", "liver").replace("tumor", "tumor")
                        for m in metrics], fontsize=8)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


# ── Vẽ histogram phân phối Dice ───────────────────────────────────────────────
def plot_dice_hist(ax, dfs, names, colors, cls="liver"):
    bins = np.linspace(0, 1, 21)
    for df, name, color in zip(dfs, names, colors):
        col = f"dice_{cls}"
        ax.hist(df[col], bins=bins, alpha=0.55, color=color,
                label=name, edgecolor="white", linewidth=0.3, zorder=3)

    ax.set_xlabel("Dice score", fontsize=8)
    ax.set_ylabel("Số ảnh", fontsize=8)
    ax.set_title(f"Phân phối Dice — {cls}", fontsize=10, fontweight="bold", pad=8)
    ax.grid(alpha=0.3, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)


# ── Vẽ radar chart ────────────────────────────────────────────────────────────
def plot_radar(ax, names, stats_list, colors):
    radar_metrics = [
        ("dice_liver",       "Dice\nliver"),
        ("iou_liver",        "IoU\nliver"),
        ("precision_liver",  "Prec\nliver"),
        ("dice_tumor",       "Dice\ntumor"),
        ("iou_tumor",        "IoU\ntumor"),
        ("precision_tumor",  "Prec\ntumor"),
    ]
    labels = [l for _, l in radar_metrics]
    keys   = [k for k, _ in radar_metrics]
    N      = len(keys)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.7", "0.8", "0.9", "1.0"], fontsize=6.5, color="#9ca3af")
    ax.grid(color="#e5e7eb", linestyle="--", alpha=0.6)
    ax.spines["polar"].set_visible(False)
    ax.set_title("Radar — 6 metrics", fontsize=10, fontweight="bold", pad=18)

    for name, s, color in zip(names, stats_list, colors):
        vals   = [s.get(k, 0) for k in keys]
        vals  += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=name)
        ax.fill(angles, vals, color=color, alpha=0.08)


# ── Vẽ near-fail stacked bar ──────────────────────────────────────────────────
def plot_fail_bars(ax, names, stats_list, colors):
    classes = ["liver", "tumor"]
    x = np.arange(len(names))
    width = 0.35

    for ci, cls in enumerate(classes):
        offset = (ci - 0.5) * width
        exc_pcts  = [s[f"excellent_{cls}_pct"] for s in stats_list]
        nf_pcts   = [s[f"near_fail_{cls}_pct"] for s in stats_list]
        mid_pcts  = [100 - e - n for e, n in zip(exc_pcts, nf_pcts)]

        bottom_exc = np.zeros(len(names))
        bottom_mid = np.array(exc_pcts)
        bottom_nf  = np.array(exc_pcts) + np.array(mid_pcts)

        label_sfx = f" ({cls})"
        ax.bar(x + offset, exc_pcts, width, color="#10b981", alpha=0.85,
               label=f"Excellent >0.9{label_sfx}" if ci == 0 else "", zorder=3)
        ax.bar(x + offset, mid_pcts, width, bottom=bottom_mid,
               color="#f59e0b", alpha=0.7,
               label=f"Mid-range{label_sfx}" if ci == 0 else "", zorder=3)
        ax.bar(x + offset, nf_pcts, width, bottom=bottom_nf,
               color="#ef4444", alpha=0.85,
               label=f"Near-fail{label_sfx}" if ci == 0 else "", zorder=3)

        for i, (nf, tot) in enumerate(zip(nf_pcts, [100]*len(names))):
            ax.text(x[i] + offset, 101, f"{nf:.1f}%",
                    ha="center", va="bottom", fontsize=7,
                    color="#ef4444", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("% ảnh", fontsize=8)
    ax.set_title("Phân bố chất lượng theo model", fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    handles = [
        Line2D([0],[0], color="#10b981", linewidth=6, label="Excellent (dice>0.9)"),
        Line2D([0],[0], color="#f59e0b", linewidth=6, label="Mid-range"),
        Line2D([0],[0], color="#ef4444", linewidth=6, label="Near-fail (dice≈0)"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")

    # marker liver / tumor
    ax.text(0.25, -0.09, "■ liver", transform=ax.transAxes, fontsize=8, color="#6b7280")
    ax.text(0.55, -0.09, "■ tumor", transform=ax.transAxes, fontsize=8, color="#6b7280")


# ── Main figure ───────────────────────────────────────────────────────────────
def build_report(dfs, names, colors, output_path):
    fig = plt.figure(figsize=(20, 24), facecolor="#f8fafc")
    fig.suptitle(
        "Liver Tumor Segmentation — Model Comparison Report",
        fontsize=16, fontweight="bold", y=0.99, color="#111827"
    )
    sub = f"Test set: {len(dfs[0]):,} images · {', '.join(names)}"
    fig.text(0.5, 0.974, sub, ha="center", fontsize=10, color="#6b7280")

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.45, wspace=0.32,
        top=0.96, bottom=0.02,
        left=0.05, right=0.97
    )

    stats_list = [compute_stats(df) for df in dfs]

    # ── Row 0: bar charts ──
    ax_dice = fig.add_subplot(gs[0, 0])
    plot_bar_group(
        ax_dice, names, stats_list, colors,
        ["dice_liver", "dice_tumor"],
        "Dice score (liver & tumor)",
        ylim=(0.55, 1.05)
    )

    ax_pr = fig.add_subplot(gs[0, 1])
    plot_bar_group(
        ax_pr, names, stats_list, colors,
        ["precision_liver", "recall_liver"],
        "Precision vs Recall — liver",
        ylim=(0.55, 1.05)
    )

    ax_std = fig.add_subplot(gs[0, 2])
    plot_bar_group(
        ax_std, names, stats_list, colors,
        ["dice_liver", "dice_tumor"],
        "Std Dice (ổn định)",
        ylim=(0, 0.55)
    )
    # override với std values
    ax_std.cla()
    for i, (name, s, color) in enumerate(zip(names, stats_list, colors)):
        vals = [s["dice_liver_std"], s["dice_tumor_std"]]
        x = np.arange(2)
        offset = (i - (len(names)-1)/2) * 0.8/len(names)
        w = 0.8/len(names) * 0.9
        bars = ax_std.bar(x + offset, vals, w, color=color, alpha=0.85, label=name, zorder=3)
        for bar, val in zip(bars, vals):
            ax_std.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, color=color, fontweight="bold")
    ax_std.set_xticks([0, 1])
    ax_std.set_xticklabels(["Liver std", "Tumor std"], fontsize=8)
    ax_std.set_ylim(0, 0.55)
    ax_std.set_title("Std Dice — độ ổn định\n(thấp hơn = tốt hơn)", fontsize=10, fontweight="bold", pad=8)
    ax_std.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax_std.spines[["top","right"]].set_visible(False)
    ax_std.tick_params(labelsize=8)

    # ── Row 1: histogram + radar + fail bars ──
    ax_hist_l = fig.add_subplot(gs[1, 0])
    plot_dice_hist(ax_hist_l, dfs, names, colors, cls="liver")

    ax_hist_t = fig.add_subplot(gs[1, 1])
    plot_dice_hist(ax_hist_t, dfs, names, colors, cls="tumor")

    ax_radar = fig.add_subplot(gs[1, 2], polar=True)
    plot_radar(ax_radar, names, stats_list, colors)
    ax_radar.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.3, -0.1))

    # ── Row 2: near-fail stacked + over-seg bars ──
    ax_fail = fig.add_subplot(gs[2, :2])
    plot_fail_bars(ax_fail, names, stats_list, colors)

    ax_overseg = fig.add_subplot(gs[2, 2])
    x = np.arange(len(names))
    w = 0.35
    for ci, cls in enumerate(["liver", "tumor"]):
        offset = (ci - 0.5) * w
        vals = [s[f"overseg_{cls}_pct"] for s in stats_list]
        bars = ax_overseg.bar(x + offset, vals, w*0.9,
                              color=colors[:len(names)], alpha=0.8, zorder=3)
        for bar, val, color in zip(bars, vals, colors):
            ax_overseg.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                            f"{val:.1f}%", ha="center", va="bottom",
                            fontsize=7, color=color, fontweight="bold")
    ax_overseg.set_xticks(x)
    ax_overseg.set_xticklabels(names, fontsize=8)
    ax_overseg.set_ylim(0, 110)
    ax_overseg.set_ylabel("% trong near-fail cases", fontsize=8)
    ax_overseg.set_title("Over-segmentation rate\nwithin near-fail cases", fontsize=10, fontweight="bold", pad=8)
    ax_overseg.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax_overseg.spines[["top","right"]].set_visible(False)
    ax_overseg.tick_params(labelsize=8)

    # ── Row 3: bảng tổng hợp full width ──
    ax_tbl = fig.add_subplot(gs[3, :])
    plot_summary_table(ax_tbl, names, stats_list, colors)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Đã lưu report: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="So sánh nhiều file CSV evaluation segmentation model"
    )
    parser.add_argument(
        "files", nargs="+",
        help="Đường dẫn các file CSV (ít nhất 2)"
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Tên model tương ứng (nếu không điền sẽ dùng tên file)"
    )
    parser.add_argument(
        "--colors", nargs="+", default=None,
        help="Màu hex tùy chỉnh, ví dụ: #3b82f6 #f43f5e"
    )
    parser.add_argument(
        "--output", default="comparison_report.png",
        help="Tên file output (mặc định: comparison_report.png)"
    )
    args = parser.parse_args()

    # Đọc file
    dfs   = []
    names = []
    for i, fpath in enumerate(args.files):
        p = Path(fpath)
        if not p.exists():
            print(f"❌ Không tìm thấy file: {fpath}")
            sys.exit(1)
        df = pd.read_csv(p)
        required = ["dice_liver", "dice_tumor", "iou_liver", "iou_tumor",
                    "precision_liver", "recall_liver", "precision_tumor", "recall_tumor",
                    "mean_dice", "mean_iou", "pixel_acc"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"❌ File {fpath} thiếu cột: {missing}")
            sys.exit(1)
        dfs.append(df)
        name = args.names[i] if args.names and i < len(args.names) else p.stem
        names.append(name)
        print(f"Đã đọc: {fpath} → {name} ({len(df):,} ảnh)")

    colors = args.colors if args.colors else DEFAULT_COLORS[:len(dfs)]

    build_report(dfs, names, colors, args.output)


if __name__ == "__main__":
    main()