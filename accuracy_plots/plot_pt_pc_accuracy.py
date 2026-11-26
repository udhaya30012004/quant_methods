"""Polished Keynote-Style Overall Accuracy Plot: Per-Tensor vs Per-Channel (BF16 vs W8A16 vs W8A8).

No W8A8 Static. Saves to both:
  - E:\\quantt\\accuracy_bf16_w8a16_w8a8_pt_vs_pc.png
  - E:\\quantt\\ppt_plots\\02_Slide5_Overall_ZeroShot_Accuracy.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LM_DIR = ROOT / "results" / "lm_eval"
OUT_ROOT = ROOT / "accuracy_bf16_w8a16_w8a8_pt_vs_pc.png"
OUT_PPT = ROOT / "ppt_plots" / "02_Slide5_Overall_ZeroShot_Accuracy.png"

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]
PRIMARY = {
    "hellaswag": "acc_norm",
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
    "piqa": "acc",
}

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8fafc",
    "savefig.facecolor": "#ffffff",
    "savefig.dpi": 240,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": "#0f172a",
    "axes.labelcolor": "#0f172a",
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.0,
    "xtick.color": "#0f172a",
    "ytick.color": "#334155",
    "font.size": 10.5,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#cbd5e1",
    "grid.linestyle": "--",
    "grid.alpha": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_avg(fmt):
    j = json.loads((LM_DIR / f"{fmt}.json").read_text(encoding="utf-8"))
    return float(np.mean([j["tasks"][t][f"{PRIMARY[t]},none"] * 100.0 for t in TASKS]))


def main():
    bf16 = load_avg("bf16")
    w8a16 = [load_avg("w8a16_pt"), load_avg("w8a16_pc")]
    w8a8 = [load_avg("w8a8_pt_dyn"), load_avg("w8a8_pc")]
    bf16_bars = [bf16, bf16]

    x = np.arange(2)
    w = 0.23

    fig, ax = plt.subplots(figsize=(10.2, 5.9))

    # Subtle alternating background tint to visually separate Per-Tensor vs Per-Channel
    ax.axvspan(-0.5, 0.5, color="#eff6ff", alpha=0.35, zorder=0)
    ax.axvspan(0.5, 1.5, color="#f0fdfa", alpha=0.35, zorder=0)
    ax.axvline(0.5, color="#94a3b8", ls=":", lw=1.2, alpha=0.85, zorder=1)

    b_bf16 = ax.bar(x - w, bf16_bars, width=w * 0.86, color="#1e293b", edgecolor="#ffffff", lw=1.2,
                    label="BF16 (Baseline)", zorder=3)
    b_w16 = ax.bar(x,      w8a16,     width=w * 0.86, color="#2563eb", edgecolor="#ffffff", lw=1.2,
                   label="W8A16 (FP8 Weights, BF16 Acts)", zorder=3)
    b_w8 = ax.bar(x + w,   w8a8,      width=w * 0.86, color="#0d9488", edgecolor="#ffffff", lw=1.2,
                  label="W8A8 (FP8 Weights, FP8 Dyn Acts)", zorder=3)

    # Crisp baseline reference line + pill badge
    ax.axhline(bf16, color="#1e293b", ls="--", lw=1.3, alpha=0.65, zorder=2)
    ax.text(1.46, bf16 + 0.03, f"BF16 = {bf16:.2f}%", ha="right", va="bottom",
            fontsize=8.5, fontweight="bold", color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.22", fc="#ffffff", ec="#cbd5e1", lw=0.8))

    # Bigger, bolder % labels + colored delta badges above every bar
    for b in b_bf16:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=11.0, fontweight="heavy", color="#1e293b", zorder=5)

    for bars, accent in ((b_w16, "#1d4ed8"), (b_w8, "#0f766e")):
        for b in bars:
            v = b.get_height()
            cx = b.get_x() + b.get_width() / 2
            ax.text(cx, v + 0.17, f"{v:.2f}%",
                    ha="center", va="bottom", fontsize=11.2, fontweight="heavy", color="#0f172a", zorder=5)
            ax.text(cx, v + 0.04, f"({v - bf16:+.2f}%)",
                    ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=accent, zorder=5)

    bar_positions = [0 - w, 0, 0 + w, 1 - w, 1, 1 + w]
    bar_labels = ["BF16", "W8A16", "W8A8", "BF16", "W8A16", "W8A8"]
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, fontsize=11.0, fontweight="bold")
    ax.tick_params(axis="x", pad=7)

    # Pill-styled bold subdivision titles cleanly below the bars
    ax.text(0.0, -0.15, "Per-Tensor Quantization", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11.5, fontweight="bold", color="#1e3a8a",
            bbox=dict(boxstyle="round,pad=0.32", fc="#dbeafe", ec="#93c5fd", lw=0.9))
    ax.text(1.0, -0.15, "Per-Channel Quantization", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11.5, fontweight="bold", color="#115e59",
            bbox=dict(boxstyle="round,pad=0.32", fc="#ccfbf1", ec="#5eead4", lw=0.9))

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Overall 5-Task Zero-Shot Accuracy (%)  ↑ better", fontweight="bold")
    ax.set_ylim(67.5, 70.25)
    ax.set_title("Overall Zero-Shot Accuracy: BF16 vs W8A16 vs W8A8 (Qwen2.5-3B)", pad=26)
    fig.text(
        0.5,
        0.91,
        "Average across HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande & PIQA (16,695 Questions)",
        ha="center",
        fontsize=9.8,
        color="#475569",
    )
    ax.legend(
        facecolor="#ffffff", edgecolor="#cbd5e1", framealpha=0.98,
        loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3,
    )

    fig.tight_layout()
    for p in (OUT_ROOT, OUT_PPT):
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", facecolor="#ffffff")
        print(f"saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
