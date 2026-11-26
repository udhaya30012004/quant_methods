"""Single-Panel Mean KL Divergence Comparison: Per-Tensor vs Per-Channel (BF16 vs W8A16 vs W8A8).

Matches the exact visual style of E:\\quantt\\accuracy_bf16_w8a16_w8a8_pt_vs_pc.png.
Saves directly to E:\\quantt\\kl_bf16_w8a16_w8a8_pt_vs_pc.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "results" / "sim"
OUT_PNG = ROOT / "kl_bf16_w8a16_w8a8_pt_vs_pc.png"

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "savefig.dpi": 220,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": "#0f172a",
    "axes.labelcolor": "#0f172a",
    "axes.edgecolor": "#94a3b8",
    "axes.linewidth": 0.9,
    "xtick.color": "#0f172a",
    "ytick.color": "#0f172a",
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#cbd5e1",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_kl(recipe):
    j = json.loads((SIM_DIR / f"{recipe}__ppl_wikitext.json").read_text(encoding="utf-8"))
    return float(j["kl_mean"])


def main():
    bf16_kl = load_kl("none")            # 0.0
    fp16_kl = load_kl("fp16cast")        # 0.00067
    w8a16 = [load_kl("w8a16_pt"), load_kl("w8a16_pc")]
    w8a8 = [load_kl("w8a8_pt"), load_kl("w8a8_pc")]
    bf16_bars = [bf16_kl, bf16_kl]

    x = np.arange(2)
    w = 0.24

    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    b_bf16 = ax.bar(x - w, bf16_bars, width=w * 0.90, color="#334155", edgecolor="#ffffff", lw=1.0, label="BF16 (Baseline = 0)")
    b_w16 = ax.bar(x,      w8a16,     width=w * 0.90, color="#2563eb", edgecolor="#ffffff", lw=1.0, label="W8A16 (FP8 Weights, BF16 Acts)")
    b_w8 = ax.bar(x + w,   w8a8,      width=w * 0.90, color="#0d9488", edgecolor="#ffffff", lw=1.0, label="W8A8 (FP8 Weights, FP8 Dyn Acts)")

    ax.axhline(fp16_kl, color="#64748b", ls="--", lw=1.2, alpha=0.85, label=f"FP16 Cast Noise Floor ({fp16_kl:.5f})")
    ax.axvline(0.5, color="#cbd5e1", ls="-", lw=1.0, alpha=0.8)

    # Labels above every bar
    for b in b_bf16:
        ax.text(
            b.get_x() + b.get_width() / 2,
            0.00035,
            "0.0000\n(Ref)",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color="#334155",
        )
    for b in b_w16:
        v = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.00028,
            f"{v:.5f}\n({v / fp16_kl:.1f}× floor)",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color="#0f172a",
        )
    for b in b_w8:
        v = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.00028,
            f"{v:.5f}\n({v / fp16_kl:.1f}× floor)",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color="#0f172a",
        )

    bar_positions = [0 - w, 0, 0 + w, 1 - w, 1, 1 + w]
    bar_labels = ["BF16", "W8A16", "W8A8", "BF16", "W8A16", "W8A8"]
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, fontsize=10.5, fontweight="bold")
    ax.tick_params(axis="x", pad=6)

    ax.text(0.0, -0.14, "Per-Tensor Quantization", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax.text(1.0, -0.14, "Per-Channel Quantization", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11.5, fontweight="bold", color="#0f172a")

    ax.set_ylabel("Mean KL(P_BF16 || P_quant)  ↓ lower is better")
    ax.set_ylim(0, 0.0138)
    ax.set_title("Mean KL Divergence vs BF16: W8A16 vs W8A8 (Qwen2.5-3B)", pad=22)
    fig.text(
        0.5,
        0.915,
        "Full-Vocabulary Distribution Shift on WikiText-2 (131,008 Tokens, Lower = Closer to BF16)",
        ha="center",
        fontsize=9.5,
        color="#475569",
    )
    ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", framealpha=0.95, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
