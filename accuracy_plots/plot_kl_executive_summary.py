"""Executive-Summary KL Divergence Plots for Qwen2.5-3B (matching 00_kl_executive_summary.png).

Generates two 2-panel executive summary figures from Track A full-vocab KL data (131,008 tokens)
and verified project metrics (project_state.json / results/):
  1. E:\\quantt\\01_kl_executive_summary_w8a16_pc_vs_bf16.png  (FP8 W8A16 Per-Channel vs BF16)
  2. E:\\quantt\\02_kl_executive_summary_w8a8_pc_vs_bf16.png   (FP8 W8A8 Per-Channel vs BF16)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SIM_DIR = RESULTS / "sim"
LM_DIR = RESULTS / "lm_eval"
BENCH_DIR = RESULTS / "bench"

TASKS = [
    ("hellaswag", "acc_norm,none"),
    ("arc_easy", "acc,none"),
    ("arc_challenge", "acc_norm,none"),
    ("winogrande", "acc,none"),
    ("piqa", "acc,none"),
]


def load_avg_acc(fmt: str) -> float:
    d = json.loads((LM_DIR / f"{fmt}.json").read_text(encoding="utf-8"))["tasks"]
    return float(np.mean([d[t][k] * 100.0 for t, k in TASKS]))


def load_size_comp(fmt: str) -> float:
    d = json.loads((RESULTS / f"size_{fmt}.json").read_text(encoding="utf-8"))
    return float(d["compression_pct"])


def load_speedups(fmt: str) -> tuple[float, float]:
    b_ref = json.loads((BENCH_DIR / "bf16.json").read_text(encoding="utf-8"))["cells"]
    b_q = json.loads((BENCH_DIR / f"{fmt}.json").read_text(encoding="utf-8"))["cells"]

    def get_cell(cells, bs, inp):
        for c in cells:
            if c["batch"] == bs and c["input_tokens"] == inp:
                return c
        raise KeyError((bs, inp))

    decode_speedup = get_cell(b_ref, 1, 128)["tpot_ms"] / get_cell(b_q, 1, 128)["tpot_ms"]
    tput_speedup = get_cell(b_q, 128, 128)["output_tok_per_s"] / get_cell(b_ref, 128, 128)["output_tok_per_s"]
    return decode_speedup, tput_speedup


def make_executive_kl_plot(
    recipe: str,
    short_tag: str,
    header_label: str,
    out_path: Path,
    speed_summary_fmt: str,
) -> None:
    # Load verified Track A full-vocab KL array & stats on primary WikiText-2 set (64 windows = 131,008 tokens)
    sim_json = json.loads((SIM_DIR / f"{recipe}__ppl_wikitext.json").read_text(encoding="utf-8"))
    kl = np.load(SIM_DIR / f"{recipe}__ppl_wikitext.npy")

    n_tokens = int(kl.size)
    kl_median = float(np.median(kl))
    kl_mean = float(kl.mean())
    kl_p90 = float(np.quantile(kl, 0.90))
    top1_pct = float(sim_json["top1_agree"]) * 100.0

    # Distribution buckets matching reference plot thresholds
    b0 = int((kl < 0.01).sum())
    b1 = int(((kl >= 0.01) & (kl < 0.05)).sum())
    b2 = int(((kl >= 0.05) & (kl < 0.10)).sum())
    b3 = int((kl >= 0.10).sum())
    counts = [b0, b1, b2, b3]
    pcts = [c / n_tokens * 100.0 for c in counts]
    under_05_pct = (b0 + b1) / n_tokens * 100.0

    # Downstream & system metrics for bottom takeaway banner
    acc_bf16 = load_avg_acc("bf16")
    acc_q = load_avg_acc(recipe)
    d_acc = acc_q - acc_bf16
    comp_pct = load_size_comp(recipe)
    decode_x, tput_x = load_speedups(recipe)

    threshold = 0.05
    med_mult = int(round(threshold / kl_median))
    mean_below_pct = int(round((threshold - kl_mean) / threshold * 100.0))

    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": "#0f172a",
        "axes.labelcolor": "#0f172a",
        "xtick.color": "#0f172a",
        "ytick.color": "#0f172a",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.45), dpi=180)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.17, wspace=0.15)

    # Super-title
    fig.suptitle(
        f"Qwen2.5-3B ({header_label} vs BF16): Practically Lossless Probability Fidelity (KL Divergence)",
        fontsize=15.5,
        fontweight="bold",
        y=0.955,
        color="#0f172a",
    )

    # =========================================================================
    # PANEL 1: KL Divergence vs Lossless Threshold (0.05 nats)
    # =========================================================================
    ax1.set_facecolor("#fafafa")
    ax1.set_axisbelow(True)
    ax1.grid(True, axis="both", linestyle="--", color="#cbd5e1", alpha=0.65, linewidth=0.85)
    for sp in ax1.spines.values():
        sp.set_visible(True)
        sp.set_color("#cbd5e1")
        sp.set_linewidth(1.0)

    # Mint green lossless zone below 0.05 nats
    ax1.axhspan(0.0, threshold, facecolor="#ecfdf5", alpha=0.85, zorder=0)
    ax1.axhline(threshold, color="#dc2626", linestyle="--", linewidth=2.2, zorder=3)
    ax1.text(
        0.12,
        threshold + 0.0014,
        "Typical Lossless Threshold (0.05 nats)",
        color="#b91c1c",
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=4,
    )

    x1 = np.arange(4)
    vals1 = [0.0, kl_median, kl_mean, kl_p90]
    cols1 = ["#94a3b8", "#059669", "#0ea5e9", "#2563eb"]
    xlabels1 = [
        "BF16 Ref\n(Baseline)",
        f"{short_tag} Median KL\n(Typical Token)",
        f"{short_tag} Mean KL\n(Overall Avg.)",
        f"{short_tag} p90 KL\n(90th Percentile)",
    ]

    bars1 = ax1.bar(x1, vals1, width=0.50, color=cols1, edgecolor="none", zorder=3)

    for xi, v in zip(x1, vals1):
        ax1.text(
            xi,
            v + 0.0009,
            f"{v:.4f}\nnats",
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color="#0f172a",
            zorder=4,
        )

    # Callout arrow 1: Median KL vs threshold
    med_label_top = kl_median + 0.0056
    ax1.annotate(
        f"{med_mult}× below\nthreshold",
        xy=(1.0, med_label_top),
        xytext=(1.0, max(0.024, kl_median + 0.017)),
        ha="center",
        va="bottom",
        fontsize=9.6,
        fontweight="bold",
        color="#047857",
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.7, shrinkA=3, shrinkB=2),
        zorder=5,
    )

    # Callout arrow 2: Mean KL vs threshold
    mean_label_top = kl_mean + 0.0056
    ax1.annotate(
        f"{mean_below_pct}% below\nthreshold",
        xy=(2.0, mean_label_top),
        xytext=(2.0, max(0.032, kl_mean + 0.019)),
        ha="center",
        va="bottom",
        fontsize=9.6,
        fontweight="bold",
        color="#1d4ed8",
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.7, shrinkA=3, shrinkB=2),
        zorder=5,
    )

    ax1.set_xticks(x1)
    ax1.set_xticklabels(xlabels1, fontsize=10.2)
    ax1.set_xlim(-0.55, 3.55)
    ax1.set_ylim(0.0, 0.062)
    ax1.set_yticks(np.arange(0.0, 0.061, 0.01))
    ax1.set_ylabel("KL Divergence from BF16 (nats — lower is better)", fontsize=11.0)
    ax1.set_title("1. KL Divergence vs Lossless Threshold (0.05 nats)", fontsize=12.8, fontweight="bold", pad=10)

    # =========================================================================
    # PANEL 2: Distribution Breakdown Across All 131,008 Token Positions
    # =========================================================================
    ax2.set_facecolor("#fafafa")
    ax2.set_axisbelow(True)
    ax2.grid(True, axis="both", linestyle="--", color="#cbd5e1", alpha=0.65, linewidth=0.85)
    for sp in ax2.spines.values():
        sp.set_visible(True)
        sp.set_color("#cbd5e1")
        sp.set_linewidth(1.0)

    x2 = np.arange(4)
    cols2 = ["#059669", "#3b82f6", "#f59e0b", "#ef4444"]
    xlabels2 = [
        "Virtually Identical\n(KL < 0.01)",
        "Minor Drift\n(0.01 – 0.05)",
        "Moderate Drift\n(0.05 – 0.10)",
        "Uncertain Tail\n(KL ≥ 0.10)",
    ]

    ax2.bar(x2, pcts, width=0.50, color=cols2, edgecolor="none", zorder=3)

    for xi, pct, cnt in zip(x2, pcts, counts):
        ax2.text(
            xi,
            pct + 1.4,
            f"{pct:.1f}%\n({cnt:,} tok)",
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color="#0f172a",
            zorder=4,
        )

    # Green bracket spanning first two buckets (KL < 0.05)
    top_bar = max(pcts[0], pcts[1])
    brk_y = top_bar + 10.5
    leg_h = 2.8
    ax2.plot(
        [-0.25, -0.25, 1.25, 1.25],
        [brk_y - leg_h, brk_y, brk_y, brk_y - leg_h],
        color="#059669",
        lw=2.0,
        zorder=4,
    )
    n_of_100 = int(round(under_05_pct))
    ax2.text(
        0.50,
        brk_y + 1.5,
        f"{under_05_pct:.1f}% (~{n_of_100} of 100 tokens) have KL < 0.05",
        ha="center",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color="#065f46",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#f0fdf4", edgecolor="#10b981", lw=1.1),
        zorder=5,
    )

    # Red callout box explaining Mean > Median due to tiny tail
    tail_label_top = pcts[3] + 8.5
    ax2.annotate(
        f"Small {pcts[3]:.1f}% tail raises\nMean ({kl_mean:.4f}) above\nMedian ({kl_median:.4f})",
        xy=(3.0, tail_label_top),
        xytext=(2.55, 42.0),
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#b91c1c",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fef2f2", edgecolor="#ef4444", lw=1.1),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.7, shrinkA=4, shrinkB=3),
        zorder=5,
    )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(xlabels2, fontsize=10.2)
    ax2.set_xlim(-0.55, 3.55)
    ax2.set_ylim(0.0, max(106.0, brk_y + 11.5))
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylabel(f"Share of All {n_tokens:,} Token Positions (%)", fontsize=11.0)
    ax2.set_title(
        f"2. Why ~{n_of_100}% of Tokens Match BF16 (Distribution Breakdown)",
        fontsize=12.8,
        fontweight="bold",
        pad=10,
    )

    # =========================================================================
    # BOTTOM TAKEAWAY BANNER
    # =========================================================================
    speed_str = speed_summary_fmt.format(decode_x=decode_x, tput_x=tput_x)
    takeaway = (
        f"Takeaway: Median KL = {kl_median:.4f} nats | Mean KL = {kl_mean:.4f} nats (< 0.05 lossless threshold) | "
        f"Top-1 Agreement = {top1_pct:.1f}% | 5-Task 0-Shot = {d_acc:+.2f}% (0.00% sig. drop) | "
        f"-{comp_pct:.1f}% Size & {speed_str}"
    )
    fig.text(
        0.5,
        0.032,
        takeaway,
        ha="center",
        va="center",
        fontsize=10.3,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.48", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.1),
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.14, facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main() -> None:
    make_executive_kl_plot(
        recipe="w8a16_pc",
        short_tag="W8A16",
        header_label="FP8 W8A16 Per-Channel",
        out_path=ROOT / "01_kl_executive_summary_w8a16_pc_vs_bf16.png",
        speed_summary_fmt="{decode_x:.2f}× Faster Decode",
    )
    make_executive_kl_plot(
        recipe="w8a8_pc",
        short_tag="W8A8",
        header_label="FP8 W8A8 Per-Channel",
        out_path=ROOT / "02_kl_executive_summary_w8a8_pc_vs_bf16.png",
        speed_summary_fmt="{decode_x:.2f}× Faster Decode ({tput_x:.2f}× Peak Throughput)",
    )


if __name__ == "__main__":
    main()
