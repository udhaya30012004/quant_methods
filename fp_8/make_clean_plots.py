"""Clean, presentation-ready comparison plots for Qwen2.5-3B FP8 PTQ -> E:\\quantt\\accuracy_plots\\.

Focuses strictly on headline comparison metrics:
  01_accuracy_comparison.png      : Benchmark accuracy (5 tasks + average) across BF16 & all FP8 methods
  02_method_axis_comparison.png   : Direct head-to-head bars: W8A16 vs W8A8, Per-Tensor vs Per-Channel, Dynamic vs Static
  03_model_size_comparison.png    : Model size (GB) & compression (%) bar chart
  04_kl_divergence_metrics.png    : Mean KL, P99 KL, and Top-1 Token Agreement comparison
  05_loss_and_perplexity.png      : Perplexity bars + loss curve across evaluation windows
  06_speed_and_latency.png        : Throughput (tok/s) & single-user decode latency (ms/tok)

Run locally:
    python fp_8/make_clean_plots.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stats import ALL_FMTS, A_NAME, FMTS, PRIMARY, ROOT, TASKS, lm_metric, load_results

OUT = ROOT / "accuracy_plots"

SHORT = {
    "bf16":           "BF16\n(Baseline)",
    "w8a16_pt":       "W8A16\nPer-Tensor",
    "w8a16_pc":       "W8A16\nPer-Channel",
    "w8a8_pc":        "W8A8\nPer-Channel",
    "w8a8_pt_dyn":    "W8A8\nPer-Tensor Dyn",
    "w8a8_pt_static": "W8A8\nPer-Tensor Static",
}
COLORS = {
    "bf16":           "#2c3e50",
    "w8a16_pt":       "#2980b9",
    "w8a16_pc":       "#5dade2",
    "w8a8_pc":        "#e67e22",
    "w8a8_pt_dyn":    "#e74c3c",
    "w8a8_pt_static": "#922b21",
}
ORDER = ["bf16", "w8a16_pc", "w8a16_pt", "w8a8_pc", "w8a8_pt_dyn", "w8a8_pt_static"]
Q_ORDER = ["w8a16_pc", "w8a16_pt", "w8a8_pc", "w8a8_pt_dyn", "w8a8_pt_static"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "legend.fontsize": 8.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.28,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {p}")


def label_bars(ax, bars, fmt="{:.2f}", dy=0.0, fontsize=8.5, weight="normal"):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight=weight,
        )


# ------------------------------------------------------------------ 01: Accuracy Comparison
def plot_01_accuracy(R):
    task_labels = ["HellaSwag", "ARC-Easy", "ARC-Challenge", "WinoGrande", "PIQA", "5-Task Avg"]
    x = np.arange(len(task_labels))
    w = 0.13

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13.0, 8.2), gridspec_kw={"height_ratios": [1.35, 1.0]}
    )

    # Top panel: grouped bar chart across all 5 tasks + average
    avgs = {}
    for i, f in enumerate(ORDER):
        vals = [lm_metric(R, f, t)[0] * 100.0 for t in TASKS]
        avg = float(np.mean(vals))
        avgs[f] = avg
        vals.append(avg)
        offs = x + (i - (len(ORDER) - 1) / 2) * w
        ax_top.bar(
            offs, vals, width=w, color=COLORS[f], edgecolor="white", lw=0.6,
            label=SHORT[f].replace("\n", " "),
        )
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(task_labels, fontsize=10, fontweight="bold")
    ax_top.set_ylabel("Accuracy (%)")
    ax_top.set_ylim(40, 84)
    ax_top.set_title("Downstream 0-Shot Benchmark Accuracy: BF16 vs All FP8 PTQ Methods")
    ax_top.legend(frameon=False, ncol=3, loc="upper right")

    # Bottom panel: zoomed 5-Task Average bar chart with exact labels
    xs = np.arange(len(ORDER))
    avg_vals = [avgs[f] for f in ORDER]
    bars = ax_bot.bar(xs, avg_vals, width=0.52, color=[COLORS[f] for f in ORDER], edgecolor="black", lw=0.7)
    ax_bot.axhline(avgs["bf16"], color="#2c3e50", ls="--", lw=1.1, alpha=0.7, label="BF16 Baseline (68.89%)")
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels([SHORT[f] for f in ORDER], fontsize=9.5)
    ax_bot.set_ylabel("Average Accuracy (%)")
    ax_bot.set_ylim(67.5, 70.2)
    ax_bot.set_title("5-Task Average Accuracy (Zoomed View — All FP8 Recipes Preserve BF16 Accuracy)")
    for b, f in zip(bars, ORDER):
        h = b.get_height()
        diff = h - avgs["bf16"]
        txt = f"{h:.2f}%" if f == "bf16" else f"{h:.2f}%\n({diff:+.2f}%)"
        ax_bot.text(b.get_x() + b.get_width() / 2, h + 0.06, txt, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_bot.legend(frameon=False, loc="lower right")

    save(fig, "01_accuracy_comparison")


# ------------------------------------------------------------------ 02: W8A16 vs W8A8 & PT vs PC
def plot_02_method_axes(R):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))

    def get_kl(f):
        return R["sim"][A_NAME.get(f, f)]["ppl_wikitext"]["stats"]["kl_mean"]

    def get_dppl(f):
        return R["trackb"][f]["ppl_wikitext"]["stats"]["ppl"] - R["trackb"]["bf16"]["ppl_wikitext"]["stats"]["ppl"]

    def get_acc(f):
        return float(np.mean([lm_metric(R, f, t)[0] * 100.0 for t in TASKS]))

    # Panel A: W8A16 vs W8A8 (Per-Channel & Per-Tensor paired)
    ax = axes[0]
    cats = ["Per-Channel\n(W8A16 vs W8A8)", "Per-Tensor\n(W8A16 vs W8A8)"]
    w8a16_kl = [get_kl("w8a16_pc"), get_kl("w8a16_pt")]
    w8a8_kl = [get_kl("w8a8_pc"), get_kl("w8a8_pt_dyn")]
    x = np.arange(2)
    b1 = ax.bar(x - 0.18, w8a16_kl, width=0.34, color="#2980b9", edgecolor="black", lw=0.7, label="W8A16 (BF16 Acts)")
    b2 = ax.bar(x + 0.18, w8a8_kl, width=0.34, color="#e74c3c", edgecolor="black", lw=0.7, label="W8A8 (FP8 Acts)")
    label_bars(ax, b1, "{:.4f}", dy=0.0003, fontsize=9, weight="bold")
    label_bars(ax, b2, "{:.4f}", dy=0.0003, fontsize=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Mean KL Divergence (lower = better)")
    ax.set_ylim(0, 0.0145)
    ax.set_title("W8A16 vs W8A8\n(Act Quantization Doubles KL)")
    ax.legend(frameon=False)

    # Panel B: Per-Tensor vs Per-Channel (within W8A16 & W8A8)
    ax = axes[1]
    cats = ["W8A16\n(PT vs PC)", "W8A8 Dynamic\n(PT vs PC)"]
    pt_ppl = [get_dppl("w8a16_pt"), get_dppl("w8a8_pt_dyn")]
    pc_ppl = [get_dppl("w8a16_pc"), get_dppl("w8a8_pc")]
    b1 = ax.bar(x - 0.18, pt_ppl, width=0.34, color="#f39c12", edgecolor="black", lw=0.7, label="Per-Tensor")
    b2 = ax.bar(x + 0.18, pc_ppl, width=0.34, color="#16a085", edgecolor="black", lw=0.7, label="Per-Channel / Token")
    label_bars(ax, b1, "+{:.3f}", dy=0.002, fontsize=9, weight="bold")
    label_bars(ax, b2, "+{:.3f}", dy=0.002, fontsize=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("WikiText ΔPPL vs BF16 (lower = better)")
    ax.set_ylim(0, 0.095)
    ax.set_title("Per-Tensor vs Per-Channel\n(PC Wins Clearly in W8A8)")
    ax.legend(frameon=False)

    # Panel C: Dynamic vs Static Activations (W8A8)
    ax = axes[2]
    names = ["W8A8\nPer-Channel\n(Dynamic)", "W8A8\nPer-Tensor\n(Dynamic)", "W8A8\nPer-Tensor\n(Static)"]
    kls = [get_kl("w8a8_pc"), get_kl("w8a8_pt_dyn"), get_kl("w8a8_pt_static")]
    cols = [COLORS["w8a8_pc"], COLORS["w8a8_pt_dyn"], COLORS["w8a8_pt_static"]]
    b = ax.bar(np.arange(3), kls, width=0.52, color=cols, edgecolor="black", lw=0.7)
    label_bars(ax, b, "{:.4f}", dy=0.0004, fontsize=9.5, weight="bold")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean KL Divergence (lower = better)")
    ax.set_ylim(0, 0.0185)
    ax.set_title("Dynamic vs Static Activations\n(Static Adds +45% KL Error)")

    fig.suptitle("Head-to-Head Quantization Design Choices (Qwen2.5-3B)", fontsize=13.5, fontweight="bold", y=1.03)
    save(fig, "02_method_axis_comparison")


# ------------------------------------------------------------------ 03: Model Size Comparison
def plot_03_model_size(R):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    xs = np.arange(len(ORDER))
    sizes_gb = [R["size"][f]["quant_bytes"] / 1e9 for f in ORDER]
    comps = [R["size"][f]["compression_pct"] for f in ORDER]
    bars = ax.bar(xs, sizes_gb, width=0.55, color=[COLORS[f] for f in ORDER], edgecolor="black", lw=0.8)

    for b, gb, cp, f in zip(bars, sizes_gb, comps, ORDER):
        lbl = f"{gb:.2f} GB\n(Baseline)" if f == "bf16" else f"{gb:.2f} GB\n(-{cp:.1f}%)"
        ax.text(b.get_x() + b.get_width() / 2, gb + 0.12, lbl, ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[f] for f in ORDER], fontsize=9.5)
    ax.set_ylabel("Checkpoint Size on Disk (GB)")
    ax.set_ylim(0, 7.3)
    ax.set_title("Model Size Comparison: BF16 vs FP8 PTQ Checkpoints (~45% Size Reduction Across All FP8 Methods)")
    save(fig, "03_model_size_comparison")


# ------------------------------------------------------------------ 04: KL Divergence Metrics
def plot_04_kl_metrics(R):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    x = np.arange(len(Q_ORDER))
    w = 0.36

    # Left: Mean KL on WikiText-2 vs UltraChat
    kl_wiki = [R["sim"][A_NAME.get(f, f)]["ppl_wikitext"]["stats"]["kl_mean"] for f in Q_ORDER]
    kl_chat = [R["sim"][A_NAME.get(f, f)]["kl_chat"]["stats"]["kl_mean"] for f in Q_ORDER]
    b1 = ax1.bar(x - w / 2, kl_wiki, width=w, color="#2e86c1", edgecolor="black", lw=0.7, label="WikiText-2")
    b2 = ax1.bar(x + w / 2, kl_chat, width=w, color="#1abc9c", edgecolor="black", lw=0.7, label="UltraChat")
    fp16_kl = R["sim"]["fp16cast"]["ppl_wikitext"]["stats"]["kl_mean"]
    ax1.axhline(fp16_kl, color="gray", ls="--", lw=1.3, label=f"FP16 Cast Noise Floor ({fp16_kl:.5f})")
    label_bars(ax1, b1, "{:.4f}", dy=0.0003, fontsize=8, weight="bold")
    label_bars(ax1, b2, "{:.4f}", dy=0.0003, fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SHORT[f] for f in Q_ORDER], fontsize=8.8)
    ax1.set_ylabel("Mean KL(P_BF16 || P_quant)  [lower = better]")
    ax1.set_ylim(0, 0.0185)
    ax1.set_title("Mean KL Divergence vs BF16 Baseline")
    ax1.legend(frameon=False, loc="upper left")

    # Right: P99 KL (tail shift) + Top-1 Token Agreement (%)
    p99 = [R["sim"][A_NAME.get(f, f)]["ppl_wikitext"]["stats"]["kl_p99"] for f in Q_ORDER]
    top1 = [R["sim"][A_NAME.get(f, f)]["ppl_wikitext"]["stats"]["top1_agree"] * 100.0 for f in Q_ORDER]
    bars = ax2.bar(x, p99, width=0.52, color=[COLORS[f] for f in Q_ORDER], edgecolor="black", lw=0.7, label="P99 KL (Tail Error)")
    label_bars(ax2, bars, "{:.3f}", dy=0.003, fontsize=9, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([SHORT[f] for f in Q_ORDER], fontsize=8.8)
    ax2.set_ylabel("99th-Percentile Token KL  [lower = better]")
    ax2.set_ylim(0, 0.155)
    ax2.set_title("Tail KL (P99) & Top-1 Token Agreement (WikiText-2)")

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.plot(x, top1, color="#1b2631", marker="o", lw=2.0, ms=7, label="Top-1 Agreement (%)")
    for xi, t1 in zip(x, top1):
        ax2_r.annotate(f"{t1:.1f}%", (xi, t1), xytext=(0, 8), textcoords="offset points",
                       ha="center", fontsize=8.5, fontweight="bold", color="#1b2631")
    ax2_r.set_ylabel("Top-1 Token Agreement (%)  [higher = better]")
    ax2_r.set_ylim(92.0, 98.5)
    ax2_r.grid(False)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    fig.suptitle("KL Divergence & Token Agreement Metrics Across FP8 Recipes", fontsize=13, fontweight="bold", y=1.02)
    save(fig, "04_kl_divergence_metrics")


# ------------------------------------------------------------------ 05: Loss & Perplexity Curves
def plot_05_loss_ppl(R):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Left: WikiText-2 Perplexity bar chart (zoomed) with exact values + delta
    xs = np.arange(len(ORDER))
    ppls = [R["trackb"][f]["ppl_wikitext"]["stats"]["ppl"] for f in ORDER]
    base_ppl = ppls[0]
    bars = ax1.bar(xs, ppls, width=0.55, color=[COLORS[f] for f in ORDER], edgecolor="black", lw=0.7)
    ax1.axhline(base_ppl, color="#2c3e50", ls="--", lw=1.0, alpha=0.65)
    for b, p, f in zip(bars, ppls, ORDER):
        txt = f"{p:.3f}" if f == "bf16" else f"{p:.3f}\n(+{p - base_ppl:.3f})"
        ax1.text(b.get_x() + b.get_width() / 2, p + 0.004, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([SHORT[f] for f in ORDER], fontsize=8.6)
    ax1.set_ylabel("WikiText-2 Perplexity (lower = better)")
    ax1.set_ylim(7.72, 7.92)
    ax1.set_title("Perplexity Comparison (Real vLLM, 64 Windows)")

    # Right: ΔLoss curve across the 64 WikiText-2 evaluation windows (sorted by BF16 window loss)
    base_w = np.asarray(R["trackb"]["bf16"]["ppl_wikitext"]["stats"]["window_losses"])
    order_idx = np.argsort(base_w)
    win_x = np.arange(1, len(base_w) + 1)

    for f in Q_ORDER:
        qw = np.asarray(R["trackb"][f]["ppl_wikitext"]["stats"]["window_losses"])
        delta = (qw - base_w)[order_idx]
        # light raw curve + smooth rolling average (window=5) for readability
        smooth = np.convolve(delta, np.ones(5) / 5, mode="valid")
        ax2.plot(win_x, delta, color=COLORS[f], lw=0.7, alpha=0.22)
        ax2.plot(win_x[2:-2], smooth, color=COLORS[f], lw=2.0, label=f"{SHORT[f].replace(chr(10), ' ')} (mean {delta.mean():+.4f})")

    ax2.axhline(0.0, color="#2c3e50", ls="--", lw=1.0, label="BF16 Baseline (ΔLoss = 0)")
    ax2.set_xlabel("Evaluation Window (sorted easiest → hardest by BF16 loss)")
    ax2.set_ylabel("Δ Cross-Entropy Loss vs BF16  (lower = better)")
    ax2.set_title("Loss Degradation Curve Across All 64 WikiText-2 Windows")
    ax2.legend(frameon=False, fontsize=8.3, loc="upper left")

    fig.suptitle("Perplexity & Cross-Entropy Loss Comparison (BF16 vs FP8 PTQ)", fontsize=13, fontweight="bold", y=1.02)
    save(fig, "05_loss_and_perplexity")


# ------------------------------------------------------------------ 06: Speed & Latency
def plot_06_speed_latency(R):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    xs = np.arange(len(ORDER))

    def get_cell(f, bs, inp):
        return next(c for c in R["bench"][f]["cells"] if c["batch"] == bs and c["input_tokens"] == inp)

    # Left: Peak Throughput at B=128 (in=128)
    tps = [get_cell(f, 128, 128)["output_tok_per_s"] for f in ORDER]
    base_tps = tps[0]
    bars = ax1.bar(xs, tps, width=0.55, color=[COLORS[f] for f in ORDER], edgecolor="black", lw=0.7)
    for b, v, f in zip(bars, tps, ORDER):
        txt = f"{v:.0f}" if f == "bf16" else f"{v:.0f}\n({v / base_tps:.2f}×)"
        ax1.text(b.get_x() + b.get_width() / 2, v + 55, txt, ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([SHORT[f] for f in ORDER], fontsize=8.8)
    ax1.set_ylabel("Output Throughput (tokens / sec)  [higher = better]")
    ax1.set_ylim(0, 3950)
    ax1.set_title("Peak Serving Throughput (Batch 128, L4 GPU)")

    # Right: Single-User Decode Latency (TPOT ms/tok at B=1)
    tpot = [get_cell(f, 1, 128)["tpot_ms"] for f in ORDER]
    base_tpot = tpot[0]
    bars = ax2.bar(xs, tpot, width=0.55, color=[COLORS[f] for f in ORDER], edgecolor="black", lw=0.7)
    for b, v, f in zip(bars, tpot, ORDER):
        txt = f"{v:.1f} ms" if f == "bf16" else f"{v:.1f} ms\n({base_tpot / v:.2f}× faster)"
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.45, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([SHORT[f] for f in ORDER], fontsize=8.8)
    ax2.set_ylabel("Time Per Output Token (ms / tok)  [lower = better]")
    ax2.set_ylim(0, 30.5)
    ax2.set_title("Single-User Decode Latency (Batch 1, L4 GPU)")

    fig.suptitle("vLLM Serving Speed & Latency on NVIDIA L4 (Ada cc 8.9)", fontsize=13, fontweight="bold", y=1.02)
    save(fig, "06_speed_and_latency")


def main():
    R = load_results()
    print(f"Generating clean presentation plots -> {OUT}")
    plot_01_accuracy(R)
    plot_02_method_axes(R)
    plot_03_model_size(R)
    plot_04_kl_metrics(R)
    plot_05_loss_ppl(R)
    plot_06_speed_latency(R)
    print("Done — 6 clean plots saved.")


if __name__ == "__main__":
    main()
