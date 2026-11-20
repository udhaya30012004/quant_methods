"""Time, Latency & Throughput Plots for Qwen2.5-3B FP8 PTQ on Modal L4.

Self-contained script: reads E:\\quantt\\results\\bench\\*.json and saves all plots
in the SAME folder (E:\\quantt\\time_plots\\).

Plots generated:
  01_decode_latency_tpot.png  : Per-token decode latency (ms/tok) across batch sizes + W8A16/W8A8 crossover
  02_prefill_time_ttft.png    : Prefill / TTFT latency for short (128) vs long (512) prompts
  03_end_to_end_latency.png   : Stacked End-to-End time breakdown (Prefill vs Decode) at B=1, 32, 128
  04_throughput_scaling.png   : Output & Total tokens/sec scaling across batch sizes (1 -> 128)
  05_speedup_heatmap.png      : Speedup vs BF16 across all 8 (Batch x Input Length) scenarios

Run locally:
    python E:\\quantt\\time_plots\\plot_time.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent / "results" / "bench"

ORDER = ["bf16", "w8a16_pc", "w8a16_pt", "w8a8_pc", "w8a8_pt_dyn", "w8a8_pt_static"]
LABEL = {
    "bf16":           "BF16 (Baseline)",
    "w8a16_pc":       "W8A16 Per-Channel",
    "w8a16_pt":       "W8A16 Per-Tensor",
    "w8a8_pc":        "W8A8 Per-Channel",
    "w8a8_pt_dyn":    "W8A8 Per-Tensor Dyn",
    "w8a8_pt_static": "W8A8 Per-Tensor Static",
}
SHORT = {
    "bf16":           "BF16\nBaseline",
    "w8a16_pc":       "W8A16\nPer-Ch",
    "w8a16_pt":       "W8A16\nPer-Ten",
    "w8a8_pc":        "W8A8\nPer-Ch",
    "w8a8_pt_dyn":    "W8A8\nPT Dyn",
    "w8a8_pt_static": "W8A8\nPT Static",
}

# Clean publication / corporate white theme (matches accuracy_plots/)
BG_FIG = "#ffffff"
BG_AX = "#ffffff"
GRID_COL = "#cbd5e1"
TEXT_MAIN = "#0f172a"
TEXT_SUB = "#475569"

COLORS = {
    "bf16":           "#334155",   # charcoal slate (baseline)
    "w8a16_pc":       "#2563eb",   # royal blue
    "w8a16_pt":       "#60a5fa",   # soft sky blue
    "w8a8_pc":        "#0d9488",   # deep teal
    "w8a8_pt_dyn":    "#d97706",   # muted amber
    "w8a8_pt_static": "#dc2626",   # academic crimson
}
MARKERS = {
    "bf16":           "o",
    "w8a16_pc":       "s",
    "w8a16_pt":       "D",
    "w8a8_pc":        "^",
    "w8a8_pt_dyn":    "v",
    "w8a8_pt_static": "P",
}

plt.rcParams.update({
    "figure.facecolor": BG_FIG,
    "axes.facecolor": BG_AX,
    "savefig.facecolor": BG_FIG,
    "savefig.dpi": 220,
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": TEXT_MAIN,
    "axes.labelcolor": TEXT_MAIN,
    "axes.edgecolor": "#94a3b8",
    "axes.linewidth": 0.9,
    "xtick.color": TEXT_MAIN,
    "ytick.color": TEXT_MAIN,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "legend.fontsize": 8.8,
    "axes.grid": True,
    "grid.color": GRID_COL,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_bench():
    data = {}
    for f in ORDER:
        p = BENCH_DIR / f"{f}.json"
        assert p.exists(), f"Missing benchmark file: {p}"
        data[f] = json.loads(p.read_text(encoding="utf-8"))
    return data


def cell(B, f, bs, inp):
    for c in B[f]["cells"]:
        if c["batch"] == bs and c["input_tokens"] == inp:
            return c
    raise KeyError((f, bs, inp))


def save(fig, name):
    fig.tight_layout()
    out = HERE / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)
    print(f"  saved {out}")


def style_legend(ax, **kwargs):
    return ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", framealpha=0.95, labelcolor=TEXT_MAIN, **kwargs)


# ------------------------------------------------------------------ 01: Decode TPOT
def plot_01_decode_tpot(B):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), gridspec_kw={"width_ratios": [1.05, 1.2]})
    xs = np.arange(len(ORDER))

    # Left: Single-user decode latency bars (B=1, in=128)
    tpots = [cell(B, f, 1, 128)["tpot_ms"] for f in ORDER]
    base = tpots[0]
    bars = ax1.bar(xs, tpots, width=0.56, color=[COLORS[f] for f in ORDER], edgecolor="#ffffff", lw=0.8)
    ax1.axhline(base, color="#334155", ls=":", lw=1.2, alpha=0.7)
    for b, v, f in zip(bars, tpots, ORDER):
        txt = f"{v:.1f} ms" if f == "bf16" else f"{v:.1f} ms\n({base / v:.2f}×)"
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, txt, ha="center", va="bottom",
                 fontsize=8.8, fontweight="bold", color=TEXT_MAIN)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([SHORT[f] for f in ORDER], fontsize=8.8)
    ax1.set_ylabel("Decode Latency — TPOT (ms / token)  ↓ better")
    ax1.set_ylim(0, 31)
    ax1.set_title("Single-User Decode Latency (Batch = 1)")

    # Right: TPOT scaling curve across Batch 1 -> 8 -> 32 -> 128 (in=128)
    batches = [1, 8, 32, 128]
    for f in ORDER:
        ys = [cell(B, f, bs, 128)["tpot_ms"] for bs in batches]
        lw = 2.4 if f in ("bf16", "w8a16_pc", "w8a8_pc") else 1.7
        ls = "--" if f in ("w8a16_pt", "w8a8_pt_dyn") else "-"
        ax2.plot(batches, ys, color=COLORS[f], marker=MARKERS[f], ms=7, lw=lw, ls=ls, label=LABEL[f])
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(batches)
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.set_xlabel("Batch Size (concurrent requests)")
    ax2.set_ylabel("Decode Latency — TPOT (ms / token)  ↓ better")
    ax2.set_title("Decode Latency Across Batch Sizes (W8A16 vs W8A8 Crossover)")
    ax2.annotate(
        "W8A8 overtakes W8A16\nat Batch 128",
        xy=(128, cell(B, "w8a8_pc", 128, 128)["tpot_ms"]),
        xytext=(30, 21.5),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.4),
        fontsize=9, fontweight="bold", color="#0f766e",
        bbox=dict(boxstyle="round,pad=0.32", fc="#f0fdfa", ec="#0d9488", lw=0.9),
    )
    style_legend(ax2, loc="upper left")

    fig.suptitle("Per-Token Decode Latency (TPOT) — Qwen2.5-3B on NVIDIA L4", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "01_decode_latency_tpot")


# ------------------------------------------------------------------ 02: Prefill / TTFT
def plot_02_prefill_ttft(B):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2))
    xs = np.arange(len(ORDER))

    # Left: Single-request TTFT (B=1, in=128 vs in=512) in ms
    w = 0.36
    p128 = [cell(B, f, 1, 128)["prefill_s_median"] * 1000 for f in ORDER]
    p512 = [cell(B, f, 1, 512)["prefill_s_median"] * 1000 for f in ORDER]
    b1 = ax1.bar(xs - w / 2, p128, width=w, color="#2563eb", edgecolor="#ffffff", lw=0.7, label="Prompt = 128 tokens")
    b2 = ax1.bar(xs + w / 2, p512, width=w, color="#d97706", edgecolor="#ffffff", lw=0.7, label="Prompt = 512 tokens")
    for b in b1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.0f}", ha="center", va="bottom", fontsize=8, color="#1e3a8a", fontweight="bold")
    for b in b2:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.0f}", ha="center", va="bottom", fontsize=8, color="#92400e", fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([SHORT[f] for f in ORDER], fontsize=8.8)
    ax1.set_ylabel("Prefill / TTFT Latency (ms)  ↓ better")
    ax1.set_ylim(0, 82)
    ax1.set_title("Single-User Time-to-First-Token (Batch = 1)")
    style_legend(ax1, loc="upper right")

    # Right: Heavy Prefill (B=128, in=512) in seconds
    heavy = [cell(B, f, 128, 512)["prefill_s_median"] for f in ORDER]
    base_h = heavy[0]
    bars = ax2.bar(xs, heavy, width=0.56, color=[COLORS[f] for f in ORDER], edgecolor="#ffffff", lw=0.8)
    ax2.axhline(base_h, color="#334155", ls=":", lw=1.2, alpha=0.7)
    for b, v, f in zip(bars, heavy, ORDER):
        sp = base_h / v
        tag = f"{v:.2f}s" if f == "bf16" else f"{v:.2f}s\n({sp:.2f}×)"
        col = "#0f766e" if sp > 1.05 else ("#b91c1c" if sp < 0.95 else TEXT_MAIN)
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.18, tag, ha="center", va="bottom",
                 fontsize=8.8, fontweight="bold", color=col)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([SHORT[f] for f in ORDER], fontsize=8.8)
    ax2.set_ylabel("Batch Prefill Time (seconds)  ↓ better")
    ax2.set_ylim(0, 10.6)
    ax2.set_title("Heavy Batch Prefill (Batch = 128, Input = 512) — W8A8 FP8 Cores Shine")

    fig.suptitle("Prefill & Time-to-First-Token (TTFT) Comparison on NVIDIA L4", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "02_prefill_time_ttft")


# ------------------------------------------------------------------ 03: Stacked End-to-End Time
def plot_03_e2e_breakdown(B):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    scenarios = [
        (1, 512, "Single User\n(B=1, in=512, out=128)"),
        (32, 512, "Medium Batch\n(B=32, in=512, out=128)"),
        (128, 512, "Heavy Batch\n(B=128, in=512, out=128)"),
    ]
    xs = np.arange(len(ORDER))

    for idx, (bs, inp, title) in enumerate(scenarios):
        ax = axes[idx]
        pre = np.array([cell(B, f, bs, inp)["prefill_s_median"] for f in ORDER])
        e2e = np.array([cell(B, f, bs, inp)["e2e_s_median"] for f in ORDER])
        dec = np.maximum(e2e - pre, 0.0)
        ax.bar(xs, pre, width=0.56, color="#d97706", edgecolor="#ffffff", lw=0.7, label="Prefill Time (s)")
        ax.bar(xs, dec, width=0.56, bottom=pre, color="#2563eb", edgecolor="#ffffff", lw=0.7, label="Decode Time (s)")
        for x_i, tot in zip(xs, e2e):
            ax.text(x_i, tot + max(e2e) * 0.02, f"{tot:.2f}s", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=TEXT_MAIN)
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[f] for f in ORDER], fontsize=8.0)
        ax.set_ylim(0, max(e2e) * 1.18)
        ax.set_title(title)
        if idx == 0:
            ax.set_ylabel("Total End-to-End Time (seconds)  ↓ better")
            style_legend(ax, loc="upper right")

    fig.suptitle("End-to-End Request Latency Breakdown: Prefill vs Decode Time", fontsize=14, fontweight="bold", y=1.03)
    save(fig, "03_end_to_end_latency")


# ------------------------------------------------------------------ 04: Throughput Scaling
def plot_04_throughput(B):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2))
    batches = [1, 8, 32, 128]

    for ax, inp, title in (
        (ax1, 128, "Output Throughput — Short Prompts (Input = 128)"),
        (ax2, 512, "Output Throughput — Long Prompts (Input = 512)"),
    ):
        for f in ORDER:
            ys = [cell(B, f, bs, inp)["output_tok_per_s"] for bs in batches]
            lw = 2.4 if f in ("bf16", "w8a16_pc", "w8a8_pc", "w8a8_pt_static") else 1.7
            ls = "--" if f in ("w8a16_pt", "w8a8_pt_dyn") else "-"
            ax.plot(batches, ys, color=COLORS[f], marker=MARKERS[f], ms=7, lw=lw, ls=ls, label=LABEL[f])
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Output Throughput (tokens / sec)  ↑ better")
        ax.set_title(title)
        top_f = "w8a8_pt_static"
        peak = cell(B, top_f, 128, inp)["output_tok_per_s"]
        base = cell(B, "bf16", 128, inp)["output_tok_per_s"]
        ax.annotate(
            f"W8A8 Peak: {peak:.0f} tok/s\n(+{(peak / base - 1) * 100:.0f}% vs BF16)",
            xy=(128, peak),
            xytext=(14, peak * 0.84),
            arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.4),
            fontsize=9, fontweight="bold", color="#0f766e",
            bbox=dict(boxstyle="round,pad=0.32", fc="#f0fdfa", ec="#0d9488", lw=0.9),
        )
        style_legend(ax, loc="upper left")

    fig.suptitle("Serving Throughput Scaling Across Batch Sizes (1 → 128)", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "04_throughput_scaling")


# ------------------------------------------------------------------ 05: Speedup Heatmap
def plot_05_speedup_heatmap(B):
    scenarios = [
        (1, 128, "B=1\nin=128"),
        (1, 512, "B=1\nin=512"),
        (8, 128, "B=8\nin=128"),
        (8, 512, "B=8\nin=512"),
        (32, 128, "B=32\nin=128"),
        (32, 512, "B=32\nin=512"),
        (128, 128, "B=128\nin=128"),
        (128, 512, "B=128\nin=512"),
    ]
    q_fmts = [f for f in ORDER if f != "bf16"]
    mat = np.zeros((len(q_fmts), len(scenarios)))
    for r, f in enumerate(q_fmts):
        for c_i, (bs, inp, _) in enumerate(scenarios):
            base_e2e = cell(B, "bf16", bs, inp)["e2e_s_median"]
            q_e2e = cell(B, f, bs, inp)["e2e_s_median"]
            mat[r, c_i] = base_e2e / q_e2e

    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0.95, vmax=1.82)
    ax.grid(False)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s[2] for s in scenarios], fontsize=9.5, fontweight="bold")
    ax.set_yticks(range(len(q_fmts)))
    ax.set_yticklabels([LABEL[f] for f in q_fmts], fontsize=10, fontweight="bold")

    for r in range(len(q_fmts)):
        for c_i in range(len(scenarios)):
            v = mat[r, c_i]
            txt_col = "#0f172a" if v < 1.48 else "#ffffff"
            ax.text(c_i, r, f"{v:.2f}×", ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=txt_col)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("End-to-End Speedup vs BF16 (higher = faster)", color=TEXT_MAIN)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MAIN)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_MAIN)
    ax.set_title("End-to-End Speedup Cheat Sheet vs BF16 (All 8 Batch × Input Workloads on L4)", pad=12)
    save(fig, "05_speedup_heatmap")


def main():
    B = load_bench()
    print(f"Generating sleek time & latency plots -> {HERE}")
    plot_01_decode_tpot(B)
    plot_02_prefill_ttft(B)
    plot_03_e2e_breakdown(B)
    plot_04_throughput(B)
    plot_05_speedup_heatmap(B)
    print("Done — 5 time/latency plots saved.")


if __name__ == "__main__":
    main()
