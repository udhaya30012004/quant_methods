"""Unified, presentation-ready Quality, Accuracy & Size plots -> E:\\quantt\\accuracy_plots\\.

Self-contained script (dark executive theme matching time_plots/) reading E:\\quantt\\results\\.
Generates 9 cleanly numbered, non-duplicate plots ready to present:
  01_ZeroShot_Benchmarks_Overview.png
  02_Accuracy_BF16_vs_W8A16.png
  03_Accuracy_BF16_vs_W8A8_Dynamic.png
  04_Accuracy_BF16_vs_W8A8_Static.png
  05_Accuracy_All_Methods_Comparison.png
  06_KL_Divergence_and_Token_Agreement.png
  07_Loss_Curve_and_Perplexity.png
  08_Head_to_Head_W8A16_vs_W8A8_and_PT_vs_PC.png
  09_Model_Size_and_Compression.png

Run locally:
    python E:\\quantt\\accuracy_plots\\plot_accuracy.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]
PRIMARY = {
    "hellaswag": "acc_norm",
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
    "piqa": "acc",
}
TASK_NAME = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "winogrande": "WinoGrande",
    "piqa": "PIQA",
}
TASK_DESC = {
    "hellaswag":     "Commonsense sentence completion (10,042 Qs)",
    "arc_easy":      "Grade-school science MCQs (2,376 Qs)",
    "arc_challenge": "Hard reasoning science MCQs (1,172 Qs)",
    "winogrande":    "Pronoun & context reasoning (1,267 Qs)",
    "piqa":          "Physical intuition QA (1,838 Qs)",
}

ALL_ORDER = ["bf16", "w8a16_pc", "w8a16_pt", "w8a8_pc", "w8a8_pt_dyn", "w8a8_pt_static"]
Q_ORDER = ["w8a16_pc", "w8a16_pt", "w8a8_pc", "w8a8_pt_dyn", "w8a8_pt_static"]
SIM_KEY = {"w8a8_pt_dyn": "w8a8_pt"}

LABEL = {
    "bf16":           "BF16 (Baseline)",
    "w8a16_pc":       "W8A16 Per-Channel",
    "w8a16_pt":       "W8A16 Per-Tensor",
    "w8a8_pc":        "W8A8 Per-Channel (Dyn)",
    "w8a8_pt_dyn":    "W8A8 Per-Tensor (Dyn)",
    "w8a8_pt_static": "W8A8 Per-Tensor (Static)",
}
SHORT = {
    "bf16":           "BF16\nBaseline",
    "w8a16_pc":       "W8A16\nPer-Ch",
    "w8a16_pt":       "W8A16\nPer-Ten",
    "w8a8_pc":        "W8A8\nPer-Ch",
    "w8a8_pt_dyn":    "W8A8\nPT Dyn",
    "w8a8_pt_static": "W8A8\nPT Static",
}

# Clean publication / corporate white theme (NeurIPS / Nature / Research Report style)
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
    "axes.titlesize": 12.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "legend.fontsize": 9.0,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID_COL,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _j(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_all():
    lm, ppl, sim, size = {}, {}, {}, {}
    for f in ALL_ORDER:
        j = _j(RES / "lm_eval" / f"{f}.json")
        sc = {t: j["tasks"][t][f"{PRIMARY[t]},none"] * 100.0 for t in TASKS}
        sc["avg"] = float(np.mean(list(sc.values())))
        lm[f] = sc
        ppl[f] = {
            "wiki": _j(RES / "trackb_ppl" / f"{f}__ppl_wikitext.json"),
            "chat": _j(RES / "trackb_ppl" / f"{f}__kl_chat.json"),
        }
        if f != "bf16":
            size[f] = _j(RES / f"size_{f}.json")
            sk = SIM_KEY.get(f, f)
            sim[f] = {
                "wiki": _j(RES / "sim" / f"{sk}__ppl_wikitext.json"),
                "chat": _j(RES / "sim" / f"{sk}__kl_chat.json"),
            }
    size["bf16"] = {"quant_bytes": size["w8a16_pt"]["bf16_bytes"], "compression_pct": 0.0}
    sim["fp16cast"] = {"wiki": _j(RES / "sim" / "fp16cast__ppl_wikitext.json")}
    return lm, ppl, sim, size


def save(fig, name):
    fig.tight_layout()
    out = HERE / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)
    print(f"  saved {out.name}")


def legend(ax, **kw):
    return ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", framealpha=0.95, labelcolor=TEXT_MAIN, **kw)


# ------------------------------------------------------------------ 01
def plot_01_zeroshot_overview(lm):
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    y = np.arange(len(TASKS))[::-1]
    bf16_vals = [lm["bf16"][t] for t in TASKS]
    best_fp8 = [max(lm[f][t] for f in Q_ORDER) for t in TASKS]
    worst_fp8 = [min(lm[f][t] for f in Q_ORDER) for t in TASKS]

    ax.barh(y, bf16_vals, height=0.50, color="#2563eb", alpha=0.88,
            edgecolor="#1e3a8a", lw=0.7, label="BF16 Baseline (0-Shot)")
    for yi, lo, hi in zip(y, worst_fp8, best_fp8):
        ax.plot([lo, hi], [yi, yi], color="#d97706", lw=4.2, solid_capstyle="round",
                label="FP8 Range (all 5 methods)" if yi == y[0] else None)
        ax.scatter([lo, hi], [yi, yi], color="#b45309", s=42, zorder=5)

    for yi, t, bv, lo, hi in zip(y, TASKS, bf16_vals, worst_fp8, best_fp8):
        ax.text(max(bv, hi) + 1.2, yi + 0.11, f"BF16: {bv:.1f}%  |  FP8: {lo:.1f}% – {hi:.1f}%",
                va="center", ha="left", fontsize=9.2, fontweight="bold", color=TEXT_MAIN)
        ax.text(max(bv, hi) + 1.2, yi - 0.18, TASK_DESC[t],
                va="center", ha="left", fontsize=8.5, color=TEXT_SUB)

    ax.set_yticks(y)
    ax.set_yticklabels([TASK_NAME[t] for t in TASKS], fontsize=10.8, fontweight="bold")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Zero-Shot Accuracy (%)  —  No Examples Given in Prompt (0-Shot)")
    ax.set_title("Zero-Shot Benchmark Suite (16,695 Total Questions on Qwen2.5-3B)", pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.55)
    ax.grid(axis="y", visible=False)
    legend(ax, loc="lower right")
    save(fig, "01_ZeroShot_Benchmarks_Overview")


# ------------------------------------------------------------------ 02, 03, 04
def grouped_3bar(lm, fmts, title, subtitle, filename, note=None):
    cols = TASKS + ["avg"]
    xlabels = [TASK_NAME[t] for t in TASKS] + ["5-Task\nAverage"]
    x = np.arange(len(cols))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    for i, f in enumerate(fmts):
        vals = [lm[f][c] for c in cols]
        offs = x + (i - (len(fmts) - 1) / 2) * w
        bars = ax.bar(offs, vals, width=w * 0.90, color=COLORS[f], edgecolor="#ffffff", lw=0.8, label=LABEL[f])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=TEXT_MAIN)

    ax.axvline(len(TASKS) - 0.5, color="#94a3b8", ls="--", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("0-Shot Accuracy (%)  ↑ higher is better")
    ax.set_ylim(38, 86)
    ax.set_title(title, pad=18)
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=9.8, color=TEXT_SUB)
    if note:
        ax.text(0.015, 0.04, note, transform=ax.transAxes, fontsize=9, color="#991b1b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec="#dc2626", lw=0.9))
    legend(ax, loc="upper right", ncol=len(fmts))
    save(fig, filename)


# ------------------------------------------------------------------ 05
def plot_05_all_methods(lm):
    cols = TASKS + ["avg"]
    titles = [TASK_NAME[t] for t in TASKS] + ["5-Task Average"]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2))
    y = np.arange(len(ALL_ORDER))[::-1]

    for idx, (col, ttl) in enumerate(zip(cols, titles)):
        ax = axes.flatten()[idx]
        vals = [lm[f][col] for f in ALL_ORDER]
        base = lm["bf16"][col]
        bars = ax.barh(y, vals, height=0.60, color=[COLORS[f] for f in ALL_ORDER], edgecolor="#ffffff", lw=0.7)
        ax.axvline(base, color="#334155", ls="--", lw=1.1, alpha=0.75)
        for b, v, f in zip(bars, vals, ALL_ORDER):
            txt = f" {v:.2f}%" if f == "bf16" else f" {v:.2f}% ({v - base:+.2f})"
            ax.text(v, b.get_y() + b.get_height() / 2, txt,
                    va="center", ha="left", fontsize=8.4, fontweight="bold", color=TEXT_MAIN)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[f] for f in ALL_ORDER] if idx % 3 == 0 else [""] * len(ALL_ORDER), fontsize=8.8)
        ax.set_xlim(min(vals) - 1.2, max(vals) + 2.8)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(ttl)
        ax.grid(axis="x", linestyle="--", alpha=0.55)
        ax.grid(axis="y", visible=False)

    fig.suptitle("All 6 Quantization Methods — Zoomed 0-Shot Accuracy Per Benchmark (BF16 = Dashed Line)",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "05_Accuracy_All_Methods_Comparison")


# ------------------------------------------------------------------ 06a & 06b
def plot_06_kl(sim):
    order4 = ["w8a16_pc", "w8a16_pt", "w8a8_pc", "w8a8_pt_dyn"]
    x = np.arange(len(order4))
    w = 0.35

    # 06a: ONLY Mean KL Divergence (clean bars, no % increase, no Static)
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    kl_w = [sim[f]["wiki"]["kl_mean"] for f in order4]
    kl_c = [sim[f]["chat"]["kl_mean"] for f in order4]
    b1 = ax.bar(x - w / 2, kl_w, width=w, color="#2563eb", edgecolor="#ffffff", lw=0.8, label="WikiText-2")
    b2 = ax.bar(x + w / 2, kl_c, width=w, color="#0d9488", edgecolor="#ffffff", lw=0.8, label="UltraChat")
    fp16 = sim["fp16cast"]["wiki"]["kl_mean"]
    ax.axhline(fp16, color="#64748b", ls="--", lw=1.3, label=f"FP16 Cast Noise Floor ({fp16:.5f})")

    for b in b1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.00025, f"{h:.4f}", ha="center", va="bottom",
                fontsize=9.2, fontweight="bold", color="#1e3a8a")
    for b in b2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.00025, f"{h:.4f}", ha="center", va="bottom",
                fontsize=9.2, fontweight="bold", color="#115e59")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[f] for f in order4], fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean KL(P_BF16 || P_quant)  ↓ lower is better")
    ax.set_ylim(0, 0.0135)
    ax.set_title("Mean KL Divergence vs BF16 Baseline (Qwen2.5-3B)", pad=12)
    legend(ax, loc="upper left")
    save(fig, "06a_Mean_KL_Divergence")

    # 06b: ONLY Top-1 Token Agreement (%) — crisp line plot
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    t1_w = [sim[f]["wiki"]["top1_agree"] * 100.0 for f in order4]
    t1_c = [sim[f]["chat"]["top1_agree"] * 100.0 for f in order4]
    fp16_t1 = sim["fp16cast"]["wiki"]["top1_agree"] * 100.0

    ax.plot(x, t1_c, color="#0d9488", marker="s", ms=8, lw=2.5, label="UltraChat")
    ax.plot(x, t1_w, color="#2563eb", marker="o", ms=8, lw=2.5, label="WikiText-2")
    ax.axhline(fp16_t1, color="#64748b", ls="--", lw=1.3, label=f"FP16 Cast Ceiling ({fp16_t1:.1f}%)")

    for xi, (vc, vw) in enumerate(zip(t1_c, t1_w)):
        ax.annotate(f"{vc:.1f}%", (xi, vc), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9.5, fontweight="bold", color="#115e59")
        ax.annotate(f"{vw:.1f}%", (xi, vw), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=9.5, fontweight="bold", color="#1e3a8a")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[f] for f in order4], fontsize=10, fontweight="bold")
    ax.set_xlim(-0.35, len(order4) - 0.65)
    ax.set_ylabel("Top-1 Token Agreement vs BF16 (%)  ↑ higher is better")
    ax.set_ylim(93.5, 98.8)
    ax.set_title("Top-1 Token Agreement vs BF16 Baseline (Qwen2.5-3B)", pad=12)
    legend(ax, loc="lower left")
    save(fig, "06b_Token_Agreement")


# ------------------------------------------------------------------ 07
def plot_07_loss_ppl(ppl):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
    xs = np.arange(len(ALL_ORDER))

    ppls = [ppl[f]["wiki"]["ppl"] for f in ALL_ORDER]
    base_ppl = ppls[0]
    bars = ax1.bar(xs, ppls, width=0.56, color=[COLORS[f] for f in ALL_ORDER], edgecolor="#ffffff", lw=0.8)
    ax1.axhline(base_ppl, color="#334155", ls="--", lw=1.1, alpha=0.75)
    for b, p, f in zip(bars, ppls, ALL_ORDER):
        txt = f"{p:.3f}" if f == "bf16" else f"{p:.3f}\n(+{p - base_ppl:.3f})"
        ax1.text(b.get_x() + b.get_width() / 2, p + 0.004, txt, ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=TEXT_MAIN)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([SHORT[f] for f in ALL_ORDER], fontsize=8.6)
    ax1.set_ylabel("WikiText-2 Perplexity  ↓ lower is better")
    ax1.set_ylim(7.72, 7.92)
    ax1.set_title("Perplexity Comparison (Real vLLM, 64 Windows)")

    base_w = np.asarray(ppl["bf16"]["wiki"]["window_losses"])
    order_idx = np.argsort(base_w)
    win_x = np.arange(1, len(base_w) + 1)
    for f in Q_ORDER:
        qw = np.asarray(ppl[f]["wiki"]["window_losses"])
        delta = (qw - base_w)[order_idx]
        smooth = np.convolve(delta, np.ones(5) / 5, mode="valid")
        ax2.plot(win_x, delta, color=COLORS[f], lw=0.7, alpha=0.20)
        ax2.plot(win_x[2:-2], smooth, color=COLORS[f], lw=2.2, label=f"{LABEL[f]} (mean {delta.mean():+.4f})")
    ax2.axhline(0.0, color="#334155", ls="--", lw=1.1, label="BF16 Baseline (ΔLoss = 0)")
    ax2.set_xlabel("Evaluation Window (sorted easiest → hardest by BF16 loss)")
    ax2.set_ylabel("Δ Cross-Entropy Loss vs BF16  ↓ lower is better")
    ax2.set_title("Loss Degradation Curve Across All 64 WikiText-2 Windows")
    legend(ax2, loc="upper left", fontsize=8.2)

    fig.suptitle("Perplexity & Cross-Entropy Loss Comparison (BF16 vs FP8 PTQ)", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "07_Loss_Curve_and_Perplexity")


# ------------------------------------------------------------------ 08
def plot_08_head_to_head(ppl, sim):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0))
    x = np.arange(2)

    # Panel A: W8A16 vs W8A8
    ax = axes[0]
    w16 = [sim["w8a16_pc"]["wiki"]["kl_mean"], sim["w8a16_pt"]["wiki"]["kl_mean"]]
    w8 = [sim["w8a8_pc"]["wiki"]["kl_mean"], sim["w8a8_pt_dyn"]["wiki"]["kl_mean"]]
    b1 = ax.bar(x - 0.18, w16, width=0.34, color="#2563eb", edgecolor="#ffffff", lw=0.8, label="W8A16 (BF16 Acts)")
    b2 = ax.bar(x + 0.18, w8, width=0.34, color="#d97706", edgecolor="#ffffff", lw=0.8, label="W8A8 (FP8 Acts)")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.0003, f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Per-Channel", "Per-Tensor"], fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean KL Divergence  ↓ better")
    ax.set_ylim(0, 0.0145)
    ax.set_title("W8A16 vs W8A8\n(FP8 Activations Double KL)")
    legend(ax, loc="upper left")

    # Panel B: Per-Tensor vs Per-Channel
    ax = axes[1]
    base_p = ppl["bf16"]["wiki"]["ppl"]
    pt = [ppl["w8a16_pt"]["wiki"]["ppl"] - base_p, ppl["w8a8_pt_dyn"]["wiki"]["ppl"] - base_p]
    pc = [ppl["w8a16_pc"]["wiki"]["ppl"] - base_p, ppl["w8a8_pc"]["wiki"]["ppl"] - base_p]
    b1 = ax.bar(x - 0.18, pt, width=0.34, color="#d97706", edgecolor="#ffffff", lw=0.8, label="Per-Tensor")
    b2 = ax.bar(x + 0.18, pc, width=0.34, color="#0d9488", edgecolor="#ffffff", lw=0.8, label="Per-Channel / Token")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.002, f"+{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["W8A16", "W8A8 Dynamic"], fontsize=10, fontweight="bold")
    ax.set_ylabel("WikiText ΔPPL vs BF16  ↓ better")
    ax.set_ylim(0, 0.095)
    ax.set_title("Per-Tensor vs Per-Channel\n(Per-Channel Wins in W8A8)")
    legend(ax, loc="upper left")

    # Panel C: Dynamic vs Static
    ax = axes[2]
    kls = [sim["w8a8_pc"]["wiki"]["kl_mean"], sim["w8a8_pt_dyn"]["wiki"]["kl_mean"], sim["w8a8_pt_static"]["wiki"]["kl_mean"]]
    cols = [COLORS["w8a8_pc"], COLORS["w8a8_pt_dyn"], COLORS["w8a8_pt_static"]]
    bars = ax.bar(np.arange(3), kls, width=0.52, color=cols, edgecolor="#ffffff", lw=0.8)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.0004, f"{h:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["W8A8\nPer-Ch Dyn", "W8A8\nPT Dyn", "W8A8\nPT Static"], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Mean KL Divergence  ↓ better")
    ax.set_ylim(0, 0.0185)
    ax.set_title("Dynamic vs Static Activations\n(Static Adds +45% KL Error)")

    fig.suptitle("Head-to-Head Quantization Design Choices (Qwen2.5-3B)", fontsize=14, fontweight="bold", y=1.03)
    save(fig, "08_Head_to_Head_W8A16_vs_W8A8_and_PT_vs_PC")


# ------------------------------------------------------------------ 09
def plot_09_model_size(size):
    order5 = ["bf16", "w8a16_pc", "w8a16_pt", "w8a8_pc", "w8a8_pt_dyn"]
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ax.set_facecolor("#f8fafc")
    xs = np.arange(len(order5))
    sizes_gb = [size[f]["quant_bytes"] / 1e9 for f in order5]
    comps = [size[f]["compression_pct"] for f in order5]
    bar_cols = ["#1e293b", "#2563eb", "#60a5fa", "#0d9488", "#d97706"]

    bars = ax.bar(xs, sizes_gb, width=0.52, color=bar_cols, edgecolor="#ffffff", lw=1.2, zorder=3)
    ax.axhline(sizes_gb[0], color="#1e293b", ls="--", lw=1.2, alpha=0.55, zorder=2)

    for b, gb, cp, f in zip(bars, sizes_gb, comps, order5):
        cx = b.get_x() + b.get_width() / 2
        if f == "bf16":
            ax.text(cx, gb + 0.38, f"{gb:.2f} GB", ha="center", va="bottom",
                    fontsize=11.5, fontweight="bold", color="#1e293b", zorder=5)
            ax.text(cx, gb + 0.12, "(Baseline)", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#475569", zorder=5)
        else:
            ax.text(cx, gb + 0.45, f"{gb:.2f} GB", ha="center", va="bottom",
                    fontsize=11.2, fontweight="bold", color="#0f172a", zorder=5)
            ax.text(cx, gb + 0.14, f"-{cp:.1f}%", ha="center", va="bottom",
                    fontsize=10.2, fontweight="bold", color="#0f766e", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.20", fc="#ccfbf1", ec="#5eead4", lw=0.8))

    # Clean savings callout on the right side
    ax.annotate(
        "−2.77 GB saved (~2.6 GiB VRAM freed)",
        xy=(2.5, sizes_gb[1] + 0.95),
        xytext=(2.5, 5.55),
        ha="center",
        fontsize=9.8,
        fontweight="bold",
        color="#0f766e",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#5eead4", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#0d9488", lw=1.3),
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[f] for f in order5], fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Checkpoint Size on Disk (GB)  ↓ smaller is better", fontweight="bold")
    ax.set_ylim(0, 7.3)
    ax.set_title("Model Size & Compression: BF16 vs FP8 PTQ (Qwen2.5-3B)", pad=14)
    save(fig, "09_Model_Size_and_Compression")


def main():
    lm, ppl, sim, size = load_all()
    print(f"Generating unified dark-theme deck -> {HERE}")
    plot_01_zeroshot_overview(lm)
    grouped_3bar(
        lm, ["bf16", "w8a16_pt", "w8a16_pc"],
        title="BF16 vs W8A16 (Weight-Only FP8: Per-Tensor & Per-Channel)",
        subtitle="Activations stay in BF16 — Both W8A16 methods match or exceed BF16 (+0.20% avg, zero significant drops)",
        filename="02_Accuracy_BF16_vs_W8A16",
    )
    grouped_3bar(
        lm, ["bf16", "w8a8_pc", "w8a8_pt_dyn"],
        title="BF16 vs W8A8 Dynamic (FP8 Weights + Dynamic FP8 Activations)",
        subtitle="Dynamic activation scaling preserves average accuracy (W8A8 Per-Channel has zero significant drops)",
        filename="03_Accuracy_BF16_vs_W8A8_Dynamic",
    )
    grouped_3bar(
        lm, ["bf16", "w8a8_pt_dyn", "w8a8_pt_static"],
        title="BF16 vs W8A8 Static (Calibrated Fixed Activation Scales)",
        subtitle="Static per-tensor scales suffer the largest HellaSwag drop (-0.43%, p < 0.01) due to activation outliers",
        filename="04_Accuracy_BF16_vs_W8A8_Static",
        note="Note: W8A8 Static is the ONLY method with a p < 0.01 drop on HellaSwag (73.56% -> 73.13%).",
    )
    plot_05_all_methods(lm)
    plot_06_kl(sim)
    plot_07_loss_ppl(ppl)
    plot_08_head_to_head(ppl, sim)
    plot_09_model_size(size)
    print("Done — 9 clean, presentation-ready plots saved.")


if __name__ == "__main__":
    main()
