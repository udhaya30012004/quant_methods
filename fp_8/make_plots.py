"""Step 8 (R): generate all 13 publication-quality plots (PNG + SVG) in results/plots/.

Run locally:
    python fp_8/make_plots.py
    python fp_8/make_plots.py --dry   # only plots 6-8 if running early
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stats import (
    ALL_FMTS, A_NAME, FMTS, PRIMARY, RES, SETS, SET_LABEL, STYLE, TASKS,
    bootstrap_mean_ci, kl_windows, lm_correct, lm_metric, load_results,
    mcnemar_exact, trackb_delta,
)

OUT = RES / "plots"
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .svg")


# ------------------------------------------------------------------ 01
def plot_01_weight_hist(R):
    smp = R["diag"].get("samples")
    if not smp:
        return
    layers = [0, 18, 35]
    projs = ["q_proj", "down_proj"]
    fig, axes = plt.subplots(len(projs), len(layers), figsize=(11.5, 5.8), sharey=False)
    for r_i, p in enumerate(projs):
        for c_i, li in enumerate(layers):
            ax = axes[r_i, c_i]
            k = f"L{li}.{p}"
            w = smp[f"{k}.w"]
            q_pt = smp[f"{k}.pt_q"]
            err_pt = np.abs(w - q_pt)
            lim = float(np.quantile(np.abs(w), 0.995))
            bins = np.linspace(-lim, lim, 90)
            ax.hist(w, bins=bins, density=True, alpha=0.55, color="#4c72b0", label="BF16 weights")
            ax.hist(q_pt, bins=bins, density=True, histtype="step", lw=1.1, color="#d62728", label="FP8 E4M3 (PT)")
            ax.set_title(f"Layer {li} — {p} (MAE={err_pt.mean():.2e})")
            if r_i == len(projs) - 1:
                ax.set_xlabel("Weight value")
            if c_i == 0:
                ax.set_ylabel("Density")
            if r_i == 0 and c_i == 0:
                ax.legend(frameon=False)
    fig.suptitle("01. Weight Distributions & FP8 E4M3 Quantization Overlay (Selected Layers)", y=1.02)
    save(fig, "01_weight_hist_fp8grid")


# ------------------------------------------------------------------ 02
def plot_02_channel_amax(R):
    w = R["diag"].get("weights")
    if not w:
        return
    rows = sorted(w["rows"], key=lambda r: -r["pc_amax_ratio_max_over_median"])
    ratios = [r["pc_amax_ratio_max_over_median"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax1.plot(range(1, len(ratios) + 1), ratios, color="#1f77b4", lw=1.6)
    ax1.set_yscale("log")
    ax1.set_xlabel("Linear layer rank (sorted by max/median channel amax)")
    ax1.set_ylabel("Per-channel amax spread (max / median)")
    ax1.set_title("Sorted Per-Channel Weight Spread (252 Linears)")
    for idx in range(3):
        r = rows[idx]
        ax1.annotate(
            f"L{r['layer']} {r['proj']} ({r['pc_amax_ratio_max_over_median']:.1f}×)",
            xy=(idx + 1, r["pc_amax_ratio_max_over_median"]),
            xytext=(idx + 22, r["pc_amax_ratio_max_over_median"] * (0.88 - 0.14 * idx)),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"), fontsize=8,
        )
    # right panel: top 12 bars
    top = rows[:12][::-1]
    labels = [f"L{r['layer']} {r['proj']}" for r in top]
    vals = [r["pc_amax_ratio_max_over_median"] for r in top]
    ax2.barh(labels, vals, color="#6baed6", edgecolor="#1f77b4")
    ax2.set_xlabel("Per-channel amax max / median")
    ax2.set_title("Top 12 Layers by Per-Channel Weight Spread")
    fig.suptitle("02. Weight Channel-Scale Spread Across All 252 Linear Layers", y=1.02)
    save(fig, "02_channel_amax_sorted")


# ------------------------------------------------------------------ 03
def plot_03_weight_sqnr(R):
    w = R["diag"].get("weights")
    if not w:
        return
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 5.8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for i, p in enumerate(PROJS):
        ax = axes_flat[i]
        sub = sorted([r for r in w["rows"] if r["proj"] == p], key=lambda r: r["layer"])
        xs = [r["layer"] for r in sub]
        ax.plot(xs, [r["sqnr_db_pt"] for r in sub], label="Per-tensor", color="#1f77b4", lw=1.4)
        ax.plot(xs, [r["sqnr_db_pc"] for r in sub], label="Per-channel", color="#d62728", ls="--", lw=1.4)
        ax.set_title(p)
        if i >= 4:
            ax.set_xlabel("Layer")
        if i in (0, 4):
            ax.set_ylabel("Weight SQNR (dB)")
    # 8th panel: flush-to-zero comparison across projections
    ax = axes_flat[7]
    pt_fz = [np.mean([r["flush_zero_frac_pt"] for r in w["rows"] if r["proj"] == p]) * 1e4 for p in PROJS]
    pc_fz = [np.mean([r["flush_zero_frac_pc"] for r in w["rows"] if r["proj"] == p]) * 1e4 for p in PROJS]
    x = np.arange(len(PROJS))
    ax.bar(x - 0.18, pt_fz, width=0.36, color="#1f77b4", label="Per-tensor")
    ax.bar(x + 0.18, pc_fz, width=0.36, color="#d62728", label="Per-channel")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_proj", "") for p in PROJS], rotation=35, fontsize=8)
    ax.set_ylabel("Flush-to-zero (×10⁻⁴)")
    ax.set_title("Mean Flush-to-Zero Rate")
    axes_flat[0].legend(frameon=False, loc="lower left")
    fig.suptitle("03. Weight Quantization SQNR (dB) by Layer & Projection (PT vs PC)", y=1.01)
    save(fig, "03_weight_sqnr_by_layer")


# ------------------------------------------------------------------ 04
def plot_04_act_outliers(R):
    a = R["diag"].get("acts")
    if not a:
        return
    rows = a["rows"]
    n_layers = max(r["layer"] for r in rows) + 1
    mat = np.zeros((n_layers, len(PROJS)))
    for r in rows:
        mat[r["layer"], PROJS.index(r["proj"])] = np.log10(max(r["outlier_factor"], 1.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    # left: per-tensor vs per-token activation SQNR averaged per layer + down_proj separately
    layers = list(range(n_layers))
    dp = {r["layer"]: r for r in rows if r["proj"] == "down_proj"}
    ax1.plot(layers, [dp[i]["act_sqnr_db_per_tensor"] for i in layers],
             color="#ff7f0e", lw=1.6, label="down_proj — per-tensor FP8")
    ax1.plot(layers, [dp[i]["act_sqnr_db_per_token"] for i in layers],
             color="#d62728", ls="--", lw=1.6, label="down_proj — per-token FP8")
    other_pt = [np.mean([r["act_sqnr_db_per_tensor"] for r in rows if r["layer"] == i and r["proj"] != "down_proj"]) for i in layers]
    other_tk = [np.mean([r["act_sqnr_db_per_token"] for r in rows if r["layer"] == i and r["proj"] != "down_proj"]) for i in layers]
    ax1.plot(layers, other_pt, color="#1f77b4", lw=1.2, alpha=0.8, label="Other 6 projs — per-tensor")
    ax1.plot(layers, other_tk, color="#6baed6", ls="--", lw=1.2, alpha=0.9, label="Other 6 projs — per-token")
    ax1.set_xlabel("Decoder layer")
    ax1.set_ylabel("Activation SQNR (dB)")
    ax1.set_title("Activation Quantization SQNR (Per-Tensor vs Per-Token)")
    ax1.legend(frameon=False)

    # right: log10 outlier factor heatmap
    im = ax2.imshow(mat, aspect="auto", cmap="inferno", origin="lower")
    ax2.set_xticks(range(len(PROJS)))
    ax2.set_xticklabels([p.replace("_proj", "") for p in PROJS], rotation=30)
    ax2.set_ylabel("Decoder layer")
    ax2.set_title("Activation Outlier Factor: log₁₀(max / median)")
    cbar = fig.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label("log₁₀(max / median)")
    ax2.grid(False)
    fig.suptitle("04. Activation Outliers & Quantization SQNR Across Depth", y=1.01)
    save(fig, "04_act_amax_by_layer")


# ------------------------------------------------------------------ 05
def plot_05_layer_error(R):
    le = R["diag"].get("layer_err")
    if not le:
        return
    acc = le["accumulated"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for f in FMTS:
        rk = A_NAME.get(f, f)
        if rk not in acc:
            continue
        st = STYLE[f]
        xs = np.arange(len(acc[rk]["cos"]))
        ax1.plot(xs, acc[rk]["cos"], color=st["color"], ls=st["ls"], marker=st["marker"],
                 ms=3.5, markevery=3, lw=1.5, label=st["label"])
        ax2.plot(xs, acc[rk]["rel_err"], color=st["color"], ls=st["ls"], marker=st["marker"],
                 ms=3.5, markevery=3, lw=1.5, label=st["label"])
    ax1.set_xlabel("Decoder layer")
    ax1.set_ylabel("Residual-stream cosine similarity vs BF16")
    ax1.set_title("Accumulated Residual Cosine Similarity")
    ax1.legend(frameon=False, loc="lower left")
    ax2.set_xlabel("Decoder layer")
    ax2.set_ylabel("Relative L2 error ||h_q - h_b|| / ||h_b||")
    ax2.set_title("Accumulated Residual Relative Error")
    fig.suptitle("05. Quantization Error Propagation Across 36 Transformer Layers", y=1.02)
    save(fig, "05_layer_output_error_depth")


# ------------------------------------------------------------------ 06
def plot_06_loss_ppl_bars(R):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    x = np.arange(len(FMTS))
    w = 0.36
    for idx, s in enumerate(SETS):
        ax = axes[idx]
        da, db, err_lo, err_hi = [], [], [], []
        for f in FMTS:
            rk = A_NAME.get(f, f)
            st_a = R["sim"][rk][s]["stats"]
            da.append(st_a["ppl_quant"] - st_a["ppl_base"])
            d = trackb_delta(R, f, s)
            db.append(d["dppl"])
            err_lo.append(max(d["dppl"] - d["dppl_lo"], 0.0))
            err_hi.append(max(d["dppl_hi"] - d["dppl"], 0.0))
        colors = [STYLE[f]["color"] for f in FMTS]
        hatches = [STYLE[f]["hatch"] for f in FMTS]
        b1 = ax.bar(x - w / 2, da, width=w, color=colors, alpha=0.55, edgecolor="black", lw=0.7, label="Track A (Sim)")
        b2 = ax.bar(x + w / 2, db, width=w, color=colors, alpha=0.95, edgecolor="black", lw=0.7, label="Track B (vLLM)")
        for bar, h in zip(b1, hatches):
            if h:
                bar.set_hatch(h)
        for bar, h in zip(b2, hatches):
            if h:
                bar.set_hatch(h)
        ax.errorbar(x + w / 2, db, yerr=[err_lo, err_hi], fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[f]["label"] for f in FMTS], rotation=22, ha="right", fontsize=8.5)
        ax.set_ylabel("ΔPPL vs BF16 (lower is better)")
        ax.set_title(f"{SET_LABEL[s]} (BF16 PPL = {R['trackb']['bf16'][s]['stats']['ppl']:.3f})")
        if idx == 0:
            ax.legend(frameon=False)
    fig.suptitle("06. Perplexity Increase (ΔPPL) — Track A Simulation vs Track B Real vLLM (95% CI)", y=1.02)
    save(fig, "06_loss_ppl_bars")


# ------------------------------------------------------------------ 07
def plot_07_kl_hist_cdf(R):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8), sharex=True)
    eps = 1e-6
    bins = np.logspace(-6, 0.8, 70)
    recs = [("fp16cast", dict(color="gray", ls=":", lw=1.4, label="FP16 cast (noise floor)"))] + [
        (A_NAME.get(f, f), dict(color=STYLE[f]["color"], ls=STYLE[f]["ls"], lw=1.5, label=STYLE[f]["label"]))
        for f in FMTS
    ]
    for col, s in enumerate(SETS):
        ax_h, ax_c = axes[0, col], axes[1, col]
        for rk, kw in recs:
            if rk not in R["sim"]:
                continue
            kl = np.clip(R["sim"][rk][s]["kl"], eps, None)
            ax_h.hist(kl, bins=bins, density=False, weights=np.ones_like(kl) / len(kl),
                      histtype="step", **kw)
            ks = np.sort(kl)
            ys = np.linspace(0, 1, len(ks), endpoint=False)
            step = max(len(ks) // 2000, 1)
            ax_c.plot(ks[::step], ys[::step], **kw)
        ax_h.set_xscale("log")
        ax_h.set_title(f"{SET_LABEL[s]} — Token-Level KL Distribution")
        ax_h.set_ylabel("Fraction of tokens")
        ax_c.set_xscale("log")
        ax_c.set_xlabel("KL(P_BF16 || P_quant)  [log scale]")
        ax_c.set_ylabel("Empirical CDF")
        ax_c.axhline(0.99, color="black", ls=":", lw=0.7, alpha=0.5)
        if col == 0:
            ax_h.legend(frameon=False, loc="upper left")
    fig.suptitle("07. Full-Vocabulary Token-Level KL Divergence Distribution & CDF", y=1.01)
    save(fig, "07_kl_hist_cdf")


# ------------------------------------------------------------------ 08
def plot_08_kl_vs_position(R):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=False)
    n_bins = 16
    for col, s in enumerate(SETS):
        ax = axes[col]
        for f in FMTS:
            rk = A_NAME.get(f, f)
            kw = kl_windows(R, rk, s)              # [64, 2047]
            T = kw.shape[1]
            edges = np.linspace(0, T, n_bins + 1, dtype=int)
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
            means = [float(kw[:, edges[i]:edges[i + 1]].mean()) for i in range(n_bins)]
            p99s = [float(np.quantile(kw[:, edges[i]:edges[i + 1]], 0.99)) for i in range(n_bins)]
            st = STYLE[f]
            ax.plot(centers, means, color=st["color"], ls=st["ls"], lw=1.6, label=f"{st['label']} (mean)")
            ax.plot(centers, p99s, color=st["color"], ls=":", lw=0.9, alpha=0.55)
        ax.set_xlabel("Token position in 2048-token window")
        ax.set_ylabel("KL divergence (solid = mean, dotted = p99)")
        ax.set_yscale("log")
        ax.set_title(f"{SET_LABEL[s]} — KL vs Context Position")
        if col == 0:
            ax.legend(frameon=False, loc="center right", fontsize=7.5)
    fig.suptitle("08. Does Quantization Drift Grow With Context Length?", y=1.02)
    save(fig, "08_kl_vs_position")


# ------------------------------------------------------------------ 09
def plot_09_task_accuracy(R):
    if "bf16" not in R["lm"]:
        return
    labels = [f"{t}\n({PRIMARY[t]})" for t in TASKS] + ["Average"]
    x = np.arange(len(labels))
    w = 0.15
    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    base_accs = [lm_metric(R, "bf16", t)[0] for t in TASKS]
    base_avg = float(np.mean(base_accs))

    for i, f in enumerate(FMTS):
        st = STYLE[f]
        diffs, ses, stars = [], [], []
        accs = []
        for t, bval in zip(TASKS, base_accs):
            v, se = lm_metric(R, f, t)
            accs.append(v)
            diffs.append((v - bval) * 100.0)
            ses.append((se or 0.0) * 100.0)
            a_c, b_c = lm_correct(R, f, t), lm_correct(R, "bf16", t)
            p = mcnemar_exact(a_c, b_c)[2] if a_c is not None and b_c is not None else 1.0
            stars.append("*" if p < 0.05 else "")
        diffs.append((float(np.mean(accs)) - base_avg) * 100.0)
        ses.append(0.0)
        stars.append("")
        offs = x + (i - (len(FMTS) - 1) / 2) * w
        bars = ax.bar(offs, diffs, width=w, color=st["color"], hatch=st["hatch"],
                      edgecolor="black", lw=0.6, label=st["label"])
        ax.errorbar(offs[:-1], diffs[:-1], yerr=ses[:-1], fmt="none", ecolor="black", elinewidth=0.7, capsize=2, alpha=0.6)
        for bx, dy, st_txt in zip(offs, diffs, stars):
            if st_txt:
                ax.text(bx, dy + (0.25 if dy >= 0 else -0.45), st_txt, ha="center", va="center", fontsize=10, fontweight="bold")

    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ Accuracy vs BF16 (percentage points)")
    ax.set_title("09. Downstream 0-Shot Accuracy Change vs BF16 (* = McNemar p < 0.05; bars = ±1 stderr)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    save(fig, "09_task_accuracy_drop")


# ------------------------------------------------------------------ 10
def plot_10_pareto(R):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for f in ALL_FMTS:
        st = STYLE[f]
        sz_mb = R["size"][f]["quant_bytes"] / 1e6
        c128 = next(c for c in R["bench"][f]["cells"] if c["batch"] == 128 and c["input_tokens"] == 128)
        c1 = next(c for c in R["bench"][f]["cells"] if c["batch"] == 1 and c["input_tokens"] == 128)
        kl = 0.0 if f == "bf16" else R["sim"][A_NAME.get(f, f)]["ppl_wikitext"]["stats"]["kl_mean"]
        avg_acc = float(np.mean([lm_metric(R, f, t)[0] for t in TASKS])) * 100.0
        ms = 80 + (sz_mb / 6200.0) * 140
        ax1.scatter(c128["output_tok_per_s"], kl, s=ms, color=st["color"], marker=st["marker"],
                    edgecolor="black", lw=0.8, label=f"{st['label']} ({sz_mb/1000:.2f} GB)", zorder=3)
        ax1.annotate(st["label"], (c128["output_tok_per_s"], kl), xytext=(6, 4), textcoords="offset points", fontsize=7.8)
        ax2.scatter(c1["tpot_ms"], avg_acc, s=ms, color=st["color"], marker=st["marker"],
                    edgecolor="black", lw=0.8, zorder=3)
        ax2.annotate(st["label"], (c1["tpot_ms"], avg_acc), xytext=(6, 4), textcoords="offset points", fontsize=7.8)

    ax1.set_xlabel("Peak Output Throughput (tok/s @ B=128, in=128)  → better")
    ax1.set_ylabel("WikiText Mean KL vs BF16  (lower is better)")
    ax1.invert_yaxis()
    ax1.set_title("Throughput vs Distribution Shift (KL)")
    ax1.legend(frameon=False, loc="lower left", fontsize=7.5)

    ax2.set_xlabel("Single-User Decode TPOT (ms/tok @ B=1)  ← better")
    ax2.set_ylabel("5-Task Average 0-Shot Accuracy (%)  → better")
    ax2.invert_xaxis()
    ax2.set_title("Low-Batch Latency vs Downstream Accuracy")
    fig.suptitle("10. Pareto Trade-offs: Size, Speed, and Quality on Modal L4", y=1.02)
    save(fig, "10_pareto_size_speed_quality")


# ------------------------------------------------------------------ 11
def plot_11_sensitivity(R):
    se = R["diag"].get("sensitivity")
    if not se:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [1.5, 1.0]})
    for rk, f in (("w8a8_pt", "w8a8_pt_dyn"), ("w8a8_pc", "w8a8_pc")):
        st = STYLE[f]
        pl = se["per_layer"][rk]
        xs = sorted(int(k) for k in pl)
        ys = [pl[str(i)]["kl_mean"] for i in xs]
        ax1.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, lw=1.5, label=st["label"])
    ax1.set_xlabel("Single quantized decoder layer (other 35 layers kept in BF16)")
    ax1.set_ylabel("Mean KL vs BF16 (8 diagnostic windows)")
    ax1.set_title("Per-Layer Quantization Sensitivity")
    ax1.legend(frameon=False)

    # right: per-projection bars
    x = np.arange(len(PROJS))
    w = 0.36
    for idx, (rk, f) in enumerate((("w8a8_pt", "w8a8_pt_dyn"), ("w8a8_pc", "w8a8_pc"))):
        st = STYLE[f]
        vals = [se["per_proj"][rk][p]["kl_mean"] for p in PROJS]
        ax2.bar(x + (idx - 0.5) * w, vals, width=w, color=st["color"], edgecolor="black", lw=0.6, label=st["label"])
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace("_proj", "") for p in PROJS], rotation=30)
    ax2.set_ylabel("Mean KL (only this projection quantized)")
    ax2.set_title("Per-Projection Sensitivity (All 36 Layers)")
    ax2.legend(frameon=False)
    fig.suptitle("11. Layer & Projection Sensitivity Analysis (Mixed-Precision Candidates)", y=1.02)
    save(fig, "11_layer_sensitivity")


# ------------------------------------------------------------------ 12
def plot_12_throughput_latency(R):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5))
    for f in ALL_FMTS:
        st = STYLE[f]
        cells = R["bench"][f]["cells"]
        c128 = sorted([c for c in cells if c["input_tokens"] == 128], key=lambda c: c["batch"])
        c512 = sorted([c for c in cells if c["input_tokens"] == 512], key=lambda c: c["batch"])
        bs = [c["batch"] for c in c128]
        ax1.plot(bs, [c["output_tok_per_s"] for c in c128], color=st["color"], ls="-",
                 marker=st["marker"], lw=1.6, label=f"{st['label']} (in=128)")
        ax1.plot(bs, [c["output_tok_per_s"] for c in c512], color=st["color"], ls=":",
                 marker=st["marker"], lw=1.1, alpha=0.65)
        ax2.plot(bs, [c["tpot_ms"] for c in c128], color=st["color"], ls=st["ls"],
                 marker=st["marker"], lw=1.6, label=st["label"])
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 8, 32, 128])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("Batch size (solid = in=128, dotted = in=512)")
    ax1.set_ylabel("Output throughput (tok/s)")
    ax1.set_title("Serving Throughput vs Batch Size")
    ax1.legend(frameon=False, fontsize=7.5)

    ax2.set_xscale("log", base=2)
    ax2.set_xticks([1, 8, 32, 128])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("Batch size (in=128, out=128)")
    ax2.set_ylabel("Decode TPOT (ms / output token)")
    ax2.set_title("Per-Token Decode Latency (W8A16 vs W8A8 Crossover)")
    ax2.legend(frameon=False, fontsize=7.8)
    fig.suptitle("12. vLLM Serving Throughput & Decode Latency on Modal L4 (Ada cc 8.9)", y=1.02)
    save(fig, "12_throughput_latency")


# ------------------------------------------------------------------ 13
def plot_13_trackA_vs_trackB(R):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    xa, xb, da_all, db_all = [], [], [], []
    for f in ALL_FMTS:
        st = STYLE[f]
        rk = "none" if f == "bf16" else A_NAME.get(f, f)
        for s_i, s in enumerate(SETS):
            pa = R["sim"][rk][s]["stats"]["ppl_quant"]
            pb = R["trackb"][f][s]["stats"]["ppl"]
            xa.append(pa)
            xb.append(pb)
            ax1.scatter(pa, pb, color=st["color"], marker=st["marker"], s=55,
                        edgecolor="black", lw=0.6, label=st["label"] if s_i == 0 else None, zorder=3)
            if f != "bf16":
                da = R["sim"][rk][s]["stats"]["delta_loss"]
                db = R["trackb"][f][s]["stats"]["loss"] - R["trackb"]["bf16"][s]["stats"]["loss"]
                da_all.append(da)
                db_all.append(db)
                ax2.scatter(da, db, color=st["color"], marker=st["marker"], s=65,
                            edgecolor="black", lw=0.6, label=st["label"] if s_i == 0 else None, zorder=3)

    lo1, hi1 = min(xa + xb) - 0.15, max(xa + xb) + 0.15
    ax1.plot([lo1, hi1], [lo1, hi1], "k--", lw=0.9, alpha=0.6, label="y = x")
    ax1.set_xlabel("Track A (PyTorch Simulated Fake-Quant) PPL")
    ax1.set_ylabel("Track B (vLLM Real Checkpoint) PPL")
    ax1.set_title("Absolute Perplexity: Track A vs Track B")
    ax1.legend(frameon=False, fontsize=7.5)

    r_val = float(np.corrcoef(da_all, db_all)[0, 1])
    lo2, hi2 = min(da_all + db_all) * 0.85, max(da_all + db_all) * 1.12
    ax2.plot([lo2, hi2], [lo2, hi2], "k--", lw=0.9, alpha=0.6, label="y = x")
    ax2.set_xlabel("Track A ΔLoss vs BF16")
    ax2.set_ylabel("Track B ΔLoss vs BF16")
    ax2.set_title(f"ΔLoss Agreement Across Tracks (Pearson r = {r_val:.4f})")
    ax2.legend(frameon=False, fontsize=7.5)
    fig.suptitle("13. Cross-Track Validation: PyTorch Simulation vs Real vLLM Serving", y=1.02)
    save(fig, "13_trackA_vs_trackB")


def main():
    dry = "--dry" in sys.argv
    R = load_results()
    print(f"Generating plots -> {OUT} (dry={dry})")
    if dry:
        plot_06_loss_ppl_bars(R)
        plot_07_kl_hist_cdf(R)
        plot_08_kl_vs_position(R)
        return
    plot_01_weight_hist(R)
    plot_02_channel_amax(R)
    plot_03_weight_sqnr(R)
    plot_04_act_outliers(R)
    plot_05_layer_error(R)
    plot_06_loss_ppl_bars(R)
    plot_07_kl_hist_cdf(R)
    plot_08_kl_vs_position(R)
    plot_09_task_accuracy(R)
    plot_10_pareto(R)
    plot_11_sensitivity(R)
    plot_12_throughput_latency(R)
    plot_13_trackA_vs_trackB(R)
    print("All 13 plots generated.")


if __name__ == "__main__":
    main()
