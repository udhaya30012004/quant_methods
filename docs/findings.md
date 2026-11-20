# FP8 Post-Training Quantization Findings — Qwen2.5-3B on NVIDIA L4

> **Scope & Constraint:** Pure Post-Training Quantization (PTQ) only — zero fine-tuning, zero QAT, zero gradient updates (`assert_inference_only()` enforced in every script). Baseline is strictly `bfloat16` (`Qwen/Qwen2.5-3B`, 36 layers, 252 quantized Linear modules; embeddings, norms, and `lm_head` kept in BF16). Hardware: Modal NVIDIA L4 (Ada Lovelace, compute capability 8.9, 24 GB VRAM).

Full summary tables and pairwise bootstrap tests: [summary.md](../results/summary/summary.md) | [summary.csv](../results/summary/summary.csv).  
All 13 visual figures (PNG & SVG): [results/plots/](../results/plots/).

---

## 1. Executive Summary & Deployment Recommendation

| Workload on NVIDIA L4 (cc 8.9) | Recommended Recipe | Why |
|---|---|---|
| **Low-batch / latency-sensitive serving** (`B ≤ 32`, interactive single-user decode) | **`w8a16_pc`** *(W8A16 per-channel weights, BF16 activations; Marlin kernel)* | **1.74× faster decode** than BF16 (`14.66` vs `25.55 ms/tok` at `B=1`), **44.9% smaller** (`3.17` vs `5.75 GiB`), lowest perplexity degradation (`+0.022` PPL on WikiText), zero statistically significant accuracy loss on any of the 5 benchmarks (`+0.20 pp` avg). |
| **High-batch / prefill-heavy throughput serving** (`B ≥ 64` or long prompts `in ≥ 512`) | **`w8a8_pc`** *(FP8_DYNAMIC: per-channel weights, dynamic per-token FP8 activations)* | **1.35× peak output throughput** (`3,374` vs `2,504 tok/s` at `B=128, in=128`), **1.34× faster prefill** (`5.78` vs `7.72 s` at `B=128, in=512`), zero statistically significant accuracy loss on any benchmark (`+0.11 pp` avg), and significantly lower KL (`0.0105` vs `0.0152`, `p < 0.001`) than static W8A8. |
| **Avoid on Qwen2.5-3B** | **`w8a8_pt_static`** *(static per-tensor calibrated activations)* | Only **+1.5% faster** than `w8a8_pc` at `B=128`, requires a 512-sample calibration pass, suffers the **worst distribution shift** (`KL = 0.0152`, `+45%` worse than `w8a8_pc`, `p < 0.001`), and is the **only recipe with a `p < 0.01` accuracy drop** on HellaSwag (`-0.43 pp`, `p < 0.01`). |

---

## 2. Hypothesis-by-Hypothesis Evaluation

### H1: Quantizing activations (W8A8 vs W8A16) causes most of the distribution shift, not quantizing weights
**Verdict: CONFIRMED (`p < 0.001`)**  
- **Evidence:** On WikiText-2 (Track A full-vocab KL), W8A16 recipes sit at `0.00564` (`w8a16_pt`) and `0.00581` (`w8a16_pc`), whereas W8A8 recipes roughly **double** mean KL to `0.01050` (`w8a8_pc`), `0.01144` (`w8a8_pt_dyn`), and `0.01524` (`w8a8_pt_static`).
- **Paired bootstrap (Track B Δloss):** `w8a16_pc − w8a8_pc` = `-0.0039 [-0.0045, -0.0032]` (`p < 0.001`) on WikiText and `-0.0055 [-0.0062, -0.0048]` (`p < 0.001`) on UltraChat.
- **Root cause (E3 diagnostics):** Weight SQNR is clean and uniform (`~31.55 dB` across all 252 layers; `03_weight_sqnr_by_layer.png`), whereas **activations exhibit extreme outliers** in early `down_proj` inputs — max/median outlier factor reaches **`1.58 × 10⁷`** in Layer 2, `5.92 × 10⁵` in Layer 1, and `5.43 × 10⁵` in Layer 3 (`04_act_amax_by_layer.png`).
- **Noise-floor context:** All FP8 recipes sit `8×–23×` above the harmless BF16 → FP16 cast noise floor (`mean KL = 0.00067`, `top-1 = 98.03%`; `07_kl_hist_cdf.png`).

### H2: Per-channel weight scaling dramatically beats per-tensor in W8A16
**Verdict: PARTIALLY REFUTED (negligible practical difference in W8A16; small but real advantage in W8A8)**  
- **Evidence (W8A16):** Across all 252 Linear layers, mean weight SQNR is **`31.54 dB` (per-tensor) vs `31.56 dB` (per-channel)** — virtually identical (`03_weight_sqnr_by_layer.png`). While per-channel cuts flush-to-zero weights by `8.5×` (`5.87×10⁻⁵` → `6.90×10⁻⁶`) and rescues outlier channels in early `up_proj` layers (up to `44.5×` max/median spread in Layer 1; `02_channel_amax_sorted.png`), Track B WikiText Δloss between `w8a16_pt` and `w8a16_pc` is **not statistically significant** (`+0.0007 [-0.0003, +0.0016]`, `ns`). On UltraChat, `w8a16_pc` wins modestly (`-0.0019`, `p < 0.001`), while Track A mean KL is actually marginally lower for `w8a16_pt` (`0.00564` vs `0.00581`).
- **Evidence (W8A8):** Combining per-channel weights with **per-token dynamic activations** (`w8a8_pc`) clearly beats per-tensor dynamic (`w8a8_pt_dyn`): WikiText Track B Δloss `-0.0027 [-0.0041, -0.0014]` (`p < 0.001`) and Track A KL `-0.00094` (`p < 0.001`). The gain comes primarily from per-token activation scaling isolating token spikes rather than from weight granularity.

### H3: Static per-tensor activation calibration matches dynamic scaling without runtime overhead
**Verdict: REFUTED (`p < 0.001` worse quality; `<2%` speed benefit)**  
- **Quality:** Even on **in-distribution UltraChat** (same dataset family as the 512-window calibration set), `w8a8_pt_static` is significantly worse than `w8a8_pt_dyn` (Track B Δloss `+0.0024`, `p < 0.001`; Track A KL `0.01344` vs `0.01007`, `p < 0.001`) and `w8a8_pc` (`KL 0.01344` vs `0.00928`). On WikiText-2 (out-of-distribution relative to calibration), `p99 KL` jumps to **`0.1274`** (vs `0.0816` for `w8a8_pc`) and HellaSwag drops **`-0.43 pp` (`p < 0.01`, McNemar)** — the only `p < 0.01` downstream drop in the entire study (`09_task_accuracy_drop.png`).
- **Speed:** At `B=128, in=128`, `w8a8_pt_static` delivers `3,426.7 tok/s` vs `3,373.5 tok/s` for `w8a8_pc` (**+1.6%**), and at `B=128, in=512` it is virtually identical (`1,615.0` vs `1,612.6 tok/s`, **+0.15%**). Static calibration does not pay for itself on L4.

### H4: Quantization error accumulates monotonically with depth and concentrates in specific layers/projections
**Verdict: CONFIRMED (`05_layer_output_error_depth.png`, `11_layer_sensitivity.png`)**  
- **Depth accumulation:** Final-block residual-stream cosine similarity vs BF16 follows the exact same monotonic ordering as PPL and KL:
  - `w8a16_pc`: `0.99880` (`rel_err 0.0483`)
  - `w8a16_pt`: `0.99877` (`rel_err 0.0483`)
  - `w8a8_pc`: `0.99785` (`rel_err 0.0644`)
  - `w8a8_pt_dyn`: `0.99765` (`rel_err 0.0673`)
  - `w8a8_pt_static`: `0.99702` (`rel_err 0.0756`)
- **Single-layer & per-projection sensitivity:** Quantizing one layer at a time reveals a **U-shaped sensitivity profile**: Layer **35** (final block, KL `0.00201`), Layer **0** (`0.00148`), Layer **2** (`0.00132`), and Layers **30–31** (`~0.0012`) dominate. Across projection types, **`down_proj` is by far the most sensitive** (`KL 0.00452` for `w8a8_pt`, `0.00400` for `w8a8_pc`), followed by `gate_proj` (`0.00320`) and `up_proj` (`0.00304`), directly matching the activation outlier heatmap. Keeping just `down_proj` in Layers 0, 2, and 35 in BF16 is the highest-leverage mixed-precision lever if tighter fidelity is ever needed.
- **Context-position stability (`08_kl_vs_position.png`):** Token-level KL stays flat across the 2048-token window rather than compounding with sequence length.

### H5: Downstream 0-shot accuracy is largely preserved across FP8 PTQ, even when KL/PPL shifts are statistically significant
**Verdict: CONFIRMED (`09_task_accuracy_drop.png`)**  
- Across 16,695 total evaluation examples (`hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`, `piqa`), BF16 averages **`68.89%`** and all five FP8 checkpoints average **`69.01%–69.32%`** (`+0.11` to `+0.42 pp`, well within task noise).
- Paired **exact McNemar tests** against BF16 across all 25 `(recipe × task)` cells show **no statistically significant degradation** for `w8a16_pt`, `w8a16_pc`, or `w8a8_pc` on any task. Only `w8a8_pt_dyn` (`-0.30 pp`, `p < 0.05`) and `w8a8_pt_static` (`-0.43 pp`, `p < 0.01`) show significant drops on HellaSwag (`N = 10,042`, the highest-powered benchmark).
- **Takeaway:** Standard 5-task lm-eval benchmarks are insensitive to small FP8 differences (`ΔPPL < 0.08`); **full-vocab KL and paired window-level Δloss** are far sharper instruments for comparing PTQ recipes.

### H6: W8A16 vs W8A8 exhibits a clear hardware speed crossover on NVIDIA L4 (Ada cc 8.9)
**Verdict: CONFIRMED (`12_throughput_latency.png`, `10_pareto_size_speed_quality.png`)**  
- **Memory-bandwidth regime (small-batch decode, `B ≤ 32`):** **W8A16 (Marlin)** is fastest — `14.66 ms/tok` at `B=1` (**1.74×** faster than BF16's `25.55 ms/tok` and **12% faster** than W8A8's `16.42 ms/tok`), because decode is memory-bandwidth-bound and Marlin avoids dynamic activation quantization overhead.
- **Compute-bound regime (large batch / long prefill, `B = 128` or `in = 512`):** **W8A8 (native Ada FP8 Tensor Cores)** wins decisively — prefill at `B=128, in=512` drops from `7.72 s` (BF16) to **`5.78 s`** (**1.34× faster**), whereas W8A16 *slows down* to `8.74 s` due to on-the-fly FP8→BF16 weight dequantization overhead. Peak output throughput at `B=128, in=128` reaches **`3,374–3,427 tok/s`** (**+35%–37%** over BF16) vs `2,860–2,891 tok/s` (+15%) for W8A16.
- **Checkpoint compression:** All 5 FP8 checkpoints shrink identically by **44.9%–45.0%** (`6,171.9 MB` → `3,397.4–3,399.5 MB`, ~`2.6 GiB` freed on GPU for KV cache; spread across recipes is only `0.061%` from scale tensors).

### H7: PyTorch fake-quantization (Track A) faithfully predicts real vLLM FP8 serving (Track B)
**Verdict: CONFIRMED (`13_trackA_vs_trackB.png`, Gate 1–3 PASS)**  
- **Weight validation gate (`validate_sim.py`):** Comparing Track A fake-quant weights against dequantized Track B checkpoint weights across all 252 layers yields `97.0%` exact FP8 match and `RMSE = 4.0×10⁻⁴`. The `3.32×10⁻³` max relative scale difference is explained by `llm-compressor` storing `weight_scale` in `bfloat16` (`1 ULP = 2⁻⁸ ≈ 3.91×10⁻³`) vs `float32` in Track A.
- **Perplexity & Δloss agreement:** BF16 WikiText PPL matches within **`0.06%`** (`7.778` Track A vs `7.782` Track B), and all 10 `(recipe × dataset)` Δloss pairs match in sign, rank order, and magnitude (`Gate 3 PASS`).

---

## 3. Methodological Notes & Limitations
1. **vLLM W8A16 Marlin pin:** vLLM 0.34.0's default Humming backend failed NVRTC JIT compilation for compressed-tensors W8A16 FP8 on L4; pinning only `w8a16_fp8` to `"marlin"` via `KernelConfig` resolved it cleanly while leaving W8A8 on native automatic FP8 Tensor Core dispatch.
2. **EngineCore log capture (Gate 6):** vLLM V1 runs `EngineCore` in a subprocess whose logs bypass Python's `MainProcess` logging handler, so `kernel_log` and `model_weight_gib` were not captured automatically at runtime. Checkpoint weight sizes are taken directly from safetensors inspection (`results/size_*.json`), and W8A8 FP8 Tensor Core execution is confirmed by the `1.34×` prefill speedup at `B=128, in=512`.
3. **Greedy exact-match (`sanity25`):** 25-prompt greedy exact match (`6/25–8/25`, mean prefix `0.39–0.58`) is chaotic because a single near-tie token flip diverges all subsequent tokens; it serves as a qualitative sanity check rather than a ranking metric.
4. **Next phase (INT8):** The identical evaluation harness (`sim_eval.py`, `eval_ppl_vllm.py`, `layer_diag.py`, `bench_speed.py`, `eval_lm.py`, `stats.py`, `make_plots.py`, `make_summary.py`) is ready to benchmark INT8 PTQ recipes against these exact BF16 and FP8 baselines.
