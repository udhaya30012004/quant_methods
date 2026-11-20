# FP8 Post-Training Quantization — Master Comparison (`Qwen/Qwen2.5-3B` on NVIDIA L4)

> **Setup:** Pure Post-Training Quantization (PTQ) only — zero fine-tuning, zero QAT, zero gradient updates. Baseline is strictly `bfloat16` (`Qwen/Qwen2.5-3B`, 36 layers, 252 quantized Linear layers; embeddings, norms, and `lm_head` kept in BF16). All real-serving numbers measured on **Modal NVIDIA L4 (Ada Lovelace, cc 8.9)** via `vLLM 0.34.0`.

---

## 1. Master Side-by-Side Comparison (All Metrics)

| Method | Weights / Activations | Calibration Needed? | Size (GB) | Size Saved | WikiText PPL (Δ) | Chat PPL (Δ) | Mean KL (Wiki) | P99 KL (Wiki) | Top-1 Agree | 5-Task Avg Acc | B=1 Decode (ms/tok) | B=128 Throughput (tok/s) | B=128 Prefill (in=512) | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **BF16 (Baseline)** | BF16 / BF16 | No | 6.17 GB | 0.0% | 7.782 (—) | 4.144 (—) | 0.00000 | 0.000 | 100.0% | 68.89% | 25.55 ms (1.00×) | 2,504 (1.00×) | 7.72 s (1.00×) | Reference baseline |
| **`w8a16_pc`** | FP8 Per-Ch / BF16 | No | 3.40 GB | **44.9%** | **7.804 (+0.022)** | **4.157 (+0.013)** | 0.00581 | 0.045 | 96.2% | 69.09% (ns) | **14.66 ms (1.74×)** | 2,860 (1.14×) | 8.76 s (0.88×) | **Best for low-batch decode (`B ≤ 32`) & max quality** |
| **`w8a16_pt`** | FP8 Per-Ten / BF16 | No | 3.40 GB | **45.0%** | 7.809 (+0.027) | 4.165 (+0.021) | **0.00564** | **0.043** | **96.3%** | 69.09% (ns) | **14.66 ms (1.74×)** | 2,891 (1.15×) | 8.74 s (0.88×) | Tied with `w8a16_pc` on speed & WikiText |
| **`w8a8_pc`** *(FP8_DYNAMIC)* | FP8 Per-Ch / FP8 Per-Tok Dyn | No | 3.40 GB | **44.9%** | 7.834 (+0.052) | 4.180 (+0.036) | 0.01050 | 0.082 | 94.7% | 69.01% (ns) | 16.42 ms (1.56×) | **3,374 (1.35×)** | **5.78 s (1.34×)** | **Best for high-batch (`B ≥ 64`) & long prefill** |
| **`w8a8_pt_dyn`** | FP8 Per-Ten / FP8 Per-Ten Dyn | No | 3.40 GB | **45.0%** | 7.855 (+0.073) | 4.183 (+0.039) | 0.01144 | 0.089 | 94.6% | 69.32% (`*` HS) | 16.35 ms (1.56×) | 3,397 (1.36×) | 5.82 s (1.33×) | Slightly worse KL/PPL than `w8a8_pc` |
| **`w8a8_pt_static`** | FP8 Per-Ten / FP8 Per-Ten Static | **Yes (512 wins)** | 3.40 GB | **45.0%** | 7.866 (+0.084) | 4.193 (+0.049) | 0.01524 | 0.127 | 94.0% | 69.16% (`**` HS) | 16.04 ms (1.59×) | 3,427 (1.37×) | 5.78 s (1.34×) | **Avoid** — worst KL (+45%) & `p<0.01` HellaSwag drop |

*Noise floor reference (harmless BF16 → FP16 cast): WikiText Mean KL = `0.00067`, Top-1 Agreement = `98.03%`.*

---

## 2. Zero-Shot Downstream Accuracy Breakdown (16,695 Questions)

Significance vs BF16 computed via **paired exact McNemar test** (`ns` = not significant, `*` = `p < 0.05`, `**` = `p < 0.01`):

| Method | HellaSwag (`acc_norm`, 10,042 Qs) | ARC-Easy (`acc`, 2,376 Qs) | ARC-Challenge (`acc_norm`, 1,172 Qs) | WinoGrande (`acc`, 1,267 Qs) | PIQA (`acc`, 1,838 Qs) | 5-Task Average | Δ vs BF16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BF16 (Baseline)** | 73.56% | 77.31% | 46.76% | 68.43% | 78.40% | **68.89%** | — |
| **`w8a16_pc`** | 73.56% (+0.00 ns) | 77.19% (-0.13 ns) | 47.44% (+0.68 ns) | 68.82% (+0.39 ns) | 78.45% (+0.05 ns) | **69.09%** | +0.20% |
| **`w8a16_pt`** | 73.46% (-0.10 ns) | 77.61% (+0.29 ns) | 47.18% (+0.43 ns) | 68.90% (+0.47 ns) | 78.29% (-0.11 ns) | **69.09%** | +0.20% |
| **`w8a8_pc`** | 73.52% (-0.04 ns) | 77.15% (-0.17 ns) | 46.93% (+0.17 ns) | 68.98% (+0.55 ns) | 78.45% (+0.05 ns) | **69.01%** | +0.11% |
| **`w8a8_pt_dyn`** | 73.26% (-0.30 `*`) | 77.48% (+0.17 ns) | 48.04% (+1.28 `*`) | 69.61% (+1.18 ns) | 78.18% (-0.22 ns) | **69.32%** | +0.42% |
| **`w8a8_pt_static`** | **73.13% (-0.43 `**`)** | 77.74% (+0.42 ns) | 47.70% (+0.94 ns) | 69.06% (+0.63 ns) | 78.18% (-0.22 ns) | **69.16%** | +0.27% |

---

## 3. Serving Speed & Latency Comparison (vLLM on NVIDIA L4, Output = 128 Tokens)

| Method | B=1 TPOT (in=128) | B=1 tok/s (in=128) | B=8 tok/s (in=128) | B=32 tok/s (in=128) | B=128 tok/s (in=128) | B=128 tok/s (in=512) | B=128 Prefill (in=512) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BF16 (Baseline)** | 25.55 ms | 39.0 | 292.2 | 970.9 | 2,503.6 | 1,223.3 | 7.72 s |
| **`w8a16_pc`** | **14.66 ms** | **67.8** | **493.9** | **1,510.8** | 2,859.5 (1.14×) | 1,219.6 (1.00×) | 8.76 s (0.88×) |
| **`w8a16_pt`** | **14.66 ms** | **67.8** | **494.6** | **1,509.0** | 2,891.2 (1.15×) | 1,222.0 (1.00×) | 8.74 s (0.88×) |
| **`w8a8_pc`** | 16.42 ms | 60.7 | 451.4 | 1,459.9 | **3,373.5 (1.35×)** | **1,612.6 (1.32×)** | **5.78 s (1.34×)** |
| **`w8a8_pt_dyn`** | 16.35 ms | 60.9 | 453.3 | 1,460.9 | **3,396.6 (1.36×)** | **1,600.2 (1.31×)** | 5.82 s (1.33×) |
| **`w8a8_pt_static`** | 16.04 ms | 62.1 | 461.6 | 1,472.9 | **3,426.7 (1.37×)** | **1,615.0 (1.32×)** | **5.78 s (1.34×)** |

---

## 4. Head-to-Head Design Comparisons

| Comparison Axis | Winner | Quantitative Difference | Why |
|---|---|---|---|
| **W8A16 vs W8A8 (Quality)** | **W8A16** | ~2× lower Mean KL (`0.0058` vs `0.0105`, `p < 0.001`) | Weights are smooth (`31.5 dB` SQNR), while activations have extreme outliers (`1.6 × 10⁷` max/median in Layer 2 `down_proj`). |
| **W8A16 vs W8A8 (Speed)** | **Split by Batch Size** | W8A16 **1.74×** at `B=1`; W8A8 **1.35×** at `B=128` & **1.34×** prefill | `B ≤ 32` is memory-bandwidth bound (Marlin W8A16 wins); `B = 128` / long prefill is compute bound (native Ada FP8 Tensor Cores win). |
| **Per-Channel vs Per-Tensor (W8A16)** | **Tie (slight PC edge)** | WikiText Δloss `ns`; Chat Δloss `-0.0019` (`p < 0.001`); same speed | Weight SQNR is identical (`31.56` vs `31.54 dB`), though PC cuts flush-to-zero weights by 8.5×. |
| **Per-Channel/Token vs Per-Tensor (W8A8)** | **Per-Channel / Token (`w8a8_pc`)** | WikiText Δloss `-0.0027` (`p < 0.001`), KL `-0.00094` (`p < 0.001`) | Per-token dynamic activation scaling isolates individual outlier tokens. |
| **Dynamic vs Static Activations (W8A8)** | **Dynamic (`w8a8_pc` / `w8a8_pt_dyn`)** | Static adds **+45% KL error** (`0.0152` vs `0.0105`, `p < 0.001`) for **<2% speed** | Fixed calibration scales clip extreme runtime activation spikes even on in-distribution chat data. |

---

## 5. Final Deployment Recommendation

1. **Interactive / Low-Batch Serving (`B ≤ 32`):** Deploy **`w8a16_pc`** (Marlin kernel) — **45% smaller**, **1.74× faster single-user decode**, lowest perplexity & KL shift, zero accuracy loss.
2. **High-Concurrency / Long-Prompt Serving (`B ≥ 64` or `in ≥ 512`):** Deploy **`w8a8_pc`** (`FP8_DYNAMIC`) — **45% smaller**, **1.35× peak throughput**, **1.34× faster prefill**, zero accuracy loss, no calibration needed.
3. **Skip `w8a8_pt_static`** — requires calibration, causes the worst distribution shift and the only `p < 0.01` benchmark drop, with negligible speed benefit over `w8a8_pc`.

*Supporting visuals:* [accuracy_plots/](accuracy_plots/) | [time_plots/](time_plots/) | [docs/findings.md](docs/findings.md) | [results/summary/summary.md](results/summary/summary.md)
