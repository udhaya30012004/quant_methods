# FP8 W8A8 Dynamic (`FP8_DYNAMIC`) — Exact Step-by-Step Procedure for Gemma 3 27B

> **Scope:** Strictly Post-Training Quantization (PTQ) only — **FP8 W8A8 Dynamic (`FP8_DYNAMIC`)**:
> - **Weights:** 8-bit FP8 (`float8_e4m3fn`), symmetric, **per-channel**, static RTN (no calibration).
> - **Activations:** 8-bit FP8 (`float8_e4m3fn`), symmetric, **dynamic per-token** at runtime (no calibration).
> - **Baseline dtype:** Strictly `bfloat16` (never `float16`).
> - **Target model:** `google/gemma-3-27b-it` (high-concurrency serving on vLLM).

---

## Step 0: Set Up Separate Modal Environments & Smoke-Test GPU

Use **two separate Modal images** (so `llmcompressor` and `vllm` never conflict on `torch`/`transformers`) and a persistent Modal Volume (`/vol`).

- **GPU requirement:** Compute Capability **≥ 8.9** with enough VRAM for 27B:
  - **`H100` (80 GB)** for quantization and BF16 vs FP8 apples-to-apples benchmarking (BF16 is ~55 GB, FP8 is ~29 GB).
  - Do **not** use A100 (cc 8.0, no native FP8) or L4 (24 GB, OOM).
- **Secrets:** Attach Hugging Face token via Modal Secret (`HF_TOKEN`) because `google/gemma-3-27b-it` is gated.
- **Image 1 (`quant_image`):** `llmcompressor`, `transformers` (supporting `Gemma3ForConditionalGeneration`), `accelerate`, `safetensors`.
- **Image 2 (`vllm_image`):** `vllm`, `lm-eval`.
- **Smoke test both images** before running anything:
  - Verify `torch.cuda.get_device_capability(0) >= (8, 9)` (native FP8 supported).
  - Verify `from transformers import Gemma3ForConditionalGeneration` and Volume mount `/vol`.

---

## Step 1: Download BF16 Baseline to Modal Volume (CPU Only)

Download `google/gemma-3-27b-it` into `/vol/models/gemma-3-27b-it-bf16` via `huggingface_hub.snapshot_download` and call `volume.commit()`.

- Include: `*.safetensors`, `*.json`, `*.txt`, `*.jinja`, `*.model`.
- Verify `config.json` shows `bfloat16` and record total BF16 `.safetensors` size in bytes.

---

## Step 2: Prepare Frozen Evaluation Datasets (CPU Only)

Build and save deterministic evaluation sets on `/vol/data/` using the target model's tokenizer (seed `42`, window length `2048` tokens):
1. `ppl_wikitext.npy`: 64 frozen 2048-token windows from WikiText-2 test split.
2. `kl_chat.npy`: 64 frozen 2048-token windows from UltraChat (`HuggingFaceH4/ultrachat_200k`).
3. `prompts.json`: 25 frozen test prompts for greedy generation sanity checks.

*(No calibration dataset is required because `FP8_DYNAMIC` is 100% data-free.)*

---

## Step 3: Quantize BF16 → FP8 W8A8 Dynamic (`FP8_DYNAMIC`)

Run one-shot PTQ on GPU (`H100`) using `quant_image`. Enforce `torch.set_grad_enabled(False)` and `model.eval()`.

```python
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

SRC = "/vol/models/gemma-3-27b-it-bf16"
DST = "/vol/models/gemma-3-27b-it-w8a8-pc"

# Mandatory ignore list for Gemma 3 (multimodal) + output head
QUANT_IGNORE = [
    "re:.*lm_head",
    "re:.*vision_tower.*",
    "re:.*multi_modal_projector.*",
]

torch.manual_seed(42)
torch.set_grad_enabled(False)

model = Gemma3ForConditionalGeneration.from_pretrained(
    SRC, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
processor = AutoProcessor.from_pretrained(SRC)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=QUANT_IGNORE,
)

# Data-free one-shot PTQ (no calibration dataset)
oneshot(model=model, recipe=recipe)

# Save compressed FP8 checkpoint + processor/tokenizer
model.save_pretrained(DST, save_compressed=True)
processor.save_pretrained(DST)
```

Call `volume.commit()`.

---

## Step 4: Audit the Saved FP8 Checkpoint

Inspect every `.safetensors` shard and `config.json` in `DST` and save `size_report.json`:
1. Confirm `config.json` has `quantization_config.quant_method == "compressed-tensors"` and `format == "float-quantized"`.
2. Confirm `torch.float8_e4m3fn` tensors exist and total `.safetensors` size shrank by **~45%–48%** vs BF16.
3. Confirm `weight_scale` tensors exist for all language-model Linear layers and **`input_scale` count == 0** (proving dynamic activations).
4. Confirm **zero** `weight_scale` tensors appear in `lm_head`, `vision_tower`, or `multi_modal_projector`.

---

## Step 5: Evaluate Perplexity on Frozen Windows (`vLLM`)

Using `vllm_image` on the same GPU (`H100`), run both **`bf16`** and **`w8a8_pc`** through `vLLM`:
- Load with `dtype="bfloat16"`, `max_model_len=2048 + 8`, `seed=42` (let vLLM auto-select native FP8 Tensor Core kernels for W8A8).
- Feed `ppl_wikitext.npy` and `kl_chat.npy` (64 windows each) via `TokensPrompt` with `SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0, seed=42)`.
- Extract per-token NLL (`-logprob`), window-level losses, mean loss, and `PPL = exp(mean_loss)`.
- Save per-token NLL `.npy` and summary `.json` for both `bf16` and `w8a8_pc`, and compute paired bootstrap 95% CI on window-level `Δloss`.

---

## Step 6: Run Greedy Generation Sanity Check (`vLLM`)

Run both **`bf16`** and **`w8a8_pc`** on the 25 frozen prompts (`prompts.json`) with `SamplingParams(temperature=0.0, max_tokens=128, seed=42)`:
- Compare generated `token_ids` against `bf16` (exact match count, first divergence token, mean matching prefix fraction).
- Also run a quick multimodal sanity check (image + text prompt) to confirm `vision_tower` and `multi_modal_projector` work cleanly.

---

## Step 7: Benchmark High-Concurrency Speed & Memory (`vLLM`)

Run **`bf16`** and **`w8a8_pc`** **sequentially** (never in parallel) on the exact same GPU (`H100`) using `vllm_image`:
- Load with `dtype="bfloat16"`, `gpu_memory_utilization=0.90`, `max_num_seqs=max(batches)`, `enable_prefix_caching=False`, `seed=42`.
- Use fixed-seed synthetic `TokensPrompt` batches (identical token IDs across both formats) and `SamplingParams(temperature=0.0, ignore_eos=True, seed=42)`.
- Benchmark grid (high-concurrency focus):
  - `batches = [1, 8, 32, 64, 128, 256]`
  - `input_tokens = [128, 512, 2048]`
  - `output_tokens = 128`
  - `warmup = 2`, `repeats = 5` (report median)
- Per `(batch, input_tokens)` cell, time:
  1. `prefill_s`: `generate(max_tokens=1)`
  2. `e2e_s`: `generate(max_tokens=128)`
  3. `tpot_ms = (e2e_s - prefill_s) / 127 * 1000`
  4. `output_tok_per_s = batch * 128 / e2e_s`
  5. `total_tok_per_s = batch * (input_tokens + 128) / e2e_s`
- Record GPU VRAM used (`torch.cuda.mem_get_info(0)`) and max concurrent KV-cache capacity.

---

## Step 8: Evaluate Downstream Zero-Shot Accuracy (`lm-eval` + `vLLM`)

Using `vllm_image`, run `lm-evaluation-harness` (`lm_eval.models.vllm_causallms.VLLM`) on both **`bf16`** and **`w8a8_pc`**:
- Tasks (0-shot, `seed=42`, `log_samples=True`):
  - `hellaswag` (`acc_norm`)
  - `arc_easy` (`acc`)
  - `arc_challenge` (`acc_norm`)
  - `winogrande` (`acc`)
  - `piqa` (`acc`)
- Save per-example correctness arrays (`__samples.npz`) and run **paired exact McNemar tests** (`w8a8_pc` vs `bf16`) per task to verify zero statistically significant degradation.

---

## Step 9: Publish Verified Checkpoint to Hugging Face (CPU Only)

Upload directly from the Modal Volume (`/vol/models/gemma-3-27b-it-w8a8-pc`) with `HF_XET_HIGH_PERFORMANCE=1`:
1. Verify `.safetensors` headers and `config.json` `quantization_config` before uploading.
2. Upload allowlisted checkpoint files (`*.safetensors`, `model.safetensors.index.json`, `config.json`, `generation_config.json`, `processor_config.json`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model`, `chat_template.*`, `recipe.yaml`).
3. Exclude `size_report.json`.
4. Commit `README.md` model card (with `base_model: google/gemma-3-27b-it`, `base_model_relation: quantized`, benchmark metrics, and vLLM quickstart).
