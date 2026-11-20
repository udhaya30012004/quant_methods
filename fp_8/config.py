"""Central configuration for the PTQ study (BF16 baseline vs FP8 recipes, INT8 later).

POST-TRAINING QUANTIZATION ONLY: no QAT, no fine-tuning, no gradients anywhere.

Every other script imports from here so all formats are compared on identical settings.
Run directly (`python fp_8/config.py`) to print and sanity-check the config.
"""
from pathlib import Path

# ---------------------------------------------------------------- model
MODEL_ID = "Qwen/Qwen2.5-3B"               # BF16 baseline from Hugging Face (base model)
MODEL_SHORT = "qwen2.5-3b"               # used in folder / file names

# ---------------------------------------------------------------- Modal
APP_NAME = "quantt-fp8"
GPU = "L4"                  # Ada GPU, native FP8. SAME GPU for every format.
VOLUME_NAME = "quant-models"
VOLUME_MOUNT = "/vol"       # where the Volume is mounted inside the container

# paths INSIDE the Modal container (on the Volume)
REMOTE_HF_CACHE = f"{VOLUME_MOUNT}/hf-cache"
REMOTE_MODELS = f"{VOLUME_MOUNT}/models"
REMOTE_RESULTS = f"{VOLUME_MOUNT}/results"

# Track A recipes (sim_eval.py): FP8 = E4M3. This table defines the simulated recipes.
RECIPES = {
    "w8a16_pt":       dict(w="fp8", a="bf16", wscale="per_tensor",  ascale="none"),
    "w8a16_pc":       dict(w="fp8", a="bf16", wscale="per_channel", ascale="none"),
    "w8a8_pt":        dict(w="fp8", a="fp8",  wscale="per_tensor",  ascale="per_tensor_dynamic"),
    "w8a8_pc":        dict(w="fp8", a="fp8",  wscale="per_channel", ascale="per_token_dynamic"),  # == FP8_DYNAMIC
    "w8a8_pt_static": dict(w="fp8", a="fp8",  wscale="per_tensor",  ascale="per_tensor_static"),  # uses Track B checkpoint scales
}

# Track B recipes (llm-compressor checkpoints).
# w8a8_pt_dyn mirrors Track A's dynamic per-tensor w8a8_pt; w8a8_pt_static uses
# calibrated static per-tensor activation scales. Keep distinct IDs so cross-track
# comparisons never mix dynamic and static activation scaling.
CHECKPOINT_RECIPES = {
    "w8a16_pt":       dict(w="fp8", a="bf16", wscale="per_tensor",  ascale="none"),
    "w8a16_pc":       dict(w="fp8", a="bf16", wscale="per_channel", ascale="none"),
    "w8a8_pc":        dict(w="fp8", a="fp8",  wscale="per_channel", ascale="per_token_dynamic"),
    "w8a8_pt_dyn":    dict(w="fp8", a="fp8",  wscale="per_tensor",  ascale="per_tensor_dynamic"),
    "w8a8_pt_static": dict(w="fp8", a="fp8",  wscale="per_tensor",  ascale="per_tensor_static"),
}

# format name -> checkpoint dir on the Volume. "bf16" is the untouched baseline.
FORMATS = {"bf16": f"{REMOTE_MODELS}/{MODEL_SHORT}-bf16"}
FORMATS.update({r: f"{REMOTE_MODELS}/{MODEL_SHORT}-{r.replace('_', '-')}" for r in CHECKPOINT_RECIPES})
ALL_FORMATS = ["bf16", *CHECKPOINT_RECIPES]
# FORMATS["int8_..."] = ...   # phase 6

DTYPE = "bfloat16"          # baseline weights AND activations. Never float16.
REMOTE_DATA = f"{VOLUME_MOUNT}/data"

# ---------------------------------------------------------------- local paths
ROOT = Path(__file__).resolve().parent.parent      # E:\quantt
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SUMMARY_DIR = RESULTS_DIR / "summary"

# ---------------------------------------------------------------- experiment
SEED = 42
MAX_MODEL_LEN = 2048
GPU_MEM_UTIL = 0.85

# FP8 recipe (llm-compressor)
FP8_SCHEME = "FP8_DYNAMIC"   # only used by the default w8a8_pc path until phase 1
QUANT_IGNORE = ["lm_head"]

# accuracy tasks (lm-evaluation-harness; WikiText PPL is evaluated in eval_ppl_vllm.py on frozen windows)
EVAL_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]
LM_EVAL_FEWSHOT = 0
LM_EVAL_BATCH = "auto"
EVAL_MMLU = False           # optional second pass: --tasks mmlu --fewshot 5

# speed benchmark
BENCH_BATCH_SIZES = [1, 8, 32, 128]
BENCH_INPUT_TOKENS = [128, 512]
BENCH_OUTPUT_TOKENS = 128
BENCH_REPEATS = 5
BENCH_WARMUP = 2


def assert_inference_only(model=None) -> None:
    """PTQ guard: no autograd, no trainable params. Call at the start of every Modal function."""
    import torch

    torch.set_grad_enabled(False)
    assert not torch.is_grad_enabled(), "grad must be disabled (PTQ only)"
    if model is not None:
        model.eval()
        bad = [n for n, p in model.named_parameters() if p.requires_grad]
        if bad:
            for _, p in model.named_parameters():
                p.requires_grad_(False)


if __name__ == "__main__":
    print(f"Model      : {MODEL_ID}")
    print(f"GPU        : {GPU}")
    print(f"dtype      : {DTYPE}")
    print(f"Volume     : {VOLUME_NAME} -> {VOLUME_MOUNT}")
    print(f"Formats    : {list(FORMATS)}")
    print(f"Root       : {ROOT}")
    for name, p in [("results", RESULTS_DIR), ("plots", PLOTS_DIR), ("summary", SUMMARY_DIR)]:
        print(f"{name:<10} : {p}  exists={p.exists()}")
