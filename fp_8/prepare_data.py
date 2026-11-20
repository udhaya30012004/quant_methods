"""Step 0b: build the frozen evaluation / calibration data on the Modal Volume (CPU only).

Everything is tokenized once with the model's tokenizer and saved as .npy so every
recipe (bf16 baseline and all PTQ variants) is evaluated on byte-identical inputs.

  ppl_wikitext : WikiText-2 test      -> 64 x 2048 tokens   (PPL / loss / KL)
  kl_chat      : UltraChat test_sft   -> 64 x 2048 tokens   (KL on a 2nd domain)
  calib_chat   : UltraChat train_sft  -> 512 samples        (static-activation calibration ONLY; no training)
  prompts      : 25 fixed prompts                           (greedy output diff)

Run from E:\\quantt\\fp_8 (needs the model downloaded first, for its tokenizer):
    modal run prepare_data.py
"""
import modal

import config
from modal_app import app, quant_image, volume, VOLUMES

DATA_DIR = f"{config.VOLUME_MOUNT}/data"
N_WINDOWS = 64
WINDOW = 2048
N_CALIB = 512

PROMPTS = [
    # factual
    "What is the capital of France? Answer in one sentence.",
    "Who wrote the novel 'Pride and Prejudice'?",
    "What is the boiling point of water at sea level in Celsius?",
    "Name the largest planet in our solar system and one fact about it.",
    "In what year did World War II end?",
    # explanation
    "Explain what quantization of a neural network means in two sentences.",
    "Explain the difference between a process and a thread.",
    "Why is the sky blue? Keep it under three sentences.",
    "What does a hash table do and why is lookup fast?",
    "Summarize what overfitting is in machine learning.",
    # code
    "Write a Python function that returns the n-th Fibonacci number.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Write a SQL query that returns the top 3 highest salaries from a table named employees.",
    "Write a JavaScript function that reverses an array without using reverse().",
    "Write a Python function that merges two sorted lists into one sorted list.",
    # math / reasoning
    "What is 17 * 23? Show the calculation.",
    "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
    "A box has 3 red and 5 blue balls. What is the probability of drawing a red ball?",
    "Solve for x: 3x + 7 = 25.",
    "Is 91 a prime number? Explain.",
    # translation / language
    "Translate to French: The weather is nice today.",
    "Translate to Spanish: Where is the nearest train station?",
    "Translate to German: I would like a cup of coffee, please.",
    "Rewrite this sentence more formally: hey, can u send me the report asap?",
    "Give three synonyms for the word 'happy'.",
]


def _windows(token_ids, n, size):
    """Cut a long token stream into n non-overlapping windows of `size` tokens."""
    import numpy as np

    need = n * size
    assert len(token_ids) >= need, f"only {len(token_ids)} tokens, need {need}"
    return np.asarray(token_ids[:need], dtype=np.int32).reshape(n, size)


@app.function(image=quant_image, volumes=VOLUMES, timeout=3600)
def build() -> dict:
    import hashlib
    import json
    import os

    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    volume.reload()
    os.makedirs(DATA_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(config.FORMATS["bf16"])

    def chat_ids(messages):
        # render to text first, then tokenize: apply_chat_template(tokenize=True) returns a dict in new transformers
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return tok(text, add_special_tokens=False)["input_ids"]

    manifest = {"model_id": config.MODEL_ID, "window": WINDOW, "sets": {}}

    def save(name, arr, source):
        path = f"{DATA_DIR}/{name}.npy"
        np.save(path, arr)
        manifest["sets"][name] = {
            "source": source,
            "shape": list(arr.shape),
            "tokens": int(arr.size),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
        print(f"{name:<14} shape={arr.shape} tokens={arr.size}  <- {source}")

    # ---- 1. WikiText-2 test (standard PPL protocol: join with blank lines, tokenize once)
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(wt["text"]), add_special_tokens=False)["input_ids"]
    save("ppl_wikitext", _windows(ids, N_WINDOWS, WINDOW), "wikitext-2-raw-v1/test")

    # ---- 2. UltraChat test_sft (second domain): concatenate rendered chats until enough tokens
    uc_test = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    stream = []
    for row in uc_test:
        stream += chat_ids(row["messages"])
        if len(stream) >= N_WINDOWS * WINDOW:
            break
    save("kl_chat", _windows(stream, N_WINDOWS, WINDOW), "ultrachat_200k/test_sft")

    # ---- 3. calibration set: different split from every eval set. Activation range statistics only.
    uc_train = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=config.SEED)
    calib = []
    for row in uc_train.select(range(N_CALIB)):
        t = chat_ids(row["messages"])[:WINDOW]
        calib.append(t)
    flat = np.zeros((N_CALIB, WINDOW), dtype=np.int32)
    lens = np.zeros(N_CALIB, dtype=np.int32)
    for i, t in enumerate(calib):
        flat[i, : len(t)] = t
        lens[i] = len(t)
    save("calib_chat", flat, "ultrachat_200k/train_sft (seed=42)")
    np.save(f"{DATA_DIR}/calib_chat_lens.npy", lens)

    # ---- 4. fixed prompts
    with open(f"{DATA_DIR}/prompts.json", "w") as f:
        json.dump(PROMPTS, f, indent=2)
    manifest["sets"]["prompts"] = {"source": "hand-written", "count": len(PROMPTS)}

    with open(f"{DATA_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    import json

    m = build.remote()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "data_manifest.json"
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"manifest saved -> {out}")
