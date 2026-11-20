"""Step 1: download the baseline model into the Modal Volume (CPU only, no GPU cost).

Run from E:\\quantt\\fp_8:
    modal run download_model.py
"""
import modal

import config
from modal_app import app, quant_image, volume, VOLUMES


@app.function(image=quant_image, volumes=VOLUMES, timeout=1800)
def download() -> dict:
    import json
    import os

    from huggingface_hub import snapshot_download

    dest = config.FORMATS["bf16"]
    os.makedirs(dest, exist_ok=True)

    snapshot_download(
        repo_id=config.MODEL_ID,
        local_dir=dest,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "merges.txt"],
    )
    volume.commit()  # persist to the Volume

    files = {}
    for name in sorted(os.listdir(dest)):
        path = os.path.join(dest, name)
        if os.path.isfile(path):
            files[name] = os.path.getsize(path)

    with open(os.path.join(dest, "config.json")) as f:
        cfg = json.load(f)

    print(f"\nSaved to      : {dest}")
    for name, size in files.items():
        print(f"  {name:<28} {size / 1e6:>10.2f} MB")
    print(f"total         : {sum(files.values()) / 1e6:.2f} MB")
    print(f"stored dtype  : {cfg.get('torch_dtype') or cfg.get('dtype')}")
    print(f"architecture  : {cfg.get('architectures')}")
    print(f"layers/hidden : {cfg.get('num_hidden_layers')} / {cfg.get('hidden_size')}")
    print(f"tied embeds   : {cfg.get('tie_word_embeddings')}")
    return files


@app.local_entrypoint()
def main() -> None:
    download.remote()
