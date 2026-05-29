"""Resilient MuseTalk weight puller.

The download_weights.bat -> huggingface-cli flow keeps aborting partway
through medium-sized files on this connection. snapshot_download with
explicit retries + resume picks up where it left off.
"""
from __future__ import annotations

import os
import sys
import time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")

BUNDLES = [
    # (repo_id, local subdir, allow_patterns or None)
    ("TMElyralab/MuseTalk",            MODELS,                            None),
    ("stabilityai/sd-vae-ft-mse",       os.path.join(MODELS, "sd-vae"),     ["config.json", "diffusion_pytorch_model.bin"]),
    ("openai/whisper-tiny",             os.path.join(MODELS, "whisper"),    ["config.json", "pytorch_model.bin", "preprocessor_config.json"]),
    ("yzd-v/DWPose",                    os.path.join(MODELS, "dwpose"),     ["dw-ll_ucoco_384.pth"]),
    ("ByteDance/LatentSync",            os.path.join(MODELS, "syncnet"),    ["latentsync_syncnet.pt"]),
    ("ManyOtherFunctions/face-parse-bisent", os.path.join(MODELS, "face-parse-bisent"), ["79999_iter.pth", "resnet18-5c106cde.pth"]),
]

MAX_RETRIES = 8

def pull(repo_id: str, local_dir: str, allow_patterns):
    os.makedirs(local_dir, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[{repo_id}] attempt {attempt}/{MAX_RETRIES} -> {local_dir}", flush=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                max_workers=2,
                etag_timeout=60,
            )
            print(f"[{repo_id}] OK", flush=True)
            return True
        except (LocalEntryNotFoundError, Exception) as exc:
            wait = min(2 ** attempt, 30)
            print(f"[{repo_id}] FAIL attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            if attempt == MAX_RETRIES:
                print(f"[{repo_id}] GIVING UP after {MAX_RETRIES} attempts", flush=True)
                return False
            print(f"[{repo_id}] retrying in {wait}s", flush=True)
            time.sleep(wait)
    return False

def main():
    failed = []
    for repo_id, local_dir, allow in BUNDLES:
        ok = pull(repo_id, local_dir, allow)
        if not ok:
            failed.append(repo_id)
    print("=== SUMMARY ===", flush=True)
    print(f"OK: {len(BUNDLES) - len(failed)} / {len(BUNDLES)}", flush=True)
    if failed:
        print(f"FAILED: {failed}", flush=True)
        sys.exit(1)
    print("ALL WEIGHTS DOWNLOADED", flush=True)

if __name__ == "__main__":
    main()
