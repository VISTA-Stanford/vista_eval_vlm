"""Find the largest request the Stanford Health Care AI Hub lets through from this machine.

Synthetic images only (random noise JPEGs, no patient data). Each request carries 50 images
(Azure's per-request limit) at a growing size, so the total body runs from ~50 KB to ~20 MB;
images cost the same ~300 tokens whatever their bytes, so the probe is cheap. A reply that is
the firewall's "Request Rejected" page (HTTP 200, HTML) means the gateway blocked it before the
model; a JSON reply means it reached the model. From phil-sllm-01 (GCP) the block starts at
~48 KB (10-15). Prints one line per size and the largest that passed.

    .venv/bin/python scripts/aihub_body_cap_probe.py [--model gpt-6-astra]
Pass: the line "largest passing request" names a size of at least 5 MB (a 25 + 25 slice pair is ~5 MB).
"""

import argparse
import io
import socket
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.aihub_openai import AIHubOpenAIAdapter  # noqa: E402

CHOICES = ["Stable", "Progression", "Response", "Mixed Response"]


def noise_jpeg(side, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (side, side, 3), dtype=np.uint8), "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--key-file", default="~/.config/vista/aihub.env")
    args = ap.parse_args()
    print(f"host {socket.gethostname()}")
    a = AIHubOpenAIAdapter(args.model, cache_dir="/tmp/aihub_probe_cache_unused")
    a.configure({"key_file": args.key_file, "max_retries": 1})
    a.load()
    best = 0
    # First a small request that must pass (proves the probe and key work), then 50 images growing.
    for n, side in ((5, 8), (50, 8), (50, 24), (50, 64), (50, 128), (50, 256), (50, 512)):
        imgs = [noise_jpeg(side, i) for i in range(n)]
        a.response_cache = Path(f"/tmp/aihub_probe_nocache_{side}_{np.random.randint(1e9)}")  # always a real call
        out = a._one({"question": "Synthetic noise test images, not medical. Pick any option.", "images": imgs}, 200, CHOICES)
        size = (out.get("usage") or {}).get("request_bytes")
        ok = not out.get("error")
        print(f"{n} images at {side}px: request {size:,} bytes -> {'reached the model' if ok else out['error'][:120]}")
        if ok:
            best = max(best, size)
        else:
            break
    if not best:
        print("even the small first request failed: the key or the route is broken, not the size")
    print(f"largest passing request: {best / 1e6:.3f} MB")
    sys.exit(0 if best >= 5e6 else 1)


if __name__ == "__main__":
    main()
