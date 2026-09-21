"""Content-addressed response cache.

Keyed by everything that could change a response: model, elicitation mode, sampling temperature,
member index, the prompt text and the bytes of the image. A cache hit is therefore safe by
construction -- there is no way to change the question and silently read back the old answer.

This exists because the VLM arm is the only part of the project that costs money, and a re-analysis
of a completed run must not re-pay for it.
"""
import hashlib
import json
import os

from ..utils.io import ROOT

CACHE_DIR = os.path.join(ROOT, "experiments", "logs", "vlm_cache")


def key(model: str, elicitation: str, temperature: float, member: int,
        prompt: str, image_bytes: bytes = b"") -> str:
    h = hashlib.sha256()
    for part in (model, elicitation, f"{temperature:.4f}", str(member), prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    h.update(image_bytes)
    return h.hexdigest()


def get(k: str):
    path = os.path.join(CACHE_DIR, k[:2], f"{k}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def put(k: str, payload: dict) -> None:
    path = os.path.join(CACHE_DIR, k[:2], f"{k}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh)
