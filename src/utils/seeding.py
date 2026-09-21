"""Deterministic seed derivation.

Every stochastic component draws from a generator derived from one root seed and a string path, so
that adding an arm to an experiment cannot change the draws of the arms beside it. That property is
what makes two runs of `validate_instrument.py` comparable after new checks are added to it.
"""
import hashlib

import numpy as np


def derive(root: int, *path: str) -> int:
    """Derive a 63-bit seed from a root seed and a slash-free label path.

    Deterministic across processes and platforms: it hashes the text, so it does not depend on
    Python's salted `hash()`.
    """
    key = f"{int(root)}:" + "/".join(str(p) for p in path)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def rng(root: int, *path: str) -> np.random.Generator:
    """A `np.random.Generator` for one named stream."""
    return np.random.default_rng(derive(root, *path))
