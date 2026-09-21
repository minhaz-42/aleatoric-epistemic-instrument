"""Scene splits, and the leakage guard that makes them worth having.

Three disjoint splits. `fit` is for anything that estimates a parameter from data -- a calibration
temperature, a routing threshold. `calib` is for choosing a budget or an operating point. `test`
carries every reported number and is touched once.

The split is by scene identity, assigned by hashing the scene id, so it is stable under changes to
the sampling order and under changes to how many scenes are drawn. Adding scenes to a run does not
move existing scenes between splits, which is what lets a run be extended without invalidating a
threshold fitted on an earlier one.
"""
import hashlib

import numpy as np

SPLITS = ("fit", "calib", "test")
DEFAULT_WEIGHTS = (0.3, 0.2, 0.5)


def assign(scene_id: int, weights=DEFAULT_WEIGHTS, salt: str = "predictive-evidence") -> str:
    """Stable split assignment for one scene."""
    h = hashlib.sha256(f"{salt}:{int(scene_id)}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2 ** 64
    edges = np.cumsum(np.asarray(weights, dtype=float) / np.sum(weights))
    return SPLITS[int(np.searchsorted(edges, u, side="right"))]


def split_scenes(scenes, weights=DEFAULT_WEIGHTS) -> dict:
    out = {s: [] for s in SPLITS}
    for sc in scenes:
        out[assign(sc.scene_id, weights)].append(sc)
    return out


def assert_disjoint(split_map: dict) -> None:
    """Refuse to proceed if any scene appears in two splits.

    Cheap, and it runs in the protocol rather than in a test, because the failure it catches is one
    that silently improves every number rather than breaking anything.
    """
    seen = {}
    for name, scenes in split_map.items():
        for sc in scenes:
            if sc.scene_id in seen:
                raise AssertionError(
                    f"scene {sc.scene_id} appears in both {seen[sc.scene_id]!r} and {name!r}")
            seen[sc.scene_id] = name
