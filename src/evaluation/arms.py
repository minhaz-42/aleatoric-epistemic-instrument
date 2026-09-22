"""Experiment cores that more than one script needs to run identically.

A measurement that two scripts perform is one experiment, so it has one implementation. The prefix
arm is reported at a single seed by `run_prefix.py` and re-run across seeds by
`run_robustness.py`; keeping the computation here is what makes the second a replication of the
first rather than a second experiment that resembles it.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..baselines.predictors import LogitNoise
from ..forecasting.task import evidence_of
from ..forecasting.world import entropy, kl
from ..temporal.prefix import nested_scene, prefix_sweep
from ..uncertainty.decomposition import ensemble_split

LENGTHS = (1, 2, 4, 8, 16, 32)
BASES = (1, 2, 4, 8)
CHEAP_SIGMA, EXPENSIVE_SIGMA = 1.20, 0.35


def prefix_arm(base, world_key: str, seed: int, n_scenes: int, n_members: int,
               reveal=None, bases=BASES, lengths=LENGTHS):
    """The prefix experiment: two actions on the same forecast, at several prefix lengths.

    Returns `(frame, sweep)`. One row per (base length, scene), carrying the cheap forecaster's
    uncertainty signals and the true gain from each of the two actions available to it -- escalate
    the model, or extend the prefix. `sweep` is the exact floor by length, enumerated.
    """
    from ..utils.seeding import rng

    sweep = prefix_sweep(base, lengths, reveal)
    r = rng(seed, "prefix", world_key)
    scenes = [nested_scene(base, r, i, lengths, reveal) for i in range(n_scenes)]

    rows = []
    for L in bases:
        Lx = L * 2
        wL, wX = scenes[0][L][0], scenes[0][Lx][0]
        cheap = LogitNoise(wL, CHEAP_SIGMA, rho=0.3, seed=seed)
        expens = LogitNoise(wL, EXPENSIVE_SIGMA, rho=0.3, seed=seed + 1)
        cheapX = LogitNoise(wX, CHEAP_SIGMA, rho=0.3, seed=seed)

        for sc in scenes:
            _, s = sc[L]
            _, sx = sc[Lx]
            ev, evx = evidence_of(s, wL), evidence_of(sx, wX)
            qc = ensemble_split(np.stack([cheap.predict(ev, m) for m in range(n_members)]))
            qe = ensemble_split(np.stack([expens.predict(ev, m) for m in range(n_members)]))
            qx = ensemble_split(np.stack([cheapX.predict(evx, m) for m in range(n_members)]))

            a_L, a_X = entropy(s.posterior), entropy(sx.posterior)
            k_c = kl(s.posterior, qc.mean_prediction)
            k_e = kl(s.posterior, qe.mean_prediction)
            k_x = kl(sx.posterior, qx.mean_prediction)
            rows.append({
                "world": world_key, "base_L": L, "ext_L": Lx, "scene_id": s.scene_id,
                "aleatoric_L": a_L, "aleatoric_ext": a_X,
                "total_hat": qc.total, "epistemic_hat": qc.epistemic_hat,
                # the two gains, same cheap forecaster, two different actions
                "gain_model": k_c - k_e,
                "gain_prefix": (a_L + k_c) - (a_X + k_x),
                "gain_prefix_aleatoric_part": a_L - a_X,
            })
    return pd.DataFrame(rows), sweep


def prefix_reduce(frame: pd.DataFrame, sweep: dict, bases=BASES) -> list:
    """One record per base length: the mean gains, their shares, and the two correlations."""
    out = []
    for L in bases:
        g = frame[frame.base_L == L]
        rec = {
            "base_L": int(L), "ext_L": int(L * 2),
            "exact_floor_L": sweep[L], "exact_floor_ext": sweep[L * 2],
            "mean_gain_model": float(g.gain_model.mean()),
            "mean_gain_prefix": float(g.gain_prefix.mean()),
            "mean_gain_prefix_aleatoric_part": float(g.gain_prefix_aleatoric_part.mean()),
            "corr_total_model": float(spearmanr(g.total_hat, g.gain_model).statistic),
            "corr_total_prefix": float(spearmanr(g.total_hat, g.gain_prefix).statistic),
            "corr_bald_model": float(spearmanr(g.epistemic_hat, g.gain_model).statistic),
            "corr_bald_prefix": float(spearmanr(g.epistemic_hat, g.gain_prefix).statistic),
        }
        rec["aleatoric_share"] = (rec["mean_gain_prefix_aleatoric_part"] / rec["mean_gain_prefix"]
                                  if abs(rec["mean_gain_prefix"]) > 1e-12 else float("nan"))
        out.append(rec)
    return out
