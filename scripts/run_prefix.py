#!/usr/bin/env python3
"""Claim G: the irreducible term cancels for model choice but NOT for prefix choice.

An uncertain forecast admits two responses. Spend more on the model, or spend more on the evidence.
The project's central identity says the aleatoric term cancels out of the first decision:

    gain(escalate model)   =  KL(P*_L || Q_c) − KL(P*_L || Q_e)          H(P*_L) cancels

It does not cancel out of the second, because a longer prefix changes P* itself:

    gain(extend prefix)    =  [H(P*_L) + KL(P*_L || Q)] − [H(P*_L') + KL(P*_L' || Q)]

The leading term there is H(P*_L) − H(P*_L'), which is exactly the irreducible entropy that carries
no information about model escalation. So the quantity that confounds one decision is genuine signal
for the other, and a system that ranks both by the same confidence score has to be wrong about one
of them.

This is the only measurement in the project that is specific to video rather than to classifiers in
general: an image classifier has no prefix to extend.

Masks are nested across lengths (`temporal.prefix.nested_scene`) — a longer prefix reveals a
superset, because evidence does not disappear as you keep watching.

    python3 scripts/run_prefix.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.predictors import LogitNoise
from src.forecasting.task import evidence_of
from src.forecasting.world import entropy, kl
from src.temporal.prefix import nested_scene, prefix_sweep
from src.uncertainty.decomposition import ensemble_split
from src.utils.io import write_records, write_summary
from src.utils.seeding import rng

SEED = 20260921
LENGTHS = (1, 2, 4, 8, 16, 32)
BASES = (1, 2, 4, 8)
CHEAP_SIGMA, EXPENSIVE_SIGMA = 1.20, 0.35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=3000)
    ap.add_argument("--members", type=int, default=12)
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--world", default="junction", choices=["junction", "kitchen"])
    args = ap.parse_args()

    from src.forecasting.scenarios import KITCHEN_REVEAL, junction_world, kitchen_world
    base = {"junction": junction_world, "kitchen": kitchen_world}[args.world](args.tau)
    reveal = KITCHEN_REVEAL if args.world == "kitchen" else None

    print(f"world {base.spec.name}   scenes {args.scenes}   members {args.members}\n")
    print("Exact irreducible floor by prefix length (enumerated, not sampled):")
    sweep = prefix_sweep(base, LENGTHS, reveal)
    for L in LENGTHS:
        print(f"   L={L:3d}   E[H(P*)] = {sweep[L]:.4f} nats")

    r = rng(args.seed, "prefix", args.world)
    scenes = [nested_scene(base, r, i, LENGTHS, reveal) for i in range(args.scenes)]

    rows = []
    for L in BASES:
        Lx = L * 2
        wL, wX = scenes[0][L][0], scenes[0][Lx][0]
        cheap = LogitNoise(wL, CHEAP_SIGMA, rho=0.3, seed=args.seed)
        expens = LogitNoise(wL, EXPENSIVE_SIGMA, rho=0.3, seed=args.seed + 1)
        cheapX = LogitNoise(wX, CHEAP_SIGMA, rho=0.3, seed=args.seed)

        for sc in scenes:
            _, s = sc[L]
            _, sx = sc[Lx]
            ev, evx = evidence_of(s, wL), evidence_of(sx, wX)
            qc = ensemble_split(np.stack([cheap.predict(ev, m) for m in range(args.members)]))
            qe = ensemble_split(np.stack([expens.predict(ev, m) for m in range(args.members)]))
            qx = ensemble_split(np.stack([cheapX.predict(evx, m) for m in range(args.members)]))

            a_L, a_X = entropy(s.posterior), entropy(sx.posterior)
            k_c = kl(s.posterior, qc.mean_prediction)
            k_e = kl(s.posterior, qe.mean_prediction)
            k_x = kl(sx.posterior, qx.mean_prediction)
            rows.append({
                "world": args.world, "base_L": L, "ext_L": Lx, "scene_id": s.scene_id,
                "aleatoric_L": a_L, "aleatoric_ext": a_X,
                "total_hat": qc.total, "epistemic_hat": qc.epistemic_hat,
                # the two gains, same cheap forecaster, two different actions
                "gain_model": k_c - k_e,
                "gain_prefix": (a_L + k_c) - (a_X + k_x),
                "gain_prefix_aleatoric_part": a_L - a_X,
            })

    frame = pd.DataFrame(rows)
    summary = []
    print("\nWhich uncertainty signal tracks which decision?")
    print("corr(score, gain), Spearman. The same cheap forecaster, two different actions.\n")
    head = (f"{'base L':>7} {'→ L':>5} {'E[H(P*)]':>9} "
            f"{'total→model':>12} {'total→prefix':>13} {'BALD→model':>11} {'BALD→prefix':>12}")
    print(head); print("-" * len(head))
    for L in BASES:
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
        summary.append(rec)
        print(f"{L:7d} {L*2:5d} {sweep[L]:9.4f} "
              f"{rec['corr_total_model']:+12.3f} {rec['corr_total_prefix']:+13.3f} "
              f"{rec['corr_bald_model']:+11.3f} {rec['corr_bald_prefix']:+12.3f}")

    print("\nHow much of the prefix-extension gain is the irreducible term that cancels for models?")
    for rec in summary:
        share = (rec["mean_gain_prefix_aleatoric_part"] / rec["mean_gain_prefix"]
                 if abs(rec["mean_gain_prefix"]) > 1e-12 else float("nan"))
        print(f"   L={rec['base_L']:2d}→{rec['ext_L']:2d}: prefix gain {rec['mean_gain_prefix']:.4f} nats, "
              f"of which {rec['mean_gain_prefix_aleatoric_part']:.4f} ({share:.0%}) is ΔH(P*)")

    tag = "prefix" if args.world == "junction" else f"prefix_{args.world}"
    write_records(tag, frame)
    write_summary(tag, {"world": base.spec.name, "seed": args.seed,
                             "n_scenes": args.scenes, "n_members": args.members,
                             "exact_floor_sweep": {int(k): v for k, v in sweep.items()},
                             "rows": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
