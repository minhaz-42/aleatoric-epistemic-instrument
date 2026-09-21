#!/usr/bin/env python3
"""How well the computable decomposition recovers the exact one, per forecaster.

For each forecaster in the standard suite this reports, side by side, the two terms that can only
be separated in a synthetic world: the exact aleatoric floor H(P*) and epistemic error KL(P*||Q),
and the ensemble estimates that stand in for them on real data. The question it answers is not
"which forecaster is best" -- the ranking is fixed by construction -- but "how far off is the
estimate, and in which direction".

    python3 scripts/run_decomposition.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.predictors import standard_suite
from src.evaluation.metrics import accuracy, brier, ece, nll
from src.evaluation.protocol import score
from src.forecasting.scenarios import junction_world
from src.temporal.splits import assert_disjoint, split_scenes
from src.utils.io import write_records, write_summary
from src.utils.seeding import rng

SEED = 20260921


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=4000)
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--members", type=int, default=12)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    world = junction_world(args.tau)
    scenes = world.sample_scenes(rng(args.seed, "decomposition"), args.scenes)
    splits = split_scenes(scenes)
    assert_disjoint(splits)
    test = splits["test"]

    print(f"world {world.spec.name}   test scenes {len(test)}   members {args.members}")
    print(f"exact E[H(P*)] = {world.expected_aleatoric():.4f} nats\n")

    all_rows, summary = [], []
    hdr = (f"{'forecaster':30s} {'alea_true':>9} {'alea_hat':>9} {'epi_true':>9} "
           f"{'epi_hat':>9} {'NLL':>7} {'acc':>6} {'ECE':>6}")
    print(hdr); print("-" * len(hdr))

    for f in standard_suite(world):
        n = args.members if f.stochastic else 1
        rows = score(world, test, f, n_members=n)
        probs = np.stack([r.pop("_q") for r in rows])
        labels = np.array([r["y"] for r in rows])
        rec = {
            "forecaster": f.name, "stochastic": f.stochastic, "n_members": n,
            "aleatoric_true": float(np.mean([r["aleatoric_true"] for r in rows])),
            "aleatoric_hat": float(np.mean([r["aleatoric_hat"] for r in rows])),
            "epistemic_true": float(np.mean([r["epistemic_true"] for r in rows])),
            "epistemic_hat": float(np.mean([r["epistemic_hat"] for r in rows])),
            "total_hat": float(np.mean([r["total_hat"] for r in rows])),
            "expected_nll": float(np.mean([r["expected_nll"] for r in rows])),
            "realised_nll": nll(probs, labels), "brier": brier(probs, labels),
            "accuracy": accuracy(probs, labels), "ece": ece(probs, labels),
        }
        summary.append(rec)
        all_rows.extend(rows)
        print(f"{f.name:30s} {rec['aleatoric_true']:9.4f} {rec['aleatoric_hat']:9.4f} "
              f"{rec['epistemic_true']:9.4f} {rec['epistemic_hat']:9.4f} "
              f"{rec['realised_nll']:7.4f} {rec['accuracy']:6.3f} {rec['ece']:6.3f}")

    frame = pd.DataFrame(all_rows)
    write_records("decomposition", frame)
    write_summary("decomposition", {
        "world": world.spec.name, "seed": args.seed, "n_test": len(test),
        "n_members": args.members, "expected_aleatoric_exact": world.expected_aleatoric(),
        "forecasters": summary,
    })

    print("\nReading of the aleatoric column: alea_true is a property of the world and the prefix,")
    print("so it is identical for every forecaster by construction. alea_hat is the ensemble")
    print("estimate of it, and the gap between the two columns is the estimate's bias.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
