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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.arms import BASES, LENGTHS, prefix_arm, prefix_reduce
from src.utils.io import write_records, write_summary

SEED = 20260921


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
    frame, sweep = prefix_arm(base, args.world, args.seed, args.scenes, args.members, reveal)

    print("Exact irreducible floor by prefix length (enumerated, not sampled):")
    for L in LENGTHS:
        print(f"   L={L:3d}   E[H(P*)] = {sweep[L]:.4f} nats")

    summary = prefix_reduce(frame, sweep, BASES)
    print("\nWhich uncertainty signal tracks which decision?")
    print("corr(score, gain), Spearman. The same cheap forecaster, two different actions.\n")
    head = (f"{'base L':>7} {'→ L':>5} {'E[H(P*)]':>9} "
            f"{'total→model':>12} {'total→prefix':>13} {'BALD→model':>11} {'BALD→prefix':>12}")
    print(head); print("-" * len(head))
    for rec in summary:
        print(f"{rec['base_L']:7d} {rec['ext_L']:5d} {rec['exact_floor_L']:9.4f} "
              f"{rec['corr_total_model']:+12.3f} {rec['corr_total_prefix']:+13.3f} "
              f"{rec['corr_bald_model']:+11.3f} {rec['corr_bald_prefix']:+12.3f}")

    print("\nHow much of the prefix-extension gain is the irreducible term that cancels for models?")
    for rec in summary:
        share = rec["aleatoric_share"]
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
