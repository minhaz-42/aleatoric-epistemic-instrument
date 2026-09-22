#!/usr/bin/env python3
"""Does confidence-based routing fail the same way whatever makes the cheap model worse?

It does not, and this is the experiment that qualifies the headline.

`run_routing.py` sweeps the aleatoric floor with one cheap/expensive pair and finds that ranking by
total predictive uncertainty inverts -- passing zero and becoming worse than random as the floor
rises. Taken alone that invites the reading that the floor is the whole story. It is not. The
inversion depends on the relationship between the cheap forecaster's confidence and its error, and
that relationship is a property of WHY the cheap forecaster is worse:

  missing evidence    the cheap model cannot see a cue. Its uncertainty is honest -- it is uncertain
                      where it lacks evidence, and that is also where a better model gains. The
                      correlation decays toward zero as the floor rises but does not invert.
  injected noise      the cheap model reads the evidence unreliably. Its confidence is unrelated to
                      where it errs, and the floor dominates: the correlation inverts.
  miscalibration      the cheap model is overconfident. Its entropy is LOW exactly where it is most
                      wrong, so ranking by high entropy systematically selects the scenes it already
                      handles. This inverts at every floor, and inverts hardest.

The third is the case that matters outside the synthetic world, because overconfidence is the
documented failure mode of neural classifiers rather than a hypothetical one.

    python3 scripts/run_mechanisms.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.mechanisms import SEED, mechanisms
from src.evaluation.protocol import score_pair
from src.forecasting.scenarios import junction_world, kitchen_world
from src.routing.curves import DEFAULT_BUDGETS, efficiency, gain_correlation, skill
from src.routing.policies import standard_policies
from src.temporal.splits import split_scenes
from src.utils.io import write_records, write_summary
from src.utils.seeding import rng

TAUS = (0.35, 0.70, 1.30, 2.00, 3.50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=4000)
    ap.add_argument("--members", type=int, default=12)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--world", default="junction", choices=["junction", "kitchen"])
    args = ap.parse_args()
    make_world = {"junction": junction_world, "kitchen": kitchen_world}[args.world]
    MECHANISMS = mechanisms(args.seed)

    records, summary = [], []
    print("corr(score, true gain) for each routing policy, by why the cheap forecaster is worse.")
    print("Negative means the policy is worse than routing at random.")
    print("nan for epistemic_bald means the cheap forecaster is deterministic: a single model has")
    print("no ensemble spread, so the estimate is identically zero and induces no ranking at all.")
    print("That is the most common deployment, and it is why `evidence_ensemble` is included.\n")

    for mech, (desc, make) in MECHANISMS.items():
        print(f"{mech}  --  {desc}")
        head = f"  {'tau':>5} {'E[H(P*)]':>9} " + " ".join(
            f"{p.name[:14]:>14}" for p in standard_policies() if p.name != "oracle_gain")
        print(head); print("  " + "-" * (len(head) - 2))
        for tau in TAUS:
            world = make_world(tau)
            scenes = split_scenes(world.sample_scenes(
                rng(args.seed, "mechanisms", args.world, mech, f"tau{tau}"), args.scenes))["test"]
            cheap, expensive = make(world)
            n = args.members if (cheap.stochastic or expensive.stochastic) else 1
            rows = score_pair(world, scenes, cheap, expensive, n_members=n)

            row = {"world": args.world, "mechanism": mech, "tau": tau, "n_members": n,
                   "expected_aleatoric": world.expected_aleatoric(),
                   "cheap": cheap.name, "expensive": expensive.name,
                   "mean_gain": float(np.mean([r["epistemic_true"]
                                               - r["epistemic_true_expensive"] for r in rows]))}
            effs = {}
            for p in standard_policies(seed=args.seed):
                sc = p.scores(rows)
                e = efficiency(rows, sc, DEFAULT_BUDGETS)
                e["corr"] = gain_correlation(rows, sc)
                effs[p.name] = e
            for name, e in effs.items():
                row[f"corr_{name}"] = e["corr"]
                row[f"skill_{name}"] = skill(e["auc_efficiency"],
                                             effs["random"]["auc_efficiency"])
            records.append(row)
            summary.append(row)
            print(f"  {tau:5.2f} {row['expected_aleatoric']:9.4f} " + " ".join(
                f"{row['corr_' + p.name]:14.3f}" for p in standard_policies()
                if p.name != "oracle_gain"))
        print()

    tag = "mechanisms" if args.world == "junction" else f"mechanisms_{args.world}"
    write_records(tag, pd.DataFrame(records))
    write_summary(tag, {"world": args.world, "seed": args.seed, "taus": list(TAUS),
                        "mechanisms": {k: v[0] for k, v in MECHANISMS.items()},
                        "rows": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
