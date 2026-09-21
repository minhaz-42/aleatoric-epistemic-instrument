#!/usr/bin/env python3
"""The headline experiment: what a routing policy is allowed to see, and what that costs.

Two sweeps, each isolating one variable.

TAU sweep, at fixed ensemble correlation. The world's temperature sets the irreducible floor
without touching how wrong either forecaster is. The prediction is that ranking by total predictive
uncertainty degrades as the floor rises, because the floor is a growing share of a quantity that is
by construction irrelevant to the escalation decision, while ranking by epistemic uncertainty does
not.

RHO sweep, at fixed world. Ensemble members go from independent to identical. The prediction is
that the epistemic policy's advantage is not free: it depends on an estimate that V6 shows collapses
as members correlate, so it should lose its advantage exactly where a real ensemble of
similarly-pretrained models sits.

    python3 scripts/run_routing.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.predictors import LogitNoise
from src.evaluation.protocol import score_pair
from src.forecasting.scenarios import TAU_SWEEP, junction_world
from src.routing.curves import DEFAULT_BUDGETS, efficiency, gain_correlation, skill
from src.routing.policies import standard_policies
from src.temporal.splits import assert_disjoint, split_scenes
from src.utils.io import write_records, write_summary
from src.utils.seeding import rng

SEED = 20260921
RHO_SWEEP = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
CHEAP_SIGMA, EXPENSIVE_SIGMA = 1.20, 0.35


def one_setting(world, scenes, rho, members):
    """Score every policy in one (world, correlation) setting.

    Reports two numbers per policy. `corr` is the Spearman correlation between the policy's score
    and the true escalation gain -- the diagnostic, with a clean zero and a clean sign. `skill` is
    the budgeted efficiency rescaled so random routing is 0 and the oracle is 1 -- the
    deployment-facing number. They can disagree at the margin because one is rank-only and the
    other is budget-weighted, and both are kept.
    """
    cheap = LogitNoise(world, CHEAP_SIGMA, rho=rho, seed=SEED)
    expensive = LogitNoise(world, EXPENSIVE_SIGMA, rho=rho, seed=SEED + 1)
    rows = score_pair(world, scenes, cheap, expensive, n_members=members)
    raw = {}
    for p in standard_policies(seed=SEED):
        sc = p.scores(rows)
        e = efficiency(rows, sc, DEFAULT_BUDGETS)
        e["corr"] = gain_correlation(rows, sc)
        raw[p.name] = e
    auc_random = raw["random"]["auc_efficiency"]
    for name, e in raw.items():
        e["skill"] = skill(e["auc_efficiency"], auc_random)
    return rows, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=4000)
    ap.add_argument("--members", type=int, default=12)
    ap.add_argument("--rho", type=float, default=0.30, help="ensemble correlation for the tau sweep")
    ap.add_argument("--tau", type=float, default=0.90, help="world temperature for the rho sweep")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    records, summary = [], {"tau_sweep": [], "rho_sweep": [],
                            "cheap_sigma": CHEAP_SIGMA, "expensive_sigma": EXPENSIVE_SIGMA,
                            "n_members": args.members, "seed": args.seed}
    policies = [p.name for p in standard_policies()]

    # ---- sweep 1: the aleatoric floor ---------------------------------------------------
    print(f"TAU SWEEP  (ensemble correlation fixed at rho={args.rho})")
    print("Each cell is  corr / skill.  corr is the Spearman correlation between the policy's")
    print("score and the true escalation gain. skill is the budgeted efficiency rescaled so that")
    print("random routing scores 0 and the oracle scores 1. Negative means worse than random.\n")
    head = f"{'tau':>5} {'E[H(P*)]':>9} " + " ".join(f"{p[:14]:>14}" for p in policies
                                                      if p not in ("oracle_gain", "random"))
    print(head); print("-" * len(head))
    for tau in TAU_SWEEP:
        world = junction_world(tau)
        scenes = split_scenes(world.sample_scenes(rng(args.seed, "routing", f"tau{tau}"),
                                                  args.scenes))["test"]
        _, effs = one_setting(world, scenes, args.rho, args.members)
        row = {"tau": tau, "expected_aleatoric": world.expected_aleatoric(),
               "n_test": len(scenes), "rho": args.rho}
        for name, e in effs.items():
            row[f"auc_{name}"] = e["auc_efficiency"]
            row[f"skill_{name}"] = e["skill"]
            row[f"corr_{name}"] = e["corr"]
        row["loss_cheap"] = effs["random"]["loss_all_cheap"]
        row["loss_expensive"] = effs["random"]["loss_all_expensive"]
        summary["tau_sweep"].append(row)
        records.append({**row, "sweep": "tau"})
        print(f"{tau:5.2f} {row['expected_aleatoric']:9.4f} " +
              " ".join(f"{row['corr_' + p]:+7.3f}/{row['skill_' + p]:+6.2f}"
                       for p in policies if p not in ("oracle_gain", "random")))

    # ---- sweep 2: ensemble correlation --------------------------------------------------
    print(f"\nRHO SWEEP  (world fixed at tau={args.tau}, E[H(P*)]="
          f"{junction_world(args.tau).expected_aleatoric():.4f})\n")
    head = f"{'rho':>5} " + " ".join(f"{p[:14]:>14}" for p in policies
                                      if p not in ("oracle_gain", "random"))
    print(head); print("-" * len(head))
    world = junction_world(args.tau)
    scenes = split_scenes(world.sample_scenes(rng(args.seed, "routing", "rho"), args.scenes))["test"]
    for rho in RHO_SWEEP:
        _, effs = one_setting(world, scenes, rho, args.members)
        row = {"rho": rho, "tau": args.tau, "n_test": len(scenes)}
        for name, e in effs.items():
            row[f"auc_{name}"] = e["auc_efficiency"]
            row[f"skill_{name}"] = e["skill"]
            row[f"corr_{name}"] = e["corr"]
        summary["rho_sweep"].append(row)
        records.append({**row, "sweep": "rho"})
        print(f"{rho:5.2f} " + " ".join(f"{row['corr_' + p]:+7.3f}/{row['skill_' + p]:+6.2f}"
                                        for p in policies if p not in ("oracle_gain", "random")))

    write_records("routing", pd.DataFrame(records))
    write_summary("routing", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
