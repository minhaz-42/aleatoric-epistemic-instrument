#!/usr/bin/env python3
"""How much of every reported number is the seed?

Each result elsewhere in this project is one run at one seed, and each headline claim is a claim
about a SIGN -- that BALD collapses while the error it estimates rises, that confidence routing
inverts under miscalibration and does not under missing evidence, that almost all of the
prefix-extension gain is the term which cancels from model escalation. A sign claim from a single
run is not yet a claim, so this script re-runs the three claim families across seeds and reports
the spread, and separately bootstraps the published run to separate scene-sampling noise from
seed-to-seed variation.

Two sources of variation, deliberately kept apart:

  seeds       re-draws the scenes AND the forecasters' noise streams. The honest replication, and
              the number to quote. Reported as mean +/- sd over `--seeds` runs, with the fraction
              of runs whose sign agrees with the published one.
  bootstrap   resamples the scenes of the published run only. It answers the narrower question of
              how precisely that one run's correlation is pinned, and it is the interval to quote
              when the claim is about a single reported figure.

Every claim family also runs in both worlds, so the table says whether a sign survives re-seeding
and re-worlding at once.

    python3 scripts/run_robustness.py               # ~7 min, CPU only
    python3 scripts/run_robustness.py --seeds 3 --bootstrap 200    # ~4 min, for a quick check
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.arms import BASES, prefix_arm, prefix_reduce
from src.evaluation.mechanisms import mechanisms
from src.evaluation.protocol import score_pair, v6_ensemble_confound
from src.forecasting.scenarios import KITCHEN_REVEAL, junction_world, kitchen_world
from src.routing.curves import gain_correlation
from src.routing.policies import EpistemicBALD, TotalUncertainty
from src.temporal.splits import split_scenes
from src.utils.io import TABLES, write_records, write_summary
from src.utils.seeding import rng

SEED = 20260921
TAU_LO, TAU_HI = 0.35, 3.50
WORLDS = {"junction": (junction_world, 0.90, None),
          "kitchen": (kitchen_world, 0.70, KITCHEN_REVEAL)}


def spearman(a, b) -> float:
    """Spearman by ranking then correlating. Same value as `scipy.stats.spearmanr`, fast enough to
    bootstrap a thousand times per setting."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def boot_ci(score, gain, n_boot: int, seed: int, level: float = 0.95):
    """Percentile bootstrap interval for one Spearman correlation, resampling scenes.

    The scenes are the unit of resampling because they are the unit that was sampled: the ensemble
    members are not a sample from a population, they are part of the forecaster.
    """
    if n_boot <= 0:
        return (float("nan"), float("nan"))
    r = rng(seed, "bootstrap")
    n = len(score)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = r.integers(0, n, n)
        vals[i] = spearman(np.asarray(score)[idx], np.asarray(gain)[idx])
    lo, hi = np.nanpercentile(vals, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def agg(values, published=None) -> dict:
    """mean, sd, range, and whether the sign held -- the four numbers a sign claim needs."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    ref = published if published is not None else (v[0] if len(v) else float("nan"))
    out = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else float("nan"),
           "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
           "min": float(v.min()) if len(v) else float("nan"),
           "max": float(v.max()) if len(v) else float("nan")}
    out["sign_agreement"] = (float(np.mean(np.sign(v) == np.sign(ref)))
                             if len(v) and np.isfinite(ref) else float("nan"))
    return out


# -------------------------------------------------------------------------------------------
# family 1: the correlated-ensemble confound. Does the crossing survive re-seeding?
# -------------------------------------------------------------------------------------------
def family_confound(seeds, n_scenes, members) -> list:
    rows = []
    for world_key, (make, tau, _) in WORLDS.items():
        world = make(tau)
        for s in seeds:
            scenes = world.sample_scenes(rng(s, "robustness", "confound", world_key), n_scenes)
            table = v6_ensemble_confound(world, scenes, n_members=members).detail["table"]
            lo, hi = table[0], table[-1]
            cross = next((t["rho"] for t in table
                          if t["epistemic_hat"] <= t["epistemic_true"]), float("nan"))
            rows.append({
                "family": "confound", "world": world_key, "seed": s, "setting": "rho 0 -> 1",
                "single_lo": lo["single_member_kl"], "single_hi": hi["single_member_kl"],
                "true_lo": lo["epistemic_true"], "true_hi": hi["epistemic_true"],
                "bald_lo": lo["epistemic_hat"], "bald_hi": hi["epistemic_hat"],
                "true_rise_factor": hi["epistemic_true"] / max(lo["epistemic_true"], 1e-12),
                "overstate_at_rho0": lo["ratio_hat_over_true"],
                "crossing_rho": cross,
            })
            print(f"  confound {world_key:9s} seed {s}: true x{rows[-1]['true_rise_factor']:5.1f}, "
                  f"BALD {lo['epistemic_hat']:.4f} -> {hi['epistemic_hat']:.1e}, "
                  f"cross at rho={cross}")
    return rows


# -------------------------------------------------------------------------------------------
# family 2: the mechanism map. Does each mechanism's sign survive re-seeding, at both floors?
# -------------------------------------------------------------------------------------------
def family_mechanisms(seeds, n_scenes, members, n_boot) -> list:
    rows = []
    for world_key, (make, _, _) in WORLDS.items():
        for mech in mechanisms():
            for tau in (TAU_LO, TAU_HI):
                for s in seeds:
                    world = make(tau)
                    scenes = split_scenes(world.sample_scenes(
                        rng(s, "mechanisms", world_key, mech, f"tau{tau}"), n_scenes))["test"]
                    cheap, expensive = mechanisms(s)[mech][1](world)
                    n = members if (cheap.stochastic or expensive.stochastic) else 1
                    scored = score_pair(world, scenes, cheap, expensive, n_members=n)
                    gain = np.array([r["epistemic_true"] - r["epistemic_true_expensive"]
                                     for r in scored])
                    total = TotalUncertainty().scores(scored)
                    bald = EpistemicBALD().scores(scored)
                    row = {"family": "mechanism", "world": world_key, "seed": s,
                           "setting": f"{mech}@tau{tau}", "mechanism": mech, "tau": tau,
                           "n_test": len(scored),
                           "corr_total": gain_correlation(scored, total),
                           "corr_bald": gain_correlation(scored, bald)}
                    # the published run is the only one worth bootstrapping: the interval is a
                    # statement about it, not about the family
                    if s == SEED and n_boot:
                        lo, hi = boot_ci(total, gain, n_boot, s)
                        row["corr_total_ci_lo"], row["corr_total_ci_hi"] = lo, hi
                        if np.isfinite(row["corr_bald"]):
                            blo, bhi = boot_ci(bald, gain, n_boot, s + 1)
                            row["corr_bald_ci_lo"], row["corr_bald_ci_hi"] = blo, bhi
                    rows.append(row)
                got = [r for r in rows if r["setting"] == f"{mech}@tau{tau}"
                       and r["world"] == world_key]
                a = agg([r["corr_total"] for r in got], published=got[0]["corr_total"])
                print(f"  mech {world_key:9s} {mech:18s} tau={tau:4.2f}: "
                      f"total {a['mean']:+.3f} +/- {a['sd']:.3f} "
                      f"[{a['min']:+.3f},{a['max']:+.3f}] sign {a['sign_agreement']:.0%}")
    return rows


# -------------------------------------------------------------------------------------------
# family 3: the prefix arm. Does the aleatoric share, and the sign pair, survive re-seeding?
# -------------------------------------------------------------------------------------------
def family_prefix(seeds, n_scenes, members) -> list:
    rows = []
    for world_key, (make, tau, reveal) in WORLDS.items():
        base = make(tau)
        for s in seeds:
            frame, sweep = prefix_arm(base, world_key, s, n_scenes, members, reveal)
            for rec in prefix_reduce(frame, sweep, BASES):
                rows.append({"family": "prefix", "world": world_key, "seed": s,
                             "setting": f"L{rec['base_L']}->{rec['ext_L']}", **rec})
            shares = [r["aleatoric_share"] for r in rows if r["seed"] == s
                      and r["world"] == world_key]
            print(f"  prefix {world_key:9s} seed {s}: aleatoric share "
                  f"{min(shares):.0%}-{max(shares):.0%}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds, starting at the published one")
    ap.add_argument("--bootstrap", type=int, default=1000, help="resamples for the published run's CI")
    ap.add_argument("--scenes", type=int, default=4000)
    ap.add_argument("--prefix-scenes", type=int, default=3000)
    ap.add_argument("--validation-scenes", type=int, default=2000)
    ap.add_argument("--members", type=int, default=12)
    args = ap.parse_args()
    seeds = [SEED + i for i in range(args.seeds)]

    print(f"seeds {seeds}   bootstrap {args.bootstrap} resamples of the published run\n")
    print("family 1: the correlated-ensemble confound")
    rows = family_confound(seeds, args.validation_scenes, args.members)
    print("\nfamily 2: the mechanism map")
    rows += family_mechanisms(seeds, args.scenes, args.members, args.bootstrap)
    print("\nfamily 3: the prefix arm")
    rows += family_prefix(seeds, args.prefix_scenes, args.members)

    frame = pd.DataFrame(rows)
    write_records("robustness", frame)

    # ---- reduce to one record per claim, and say whether the sign held --------------------
    out = []
    for (world, setting), g in frame[frame.family == "confound"].groupby(["world", "setting"]):
        for q in ("true_rise_factor", "bald_hi", "overstate_at_rho0", "crossing_rho"):
            out.append({"family": "confound", "world": world, "setting": setting, "quantity": q,
                        **agg(g[q].tolist(), published=g[g.seed == SEED][q].iloc[0])})
    mech = frame[frame.family == "mechanism"]
    for (world, setting), g in mech.groupby(["world", "setting"]):
        pub = g[g.seed == SEED]
        for q in ("corr_total", "corr_bald"):
            rec = {"family": "mechanism", "world": world, "setting": setting, "quantity": q,
                   "published": float(pub[q].iloc[0]) if np.isfinite(pub[q].iloc[0]) else float("nan"),
                   **agg(g[q].tolist(), published=pub[q].iloc[0])}
            for side in ("lo", "hi"):
                col = f"{q}_ci_{side}"
                if col in pub and np.isfinite(pub[col].iloc[0]):
                    rec[f"ci_{side}"] = float(pub[col].iloc[0])
            out.append(rec)
    pre = frame[frame.family == "prefix"]
    for (world, setting), g in pre.groupby(["world", "setting"]):
        pub = g[g.seed == SEED]
        for q in ("aleatoric_share", "corr_total_model", "corr_total_prefix"):
            out.append({"family": "prefix", "world": world, "setting": setting, "quantity": q,
                        "published": float(pub[q].iloc[0]),
                        **agg(g[q].tolist(), published=pub[q].iloc[0])})

    table = pd.DataFrame(out)
    table.to_csv(os.path.join(TABLES, "robustness.csv"), index=False)

    # ---- the four sentences this script exists to be able to write ------------------------
    def pick(family, quantity, world=None, setting=None):
        m = (table.family == family) & (table.quantity == quantity)
        if world:
            m &= table.world == world
        if setting:
            m &= table.setting == setting
        return table[m]

    print("\n" + "=" * 94)
    print("SEED STABILITY OF THE CLAIMS  (mean +/- sd over "
          f"{len(seeds)} seeds; sign = fraction of seeds agreeing with the published sign)")
    print("=" * 94)
    rise = pick("confound", "true_rise_factor")
    for _, r in rise.iterrows():
        print(f"  confound {r.world:9s}: true error rises x{r['mean']:.1f} +/- {r['sd']:.1f} "
              f"(range {r['min']:.1f}-{r['max']:.1f}); BALD at rho=1 is "
              f"{pick('confound', 'bald_hi', r.world)['max'].iloc[0]:.1e} at worst")
    signs = pick("mechanism", "corr_total")
    print(f"  mechanism map: {int((signs.sign_agreement >= 1.0).sum())}/{len(signs)} "
          "(world, mechanism, floor) cells keep the published sign at every seed")
    worst = signs.loc[signs.sign_agreement.idxmin()]
    print(f"                 weakest cell: {worst.world}/{worst.setting} at "
          f"{worst['sign_agreement']:.0%} ({worst['mean']:+.3f} +/- {worst['sd']:.3f})")
    bald = pick("mechanism", "corr_bald")
    bald = bald[bald.setting.str.startswith("evidence_ensemble")]
    print(f"  BALD on the well-posed ensemble: {bald['mean'].min():+.3f} to {bald['mean'].max():+.3f} "
          f"across worlds and floors, sd <= {bald['sd'].max():.3f}, sign "
          f"{bald.sign_agreement.min():.0%}")
    share = pick("prefix", "aleatoric_share")
    print(f"  prefix: the aleatoric share of the prefix gain is "
          f"{share['min'].min():.0%}-{share['max'].max():.0%} over every seed, world and length "
          f"(sd <= {share['sd'].max():.3f})")
    pm = pick("prefix", "corr_total_model")
    pp = pick("prefix", "corr_total_prefix")
    print(f"          the prefix-decision correlation stays positive in "
          f"{int((pp.sign_agreement >= 1.0).sum())}/{len(pp)} cells; the model-decision "
          f"correlation keeps its sign in {int((pm.sign_agreement >= 1.0).sum())}/{len(pm)}")

    write_summary("robustness", {
        "seeds": seeds, "n_bootstrap": args.bootstrap, "scenes": args.scenes,
        "prefix_scenes": args.prefix_scenes, "validation_scenes": args.validation_scenes,
        "n_members": args.members, "tau_lo": TAU_LO, "tau_hi": TAU_HI,
        "rows": out,
        "headline": {
            "mechanism_cells": int(len(signs)),
            "mechanism_cells_sign_stable": int((signs.sign_agreement >= 1.0).sum()),
            "prefix_share_min": float(share["min"].min()),
            "prefix_share_max": float(share["max"].max()),
            "prefix_cells": int(len(pp)),
            "prefix_positive_stable": int((pp.sign_agreement >= 1.0).sum()),
            "prefix_model_stable": int((pm.sign_agreement >= 1.0).sum()),
            "bald_ensemble_min_mean": float(bald["mean"].min()),
            "bald_ensemble_max_mean": float(bald["mean"].max()),
            "bald_ensemble_max_sd": float(bald["sd"].max()),
            "confound_rise_min": float(rise["min"].min()),
            "confound_rise_max": float(rise["max"].max()),
        },
    })
    print(f"\nwrote experiments/results/robustness/ and tables/robustness.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
