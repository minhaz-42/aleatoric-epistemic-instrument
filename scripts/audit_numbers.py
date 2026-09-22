#!/usr/bin/env python3
"""Check every number the documentation quotes against the summaries on disk.

This exists because the documentation drifted twice. The first time, figures quoted from a
1200-scene validation run while the default is 2000. The second time, making the experiment scripts
world-generic changed the seed-stream paths (`rng(seed, "mechanisms", world, ...)`), which changed
the junction scenes and moved 19 documented numbers — including one that changed a claim's
narrative, not just its digits.

A documented number is a claim, and a claim that no longer reproduces is a defect. Run this after
any change to a script, a seed, or a world, and before quoting anything.

    python3 scripts/audit_numbers.py

Exit code 0 if every value matches, 1 otherwise.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import RESULTS, read_summary


def _mech(name):
    out = collections.defaultdict(dict)
    for row in read_summary(name)["rows"]:
        out[row["mechanism"]][row["tau"]] = row
    return out


def build():
    """(label -> (measured, documented)). Documented values are what the .md files say."""
    vj, vk = read_summary("validation"), read_summary("validation_kitchen")
    dec, rt = read_summary("decomposition"), read_summary("routing")
    Mj, Mk = _mech("mechanisms"), _mech("mechanisms_kitchen")
    PJ = {r["base_L"]: r for r in read_summary("prefix")["rows"]}
    PK = {r["base_L"]: r for r in read_summary("prefix_kitchen")["rows"]}
    v6j = [c for c in vj["checks"] if c["id"] == "V6"][0]["detail"]["table"]
    v6k = [c for c in vk["checks"] if c["id"] == "V6"][0]["detail"]["table"]
    fc = {f["forecaster"]: f for f in dec["forecasters"]}
    tau = {r["tau"]: r for r in rt["tau_sweep"]}
    rnd = lambda v, n: round(float(v), n)

    F = {
        "validation junction 7/7": (vj["n_passed"], 7),
        "validation kitchen 7/7": (vk["n_passed"], 7),
        "V3 measured": (rnd([c for c in vj["checks"] if c["id"] == "V3"][0]["detail"]["measured"], 6), 0.382810),
        "V3 exact": (rnd([c for c in vj["checks"] if c["id"] == "V3"][0]["detail"]["exact"], 6), 0.383812),
        "V6 junction rise": (rnd(v6j[-1]["epistemic_true"] / v6j[0]["epistemic_true"], 1), 10.0),
        "V6 junction overstate": (rnd(v6j[0]["ratio_hat_over_true"], 1), 8.1),
        "V6 junction single lo": (rnd(v6j[0]["single_member_kl"], 4), 0.1569),
        "V6 junction single hi": (rnd(v6j[-1]["single_member_kl"], 4), 0.1491),
        "V6 junction true lo": (rnd(v6j[0]["epistemic_true"], 4), 0.0148),
        "V6 junction true hi": (rnd(v6j[-1]["epistemic_true"], 4), 0.1491),
        "V6 junction bald lo": (rnd(v6j[0]["epistemic_hat"], 4), 0.1196),
        "V6 kitchen rise": (rnd(v6k[-1]["epistemic_true"] / v6k[0]["epistemic_true"], 1), 9.5),
        "kitchen floor tau=0.70": (rnd(vk["expected_aleatoric"], 4), 0.7792),
        "junction exact floor": (rnd(dec["expected_aleatoric_exact"], 4), 0.9919),
        "aleatoric true (all forecasters)": (rnd(fc["bayes_optimal"]["aleatoric_true"], 4), 0.9897),
        "tempered T=0.5 alea_hat": (rnd(fc["tempered[T=0.5]"]["aleatoric_hat"], 4), 0.6543),
        "tempered T=2 alea_hat": (rnd(fc["tempered[T=2]"]["aleatoric_hat"], 4), 1.2348),
        "marginal alea_hat": (rnd(fc["marginal"]["aleatoric_hat"], 4), 1.3758),
        "marginal epi_true": (rnd(fc["marginal"]["epistemic_true"], 4), 0.3862),
        "routing total tau=0.35": (rnd(tau[0.35]["corr_total_uncertainty"], 3), 0.336),
        "routing total tau=3.5": (rnd(tau[3.5]["corr_total_uncertainty"], 3), -0.590),
        "routing bald tau=0.35": (rnd(tau[0.35]["corr_epistemic_bald"], 3), 0.294),
        "routing bald tau=3.5": (rnd(tau[3.5]["corr_epistemic_bald"], 3), -0.223),
    }

    doc_mech = {
        "junction": {"missing_evidence": (0.158, 0.043), "injected_noise": (0.375, -0.613),
                     "overconfidence": (-0.443, -0.992), "underconfidence": (-0.307, -0.978),
                     "evidence_ensemble": (-0.277, -0.460), "mixed": (0.405, -0.494)},
        "kitchen": {"missing_evidence": (0.190, 0.168), "injected_noise": (0.306, -0.574),
                    "overconfidence": (-0.223, -1.000), "underconfidence": (-0.722, -0.997),
                    "evidence_ensemble": (-0.441, -0.624), "mixed": (0.306, -0.495)},
    }
    for world, M in (("junction", Mj), ("kitchen", Mk)):
        for mech, (lo, hi) in doc_mech[world].items():
            F[f"{world} {mech} tau=0.35"] = (rnd(M[mech][0.35]["corr_total_uncertainty"], 3), lo)
            F[f"{world} {mech} tau=3.5"] = (rnd(M[mech][3.5]["corr_total_uncertainty"], 3), hi)
    for world, M, lo, hi in (("junction", Mj, 0.557, 0.663), ("kitchen", Mk, 0.819, 0.843)):
        F[f"{world} evidence_ensemble BALD tau=0.35"] = (rnd(M["evidence_ensemble"][0.35]["corr_epistemic_bald"], 3), lo)
        F[f"{world} evidence_ensemble BALD tau=3.5"] = (rnd(M["evidence_ensemble"][3.5]["corr_epistemic_bald"], 3), hi)

    doc_pre = {"junction": (PJ, {1: (-0.189, 0.151), 2: (0.001, 0.135),
                                 4: (0.084, 0.117), 8: (0.127, 0.070)}),
               "kitchen": (PK, {1: (-0.490, 0.059), 2: (-0.346, 0.052),
                                4: (-0.220, 0.083), 8: (-0.133, 0.060)})}
    for world, (P, exp) in doc_pre.items():
        for L, (m, pf) in exp.items():
            F[f"{world} prefix L={L} model"] = (rnd(P[L]["corr_total_model"], 3), m)
            F[f"{world} prefix L={L} prefix"] = (rnd(P[L]["corr_total_prefix"], 3), pf)
    return F


def main():
    needed = ["validation", "validation_kitchen", "decomposition", "routing",
              "mechanisms", "mechanisms_kitchen", "prefix", "prefix_kitchen"]
    missing = [n for n in needed if not os.path.exists(os.path.join(RESULTS, n, "summary.json"))]
    if missing:
        print("cannot audit: missing results for " + ", ".join(missing))
        print("run the experiments listed in research/04_instrument.md section 9 first")
        return 1

    F = build()
    bad = [(k, m, d) for k, (m, d) in F.items()
           if (abs(m - d) > 1e-9 if isinstance(d, float) else m != d)]
    for k, m, d in bad:
        print(f"  MISMATCH  {k}: documentation says {d}, run gives {m}")
    print(f"{len(F) - len(bad)}/{len(F)} documented numbers verified"
          + ("  -- ALL MATCH" if not bad else ""))
    if bad:
        print("\nA documented number that no longer reproduces is a defect. Either the change was")
        print("intended -- in which case re-quote the documentation and say so in the research log --")
        print("or it was not, in which case find out what moved.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
