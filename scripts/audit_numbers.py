#!/usr/bin/env python3
"""Check every number the documentation quotes against the summaries on disk.

This exists because the documentation drifted twice. The first time, figures quoted from a
1200-scene validation run while the default is 2000. The second time, making the experiment scripts
world-generic changed the seed-stream paths (`rng(seed, "mechanisms", world, ...)`), which changed
the junction scenes and moved 19 documented numbers — including one that changed a claim's
narrative, not just its digits.

A documented number is a claim, and a claim that no longer reproduces is a defect. Run this after
any change to a script, a seed, or a world, and before quoting anything.

The hand-listed numbers below were not enough. They pinned the ENDPOINTS of the V6 table and not
its interior, and a later re-quote shifted the interior rows by one while leaving the endpoints
right -- so the audit passed on a table that had four wrong rows in it (defect D10). The fix is not
a longer hand-list: a documented table is now PARSED out of the markdown and every cell in it is
re-measured, so the audit covers what the document says rather than what someone remembered to copy
into this file.

    python3 scripts/audit_numbers.py

Exit code 0 if every value matches, 1 otherwise.
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import RESULTS, ROOT, read_summary

RESEARCH = os.path.join(ROOT, "research")


def md_table(path: str, header_key: str) -> list:
    """The rows of the first markdown table whose header contains `header_key`.

    Returns a list of cell lists, with emphasis markers stripped. Raises if the table is not there,
    because a missing table means the document was restructured and the audit is now checking
    nothing -- which must fail loudly rather than silently pass.
    """
    lines = open(os.path.join(RESEARCH, path)).read().splitlines()
    head = next((i for i, l in enumerate(lines) if l.startswith("|") and header_key in l), None)
    if head is None:
        raise AssertionError(f"{path}: no table header containing {header_key!r}")
    rows = []
    for line in lines[head + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip().replace("**", "").replace("`", "") for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def num(cell: str) -> float:
    """A number as the documents write them: unicode minus, a leading +, a trailing % or x."""
    t = (cell.replace("\u2212", "-").replace("\u00d7", "").replace("%", "")
             .replace("+", "").replace(",", "").strip())
    return float(t)


def like(measured: float, doc: str):
    """(measured, documented) rounded the way the document wrote it, so the comparison is fair.

    Scientific notation is compared as a formatted string: a document that writes 3.7e-19 is
    claiming one mantissa digit, and demanding more of it would be inventing precision.
    """
    d = doc.replace("\u2212", "-").replace("+", "").replace("%", "").strip()
    if "e" in d.lower():
        mant = d.lower().split("e")[0]
        dp = len(mant.split(".")[1]) if "." in mant else 0
        return (f"{measured:.{dp}e}", f"{float(d):.{dp}e}")
    dp = len(d.split(".")[1]) if "." in d else 0
    return (round(float(measured), dp), round(float(d), dp))


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

    # ---- tables, parsed from the documents rather than copied into this file -------------
    INSTR = "04_instrument.md"

    def cells(header_key, path=INSTR):
        return md_table(path, header_key)

    # the floor by temperature, which is a single row of values under a tau header
    tau_head = cells("| τ | 0.35 |")[0]
    for col, t in enumerate((0.35, 0.50, 0.70, 0.90, 1.30, 2.00, 3.50)):
        F[f"table: floor at tau={t}"] = like(tau[t]["expected_aleatoric"], tau_head[col + 1])

    # V4's evidence ladder, exact and measured, rung by rung
    rungs = [c for c in vj["checks"] if c["id"] == "V4"][0]["detail"]["rungs"]
    for row, rung in zip(cells("| cues read |"), rungs):
        F[f"table: V4 ladder {row[0]} exact"] = like(rung["exact"], row[1])
        F[f"table: V4 ladder {row[0]} measured"] = like(rung["measured"], row[2])

    # V6, every cell of it -- this is the table that drifted
    for row, t in zip(cells("| ρ | one member"), v6j):
        r = f"rho={t['rho']:.2f}"
        F[f"table: V6 {r} single"] = like(t["single_member_kl"], row[1])
        F[f"table: V6 {r} true"] = like(t["epistemic_true"], row[2])
        F[f"table: V6 {r} bald"] = like(t["epistemic_hat"], row[3])

    # the decomposition table, every forecaster and every column
    dec_keys = {"bayes_optimal": "bayes_optimal", "tempered T=0.5": "tempered[T=0.5]",
                "tempered T=2": "tempered[T=2]",
                "partial[heading,speed]": "partial[heading,speed]",
                "partial[heading]": "partial[heading]",
                "logit_noise σ=0.75 ρ=0": "logit_noise[sigma=0.75,rho=0]",
                "marginal": "marginal", "uniform": "uniform"}
    for row in cells("| forecaster | alea"):
        f = fc[dec_keys[row[0]]]
        for col, key in ((1, "aleatoric_true"), (2, "aleatoric_hat"), (3, "epistemic_true"),
                         (4, "epistemic_hat"), (5, "accuracy"), (6, "ece")):
            F[f"table: decomposition {row[0]} {key}"] = like(f[key], row[col])

    # the routing floor sweep
    for row in cells("| τ | E[H(P*)] | epistemic (BALD) |"):
        t = float(row[0])
        F[f"table: routing tau={t} floor"] = like(tau[t]["expected_aleatoric"], row[1])
        F[f"table: routing tau={t} bald"] = like(tau[t]["corr_epistemic_bald"], row[2])
        F[f"table: routing tau={t} total"] = like(tau[t]["corr_total_uncertainty"], row[3])

    # the mechanism map, both worlds, every cell
    mech_rows = {"missing evidence": "missing_evidence", "injected noise": "injected_noise",
                 "overconfidence (T=0.4)": "overconfidence",
                 "underconfidence (T=2.5)": "underconfidence",
                 "evidence ensemble (cue dropout)": "evidence_ensemble", "mixed": "mixed"}
    for row in cells("| why the cheap model is worse |"):
        mech = mech_rows[row[0]]
        for col, (label, key) in enumerate((("total", "corr_total_uncertainty"),
                                            ("bald", "corr_epistemic_bald"))):
            lo, hi = [x.strip() for x in row[col + 1].split("→")] if "→" in row[col + 1] else (None, None)
            if lo is None or lo == "n/a":
                continue
            F[f"table: mech junction {mech} {label} lo"] = like(Mj[mech][0.35][key], lo)
            F[f"table: mech junction {mech} {label} hi"] = like(Mj[mech][3.5][key], hi)

    # the prefix table: share, then the two correlations, for both worlds in one table
    # the table lists junction's four lengths and then kitchen's, under two sub-headers that are
    # not table rows, so position is what distinguishes the worlds
    pre_rows = [r for r in cells("| prefix | share of prefix gain") if "→" in r[0]]
    assert len(pre_rows) == 8, f"prefix table has {len(pre_rows)} data rows, expected 8"
    for k, row in enumerate(pre_rows):
        L = int(row[0].split("→")[0].strip())
        P = PJ if k < 4 else PK          # the table lists junction's four rows, then kitchen's
        w = "junction" if k < 4 else "kitchen"
        F[f"table: prefix {w} L={L} share"] = like(100 * P[L]["aleatoric_share"], row[1])
        F[f"table: prefix {w} L={L} model"] = like(P[L]["corr_total_model"], row[2])
        F[f"table: prefix {w} L={L} prefix"] = like(P[L]["corr_total_prefix"], row[3])

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
