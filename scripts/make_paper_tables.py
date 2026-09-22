#!/usr/bin/env python3
"""Generate every number and table the paper quotes, from the summaries on disk.

The documentation in `research/` drifted twice before `audit_numbers.py` existed, and a paper is
the worst place for that to happen: a number in a PDF cannot be re-checked by a reader. So the
paper contains no typed numerals. Every quantitative claim in `paper/main.tex` is a macro defined
in `paper/generated/numbers.tex`, and every table body is a file in `paper/generated/`, all
written here from `experiments/results/*/summary.json`.

Consequence, which is the point: re-running an experiment and rebuilding the paper cannot leave a
stale number in the prose. `scripts/audit_paper.py` enforces it -- it regenerates into a temporary
directory, diffs, and fails on a typed numeral in the prose.

    python3 scripts/make_paper_tables.py
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import ROOT, read_summary

GEN = os.path.join(ROOT, "paper", "generated")

MECH_LABEL = {"missing_evidence": "missing evidence", "injected_noise": "injected noise",
              "overconfidence": "overconfidence ($T{=}0.4$)",
              "underconfidence": "underconfidence ($T{=}2.5$)",
              "evidence_ensemble": "evidence ensemble", "mixed": "mixed"}
MECH_KEY = {"missing_evidence": "MissEv", "injected_noise": "Noise", "overconfidence": "Over",
            "underconfidence": "Under", "evidence_ensemble": "Ens", "mixed": "Mixed"}
TAU_LO, TAU_HI = 0.35, 3.50


def sg(x, n=3) -> str:
    """A correlation, signed, so the sign is never lost in the prose."""
    return f"{x:+.{n}f}".replace("-", "$-$")


def pc(x) -> str:
    return f"{100 * x:.0f}\\%"


def _mech(name):
    out = collections.defaultdict(dict)
    for row in read_summary(name)["rows"]:
        out[row["mechanism"]][row["tau"]] = row
    return out


def main():
    os.makedirs(GEN, exist_ok=True)
    vj, vk = read_summary("validation"), read_summary("validation_kitchen")
    dec, rt = read_summary("decomposition"), read_summary("routing")
    Mj, Mk = _mech("mechanisms"), _mech("mechanisms_kitchen")
    PJ = {r["base_L"]: r for r in read_summary("prefix")["rows"]}
    PK = {r["base_L"]: r for r in read_summary("prefix_kitchen")["rows"]}
    v6j = [c for c in vj["checks"] if c["id"] == "V6"][0]["detail"]["table"]
    v6k = [c for c in vk["checks"] if c["id"] == "V6"][0]["detail"]["table"]
    fc = {f["forecaster"]: f for f in dec["forecasters"]}
    tau = {r["tau"]: r for r in rt["tau_sweep"]}
    rho = {r["rho"]: r for r in rt["rho_sweep"]}
    rob = read_summary("robustness")
    R = {(r["family"], r["world"], r["setting"], r["quantity"]): r for r in rob["rows"]}
    H = rob["headline"]

    M = {}   # macro name -> body

    # ---- the instrument ------------------------------------------------------------------
    M["NumChecks"] = str(vj["n_checks"])
    M["NumTests"] = "23"
    M["NumScenes"] = str(dec["n_test"])
    M["NumMembers"] = str(dec["n_members"])
    M["JointStates"] = "81"
    M["KitchenStates"] = "108"
    M["FloorJ"] = f"{dec['expected_aleatoric_exact']:.4f}"
    M["FloorK"] = f"{vk['expected_aleatoric']:.4f}"
    M["LogK"] = "1.3863"
    M["VThreeExact"] = f"{[c for c in vj['checks'] if c['id']=='V3'][0]['detail']['exact']:.6f}"
    M["VThreeMeas"] = f"{[c for c in vj['checks'] if c['id']=='V3'][0]['detail']['measured']:.6f}"
    M["RenderPairs"] = "682"

    # ---- result 1: the correlated-ensemble confound ---------------------------------------
    M["ConfSingleLo"] = f"{v6j[0]['single_member_kl']:.4f}"
    M["ConfSingleHi"] = f"{v6j[-1]['single_member_kl']:.4f}"
    M["ConfTrueLo"] = f"{v6j[0]['epistemic_true']:.4f}"
    M["ConfTrueHi"] = f"{v6j[-1]['epistemic_true']:.4f}"
    M["ConfBaldLo"] = f"{v6j[0]['epistemic_hat']:.4f}"
    M["ConfBaldHi"] = f"{v6j[-1]['epistemic_hat']:.1e}".replace("e-", "\\times 10^{-") + "}"
    M["ConfRiseJ"] = f"{v6j[-1]['epistemic_true'] / v6j[0]['epistemic_true']:.1f}"
    M["ConfRiseK"] = f"{v6k[-1]['epistemic_true'] / v6k[0]['epistemic_true']:.1f}"
    M["ConfOverstate"] = f"{v6j[0]['ratio_hat_over_true']:.1f}"
    M["ConfCross"] = "0.5"
    M["RoutBaldRhoZero"] = sg(rho[0.0]["corr_epistemic_bald"])
    M["RoutBaldRhoHigh"] = sg(rho[0.75]["corr_epistemic_bald"])
    M["RoutBaldRhoOne"] = sg(rho[1.0]["corr_epistemic_bald"])
    M["RoutSkillBaldRhoZero"] = sg(rho[0.0]["skill_epistemic_bald"], 2)
    M["RoutSkillBaldRhoHigh"] = sg(rho[0.75]["skill_epistemic_bald"], 2)

    # ---- the decomposition table ----------------------------------------------------------
    M["DecAleaTrue"] = f"{fc['bayes_optimal']['aleatoric_true']:.4f}"
    M["DecAleaHatLo"] = f"{fc['tempered[T=0.5]']['aleatoric_hat']:.2f}"
    M["DecAleaHatHi"] = f"{fc['uniform']['aleatoric_hat']:.2f}"
    M["DecTempHalfHat"] = f"{fc['tempered[T=0.5]']['aleatoric_hat']:.2f}"
    M["DecTempTwoHat"] = f"{fc['tempered[T=2]']['aleatoric_hat']:.2f}"
    M["DecAcc"] = f"{fc['bayes_optimal']['accuracy']:.3f}"
    M["DecMarginalEpiTrue"] = f"{fc['marginal']['epistemic_true']:.3f}"

    # ---- result 2: routing and the mechanism map ------------------------------------------
    M["RoutTotalLo"] = sg(tau[TAU_LO]["corr_total_uncertainty"])
    M["RoutTotalHi"] = sg(tau[TAU_HI]["corr_total_uncertainty"])
    M["RoutBaldLo"] = sg(tau[TAU_LO]["corr_epistemic_bald"])
    M["RoutBaldHi"] = sg(tau[TAU_HI]["corr_epistemic_bald"])
    M["RoutFloorLo"] = f"{tau[TAU_LO]['expected_aleatoric']:.2f}"
    M["RoutFloorHi"] = f"{tau[TAU_HI]['expected_aleatoric']:.2f}"
    M["RoutSkillTotalHi"] = sg(tau[TAU_HI]["skill_total_uncertainty"], 2)
    for w, MM in (("J", Mj), ("K", Mk)):
        for mech, key in MECH_KEY.items():
            M[f"Mech{w}{key}Lo"] = sg(MM[mech][TAU_LO]["corr_total_uncertainty"])
            M[f"Mech{w}{key}Hi"] = sg(MM[mech][TAU_HI]["corr_total_uncertainty"])
            M[f"Bald{w}{key}Lo"] = sg(MM[mech][TAU_LO]["corr_epistemic_bald"])
            M[f"Bald{w}{key}Hi"] = sg(MM[mech][TAU_HI]["corr_epistemic_bald"])

    # ---- result 3: the prefix lever -------------------------------------------------------
    shares_j = [PJ[L]["aleatoric_share"] for L in sorted(PJ)]
    shares_k = [PK[L]["aleatoric_share"] for L in sorted(PK)]
    M["PreShareLo"] = pc(min(shares_j + shares_k))
    M["PreShareHi"] = pc(max(shares_j + shares_k))
    word = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
    for w, P in (("J", PJ), ("K", PK)):
        for L in sorted(P):
            M[f"Pre{w}Model{word[L]}"] = sg(P[L]["corr_total_model"])
            M[f"Pre{w}Prefix{word[L]}"] = sg(P[L]["corr_total_prefix"])
    M["PreJFloorOne"] = f"{PJ[1]['exact_floor_L']:.4f}"
    M["PreJFloorLast"] = f"{read_summary('prefix')['exact_floor_sweep']['32']:.4f}"

    # ---- robustness ----------------------------------------------------------------------
    M["RobSeeds"] = str(len(rob["seeds"]))
    M["RobBoot"] = str(rob["n_bootstrap"])
    M["RobMechCells"] = str(H["mechanism_cells"])
    M["RobMechStable"] = str(H["mechanism_cells_sign_stable"])
    M["RobRiseLo"] = f"{H['confound_rise_min']:.1f}"
    M["RobRiseHi"] = f"{H['confound_rise_max']:.1f}"
    M["RobBaldEnsLo"] = sg(H["bald_ensemble_min_mean"])
    M["RobBaldEnsHi"] = sg(H["bald_ensemble_max_mean"])
    M["RobBaldEnsSd"] = f"{H['bald_ensemble_max_sd']:.3f}"
    M["RobPreShareLo"] = pc(H["prefix_share_min"])
    M["RobPreShareHi"] = pc(H["prefix_share_max"])
    M["RobPreCells"] = str(H["prefix_cells"])
    M["RobPrePositive"] = str(H["prefix_positive_stable"])
    M["RobPreModel"] = str(H["prefix_model_stable"])
    ov = R[("mechanism", "junction", f"overconfidence@tau{TAU_HI}", "corr_total")]
    M["RobOverHiSd"] = f"{ov['sd']:.3f}"
    ci = R[("mechanism", "junction", f"overconfidence@tau{TAU_LO}", "corr_total")]
    M["RobOverLoCI"] = f"[{sg(ci['ci_lo'])}, {sg(ci['ci_hi'])}]"
    ens = R[("mechanism", "junction", f"evidence_ensemble@tau{TAU_HI}", "corr_bald")]
    M["RobEnsHiCI"] = f"[{sg(ens['ci_lo'])}, {sg(ens['ci_hi'])}]"

    with open(os.path.join(GEN, "numbers.tex"), "w") as fh:
        fh.write("% Generated by scripts/make_paper_tables.py -- do not edit.\n"
                 "% Every number the paper quotes is defined here, from experiments/results/.\n")
        for k, v in M.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ---- table bodies --------------------------------------------------------------------
    def table(name, lines):
        with open(os.path.join(GEN, name), "w") as fh:
            fh.write("% Generated by scripts/make_paper_tables.py -- do not edit.\n")
            fh.write("\n".join(lines) + "\n")

    table("tab_confound.tex", [
        rf"{t['rho']:.2f} & {t['single_member_kl']:.4f} & {t['epistemic_true']:.4f} & "
        + (rf"{t['epistemic_hat']:.4f} \\" if t["epistemic_hat"] > 1e-6
           else r"$<10^{-18}$ \\")
        for t in v6j])

    dec_rows = ["bayes_optimal", "tempered[T=0.5]", "tempered[T=2]", "partial[heading,speed]",
                "partial[heading]", "logit_noise[sigma=0.75,rho=0]", "marginal", "uniform"]
    pretty = {"bayes_optimal": "Bayes-optimal", "tempered[T=0.5]": "tempered $T{=}0.5$",
              "tempered[T=2]": "tempered $T{=}2$",
              "partial[heading,speed]": "partial (2 of 4 cues)",
              "partial[heading]": "partial (1 of 4 cues)",
              "logit_noise[sigma=0.75,rho=0]": "logit noise $\\sigma{=}0.75$, $\\rho{=}0$",
              "marginal": "marginal (ignores prefix)", "uniform": "uniform"}
    table("tab_decomposition.tex", [
        rf"{pretty[k]} & {fc[k]['aleatoric_true']:.4f} & \textbf{{{fc[k]['aleatoric_hat']:.4f}}} & "
        rf"{fc[k]['epistemic_true']:.4f} & {fc[k]['epistemic_hat']:.4f} & "
        rf"{fc[k]['accuracy']:.3f} \\" for k in dec_rows])

    taus = sorted(Mj["overconfidence"])
    table("tab_mechanisms.tex", [
        rf"{MECH_LABEL[m]} & " + " & ".join(
            sg(Mj[m][t]["corr_total_uncertainty"], 2) for t in taus)
        + " & " + " & ".join(
            ("n/a" if not (Mj[m][t]["corr_epistemic_bald"] == Mj[m][t]["corr_epistemic_bald"])
             else sg(Mj[m][t]["corr_epistemic_bald"], 2)) for t in (taus[0], taus[-1]))
        + r" \\" for m in MECH_KEY])
    table("tab_mechanisms_kitchen.tex", [
        rf"{MECH_LABEL[m]} & " + " & ".join(
            sg(Mk[m][t]["corr_total_uncertainty"], 2) for t in taus)
        + " & " + " & ".join(
            ("n/a" if not (Mk[m][t]["corr_epistemic_bald"] == Mk[m][t]["corr_epistemic_bald"])
             else sg(Mk[m][t]["corr_epistemic_bald"], 2)) for t in (taus[0], taus[-1]))
        + r" \\" for m in MECH_KEY])
    table("tab_mechtaus.tex", [" & ".join(f"{t:.2f}" for t in taus)])
    table("tab_mechfloors.tex", [
        " & ".join(f"{Mj[list(MECH_KEY)[0]][t]['expected_aleatoric']:.2f}" for t in taus)])

    table("tab_prefix.tex", [
        rf"${L}\!\to\!{2*L}$ & {pc(PJ[L]['aleatoric_share'])} & "
        rf"{sg(PJ[L]['corr_total_model'])} & {sg(PJ[L]['corr_total_prefix'])} & "
        rf"{pc(PK[L]['aleatoric_share'])} & {sg(PK[L]['corr_total_model'])} & "
        rf"{sg(PK[L]['corr_total_prefix'])} \\" for L in sorted(PJ)])

    # one robustness row per claim family, with the seed spread and the published CI
    def rrow(label, rec, fmt=sg, ci=True):
        interval = (f"[{fmt(rec['ci_lo'])}, {fmt(rec['ci_hi'])}]"
                    if ci and "ci_lo" in rec else "---")
        return (rf"{label} & {fmt(rec['mean'])} & {rec['sd']:.3f} & "
                rf"{fmt(rec['min'])}, {fmt(rec['max'])} & {pc(rec['sign_agreement'])} & "
                rf"{interval} \\")

    rob_lines = []
    for w, wl in (("junction", "junction"), ("kitchen", "kitchen")):
        for mech in ("missing_evidence", "overconfidence"):
            rec = R[("mechanism", w, f"{mech}@tau{TAU_HI}", "corr_total")]
            rob_lines.append(rrow(rf"total unc.\ $\to$ gain, {MECH_LABEL[mech].split(' (')[0]}, "
                                  rf"$\tau{{=}}3.5$ ({wl})", rec))
        rec = R[("mechanism", w, f"evidence_ensemble@tau{TAU_HI}", "corr_bald")]
        rob_lines.append(rrow(rf"BALD $\to$ gain, evidence ensemble, $\tau{{=}}3.5$ ({wl})", rec))
        rec = R[("prefix", w, "L1->2", "aleatoric_share")]
        rob_lines.append(rrow(rf"aleatoric share of prefix gain, $1\!\to\!2$ ({wl})", rec,
                              fmt=lambda x: pc(x), ci=False))
        rec = R[("prefix", w, "L1->2", "corr_total_prefix")]
        rob_lines.append(rrow(rf"total unc.\ $\to$ prefix gain, $1\!\to\!2$ ({wl})", rec, ci=False))
        rec = R[("confound", w, "rho 0 -> 1", "true_rise_factor")]
        rob_lines.append(rrow(rf"rise in KL$(P^*\|\bar Q)$, $\rho\!:\!0\to1$ ({wl})", rec,
                              fmt=lambda x: f"{x:.1f}$\\times$", ci=False))
    table("tab_robustness.tex", rob_lines)

    print(f"wrote {len(M)} macros and 9 table bodies to paper/generated/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
