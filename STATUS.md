# PROJECT STATUS AND RESUME GUIDE

> **2026-10-02: retargeted to TMLR** (AISTATS 2027 abstract deadline had passed). `paper/main.tex` now uses the official TMLR style; builds clean, 10 pages. Ready to submit: see `SUBMISSION_TMLR.md`.

**COMPLETE as of 2026-09-22. Nothing is running, nothing is half-written, no defect is open.**
**Read this file, then `paper/main.pdf`, then `research/02_novelty_analysis.md`.**

The decision this project was paused on has been taken — option 1, write up what survived — and the
write-up exists. Every number in every document is re-measured against the runs on disk (284 of
them), the surviving claims are seed-swept and bootstrapped, and the paper contains no typed
numerals at all. **The literature gate fired and refuted three of the seven claims**; that remains
the single most important fact in this file, and it is what the paper is written around rather than
in spite of.

## 1. One-paragraph summary

The project asks whether a forecaster's uncertainty about what happens next in a video is
irreducible (the clip does not determine the future) or reducible (the model cannot read the clip),
and what it costs a system to confuse them. Because the irreducible part is unobservable on real
video, a synthetic world was built in which the true conditional `P*` is exact in closed form, and
every estimator was validated against it before any model was loaded. The instrument passes **7/7
validation checks in each of two structurally different worlds** and **23/23 regression tests**.
Five result sets are measured, reproducible, reproduced in a second world, and re-run across five
seeds. The literature pass refuted three of the seven claims: the estimator result and the routing
result are prior work (arXiv:2606.12646 and arXiv:2605.07805, both 2026). What survives — and what
the paper claims — is the correlated-ensemble regime both papers assume away, the mechanism map
against a random baseline, and the prefix result. **No vision-language model has ever been
called**; that arm is hard-blocked, with no credential on this machine.

## 2. What is left

Two things, and neither is blocked on a decision.

1. **Submit.** `paper/main.pdf` is a 10-page workshop paper: a 5-page body, two pages of
   references, and six appendix sections. One `\usepackage` line swaps `neuripsish.sty` for the venue's own style file,
   and `\author{Anonymous submission}` is the anonymous form. `paper/README.md` states the venue
   assumption explicitly so it can be corrected, and says what to move for a strict 4-page limit.
2. **The VLM pilot, if a credential ever appears.** Written, wired, costed at ~$1.14, refuses to
   run without `--confirm`. It is the only setting where a real model can be handed a true
   `KL(P*‖Q)`. Predictions P1–P3 are in `research/03_research_decision.md` §6, falsifiable and
   untested.

Option 2 from the decision document — pivoting onto the prefix result with a real anticipation
dataset — is untouched and remains the natural continuation if the project runs longer.

## 3. Complete and verified

### Documents

| item | file | state |
|---|---|---|
| The paper | `paper/main.pdf` | built, 10 pages (5-page body), audited |
| Problem formulation, the decomposition, scope | `research/01_problem_and_position.md` | final |
| Novelty claims A–G and their verdicts, plus bibliography | `research/02_novelty_analysis.md` | final, revision 2026-09-22, **3 claims refuted**, every reference re-read from its own record |
| The decision, and the fact that it was taken | `research/03_research_decision.md` | **decided: option 1** |
| Instrument specification and all results | `research/04_instrument.md` | final, revision 2026-09-22, §5.5 is the seed sweep |
| Defect register D1–D10, E1 | `research/05_defect_register.md` | current, all closed |
| Dated research log | `research/06_research_log.md` | current |

### Code, all written, tested and exercised

`src/forecasting/` — `world.py` (the closed-form world: exact posterior by enumeration, exact
expected aleatoric, exact partial-evidence excess, mutual information), `scenarios.py` (the
`junction` and `kitchen` worlds plus fully-observed and blind controls), `task.py` (the `Evidence`
object a forecaster receives, which structurally cannot contain a hidden cue).

`src/baselines/predictors.py` — seven forecasters with closed-form error: `BayesOptimal`,
`Uniform`, `Marginal`, `PartialEvidence`, `Tempered`, `LogitNoise` (with the ρ correlation control),
`CueDropout`.

`src/uncertainty/` — `decomposition.py` (both decompositions side by side), `calibration.py`.
`src/routing/` — `policies.py` (seven policies, differing only in what they may read),
`curves.py` (budgeted curves, the random-normalised skill score, the gain correlation).
`src/temporal/` — `prefix.py` (prefix length as a lever, exact), `splits.py` (stable splits + guard).
`src/video/` — `render.py` (the hidden-cue contract), `clip.py` (the interface real video must meet).
`src/vlm/` — `client.py`, `prompt.py`, `cache.py`, `predictor.py`. **Never run.**
`src/evaluation/` — `metrics.py`, `protocol.py` (scoring plus checks V1–V7), `mechanisms.py` (the
six deficit mechanisms, seed-parameterised), `arms.py` (the prefix arm's core). The last two exist
so that `run_robustness.py` re-runs *the same* experiment rather than one that resembles it.

### Measured results on disk

All under `experiments/results/`, reproducible with no arguments.

**Validation: 7/7 in BOTH worlds.** V1 Bayes-optimal epistemic exactly 0.000e+00. V2 uniform scores
log 4 to 4.4e-16. V3 marginal's excess loss 0.382810 against an exact 0.383812 (within 3 sem =
0.015). V4 monotone with all 5 rungs matching their enumerated values. V5 floor non-increasing in
prefix length. V6 below. V7 oracle routing dominates at every budget, violation 0.000e+00.

**V6, the confound control.** At fixed noise scale, as ensemble members go from independent to
identical: one member's error is flat (0.1569 → 0.1491), the deployed ensemble's **true** error
rises **10.0×** (0.0148 → 0.1491), and the **estimate** falls to zero (0.1196 → +3.7e-19). The
estimate and the quantity it stands in for move in opposite directions and cross near ρ = 0.5.

**The estimated aleatoric term is not a property of the world.** True 0.9897 for every forecaster;
estimate swings 0.65 → 1.39. Two forecasters with identical accuracy (0.575) and identical true
aleatoric report 0.65 and 1.23. The estimated epistemic term is exactly 0 for every deterministic
forecaster, including one whose true epistemic error is 0.386 nats.

**Routing inverts.** corr(total uncertainty, true gain) goes +0.336 → **−0.590** as the floor rises;
the epistemic estimate falls 1.8× less far, +0.294 → −0.223. Both cross zero near τ = 1.3.

**But it depends on the mechanism.** Missing evidence +0.158 → +0.043 (decays, never inverts).
Overconfidence −0.443 → **−0.992** (inverts at every floor). Cue-dropout ensemble: BALD **+0.557 →
+0.663**, completely floor-invariant, while total uncertainty is worse than random everywhere.
All of it reproduces in the second world, where the ensemble result is stronger still (+0.819 →
+0.843).

**Prefix (claim G, the one video-specific result).** 92–97% of the gain from extending the prefix is
exactly the term that cancels from model escalation, in both worlds at every length. Whether the two
correlations have opposite signs depends on the world: in the non-additive `kitchen` they do at every
prefix length (−0.490 → −0.133 for the model decision against +0.059 → +0.060 for the prefix
decision); in `junction` only at the shortest (−0.189 against +0.151).

**Generality.** All of the above reproduces in `kitchen`: 3 outcomes, 5 cues, mixed alphabets, no
symmetry, and **non-additive** evidence. Nothing reverses sign; BALD on a well-posed ensemble
improves to +0.819 → +0.843 and the prefix sign-opposition more than doubles.

**Seed stability.** Five seeds re-drawing the scenes *and* the forecasters' noise streams, plus a
1000-resample bootstrap of the published run. **24 of 24** (world, mechanism, floor) cells keep
their published sign at every seed; the ρ rise is 9.8–10.6×; BALD on the well-posed ensemble
averages +0.563 to +0.840 with seed sd ≤ 0.012; the prefix share is 92–98% everywhere. **One cell
is not sign-stable and is reported as such:** junction's model-decision correlation at L=2→4,
published +0.001, is −0.014 ± 0.020 — indistinguishable from zero. `research/04_instrument.md` §5.5.

**Renderer.** 682 scene pairs differing only in a hidden cue render with maximum pixel difference
exactly 0.0.

## 4. How to pick this up

```
cd /Users/tanvir/Desktop/Research/PredictiveEvidence
python3 -c "import sys; sys.path.insert(0,'.'); import src; print(src.__version__)"
python3 tests/test_instrument.py          # ~17 s, must print 23/23 passed
python3 scripts/validate_instrument.py    # ~8 s,  must print 7/7 checks passed
```

Then reproduce everything currently claimed, about 90 s on CPU:

```
python3 scripts/run_decomposition.py
python3 scripts/run_routing.py
python3 scripts/run_mechanisms.py
python3 scripts/run_prefix.py
python3 scripts/render_examples.py
python3 scripts/make_figures.py
python3 scripts/audit_numbers.py          # must print 284/284 documented numbers verified
```

The generality check, and the seed sweep:

```
python3 scripts/validate_instrument.py --world kitchen --tau 0.70   # must print 7/7
python3 scripts/run_mechanisms.py --world kitchen
python3 scripts/run_prefix.py --world kitchen --tau 0.70
python3 scripts/run_robustness.py         # ~7 min; must print 24/24 sign-stable
```

The paper:

```
python3 scripts/build_paper.py            # regenerate its numbers, compile, count pages
python3 scripts/audit_paper.py            # must print PASSED
```

`audit_numbers.py` and `audit_paper.py` are the two gates. The first parses the tables out of the
research documents and re-measures every cell; the second regenerates the paper's numbers, diffs
them, and fails on any numeral typed into the prose that is not a declared experimental setting.
Both are verified to fail on purpose. Run them after touching a script, a seed or a world, and
before quoting anything anywhere.

No virtual environment: system Python 3.13.5 at
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`. Versions pinned in
`requirements.txt`. Nothing was installed for this project — `numpy`, `scipy`, `pandas`,
`matplotlib`, `PyYAML`, `anthropic` and `openai` were already present. LaTeX is a build-time
dependency of the paper only (`latexmk`, `pdflatex`); nothing else needs it.

## 5. Compute and machine state

Trivial, and that is a design property rather than luck. Everything is CPU-only, single process,
about ninety seconds end to end for the results and about seven minutes more for the seed sweep. The
closed-form quantities are enumerated over 81 cue states × 16 mask patterns, so exactness costs
nothing. No GPU was used, no model was loaded, and `data/` is empty.

The sibling project `~/Desktop/Research/CompositionalAudio` is paused and holds nothing.

## 6. Known unknowns

- **The literature risk is RETIRED.** It was paid, and it cost three claims.
- **The generality risk is RETIRED.** Every result reproduces in a second world that differs in
  outcome count, cue count, alphabet, symmetry and — most importantly — in being **non-additive**.
  Nothing reversed sign.
- **The seed risk is RETIRED**, with one named exception that is reported as noise rather than as a
  result (§3, junction L=2→4).
- **Whether a real VLM behaves as predicted.** P1–P3 are falsifiable and untested. In particular,
  whether repeated calls to one model form a high-ρ ensemble is an assumption, made more likely by
  E1 but not measured. This is the one open question that new information could settle.
- **Whether the synthetic-to-real argument persuades.** `research/01_problem_and_position.md` §3
  argues that showing an estimator fails where truth is known is informative about where it is not.
  That is an argument, a reviewer may not accept it, and §7 of the paper says so in those words.
- **Whether a workshop takes it.** The material is narrow by the project's own assessment. That is a
  judgement about worth, not a risk to the work.
