# PROJECT STATUS AND RESUME GUIDE

**PAUSED 2026-09-21 at one decision and nothing else. Nothing is running.**
**Read this file, then `research/02_novelty_analysis.md` §0, then `research/04_instrument.md`.**

Nothing is half-written and nothing is known-broken. The defect register is closed. Every number in
the documentation was re-measured against the current code. What remains is a scope judgement, and
it belongs to the user rather than to more engineering.

## 1. One-paragraph summary

The project asks whether a forecaster's uncertainty about what happens next in a video is
irreducible (the clip does not determine the future) or reducible (the model cannot read the clip),
and what it costs a system to confuse them. Because the irreducible part is unobservable on real
video, a synthetic world was built in which the true conditional `P*` is exact in closed form, and
every estimator was validated against it before any model was loaded. The instrument passes 7/7
validation checks and 21/21 regression tests. Three results are measured and reproducible. **No
vision-language model has ever been called, no video has been downloaded, and — the important one —
no literature has been verified.** The decision gate has fired and is waiting.

## 2. Where exactly it paused

At the decision in `research/03_research_decision.md` §2, and only there.

- everything in section 3 below is finished, tested and reproducible in ~90 s on CPU;
- every defect is **closed** (`research/05_defect_register.md`), with one standing external
  constraint (E1, the removal of `temperature` from current models) that is recorded, not open;
- **the literature has not been touched.** Every prior-art statement in the project is marked `[U]`.
  This is the largest open risk and it blocks any write-up.

## 3. Complete and verified

### Documents

| item | file | state |
|---|---|---|
| Problem formulation, the decomposition, scope | `research/01_problem_and_position.md` | final |
| Novelty claims A–G, each with verification status | `research/02_novelty_analysis.md` | final, **all claims unverified** |
| The open decision and recommended sequence | `research/03_research_decision.md` | final, **decision pending** |
| Instrument specification and all results | `research/04_instrument.md` | final, revision 2026-09-21 |
| Defect register D1–D8, E1 | `research/05_defect_register.md` | current, all closed |
| Dated research log | `research/06_research_log.md` | current |

### Code, all written, tested and exercised

`src/forecasting/` — `world.py` (the closed-form world: exact posterior by enumeration, exact
expected aleatoric, exact partial-evidence excess, mutual information), `scenarios.py` (the
`junction` world plus fully-observed and blind controls), `task.py` (the `Evidence` object a
forecaster receives, which structurally cannot contain a hidden cue).

`src/baselines/predictors.py` — seven forecasters with closed-form error: `BayesOptimal`,
`Uniform`, `Marginal`, `PartialEvidence`, `Tempered`, `LogitNoise` (with the ρ correlation control),
`CueDropout`.

`src/uncertainty/` — `decomposition.py` (both decompositions side by side), `calibration.py`.
`src/routing/` — `policies.py` (seven policies, differing only in what they may read),
`curves.py` (budgeted curves, the random-normalised skill score, the gain correlation).
`src/temporal/` — `prefix.py` (prefix length as a lever, exact), `splits.py` (stable splits + guard).
`src/video/` — `render.py` (the hidden-cue contract), `clip.py` (the interface real video must meet).
`src/vlm/` — `client.py`, `prompt.py`, `cache.py`, `predictor.py`. **Never run.**
`src/evaluation/` — `metrics.py`, `protocol.py` (scoring plus checks V1–V7).

### Measured results on disk

All under `experiments/results/`, reproducible with no arguments.

**Validation: 7/7.** V1 Bayes-optimal epistemic exactly 0.000e+00. V2 uniform scores log 4 to
4.4e-16. V3 marginal's excess loss 0.378616 against an exact 0.383812 (within 3 sem = 0.015). V4 monotone with all 5 rungs
matching their enumerated values. V5 floor non-increasing in prefix length. V6 below. V7 oracle
routing dominates at every budget, violation 0.000e+00.

**V6, the confound control.** At fixed noise scale, as ensemble members go from independent to
identical: one member's error is flat (0.1561 → 0.1498), the deployed ensemble's **true** error
rises **10.0×** (0.0150 → 0.1498), and the **estimate** falls to zero (0.1199 → −3.5e-18). The
estimate and the quantity it stands in for move in opposite directions and cross near ρ = 0.5.

**The estimated aleatoric term is not a property of the world.** True 0.9897 for every forecaster;
estimate swings 0.65 → 1.39. Two forecasters with identical accuracy (0.575) and identical true
aleatoric report 0.65 and 1.23. The estimated epistemic term is exactly 0 for every deterministic
forecaster, including one whose true epistemic error is 0.386 nats.

**Routing inverts.** corr(total uncertainty, true gain) goes +0.336 → **−0.590** as the floor rises;
the epistemic estimate falls 1.8× less far, +0.294 → −0.223. Both cross zero near τ = 1.3.

**But it depends on the mechanism.** Missing evidence +0.207 → +0.049 (decays, never inverts).
Overconfidence −0.395 → **−0.993** (inverts at every floor). Cue-dropout ensemble: BALD **+0.570 →
+0.686**, completely floor-invariant, while total uncertainty is worse than random everywhere.

**Renderer.** 682 scene pairs differing only in a hidden cue render with maximum pixel difference
exactly 0.0.

## 4. The decision that must be taken first on resuming

**This is the only open item.** Full argument in `research/03_research_decision.md`.

1. **Verify the literature, then decide.** 1–2 days. Claims B and E in
   `research/02_novelty_analysis.md` §4 determine whether the headline is a finding or a
   replication, and every downstream choice depends on that. **Recommended, and not close.**
2. **Strengthen first, verify after.** Run the prefix-length arm (free, built) and the VLM pilot
   (~$1.14). Defensible only if the literature risk is judged low, which nothing supports.
3. **Reframe as an uncertainty-estimation paper.** Honest — the measurements are about estimators,
   not video — but it drops the one structural claim that is video-specific.
4. **Stop.** The instrument is reusable and the documentation is complete.

## 5. How to resume

```
cd /Users/tanvir/Desktop/Research/PredictiveEvidence
python3 -c "import sys; sys.path.insert(0,'.'); import src; print(src.__version__)"
python3 tests/test_instrument.py          # ~17 s, must print 21/21 passed
python3 scripts/validate_instrument.py    # ~8 s,  must print 7/7 checks passed
```

Then reproduce everything currently claimed, about 90 s total, CPU only:

```
python3 scripts/run_decomposition.py
python3 scripts/run_routing.py
python3 scripts/run_mechanisms.py
python3 scripts/render_examples.py
python3 scripts/make_figures.py
```

No virtual environment: system Python 3.13.5 at
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`. Versions pinned in
`requirements.txt`. Nothing was installed for this project — `numpy`, `scipy`, `pandas`,
`matplotlib`, `PyYAML`, `anthropic` and `openai` were already present.

Then, in order:

1. **Settle the section 4 decision.** Nothing else should start before it.
2. If proceeding: **run the prefix-length arm.** Free, built, exact, and the only video-specific
   measurement available. Skipping it while keeping the video framing is the weakest position.
3. **Run the VLM pilot on the synthetic world**, not on real video. It is the only setting where a
   real model can be given a true `KL(P*‖Q)`. Predictions P1–P3 are in
   `research/03_research_decision.md` §4; P2 is the cheapest and carries the most.
4. Only then consider real video, where `P*` does not exist and the exact decomposition does not
   transfer.

## 6. Compute and machine state

Trivial, and that is a design property rather than luck. Everything is CPU-only, single process,
about ninety seconds end to end. The closed-form quantities are enumerated over 81 cue states × 16 mask
patterns, so exactness costs nothing. No GPU was used, no model was loaded, and `data/` is empty.

The sibling project `~/Desktop/Research/CompositionalAudio` is paused and holds nothing.

## 7. Known unknowns

- **The literature.** Unquantified and dominant. See `research/02_novelty_analysis.md` §0.
- **Generality across worlds.** Every result comes from one world. The mechanism sweep varies the
  axis that matters most — the *cause* of the cheap model's deficit — but the cue structure, the
  linear-logit outcome model and the four-outcome space are fixed. A second world costs a day.
- **Whether a real VLM behaves as predicted.** P1–P3 are falsifiable and untested. In particular,
  whether repeated calls to one model form a high-ρ ensemble is an assumption, made more likely by
  E1 but not measured.
- **Whether the synthetic-to-real argument persuades.** `research/01_problem_and_position.md` §3
  argues that showing an estimator fails where truth is known is informative about where it is not.
  That is an argument, and a reviewer may not accept it.
