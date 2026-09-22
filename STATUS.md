# PROJECT STATUS AND RESUME GUIDE

**PAUSED 2026-09-21b at one decision and nothing else. Nothing is running.**
**Read this file, then `research/02_novelty_analysis.md`, then `research/04_instrument.md`.**

Nothing is half-written and nothing is known-broken. The defect register is closed. Every number in
the documentation was re-measured against the current code. **The literature gate has fired and it
refuted three of the seven claims** — that is the single most important fact in this file, and it
changes what the project is worth rather than whether it works.

## 1. One-paragraph summary

The project asks whether a forecaster's uncertainty about what happens next in a video is
irreducible (the clip does not determine the future) or reducible (the model cannot read the clip),
and what it costs a system to confuse them. Because the irreducible part is unobservable on real
video, a synthetic world was built in which the true conditional `P*` is exact in closed form, and
every estimator was validated against it before any model was loaded. The instrument passes **7/7
validation checks in each of two structurally different worlds** and **23/23 regression tests**.
Four results are measured, reproducible, and reproduced in both worlds. The literature verification
pass then ran and **refuted three of the seven claims**: the estimator result and the routing result
are prior work (arXiv:2606.12646 and arXiv:2605.07805, both 2026). What survives is the
correlated-ensemble regime both papers assume away, the mechanism map, and the prefix result. **No
vision-language model has ever been called** — that arm is hard-blocked, with no credential on this
machine.

## 2. Where exactly it paused

At the decision in `research/03_research_decision.md` §3, and only there. The decision is no longer
"what should we do next" but "is what survived worth writing up".

- everything in section 3 below is finished, tested and reproducible on CPU;
- every defect is **closed** (`research/05_defect_register.md`, now D1–D9), with one standing
  external constraint (E1, the removal of `temperature` from current models) that is recorded;
- **the literature gate has fired.** Claims B, D and E are refuted; F is partially anticipated; G
  survives and is now measured. Nothing is outstanding.
- the **VLM arm is hard-blocked**, not deferred: no `ANTHROPIC_API_KEY`, no `.env`, no `ant` CLI.

## 3. Complete and verified

### Documents

| item | file | state |
|---|---|---|
| Problem formulation, the decomposition, scope | `research/01_problem_and_position.md` | final |
| Novelty claims A–G and their verdicts, plus bibliography | `research/02_novelty_analysis.md` | final, revision 2026-09-21b, **3 claims refuted** |
| The decision the verdicts force | `research/03_research_decision.md` | final, revision 2026-09-21b, **decision pending** |
| Instrument specification and all results | `research/04_instrument.md` | final, revision 2026-09-21b |
| Defect register D1–D9, E1 | `research/05_defect_register.md` | current, all closed |
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

**Validation: 7/7 in BOTH worlds.** V1 Bayes-optimal epistemic exactly 0.000e+00. V2 uniform scores log 4 to
4.4e-16. V3 marginal's excess loss 0.382810 against an exact 0.383812 (within 3 sem = 0.015). V4 monotone with all 5 rungs
matching their enumerated values. V5 floor non-increasing in prefix length. V6 below. V7 oracle
routing dominates at every budget, violation 0.000e+00.

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

**Renderer.** 682 scene pairs differing only in a hidden cue render with maximum pixel difference
exactly 0.0.

## 4. The decision that must be taken first on resuming

**This is the only open item.** Full argument in `research/03_research_decision.md`.

1. **A short methods / negative-results note on what survives** — the correlated-ensemble regime,
   the mechanism map against a random baseline, and the prefix result. Workshop, not main track.
   1–2 weeks. **Recommended.**
2. **Pivot onto the prefix result**, the only claim specific to video and the only one no paper was
   found to cover. Would need a real anticipation dataset to be credible. 4–6 weeks.
   **Recommended if the project continues at length.**
3. **Stop, and keep the instrument.** Built, validated in two worlds, documented, defect-free,
   reusable. Entirely defensible: the verification pass is itself the deliverable that prevented
   weeks of work on refuted claims.
4. **Proceed as originally framed.** Not defensible — claims B, D and E are taken.

## 5. How to resume

```
cd /Users/tanvir/Desktop/Research/PredictiveEvidence
python3 -c "import sys; sys.path.insert(0,'.'); import src; print(src.__version__)"
python3 tests/test_instrument.py          # ~17 s, must print 23/23 passed
python3 scripts/validate_instrument.py    # ~8 s,  must print 7/7 checks passed
```

Then reproduce everything currently claimed, about 90 s total, CPU only:

```
python3 scripts/run_decomposition.py
python3 scripts/run_routing.py
python3 scripts/run_mechanisms.py
python3 scripts/run_prefix.py
python3 scripts/render_examples.py
python3 scripts/make_figures.py
python3 scripts/audit_numbers.py          # must print 67/67 documented numbers verified
```

The generality check, which is what retires the second-largest risk:

```
python3 scripts/validate_instrument.py --world kitchen --tau 0.70   # must print 7/7
python3 scripts/run_mechanisms.py --world kitchen
python3 scripts/run_prefix.py --world kitchen --tau 0.70
```

No virtual environment: system Python 3.13.5 at
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`. Versions pinned in
`requirements.txt`. Nothing was installed for this project — `numpy`, `scipy`, `pandas`,
`matplotlib`, `PyYAML`, `anthropic` and `openai` were already present.

Then, in order:

1. **Settle the section 4 decision.** Nothing else should start before it.
2. **The VLM pilot, if a credential ever appears.** `python3 scripts/run_vlm_pilot.py --estimate`
   prints the cost (~$1.14) and the arm refuses to run without `--confirm`. It is the only setting
   where a real model can be given a true `KL(P*‖Q)`. Predictions P1–P3 are in
   `research/03_research_decision.md` §6.
3. Only then consider real video, where `P*` does not exist and the exact decomposition does not
   transfer — except the prefix lever, which does.

## 6. Compute and machine state

Trivial, and that is a design property rather than luck. Everything is CPU-only, single process,
about ninety seconds end to end. The closed-form quantities are enumerated over 81 cue states × 16 mask
patterns, so exactness costs nothing. No GPU was used, no model was loaded, and `data/` is empty.

The sibling project `~/Desktop/Research/CompositionalAudio` is paused and holds nothing.

## 7. Known unknowns

- **The literature risk is RETIRED.** It was paid, and it cost three claims.
- **The generality risk is RETIRED.** Every result now reproduces in a second world that differs in
  outcome count, cue count, alphabet, symmetry and — most importantly — in being **non-additive**.
  Nothing reversed sign.
- **What remains is a worth judgement, not a risk.** Is the surviving material enough for a note?
- **Whether a real VLM behaves as predicted.** P1–P3 are falsifiable and untested. In particular,
  whether repeated calls to one model form a high-ρ ensemble is an assumption, made more likely by
  E1 but not measured.
- **Whether the synthetic-to-real argument persuades.** `research/01_problem_and_position.md` §3
  argues that showing an estimator fails where truth is known is informative about where it is not.
  That is an argument, and a reviewer may not accept it.
