# The instrument: specification and validation

Revision 2026-09-21. Everything below is reproducible from `scripts/` with no arguments, CPU only,
in about 90 seconds total. No model, no network, no data files.

## 1. The world

`EvidenceWorld` (`src/forecasting/world.py`), instantiated as `junction` (`src/forecasting/scenarios.py`).

An agent approaches a four-way junction; the forecast is which exit it takes, from
`{left, straight, right, stop}`. Four cues, each a different *kind* of evidence and each differing
in how often a prefix shows it:

| cue | alphabet | prior | visibility | what it is |
|---|---|---|---|---|
| `heading` | −1, 0, +1 | uniform | 0.95 | agent orientation; strongest single cue, almost always visible |
| `speed` | −1, 0, +1 | 0.30/0.40/0.30 | 0.85 | approach speed; weak alone, interacts with `blocker` |
| `signal` | −1, 0, +1 | 0.35/0.30/0.35 | 0.55 | junction light; often outside a short prefix's view |
| `blocker` | −1, 0, +1 | 0.25/0.50/0.25 | 0.40 | obstruction on a branch; often self-occluded |

Outcome logits are linear in the cues with fixed hand-set weights (`JUNCTION_W`), softened by a
temperature `τ`. The weights are a property of the instrument, not a fitted result, and are held
constant across every experiment so that a change in a measured number is attributable to the
forecaster or the protocol rather than to the world.

The spread in visibility is deliberate. If every cue were equally visible, the observability mask
would carry no information and every scene would sit at the same aleatoric floor — the degenerate
case the routing experiment must avoid.

**`τ` is the only knob on the irreducible floor.** Exact, by enumeration over all 81 joint cue
states and all 16 mask patterns:

| τ | 0.35 | 0.50 | 0.70 | 0.90 | 1.30 | 2.00 | 3.50 |
|---|---|---|---|---|---|---|---|
| E[H(P*)] nats | 0.7212 | 0.8018 | 0.9043 | **0.9919** | 1.1196 | 1.2399 | 1.3297 |

against a ceiling of log 4 = 1.3863. τ = 0.90 is the default.

**Prefix length is the second lever.** A cue is revealed by a prefix of `L` frames with probability
`v_i · (1 − exp(−L / L_i))`, which depends on `L` alone and so keeps the mask independent of `z` —
the property equation (1) relies on. E[H(P*)] falls 1.2122 → 0.9932 as `L` goes 1 → 32.

## 2. Forecasters with known behaviour

`src/baselines/predictors.py`. Every one has a closed form known before it runs.

| forecaster | what it isolates |
|---|---|
| `BayesOptimal` | the floor: epistemic exactly 0 |
| `Uniform` | expected NLL exactly log K on every scene |
| `Marginal` | ignores the prefix; excess loss is exactly I(y; z_m) |
| `PartialEvidence(S)` | correctly specified over a smaller field of view |
| `Tempered(T)` | right ranking, wrong confidence: separates accuracy from calibration |
| `LogitNoise(σ, ρ)` | the main probe; `ρ` is the ensemble-correlation confound control |
| `CueDropout(p)` | members disagree *because evidence is missing*: the fair test of BALD |

`LogitNoise` holds each member's own marginal noise scale fixed while sweeping `ρ`, so a member's
error does not change with `ρ` — only the ensemble's ability to see it does. That invariance is what
makes the ρ sweep interpretable, and it is asserted in test T9.

## 3. Validation: V1–V7

`python3 scripts/validate_instrument.py` — **7/7 pass**, 2000 scenes (the default), ~8 s.
Every expectation was fixed by the mathematics before the check ran.

| id | claim | expected | measured |
|---|---|---|---|
| V1 | Bayes-optimal has zero epistemic error | max KL = 0 exactly | **0.000e+00** |
| V2 | uniform scores log K on every scene | 1.386294 exactly | max deviation **4.4e-16** |
| V3 | marginal's excess loss = I(y; z_m) | 0.383812 nats (exact) | **0.378616** ± 0.015 (3 sem) |
| V4 | expected epistemic monotone in cues used, and matches closed form | non-decreasing; every rung within 3 sem of its enumerated value | monotone; **5/5 rungs match** |
| V5 | longer prefix never raises the floor | every difference ≤ 0 | max increase **−1.0e-02** |
| V6 | estimate and truth move in opposite directions as members correlate | see §4 | see §4 |
| V7 | oracle routing dominates at every budget | no violation | **0.000e+00** |

The V4 ladder against its enumerated values:

| cues read | exact E[excess] | measured |
|---|---|---|
| all four | 0.000000 | 0.000000 |
| heading, speed, signal | 0.048312 | 0.048041 |
| heading, speed | 0.115331 | 0.111536 |
| heading | 0.183902 | 0.180950 |
| none | 0.383812 | 0.378616 |

## 4. V6, the confound control — the central measurement

At fixed noise scale, as ensemble members go from independent (ρ=0) to identical (ρ=1):

| ρ | one member's error | ensemble's **true** error KL(P*‖Q̄) | ensemble's **estimate** (BALD) |
|---|---|---|---|
| 0.00 | 0.1561 | 0.0150 | 0.1199 |
| 0.25 | 0.1553 | 0.0434 | 0.0906 |
| 0.50 | 0.1543 | 0.0755 | 0.0612 |
| 0.75 | 0.1531 | 0.1109 | 0.0311 |
| 0.90 | 0.1519 | 0.1338 | 0.0126 |
| 1.00 | 0.1498 | **0.1498** | **−3.5e-18** |

Read it in three parts:

1. **A single member's error is flat** (0.1561 → 0.1498). `ρ` changes how members relate to each
   other, not how wrong any one of them is. This is the control that makes the other two columns
   readable.
2. **The deployed ensemble's true error rises 10.0×.** Averaging only cancels error the members do
   not share.
3. **The estimate falls to exactly zero.** It measures disagreement, and identical members do not
   disagree.

So the estimate and the quantity it stands in for **move in opposite directions and cross** near
ρ = 0.5. At ρ = 0 the estimate *overstates* the ensemble's error by 8.0×; at ρ = 1 it reports zero
for an ensemble carrying 0.150 nats of recoverable error.

Figure: `figures/fig2_confound.png`. Table: `tables/confound_rho_sweep.csv`.

## 5. Results

### 5.1 The estimated aleatoric term is not a property of the world

`scripts/run_decomposition.py`, 2018 test scenes, 12 members. Exact E[H(P*)] = 0.9919 nats.

| forecaster | alea **true** | alea **estimate** | epi **true** | epi **estimate** | acc | ECE |
|---|---|---|---|---|---|---|
| bayes_optimal | 0.9897 | 0.9897 | 0.0000 | 0.0000 | 0.575 | 0.030 |
| tempered T=0.5 | 0.9897 | **0.6543** | 0.1480 | 0.0000 | 0.575 | 0.136 |
| tempered T=2 | 0.9897 | **1.2348** | 0.0746 | 0.0000 | 0.575 | 0.149 |
| partial[heading,speed] | 0.9897 | 1.1065 | 0.1149 | 0.0000 | 0.522 | 0.026 |
| partial[heading] | 0.9897 | 1.1769 | 0.1848 | 0.0000 | 0.482 | 0.014 |
| logit_noise σ=0.75 ρ=0 | 0.9897 | 0.9136 | 0.0147 | 0.1206 | 0.562 | 0.036 |
| marginal | 0.9897 | **1.3758** | **0.3862** | **0.0000** | 0.290 | 0.003 |
| uniform | 0.9897 | 1.3863 | 0.3966 | 0.0000 | 0.290 | 0.040 |

Two things to take from this table.

**The true aleatoric column is constant**, as it must be — it is a property of the world and the
prefix. **The estimated column swings from 0.65 to 1.39**, tracking the forecaster's confidence
rather than the world. The two tempered rows make the point cleanly: identical accuracy (0.575),
identical true aleatoric, estimates 0.65 and 1.23.

**The estimated epistemic column is exactly zero for every deterministic forecaster**, including
`marginal`, whose true epistemic error is 0.386 nats — the largest of any forecaster that reads the
prefix at all. A single deterministic model has no ensemble spread, so the estimator reports perfect
knowledge for a forecaster that ignores the evidence entirely. This is the ρ = 1 case in its most
extreme form, and it is the most common deployment.

Figure: `figures/fig1_decomposition.png`.

### 5.2 Routing: confidence-based ranking inverts as the floor rises

`scripts/run_routing.py`. Cheap σ=1.20, expensive σ=0.35, 12 members, ρ=0.3. Reported as Spearman
correlation between the policy's score and the true escalation gain of (3) — clean zero at no
signal, clean sign.

| τ | E[H(P*)] | epistemic (BALD) | total uncertainty |
|---|---|---|---|
| 0.35 | 0.7212 | +0.294 | +0.336 |
| 0.90 | 0.9919 | +0.104 | +0.136 |
| 1.30 | 1.1196 | +0.014 | +0.003 |
| 2.00 | 1.2399 | −0.148 | −0.236 |
| 3.50 | 1.3297 | **−0.223** | **−0.590** |

Both policies pass through zero near τ = 1.3 and go negative — **worse than routing at random**.
Across the sweep, total uncertainty's correlation falls 1.8× as far as the epistemic estimate's
(a span of 0.925 against 0.517), and at the highest floor it sits 2.6× deeper below zero
(−0.590 against −0.223).

The mechanism is explicable rather than mysterious. High predictive entropy selects scenes where the
ensemble has hedged toward uniform, and hedging toward uniform is close to optimal precisely when
the world is uncertain. Those are the scenes with the *least* recoverable error, so the budget goes
to exactly the wrong places.

The ρ sweep at fixed τ = 0.90 shows the other half: the epistemic estimate's advantage depends on
members being independent, and it is gone by ρ = 0.75.

Figure: `figures/fig3_routing.png`.

### 5.3 Whether confidence routing fails depends on why the cheap model is worse

`scripts/run_mechanisms.py`. This is the main qualification on §5.2 and the result that stops it
being an overclaim.

corr(score, true gain), at τ = 0.35 → τ = 3.50:

| why the cheap model is worse | total uncertainty | epistemic (BALD) |
|---|---|---|
| missing evidence | +0.207 → +0.049 | n/a (deterministic) |
| injected noise | +0.326 → −0.580 | +0.288 → −0.220 |
| **overconfidence** (T=0.4) | **−0.395 → −0.993** | n/a (deterministic) |
| underconfidence (T=2.5) | −0.263 → −0.983 | n/a (deterministic) |
| **evidence ensemble** (cue dropout) | −0.304 → −0.484 | **+0.570 → +0.686** |
| mixed | +0.396 → −0.491 | +0.355 → −0.157 |

- **Missing evidence is the benign case.** A model that cannot see a cue is honestly uncertain, and
  that is also where a better model gains. The correlation decays toward zero but never inverts.
- **Miscalibration is the catastrophic case, and it inverts at every floor.** An overconfident
  model's entropy is lowest exactly where it is most wrong, so ranking by entropy systematically
  selects the scenes it already handles. At τ=1.3 and above the correlation is below −0.94. This is
  the case that matters outside the synthetic world, because overconfidence is the documented
  failure mode of neural classifiers rather than a hypothetical one.
- **The one configuration where the epistemic estimate is well-posed, it works and is
  floor-invariant.** With members that disagree because evidence is missing and whose errors are
  independent by construction, BALD scores +0.57 to +0.69 and *does not degrade at all* as the
  irreducible floor rises — while total uncertainty is worse than random everywhere in the same
  setting. That is the positive result, and it is what makes the negative ones worth reporting: the
  aleatoric/epistemic distinction is real and actionable; the difficulty is entirely in estimating
  it.

`n/a` marks configurations where the estimate does not exist rather than being small: a single
deterministic forecaster has no ensemble spread, so BALD is identically zero and induces no ranking.

Figure: `figures/fig4_mechanisms.png`.

## 6. Rendering, and the contract it must honour

`src/video/render.py` draws a prefix as 8 top-down frames. The instrument does not need pixels —
every closed-form quantity comes from the cue vector directly, and no validation run calls this
module. It exists so the VLM arm is a real experiment rather than a description of one.

The contract is absolute: **a cue the mask hides must not be recoverable from the pixels.** A
renderer that leaked one would not add noise, it would falsify the aleatoric floor every result is
measured against. Each cue is hidden by removing the evidence for it, not by drawing it differently:

- `heading` hidden → agent drawn centred and rotationally symmetric, no nose marker
- `speed` hidden → no motion trail, and the agent holds position across the prefix
- `signal` hidden → the signal head is replaced by an opaque occluder at the same location
- `blocker` hidden → **both** branch mouths are covered

The last is the subtle one. Covering only the branch that holds the blocker would leak the cue
through which side got covered; drawing nothing would leak it through the absence of an obstruction.
Covering both unconditionally is the only option that reveals nothing.

Verified by test T10 and by `scripts/render_examples.py`: **682 scene pairs differing only in a
hidden cue render with a maximum pixel difference of exactly 0.0.** T11 asserts the converse — that
visible cues *do* change the pixels — so the renderer cannot pass T10 by being constant. Measured
agent displacement is 35 px at speed −1, 70 px at speed +1, and exactly 0 when speed is hidden.

## 7. Tests

`python3 tests/test_instrument.py` — **21/21 pass**, ~17 s.

They assert mathematics, not current output: entropy bounded by log K, KL zero exactly at equality,
marginalising every cue reproduces the marginal, tempering preserves the ranking, `LogitNoise`
members correlate at exactly ρ with a ρ-invariant marginal scale, a hidden cue cannot change a
pixel, the oracle dominates at every budget, splits are stable under adding scenes.

Two of them exist because an earlier revision asserted something false and the suite did not catch
it. **D1** pins that evidence monotonicity holds in the mean *and* is false pointwise — it asserts
both halves, so the wrong version cannot silently pass again. **D2** pins the direction of all three
quantities in the ρ sweep. See `research/05_defect_register.md`.

## 8. Reproducing

```
python3 tests/test_instrument.py          # 21/21, ~17 s
python3 scripts/validate_instrument.py    # 7/7,   ~8 s
python3 scripts/run_decomposition.py      # ~2 s
python3 scripts/run_routing.py            # ~22 s
python3 scripts/run_mechanisms.py         # ~25 s
python3 scripts/render_examples.py        # ~18 s
python3 scripts/make_figures.py           # ~1 s
```

Everything is seeded from one root (20260921) through `src/utils/seeding.py`, which derives each
stream by hashing a label path. Adding an arm to an experiment therefore cannot change the draws of
the arms beside it.
