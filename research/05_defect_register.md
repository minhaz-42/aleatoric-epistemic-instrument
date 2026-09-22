# Defect register

Every defect found in the instrument, its evidence, and what closed it. `D` numbers are stable and
never reused. Defects found by an assertion that was itself wrong are marked as such, because the
fix in those cases was to the *claim*, not to the code — and those are the ones worth reading.

| id | severity | status | what |
|---|---|---|---|
| D1 | **blocking** | **FIXED** | V4 asserted pointwise monotonicity of epistemic error in the cues read. False: 441/1200 scenes violated it. |
| D2 | **major** | **FIXED** | V6 expected the ensemble's true error to stay flat as members correlate. It rises 10.0×. |
| D3 | major | **FIXED** | `TrueAleatoric` policy documented as necessarily performing at chance. It does not. |
| D4 | major | **FIXED** | Routing efficiency metric did not place random routing at zero; the whole comparison sat in the top 40% of the range. |
| D5 | minor | **FIXED** | `gain_correlation` raised a warning and returned a meaningless value for constant scores. |
| D6 | minor | **FIXED** | Test T8's precondition used the full spread instead of the top-two gap; the world has exact ties by design. |
| D7 | minor | **FIXED** | `write_summary` crashed on `np.bool_`, losing a completed run's output. |
| D8 | cosmetic | **FIXED** | Rendered agent travelled too little for speed to be legible, and the motion trail was near-invisible. |
| D9 | major | **FIXED** | A new test asserted the irreducible floor falls monotonically with prefix length *scene by scene*. False — and it is the third instance of the same error class. |
| E1 | external | **STANDING** | `temperature` is rejected with a 400 on current Anthropic models. Not a defect; a constraint that changes the VLM arm's design. |

---

## D1 — pointwise evidence monotonicity (blocking, fixed)

**Claimed.** Epistemic error is monotone non-increasing in the set of cues a forecaster reads,
scene by scene.

**Evidence it was wrong.** The check failed on **441 of 1200 scenes**, worst single increase 0.192
nats. Re-measuring one adjacent pair of the ladder in isolation gave 28/1200 violations.

**Why it was wrong.** The data-processing inequality constrains `E[KL(P* ‖ P*_S)]` — an expectation
over scenes. It says nothing pointwise. On an individual scene a forecaster reading *fewer* cues can
land closer to the full posterior by chance, and about 2% of scenes do.

**The fix, and why it is stronger than the original.** The check moved to the mean, and gained a
closed-form reference: `World.expected_partial_excess` enumerates all 16 mask patterns × 81 cue
states and returns the exact expected excess. V4 now asserts both monotonicity *and* that every rung
matches its enumerated value within 3 sem. All five rungs match:

| cues read | exact | measured |
|---|---|---|
| all four | 0.000000 | 0.000000 |
| heading, speed, signal | 0.048312 | 0.047361 |
| heading, speed | 0.115331 | 0.114348 |
| heading | 0.183902 | 0.182313 |
| none | 0.383812 | 0.382810 |

**Pinned by** test D1, which asserts the mean version holds *and* that at least one scene violates
the pointwise version — so the wrong claim cannot silently pass again.

---

## D2 — the ensemble's true error under correlation (major, fixed)

**Claimed.** As ensemble members go from independent to identical at fixed noise scale, the true
epistemic error stays flat while the estimate collapses.

**Evidence it was wrong.** The true error of the ensemble mean rose from 0.0148 to 0.1491 nats —
**10.0×** — across the ρ sweep.

**Why it was wrong.** The claim conflated two different forecasters. A *single member's* error is
flat (0.1569 → 0.1491, and this was the intended invariance, which does hold). But the deployed
forecaster is the ensemble *mean*, and averaging only cancels error the members do not share. As
they share more, the mean stops being better than its members.

**The fix, and why the truth is a stronger result.** The corrected check tracks all three
quantities. The finding is no longer "the estimate degrades" but that **the estimate and the
quantity it stands in for move in opposite directions and cross** near ρ = 0.5. At ρ=0 the estimate
overstates by 8.1×; at ρ=1 it reports zero for an ensemble carrying 0.149 nats of recoverable error.
That is a sharper claim than the one it replaced.

**Pinned by** test D2, which asserts the direction of all three quantities.

---

## D3 — the aleatoric policy's expected performance (major, fixed)

**Claimed.** A policy ranking by `H(P*)` ranks by the term that cancels from the escalation
decision, so it must perform at or below chance.

**Evidence it was wrong.** It measures +0.277 correlation with the true gain at τ=0.35, competitive
with every deployable policy, and +0.24 in parts of the ρ sweep.

**Why it was wrong.** Cancelling from the decision means `H(P*)` has no *causal* role. It does not
mean it is statistically uncorrelated with the gain. In this world `corr(H(P*), gain)` is +0.277 at
τ=0.35 and decays to −0.015 by τ=3.5.

**The fix.** The docstring now states the distinction, gives the measured decay, and records that
the earlier prediction was wrong. The policy is retained as a diagnostic: the decay *is* the
confound becoming visible.

---

## D4 — the routing efficiency metric (major, fixed)

**Evidence.** Random routing scored 0.573–0.628 on the original `auc_efficiency`, not 0. A policy
measurably worse than random still appeared to capture 40% of the available gain.

**Why.** At budget `b`, random escalation captures `b ×` the total gain while the oracle captures
the top-`b`; the ratio integrates to roughly 0.6, not 0. The metric was correct and unreadable.

**The fix.** `routing/curves.skill` rescales so random is 0 and the oracle is 1. Both are reported;
the raw value stays in the records. The primary diagnostic is now the Spearman correlation between a
policy's score and the true gain, which has a clean zero and a clean sign — and which is what made
the inversion in `research/04_instrument.md` §5.2 legible at all.

---

## D5 — correlation on constant scores (minor, fixed)

**Evidence.** A `ConstantInputWarning` and a meaningless return when the cheap forecaster ignores
the prefix, so every confidence-derived policy scores every scene identically.

**The fix.** Returns `nan`, documented as a distinct outcome: "carries no signal" and "cannot be
ranked at all" are different findings, and the figures render the second as hatched cells labelled
*not computable* rather than as a missing measurement.

---

## D6 — the tie precondition in test T8 (minor, fixed)

**Evidence.** T8 failed at T=4.0 on scene 35, where the top two outcomes differ by 5.55e-17.

**Why.** The world is deliberately left/right symmetric — `JUNCTION_W` gives `left` and `right`
mirrored weights — so cue configurations exist where two outcomes are *exactly* tied and the argmax
is decided by float ordering. The guard checked the spread across all four outcomes rather than the
gap between the top two.

**The fix.** Guard on the top-two gap; skip genuine ties; and assert full order preservation over
the pairs that are actually ordered, which is the property tempering guarantees. An earlier draft of
this fix contained a vacuous assertion (`abs(kl(p,q)) >= 0`, always true) which was replaced with
the real pairwise check.

---

## D7 — JSON serialisation of numpy booleans (minor, fixed)

**Evidence.** `validate_instrument.py` completed all seven checks, printed them, then crashed in
`write_summary` with `TypeError: Object of type bool is not JSON serializable`, losing the run.

**Why.** `_plain` handled `np.floating` and `np.integer` but not `np.bool_`, which is not a subclass
of Python's `bool`.

**The fix.** Added to the isinstance check.

---

## D8 — renderer legibility (cosmetic, fixed)

**Evidence.** Measured agent displacement over an 8-frame prefix was under 35 px on a 256 px canvas
at every speed, and the motion trail was drawn at alpha 0.15–0.45 in a colour close to the agent's.

**Why it mattered.** Not a correctness defect — no closed-form quantity touches the renderer — but
`speed` is only readable from displacement, so an unreadable trail would have made the VLM arm test
something other than what it claims.

**The fix.** Travel raised to 35 px (slow) / 70 px (fast) / exactly 0 (hidden), trail alpha and
contrast raised. Verified by measuring the agent centroid per frame. The leak test (T10) was re-run
after the change: 682 pairs, max pixel difference exactly 0.0.

---

## D9 — pointwise prefix monotonicity (major, fixed)

**Claimed.** With nested masks, a longer prefix lowers the irreducible floor scene by scene.

**Evidence it was wrong.** Test T21 failed on its first scene: the floor went **0.9299 at L=8 to
0.9440 at L=16**.

**Why it was wrong, and why this one matters most.** Nesting has nothing to do with it. Conditioning
reduces entropy *on average* — `H(Y|X) <= H(Y)` — and never pointwise. A newly revealed cue that
points against what the shorter prefix implied genuinely raises that scene's posterior entropy. That
is correct behaviour: more evidence can make a forecaster less certain and more right.

**This is the third instance of one error class.** D1 asserted pointwise monotonicity in the cues
read; D3 assumed a term that cancels causally must be uncorrelated statistically; D9 asserted
pointwise monotonicity in prefix length. All three are the same mistake — **taking a statement that
holds in expectation and asserting it per sample.** Every information-theoretic inequality used in
this project (the data-processing inequality, conditioning-reduces-entropy, Jensen for the ensemble
split) is an expectation statement, and none of them constrains a single scene.

**Standing rule.** Any new check on an information-theoretic quantity must state explicitly whether
it is a per-scene or an expectation claim, and an expectation claim must be tested against an
enumerated closed form rather than against the sampled mean alone wherever one is computable.

**Pinned by** test D9, which asserts the mean version holds *and* that the pointwise version is
violated.

---

## E1 — `temperature` removed from current models (external, standing)

**Not a defect.** On Claude Opus 5, Sonnet 5, Opus 4.8 and Opus 4.7, the `temperature` parameter has
been removed and a request that sends one is rejected with a 400. The `SAMPLED` elicitation mode
builds an ensemble from repeated calls and was relying on temperature to set member diversity.

**Why it strengthens the project rather than weakening it.** An ensemble whose members cannot be
pushed apart is an ensemble whose members are highly correlated — the ρ → 1 regime that V6 shows
collapses the estimate. The API change pushes real deployments *further* into the regime the
instrument predicts is broken.

**Standing rule.** `src/vlm/client.py` carries a `TEMPERATURE_SUPPORTED` table; every run records
which models it used and whether their diversity was controllable, because a sampled-mode number
means something different in each case. `run_vlm_pilot.py --estimate` prints the warning before any
call is made.
