# Novelty analysis

## 0. Read this first: no literature has been verified

**No literature search has been run for this project. Not one paper has been opened, and no claim
below has been checked against a source.** Every statement about what prior work does or does not
cover is background recollection, which for this kind of question is exactly the sort of thing that
is confidently wrong.

This is the project's largest open risk and it is larger than any technical risk in the instrument.
The decomposition in `research/01_problem_and_position.md` §2 is standard and certainly not novel.
The measurements in `research/04_instrument.md` may or may not be. **Nothing here should be written
up, and no venue should be chosen, before the verification pass in §4 has run.**

Marker convention, inherited from the sibling project `~/Desktop/Research/CompositionalAudio`:

- `[V]` read from the paper's own text
- `[A]` abstract read directly
- `[S]` search summary only
- `[U]` **unverified — recollection, no source consulted**

Nothing marked `[S]` may enter a paper. Nothing marked `[U]` may enter a paper *or a plan*.
**At the time of writing, every prior-art statement in this project is `[U]`.**

## 1. The claims, and what each would need to survive

| id | claim | status |
|---|---|---|
| A | The aleatoric/epistemic decomposition of expected log loss | **known**, certainly not novel. Textbook. Used as instrumentation, claimed as nothing. |
| B | The mutual-information (BALD) estimate measures ensemble disagreement, not forecaster error, and the two come apart under member correlation | **`[U]`** — plausibly known in the active-learning literature. Highest refutation risk. |
| C | The estimated aleatoric term tracks forecaster confidence rather than the world, so it is not a property of the data | **`[U]`** — likely observed somewhere as a calibration result. |
| D | The aleatoric term cancels exactly from the escalation decision, so confidence-based routing spends budget on a provably irrelevant quantity | **`[U]`** — the algebra is trivial; whether it has been *stated as the reason routing fails* is the open question. |
| E | Confidence-based routing does not merely degrade but **inverts** as the irreducible floor rises, going below random | **`[U]`** — the measurement is ours; whether the phenomenon is reported elsewhere is unknown. |
| F | Whether it inverts depends on *why* the cheap model is worse: benign under missing evidence, catastrophic under miscalibration | **`[U]`** — this is the most likely genuine contribution, and also the least likely to have been isolated, because isolating it needs a world where the cause can be set by construction. |
| G | Video forecasting specifically: prefix length as a second, distinct lever, with the aleatoric term cancelling for model choice but **not** for prefix choice | **`[U]`** — built and exact, but not yet exercised as an experiment. |

## 2. Honest self-assessment before any search

Ranked by what I would expect to survive contact with the literature:

**Most likely to survive: F, and E as its instance.** The claim is not "uncertainty decomposition
matters" but that *the failure mode of confidence routing is determined by the cause of the cheap
model's deficit, and can be read off in advance*. Establishing that needs a world where the cause is
set by construction and the truth is known, which is precisely what a synthetic instrument buys and
what a real-video study cannot have. The specific numbers — benign (+0.21 → +0.05), catastrophic
(−0.40 → −0.99), and floor-invariant when the estimate is well-posed (+0.57 → +0.69) — are the
contribution.

**Likely already known: B and C.** The mutual-information decomposition has been studied for a
decade in active learning, and "ensemble disagreement ≠ model error under correlated members" is the
sort of thing that gets written down. If B is prior work, the project's headline moves from "this
estimator is broken" to "here is the regime map for when it breaks, measured against a known
truth" — weaker, but not dead, and F still stands.

**Weakest: A and D.** A is textbook. D is one line of algebra; if it is not in the literature that is
because it is too obvious to state, not because nobody noticed.

**Untested: G.** Built, exact, and not yet run as an experiment. It is the most video-specific claim
in a project whose other claims are about estimators in general, which is a problem for framing —
see `research/03_research_decision.md` §3.

## 3. The framing risk, independent of the literature

Everything measured so far is about **estimators**, not about **video**. The world is a synthetic
junction; the forecasters are closed-form baselines; no model and no video have been involved. A
reviewer can fairly ask why this is a video paper at all.

Three honest answers, in descending strength:

1. The aleatoric floor is unusually large and unusually *variable* in video forecasting — the future
   of a clip is genuinely undetermined far more often than the label of an image is — so the
   confound is worst exactly there.
2. Prefix length (claim G) is a real second lever that image classification does not have, and the
   cancellation in equation (3) does **not** apply to it. That is a video-specific structural point.
3. VLM ensembles share pretraining, which puts them near ρ = 1, where V6 shows the estimate
   collapses.

Answers 1 and 3 are arguments, not measurements. Answer 2 is measurable and has not been measured.
**This is the single largest thing that could be done to make the project a video paper rather than
an uncertainty-estimation paper, and it costs nothing in compute.**

## 4. The verification pass that must run before anything is written

In priority order. Each is a specific question with a decision attached, not a topic.

1. **Claim B.** Is "BALD measures disagreement, which diverges from error under correlated
   ensemble members" already established? Search: BALD; mutual-information epistemic estimation;
   deep ensembles diversity; ensemble correlation and calibration; "epistemic uncertainty
   underestimation". *If established:* demote B to instrumentation, re-headline on F.
2. **Claim E/D.** Has anyone reported confidence-based deferral or routing performing *below random*
   as irreducible uncertainty rises? Search: selective prediction; learning to defer; LLM routing /
   cascades; risk-coverage under label noise; aleatoric uncertainty and abstention.
   *If established:* the project's contribution is the mechanism map (F), not the phenomenon.
3. **Claim F.** Has the failure mode been attributed to the *cause* of the weaker model's deficit?
   This is the one I expect to be open. Search: cascade routing failure analysis; when does
   confidence-based deferral fail; miscalibration and selective risk.
4. **Claim G.** Prefix length / observation horizon as a control on irreducible uncertainty in video
   anticipation. Search: action anticipation; video prediction uncertainty; observation ratio.
5. **Synthetic-instrument precedent.** Has anyone built a synthetic world with closed-form `P*`
   specifically to audit uncertainty estimators? If yes, cite it and position against it; the
   instrument becomes a contribution only if it is not the third such thing.

## 5. What would kill the project

- **B and E both established, and F found to be a corollary rather than a finding.** Then the work
  is a replication with a cleaner instrument. Defensible as a workshop note; not a paper.
- **The synthetic construction found to be a known benchmark.** Then the instrument is not a
  contribution and everything rests on F alone.

## 6. What would make it stronger, and costs nothing

- **Run claim G.** The prefix-length arm is built and exact. It is the only video-specific
  measurement available and it is one script away.
- **Run the VLM arm on the synthetic world.** This is the arm that gives a real model a true
  `KL(P*‖Q)` — impossible on real video. It is the bridge from "estimators behave like this" to
  "this model behaves like this", and it costs about $1. See `research/03_research_decision.md`.
