# Research log

Dated, append-only. Includes what was ruled out and why.

## 2026-09-21 — project built from an empty skeleton

Started from a directory tree with no code, no commits and no specification: `src/{temporal, video,
vlm, forecasting, uncertainty, routing, baselines, evaluation, utils}`, `paper/`, `research/`,
`experiments/`, plus an `.env.example` naming an Anthropic key and an OpenAI-compatible gateway for
open VLMs. The research question was inferred from that structure and from the house style of the
sibling project `~/Desktop/Research/CompositionalAudio`, which does literature → novelty → decision
→ instrument → validation, and builds a measurement instrument validated against known-behaviour
baselines *before* loading any generative model. This project follows the same order.

**Design decision: synthetic world with closed-form `P*`.** The question is whether a forecaster's
uncertainty is irreducible or reducible, and on real video the irreducible part is unobservable —
so it could only be assumed, never measured. Finite cue alphabets and independent cues make the
marginalisation exact. Ruled out: estimating `P*` from data on a real corpus (circular — the
estimate is the thing under test) and using a known-noise label-corruption benchmark (sets the
*marginal* noise rate, not the per-scene conditional, which is the quantity that matters).

**Verified the foundation before building on it.** Fully-observed posterior equals the conditional
(0.0 difference), blind posterior equals the marginal (0.0), the decomposition identity holds to
1.8e-15, sampled outcomes match the closed form within 3σ, mutual information is 0 with nothing
observed and monotone in what is.

**First validation run: 5/7.** V4 and V6 failed. Both were my assertions, not the code — see
defects D1 and D2. The corrected V6 is a stronger claim than the one it replaced: the estimate and
the truth do not merely diverge, they move in opposite directions and cross.

**Routing: first pass unreadable.** Random routing scored ~0.6 on the efficiency metric, compressing
every comparison into the top 40% of the range (D4). Replaced the headline with the Spearman
correlation between a policy's score and the true escalation gain, which has a clean zero and a
clean sign, and added a random-normalised skill score. The inversion below zero only became legible
after this change.

**Robustness check changed the headline.** Before writing up "confidence routing inverts as the
irreducible floor rises", tested it across five different cheap/expensive mechanisms. It does not
invert under missing evidence (+0.207 → +0.049, decays but stays positive) and inverts
catastrophically under miscalibration (−0.395 → −0.993, at *every* floor). The headline was
qualified accordingly and the mechanism sweep became its own experiment. **Ruled out** the simpler
claim "the aleatoric floor breaks confidence routing" as an overclaim: the floor sets how fast it
degrades, the cause of the model's deficit sets whether it inverts at all.

**Added the cue-dropout ensemble** after noticing that three of the five mechanisms left BALD
undefined (a deterministic forecaster has no ensemble spread), which made the comparison unfair to
it. With members that disagree because evidence is missing and whose errors are independent by
construction, BALD scores +0.57 → +0.69 and is completely floor-invariant. This is the project's one
clean positive result and it exists because the first version of the experiment was unfair.

**Renderer.** Built so a hidden cue cannot reach the pixels, which is a correctness requirement
rather than a nicety: a leak would falsify the aleatoric floor every result is measured against.
Both branch mouths are occluded whenever `blocker` is hidden, because covering only the blocked one
would leak the cue through which side got covered. 682 pairs verified at exactly 0.0 difference.
Agent travel and trail contrast were raised after measuring that speed was not legible (D8).

**VLM arm written, not run.** Discovered while writing it that `temperature` is rejected with a 400
on current Anthropic models (E1). This is not a workaround problem: the sampled-elicitation mode
built its ensemble from repeated calls and relied on temperature for member diversity. Recorded as a
standing constraint, and noted that it pushes real deployments further into the high-ρ regime the
instrument predicts is broken.

**Not done, and flagged loudly:** no literature has been read. Every prior-art statement in the
project is marked `[U]` and `research/02_novelty_analysis.md` opens by saying nothing may be written
up before that changes. This is the largest open risk and it is larger than anything in the code.

Final state: 21/21 tests, 7/7 validation checks, four figures, five result tables, ~90 s to
reproduce everything end to end on CPU.
