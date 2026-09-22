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

## 2026-09-21b — literature gate fired; prefix arm run; second world added

**The literature verification pass ran, and it refuted three claims.** This is the entry that
matters. The previous revision recorded that nothing had been read and that nothing could be written
up until that changed. Two 2026 papers take most of the project's headline:

- **arXiv:2606.12646**, Robin Young, *Epistemic Uncertainty Is Not the Reducible Kind* `[V]`, proves the
  mutual-information measure and the reducibility definition are extensionally inconsistent, and
  shows ensemble disagreement "collapses to zero beneath a positive truth under consistent
  training". That is claim B, with synthetic closed-form constructions of its own — which also
  removes the idea that a synthetic instrument with a known `P*` is itself a contribution.
- **arXiv:2605.07805**, Peale, Devic, Gopalan, Wieder, Gollakota, *Flexible Routing via Uncertainty
  Decomposition* `[V]`, states in its introduction that "confidence-based methods often fail to
  account for irreducible (aleatoric) uncertainty … needlessly routing to the oracle even when it
  would also perform poorly", and builds a router on the irreducible/reducible split. That is
  claims D and E.

**Ruled out** as headline material: B, D, E. **Demoted**: C. The earlier self-assessment had
predicted B and C would likely fall and that F was "most likely to survive" — that was roughly
right, but it underestimated how completely D and E were taken.

What survives, and it is narrow: the correlated-ensemble regime both papers assume away (Peale et
al. require multiple independent annotations per input; Young assumes independently seeded members), the mechanism map measured against an explicit random baseline with the floor as a
controlled variable, and the prefix result.

**Two flagged reads, done rather than deferred.** arXiv:2409.02628 reports epistemic-uncertainty
collapse as *model complexity* rises via implicit ensembling — a real wound to the surviving
estimator claim, but a different cause from member correlation, and no routing. arXiv:2403.14715
shows label smoothing degrading selective classification through over/underconfidence — the
miscalibration half of the mechanism map, known in direction, but with no random baseline, no
controlled aleatoric variation, and no ablation separating miscalibration from evidence deficiency.

**Prefix arm run (claim G).** The one video-specific measurement, and the only claim no paper was
found to cover. 92–97% of the gain from extending the prefix is exactly `ΔH(P*)` — the term that
cancels entirely from model escalation — so the two decisions are driven by almost disjoint
quantities, in both worlds at every length. Whether the two correlations have *opposite signs* turned
out to depend on the world: throughout the sweep in the non-additive `kitchen` (−0.490 → −0.133
against +0.059 → +0.060), but only at the shortest prefix in the additive `junction` (−0.189 against
+0.151), where the model correlation turns positive by L=2→4. The claim was narrowed accordingly.
Required nested masks across prefix lengths
(`temporal.prefix.nested_scene`): a longer prefix must reveal a superset, or a negative
prefix-extension gain becomes a sampler artefact.

**Third instance of one error class (defect D9).** The new test asserted that nesting made the floor
monotone scene by scene. It does not, and nesting is irrelevant: conditioning reduces entropy on
average, never pointwise, and a newly revealed cue that contradicts the shorter prefix genuinely
raises that scene's entropy (0.9299 at L=8 → 0.9440 at L=16). Same mistake as D1 and D3. A standing
rule is now recorded: any check on an information-theoretic quantity must state whether it is a
per-scene or an expectation claim, and expectation claims are tested against an enumerated closed
form wherever one is computable.

**Second world added, because generality became the largest remaining risk once the literature risk
was paid.** `kitchen` differs in outcome count (3 vs 4), cue count (5 vs 4), alphabets (mixed vs
uniformly ternary), symmetry (none vs left/right mirrored) and — the one that matters — in having
**non-additive** evidence: pairwise cue interactions, so losing one cue can be catastrophic rather
than gradual. This required adding an optional interaction tensor to `WorldSpec`; the additive path
is bit-identical afterwards (E[H(P*)] = 0.9919, unchanged).

The instrument gives **7/7 there too**, V6 reproduces at 9.5× against 10.0×, and nothing reverses
sign. Two findings get *sharper* under non-additive evidence, for the same reason — losing a cue
matters more: BALD on a well-posed ensemble improves from +0.557–0.663 to +0.819–0.843, and the prefix
sign-opposition holds at every prefix length instead of only the shortest. One gets weaker and is
named in the write-up: `missing_evidence` no longer decays toward zero as the floor rises.

**VLM arm: hard-blocked, not deferred.** No `ANTHROPIC_API_KEY`, no `.env`, no `ant` CLI on this
machine. The arm is written, wired, costed at ~$1.14 and refuses to run without `--confirm`. Nothing
was spent and nothing could have been.

**Numbers re-audited after the refactor.** Making the scripts world-generic changed the seed-stream
paths (`rng(seed, "mechanisms", world, ...)`), so the junction scenes differ from the pre-refactor
run and 19 documented numbers moved. All were re-quoted from the canonical run; the audit script
now checks 46 documented values against the summaries on disk. The one substantive change was the
junction prefix narrative above — the drift is what caught it.

Final state: 23/23 tests, 7/7 validation checks in each of two worlds, five figures, seven result
tables. The open item is no longer "what next" but "is what survived worth writing up".
