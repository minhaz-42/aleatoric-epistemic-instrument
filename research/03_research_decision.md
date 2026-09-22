# Research decision

**Revision 2026-09-22. DECIDED: option 1. The write-up exists — `paper/main.pdf`.**

The decision below was the project's only open item for a day. It has been taken: **option 1**, a
short methods / negative-results note on what survived, which was also the recommendation. What
that took, beyond writing, was the robustness arm every sign claim needed and did not have — 5
seeds, a bootstrap of the published run, both worlds (`scripts/run_robustness.py`, §5 of
`research/04_instrument.md`) — and it held: 24/24 mechanism cells keep their sign at every seed,
with one prefix cell reported as not sign-stable because it is not. Option 2 (pivot onto the prefix
result with a real anticipation dataset) is untouched and remains the natural continuation; §7 of
this document still applies to it. The rest of this file is the argument as it stood, kept because
the reasoning is the record.

---

**Revision 2026-09-21b. The literature gate has fired and the answer changed.**

The previous revision recommended verifying the literature before anything else, and said the
project did not know whether its headline was a finding or a replication. It now knows: **largely a
replication.** Three of seven claims are refuted, one is partially anticipated, and what survives is
narrow. This document says what to do about that.

## 1. Where the project stands

Done and verified:

- a world whose predictive distribution is exact in closed form, two levers (irreducible floor `τ`,
  prefix length), **23/23 tests**, **7/7 validation checks**
- four result sets: the decomposition table, the routing sweeps, the mechanism map, the prefix arm
- a renderer that provably cannot leak a hidden cue (682 pairs, max pixel difference exactly 0.0)
- **the literature verification pass**, with verdicts in `research/02_novelty_analysis.md`

Not done, and now mostly unblocked or moot:

- the VLM arm is **hard-blocked**, not deferred: no `ANTHROPIC_API_KEY`, no `.env`, no `ant` CLI on
  this machine. It is written, wired and costed at ~$1.14; it cannot run until a credential exists.
- no real video

## 2. What the verification cost the project

| claim | before | after |
|---|---|---|
| B — BALD ≠ forecaster error | "highest refutation risk" | **refuted**, arXiv:2606.12646 |
| D — aleatoric cancels from escalation | "trivial algebra, but is it stated?" | **refuted**, arXiv:2605.07805 |
| E — confidence routing inverts | "the measurement is ours" | **refuted as a phenomenon**; the below-random measurement remains |
| F — mechanism map by deficit cause | "most likely genuine contribution" | **partially anticipated** by Peale et al.'s regime analysis on a different axis |
| G — prefix lever | "untested" | **measured, and survives** |

The earlier revision's §5 named the condition that would kill the project: *"B and E both
established, and F found to be a corollary rather than a finding."* B and E are established. F is
not quite a corollary — Peale et al.'s axis is the correlation between the two uncertainty terms,
ours is the cause of the weak model's deficit — but it is adjacent enough that it cannot carry a
paper alone.

## 3. The decision

**Option 1 — a short methods/negative-results note on what survives. TAKEN 2026-09-22.** Three things do:
the ρ-parameterised link from ensemble-member correlation to routing performance against a known
truth (a regime both papers assume away — Peale et al. require multiple independent annotations,
Young assumes independently seeded members); the mechanism map measured against an explicit
random baseline with the floor as a controlled variable; and the prefix result. Realistic venue is a
workshop, not a main track. Cost: 1–2 weeks. **Recommended.**

**Option 2 — pivot onto the prefix result.** Claim G is the only thing here that is specific to
video and the only claim no paper was found to cover. The measurement is sharp: 92–97% of the
prefix-extension gain is exactly the term that cancels from model escalation, and where evidence is
non-additive the same confidence score correlates −0.490 with one decision and +0.059 with the
other at every prefix length. That is a real and useful statement about video systems with two
spend levers. The second world is now done; it would need a real anticipation dataset to be
credible, so 4–6 weeks. **Recommended if the project
is to continue at length.**

**Option 3 — stop, and keep the instrument.** It is built, validated, documented, defect-free and
reusable, and the literature position is now recorded rather than guessed. Stopping costs nothing
further. **Entirely defensible, and better than it sounds:** the verification pass is itself the
deliverable that prevented weeks of work on refuted claims.

**Option 4 — proceed as originally framed.** Not defensible. Claims B, D and E are taken.

## 4. Recommended sequence

1. ~~**Decide between options 1, 2 and 3.**~~ **Done: option 1**, 2026-09-22. The judgement belonged
   to the user and was made by instruction to complete the project and the paper.
3. If option 2: **add a second world** (§5) and run the prefix arm on it. One day. Every result
   currently comes from one world, and the prefix claim is the one that most needs to be shown to
   generalise.
3. **The VLM arm, if a credential appears.** It is the only setting where a real model can be given
   a true `KL(P*‖Q)`. Predictions P1–P3 below are falsifiable and cost ~$1.14.

## 5. Risks, updated

- **The literature risk is now retired**, which is the single biggest change. It has been paid, and
  it cost three claims.
- **The generality risk is now the largest.** Every result comes from one world. The mechanism sweep
  varies the axis that matters most, but the cue structure, the linear-logit outcome model and the
  four-outcome space are fixed. A second world costs a day and is the cheapest remaining
  strengthening.
- **The framing risk is resolved in one direction:** the project is not a video paper unless it
  leads with claim G, because everything else is about estimators in general and has been claimed.
- **What is NOT a risk:** the closed-form quantities. Enumerated, not estimated, and pinned by 23
  tests asserting identities rather than outputs.

## 6. The VLM pilot, if it ever runs

Three falsifiable predictions, unchanged and still untested:

- **P1.** The model's stated aleatoric uncertainty will track its own confidence rather than the
  world's `H(P*)`, as every baseline in `research/04_instrument.md` §5.1 does.
- **P2.** An ensemble of repeated calls to one model will behave like a high-ρ ensemble: BALD near
  zero while `KL(P*‖Q̄)` is substantial. E1 — `temperature` removed from current models — makes this
  more likely, not less.
- **P3.** Routing on the model's confidence will correlate negatively with the true gain at high τ.

P2 is the one to run first: it is the cheapest, and it is the prediction that carries the surviving
claim in §3 option 1.

## 7. If real video is eventually wanted

`P*` does not exist there, so the exact decomposition does not transfer. What can:

- the empirical decomposition and the routing policies, unchanged
- the mechanism map as a **prediction**: measure whether the cheap model is overconfident, and the
  map says whether confidence routing will invert
- **prefix length, which is real on real video and is the claim worth carrying over**

`src/video/clip.py` defines the interface a real corpus must satisfy, with the unimplementable parts
left as unimplemented methods rather than as an oversight. No corpus has been chosen.
