# Research decision

**Status: a decision is open, and it is the only thing blocking.** The instrument is built,
validated (7/7), tested (21/21) and has produced results. What to do with them is a scope judgement.

## 1. Where the project actually stands

Done and verified:

- a world whose predictive distribution is exact in closed form, with two independent levers (the
  irreducible floor `τ`, and prefix length)
- seven closed-form baselines and a validation suite of seven checks, all passing
- three result sets: the decomposition table, the routing sweeps, the mechanism map
- a renderer that provably cannot leak a hidden cue (682 pairs, max pixel difference exactly 0.0)
- a VLM arm, written, wired, costed at ~$1.14 for a 40-scene pilot, and **never run**

Not done:

- **no literature verification of any kind** (`research/02_novelty_analysis.md` §0)
- no vision-language model has been called
- the prefix-length arm is built and exact but has not been run as an experiment
- no real video

## 2. The decision

**Option 1 — verify the literature, then decide.** The cheapest and the only one that is not a
gamble. Everything in `research/02_novelty_analysis.md` §4 is `[U]`. Until claims B and E are
checked, the project does not know whether its headline is a finding or a replication, and every
downstream choice (venue, framing, which arm to run next) depends on that. Cost: 1–2 days. **This
is the recommendation and it is not close.**

**Option 2 — strengthen first, verify after.** Run the two cheap arms that would most change the
paper: the prefix-length experiment (claim G, free) and the VLM pilot on the synthetic world (~$1).
Cost: 1–2 days. Defensible only if the literature risk is judged low, which nothing currently
supports.

**Option 3 — reframe as an uncertainty-estimation paper and drop the video framing.** The
measurements are about estimators, not video (`research/02_novelty_analysis.md` §3). Honest, and it
removes a framing a reviewer would fairly attack — at the cost of the one structural claim that is
video-specific (prefix length), which has not been run and might be the strongest thing here.

**Option 4 — stop.** The instrument is reusable, the documentation is complete, the defect register
is closed. Stopping costs nothing further and leaves a working artefact.

## 3. Recommended sequence

1. **Verify claims B and E** (`research/02_novelty_analysis.md` §4, items 1–2). Half a day each.
   Decision gate: if both are established prior work, re-headline on F (the mechanism map) or take
   option 4.
2. **Run the prefix-length arm.** Free, already built, and the only video-specific measurement
   available. It is what makes this a video paper rather than an uncertainty paper, and skipping it
   while keeping the video framing is the weakest position available.
3. **Run the VLM pilot on the synthetic world.** ~$1.14, 40 scenes, 3 members. This is the bridge
   the project needs: it gives a real model a true `KL(P*‖Q)`, which is impossible on real video.
   Specifically worth measuring: does a VLM's *stated* confidence track `H(P*)`, and does an
   ensemble of VLM calls sit near ρ=1 as predicted?
4. **Only then** decide about real video, which is where the cost is.

## 4. What the VLM pilot would test, precisely

Three predictions the instrument makes, all falsifiable, none yet tested:

- **P1.** The model's stated aleatoric uncertainty will track its own confidence rather than the
  world's `H(P*)`, as every baseline in `research/04_instrument.md` §5.1 does.
- **P2.** An ensemble of repeated calls to one model will behave like a high-ρ ensemble: BALD near
  zero while `KL(P*‖Q̄)` is substantial. E1 (`temperature` removed) makes this more likely, not less.
- **P3.** Routing on the model's confidence will correlate negatively with the true gain at high τ.

P2 is the one to run first: it is the cheapest, and it is the prediction that carries the project's
claim about real deployments.

## 5. Risks

- **The literature risk dominates everything.** Unquantified, and it is the reason for option 1.
- **The framing risk.** A synthetic junction is not video. §3 of the novelty analysis has three
  answers; only one is currently measurable and it has not been measured.
- **The generality risk.** All results come from one world. The mechanism sweep varies the *cause*
  of the cheap model's deficit, which is the axis that matters most, but the cue structure, the
  linear-logit outcome model and the four-outcome space are fixed. A second world with a different
  structure would cost a day and would materially strengthen the claims.
- **What is NOT a risk:** the closed-form quantities. Those are enumerated, not estimated, and are
  pinned by 21 tests asserting identities rather than outputs.

## 6. If real video is eventually wanted

`P*` does not exist there, so the exact decomposition does not transfer — that is the whole point of
§3 in the problem formulation. What *can* transfer:

- the empirical decomposition and the routing policies, unchanged
- the mechanism map as a **prediction**: measure whether the cheap model is overconfident and the
  map says whether confidence routing will invert
- prefix length as a controlled variable, which is real on real video

`src/video/clip.py` defines the interface a real corpus must satisfy, with the unimplementable parts
left as unimplemented methods rather than as an oversight. No corpus has been chosen and none should
be before step 1.
