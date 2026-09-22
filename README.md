# Predictive Evidence

Separating irreducible from reducible uncertainty in video forecasting — and measuring what goes
wrong when a system cannot tell them apart.

When a model is uncertain about what happens next in a clip, is that because the clip does not
determine the future, or because the model cannot read the clip? Only the second is worth spending
compute on. This project builds a world where the difference is computable exactly, validates every
estimator against it, and measures how the standard estimators fail.

**Status: complete. The instrument is built and validated, the literature gate has fired, the
surviving results are seed-swept, and the write-up exists at `paper/main.pdf`. Read `STATUS.md`
first.**

The verification pass changed the answer: **three of the seven claims are refuted by two 2026
papers**, one is partially anticipated, and what survives is narrow. That is the verification doing
its job, and it is recorded in `research/02_novelty_analysis.md`. The paper is written on what
survived and says so in its introduction.

No vision-language model has been called — that arm is hard-blocked, with no credential on this
machine. No video has been downloaded.

## Read in this order

0. `STATUS.md` — what is done and what is left.
0b. `paper/main.pdf` — the write-up, and `paper/README.md` for how it is built and why it contains
   no typed numerals.
1. `research/01_problem_and_position.md` — the question, the formal statement, why the world must be
   synthetic, and what is not claimed.
2. `research/04_instrument.md` — the instrument's specification and every measured number.
3. `research/02_novelty_analysis.md` — the claims and their verdicts after the verification pass.
   **Three are refuted.** Read this before believing anything in §"The results".
4. `research/03_research_decision.md` — the decision the verdicts forced, and the record of it
   being taken (option 1: write up what survived).
5. `research/05_defect_register.md` — every defect found, its evidence, and what closed it.
6. `research/06_research_log.md` — dated log, including what was ruled out and why.

## The results

**The estimated aleatoric term is not a property of the world.** The true irreducible entropy is
0.9897 nats for every forecaster, as it must be. The ensemble *estimate* of it swings from 0.65 to
1.39, tracking the forecaster's confidence instead. Two forecasters with identical accuracy and
identical true aleatoric uncertainty report estimates of 0.65 and 1.23.

**Confidence-based routing inverts as the irreducible floor rises.** Ranking scenes for escalation
by total predictive uncertainty correlates +0.336 with the true gain at a low floor and **−0.590** at
a high one — well below routing at random. The aleatoric term cancels exactly from the escalation
decision, but it dominates the quantity being ranked by.

**Whether it inverts depends on why the cheap model is worse.** Under missing evidence the
correlation decays but stays positive (+0.158 → +0.043). Under overconfidence it inverts at every
floor (−0.443 → **−0.992**). And in the one configuration where the epistemic estimate is well-posed
— ensemble members that disagree because evidence is missing — it works and is completely
floor-invariant (**+0.557 → +0.663**) while total uncertainty is worse than random everywhere.

**The irreducible term cancels for model choice but carries the prefix decision.** 92–97% of the
gain from extending the prefix is exactly the term that cancels from model escalation — in both
worlds, at every length. Where evidence is non-additive, ranking by confidence is then actively
wrong for the model decision (−0.490 → −0.133) while staying right for the prefix decision
(+0.059 → +0.060). An image classifier has no second decision to serve at all.

The distinction is real and actionable. The difficulty is entirely in estimating it.

**All of this reproduces in a second, structurally different world** — 3 outcomes instead of 4, five
cues instead of four, no symmetry, and **non-additive** evidence. The instrument gives 7/7 there
too, nothing reverses sign, and two findings get sharper. See `research/04_instrument.md` §6.

**And it survives re-seeding.** Five seeds, re-drawing the scenes *and* the forecasters' noise
streams, plus a 1000-resample bootstrap of the published run: **24 of 24** (world, mechanism, floor)
cells keep their sign at every seed, and the prefix share stays in 92–98% everywhere. One cell is
*not* sign-stable — junction's model-decision correlation at L=2→4, published +0.001 — and is
reported as indistinguishable from zero rather than as a sign. `research/04_instrument.md` §5.5.

## What the literature check cost

Claims about the estimator (BALD ≠ forecaster error) and about routing (the aleatoric term cancels;
confidence routing underperforms) are **prior work**: arXiv:2606.12646 and arXiv:2605.07805, both
2026. What survives is the correlated-ensemble regime both papers assume away, the mechanism map
measured against a random baseline, and the prefix result. Full verdicts, with markers, in
`research/02_novelty_analysis.md`.

## Reproducing

System Python 3.13.5, no virtualenv, CPU only, no API key, no data files. About 90 seconds total.

```
python3 tests/test_instrument.py                    # 23/23, ~17 s
python3 scripts/validate_instrument.py              # 7/7,   ~8 s
python3 scripts/run_decomposition.py                # ~2 s
python3 scripts/run_routing.py                      # ~22 s
python3 scripts/run_mechanisms.py                   # ~25 s
python3 scripts/run_prefix.py                       # ~15 s
python3 scripts/run_robustness.py                   # ~7 min, 5 seeds x 2 worlds + bootstrap
python3 scripts/render_examples.py                  # ~18 s
python3 scripts/make_figures.py                     # ~1 s
python3 scripts/audit_numbers.py                    # 284/284 documented numbers

# the generality check: the same experiments in a second, non-additive world
python3 scripts/validate_instrument.py --world kitchen --tau 0.70
python3 scripts/run_mechanisms.py --world kitchen
python3 scripts/run_prefix.py --world kitchen --tau 0.70
```

Results land in `experiments/results/<name>/` as a `records.csv` (one row per scored prediction) and
a `summary.json` (the reduced numbers the documentation quotes), so every claim traces to the rows
it came from. Figures go to `figures/` and `paper/figures/`; tables to `tables/`.

Building the paper is two more commands, and the second is a gate rather than a convenience:

```
python3 scripts/build_paper.py                  # regenerate its numbers, compile, count pages
python3 scripts/audit_paper.py                  # fails on drift or on a typed numeral
```

The VLM arm refuses to run without `--confirm`, and `--estimate` prints the cost first:

```
python3 scripts/run_vlm_pilot.py --estimate     # ~$1.14 for 40 scenes × 3 members
```

## Conventions

Literature facts carry a verification marker: `[V]` read from the paper's own text, `[A]` abstract
read directly, `[S]` search summary only, `[U]` unverified recollection. Nothing marked `[S]` may
enter a paper; nothing marked `[U]` may enter a paper or a plan. The verification pass has run and
the markers are current — see the bibliography in `research/02_novelty_analysis.md` §5.

A documented number is a claim, so `scripts/audit_numbers.py` **parses the tables out of the
research documents** and re-measures every cell — 284 numbers, 140-odd of them parsed rather than
transcribed. It works that way because of defect D10: a hand-written list of expected values passed
a table that had four wrong rows in it. The paper goes further and contains no typed numerals at all
(see `paper/README.md`).

Tests assert mathematics, not current output — entropy bounded by log K, KL zero exactly at
equality, a hidden cue unable to change a pixel — so they stay meaningful when the implementation
changes. Three of them exist specifically because an earlier revision asserted something false, and
they are written to fail on the wrong version. All three are the same error: taking a statement that
holds in expectation and asserting it per sample.

## Layout

```
src/forecasting/   the world, its closed-form inference, and the task interface
src/uncertainty/   the exact decomposition and the ensemble estimate of it
src/routing/       escalation policies, budgeted curves, and the gain correlation
src/baselines/     forecasters whose error is known before they run
src/temporal/      prefix length as a lever; scene splits and the leakage guard
src/video/         rendering a prefix to frames, under the hidden-cue contract
src/vlm/           model backends, prompts, cache, and the forecaster wrapper (never run)
src/evaluation/    metrics, the protocol and checks V1-V7, the mechanism registry, the prefix arm
```

```
scripts/run_*.py         the experiments, each reproducible with no arguments
scripts/audit_numbers.py every number in research/ re-measured against the runs
scripts/make_figures.py  figures, from the summaries rather than from a re-run
scripts/build_paper.py   the paper's numbers regenerated, then compiled
scripts/audit_paper.py   the paper cannot drift from the runs either
```
