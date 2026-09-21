# Predictive Evidence

Separating irreducible from reducible uncertainty in video forecasting — and measuring what goes
wrong when a system cannot tell them apart.

When a model is uncertain about what happens next in a clip, is that because the clip does not
determine the future, or because the model cannot read the clip? Only the second is worth spending
compute on. This project builds a world where the difference is computable exactly, validates every
estimator against it, and measures how the standard estimators fail.

**Status: the instrument is built, validated and has produced results. One decision is open and it
is the only thing blocking. Read `STATUS.md` first.**

No vision-language model has been called. No video has been downloaded. **No literature has been
verified** — that is the project's largest open risk and `research/02_novelty_analysis.md` opens by
saying so.

## Read in this order

0. `STATUS.md` — what is done, where it paused, how to resume.
1. `research/01_problem_and_position.md` — the question, the formal statement, why the world must be
   synthetic, and what is not claimed.
2. `research/04_instrument.md` — the instrument's specification and every measured number.
3. `research/02_novelty_analysis.md` — the claims, each with an honest verification status. **Every
   prior-art statement is currently unverified.**
4. `research/03_research_decision.md` — the open decision and the recommended sequence.
5. `research/05_defect_register.md` — every defect found, its evidence, and what closed it.
6. `research/06_research_log.md` — dated log, including what was ruled out and why.

## The three results

**The estimated aleatoric term is not a property of the world.** The true irreducible entropy is
0.9897 nats for every forecaster, as it must be. The ensemble *estimate* of it swings from 0.65 to
1.39, tracking the forecaster's confidence instead. Two forecasters with identical accuracy and
identical true aleatoric uncertainty report estimates of 0.65 and 1.23.

**Confidence-based routing inverts as the irreducible floor rises.** Ranking scenes for escalation
by total predictive uncertainty correlates +0.336 with the true gain at a low floor and **−0.590** at
a high one — well below routing at random. The aleatoric term cancels exactly from the escalation
decision, but it dominates the quantity being ranked by.

**Whether it inverts depends on why the cheap model is worse.** Under missing evidence the
correlation decays but stays positive (+0.207 → +0.049). Under overconfidence it inverts at every
floor (−0.395 → **−0.993**). And in the one configuration where the epistemic estimate is well-posed
— ensemble members that disagree because evidence is missing — it works and is completely
floor-invariant (**+0.570 → +0.686**) while total uncertainty is worse than random everywhere.

The distinction is real and actionable. The difficulty is entirely in estimating it.

## Reproducing

System Python 3.13.5, no virtualenv, CPU only, no API key, no data files. About 90 seconds total.

```
python3 tests/test_instrument.py          # 21/21, ~17 s
python3 scripts/validate_instrument.py    # 7/7,   ~8 s
python3 scripts/run_decomposition.py      # ~2 s
python3 scripts/run_routing.py            # ~22 s
python3 scripts/run_mechanisms.py         # ~25 s
python3 scripts/render_examples.py        # ~18 s
python3 scripts/make_figures.py           # ~1 s
```

Results land in `experiments/results/<name>/` as a `records.csv` (one row per scored prediction) and
a `summary.json` (the reduced numbers the documentation quotes), so every claim traces to the rows
it came from. Figures go to `figures/` and `paper/figures/`; tables to `tables/`.

The VLM arm refuses to run without `--confirm`, and `--estimate` prints the cost first:

```
python3 scripts/run_vlm_pilot.py --estimate     # ~$1.14 for 40 scenes × 3 members
```

## Conventions

Literature facts carry a verification marker: `[V]` read from the paper's own text, `[A]` abstract
read directly, `[S]` search summary only, `[U]` unverified recollection. Nothing marked `[S]` may
enter a paper; nothing marked `[U]` may enter a paper or a plan. **Every prior-art statement in this
project is currently `[U]`.**

Tests assert mathematics, not current output — entropy bounded by log K, KL zero exactly at
equality, a hidden cue unable to change a pixel — so they stay meaningful when the implementation
changes. Two of them exist specifically because an earlier revision asserted something false, and
they are written to fail on the wrong version.

## Layout

```
src/forecasting/   the world, its closed-form inference, and the task interface
src/uncertainty/   the exact decomposition and the ensemble estimate of it
src/routing/       escalation policies, budgeted curves, and the gain correlation
src/baselines/     forecasters whose error is known before they run
src/temporal/      prefix length as a lever; scene splits and the leakage guard
src/video/         rendering a prefix to frames, under the hidden-cue contract
src/vlm/           model backends, prompts, cache, and the forecaster wrapper (never run)
src/evaluation/    metrics and the protocol, including validation checks V1-V7
```
