# paper/

`main.pdf` — *Uncertainty-Aware Routing Needs an Ensemble You Do Not Have.* Nine pages: a five-page
body, references, and a four-section appendix.

An earlier version of this file said the directory should stay empty of prose, because nothing
could be written up while every prior-art statement in the project was marked `[U]` (unverified
recollection). That gate has fired: the literature pass ran, it refuted three of the seven claims
(`research/02_novelty_analysis.md`), and the paper is written on what survived — the
correlated-ensemble regime, the mechanism map, and the prefix lever. Every reference in `refs.bib`
was read from the paper's own arXiv record, so nothing in the bibliography is weaker than `[A]`.

## Build

```
python3 scripts/build_paper.py          # regenerate numbers, compile, report page count
python3 scripts/audit_paper.py          # fail if anything drifted or a numeral was typed
```

`build_paper.py` regenerates before it compiles, on purpose: compiling without regenerating is how
a paper ends up quoting a number no run produced.

## The paper contains no typed numerals

Every quantity in `main.tex` is a macro from `generated/numbers.tex`, and every table body is a
file in `generated/`, all written by `scripts/make_paper_tables.py` from
`experiments/results/*/summary.json`. Appendix prose that carries numbers lives in
`scripts/paper_appendix.py` for the same reason. So re-running an experiment and rebuilding cannot
leave a stale number in the prose.

This is not a style preference. Defect **D10** was four stale rows inside a table the audit had
declared clean, and a stale row in a markdown file can at least be corrected in place; in a PDF a
reader cannot re-check anything. `scripts/audit_paper.py` enforces the rule in two parts: it
regenerates into a temporary directory and diffs, and it reads `main.tex` and fails on a typed
numeral that is not a declared experimental *setting* (a temperature, a correlation, a prefix
length). Both halves are verified to fail on purpose.

## Format, and the one thing to change before submitting

`neuripsish.sty` reproduces the NeurIPS layout — 10pt Times on a 5.5 × 9 inch block — because the
official style file is not redistributable and was not on the build machine. **It is not the
official file.** To submit, drop the venue's own `.sty` into this directory and change one line in
`main.tex`:

```latex
\usepackage{neuripsish}        ->    \usepackage{neurips_2026}
```

Nothing else in `main.tex` depends on it. `\author{Anonymous submission}` is the anonymous form;
replace it with whatever block the venue asks for.

**Venue assumption, stated so it can be corrected.** The project's own recommendation was a
workshop rather than a main track (`research/03_research_decision.md` §3), and the paper is written
for a NeurIPS-style workshop on uncertainty quantification or negative results. The body is five
pages, which fits the common 8- or 9-page workshop limit with room to spare. For a strict **4-page**
limit, move §6 (Robustness) and Table 3 into the appendix and cut the second paragraph of §5; do
**not** cut §7, which is the section that keeps the paper honest about a synthetic world.

## Figures

`figures/` is written by `scripts/make_figures.py` and is safe to regenerate at any time. Only
`fig4_mechanisms.png` is used by the paper (Appendix C); the others are the project's own figures
and are kept because the research documents reference them.

## Layout

```
main.tex              the paper. No typed numerals; one \usepackage line to swap for submission
refs.bib              12 references, every one read from its own arXiv record
neuripsish.sty        unofficial NeurIPS-like layout, replaced at submission time
generated/            numbers.tex and every table body, written from experiments/results/
figures/              written by scripts/make_figures.py
main.pdf              the built paper
```
