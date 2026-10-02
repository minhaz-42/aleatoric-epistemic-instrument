# TMLR submission (prepared 2026-10-02)

- **Submit at:** https://openreview.net/group?id=TMLR (button "TMLR Submission"; invitation
  `TMLR/-/Submission`, verified live). Rolling, no deadline.
- **Why TMLR:** AISTATS 2027 was the best fit, but its mandatory abstract registration closed on
  29 Sep 2026 and the author list freezes at that point.
- **PDF:** `paper/main.pdf`, built with the official TMLR style in anonymous submission mode.

## Verified on 2026-10-02
- Clean rebuild (pdflatex, bibtex, pdflatex x2): 0 errors, 0 overfull boxes, 0 undefined references
  or citations, 0 "??" in the PDF; author-year citations (`\citet` now reads "Young (2026)").
- 10 pages: main text pages 1-5, references pages 5-6, appendices A-F pages 6-10. That makes it a
  **Regular submission**. The only figure sits in Appendix C, as it did in the earlier layout.
- Anonymous: "Anonymous authors / Paper under double-blind review"; no names, emails, repository URL
  or acknowledgements in the PDF text. All pages checked visually.
- No result was recomputed; every number still comes from `paper/generated/` as committed.
- New Figure 1 (`paper/fig_overview.tex`): a colored-pencil schematic of the setting (world, exact
  loss split, what a router can measure). It is conceptual, carries no result numbers, and builds
  from `paper/pencil.tex`. Rebuilt 2026-10-02: 10 pages, 0 overfull, 0 undefined.

## Form fields
- **Title:** Uncertainty-Aware Routing Needs an Ensemble You Do Not Have
- **Abstract:** contents of `paper/abstract.txt`
- **Submission length:** Regular submission (no more than 12 pages of main content)
- **Supplementary material:** optional. The code reproduces on CPU, so an anonymised zip of the repo
  is a strong addition; strip git history and author names first, because it is public.
- **Previous TMLR submission:** leave empty
- **Competing interests:** None
- **Human subjects reporting:** N/A
