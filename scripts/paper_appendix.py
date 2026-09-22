"""The appendix prose blocks the paper includes, kept out of `main.tex` for one reason.

`main.tex` contains no typed numerals -- every quantity is a macro generated from the run
summaries, so re-running an experiment cannot leave a stale number in the paper. The appendix has
prose with numbers in it too, so it obeys the same rule: the text lives here, the numbers stay
macros, and `scripts/make_paper_tables.py` writes both into `paper/generated/`.

These are LaTeX source strings. The macros they reference (\\JointStates, \\NumChecks, ...) are
defined in `paper/generated/numbers.tex` by the same generator.
"""

WORLDS = r"""Every result in the paper is measured in two worlds that differ on every axis that
could plausibly matter, so that a finding which depends on the construction can be told from one
that does not. \textsc{kitchen} is the important one: its evidence is \emph{non-additive}, so
losing a cue can be catastrophic rather than gradual, which is the structure most likely to break
a mechanism map built on gradual evidence loss. Cue visibilities are deliberately spread rather
than uniform --- if every cue were equally visible the observability mask would carry no
information and every scene would sit at the same irreducible floor, which is the degenerate case
a routing experiment must avoid.

\begin{table}[h]
\centering
\small
\caption{The two worlds. Only the second admits interactions between cues, and the instrument's
checks are world-independent by construction rather than by configuration.}
\begin{tabular}{lcc}
\toprule
& \textsc{junction} & \textsc{kitchen} \\
\midrule
outcomes & 4 & 3 \\
cues & 4 & 5 \\
cue alphabets & uniformly ternary & mixed binary and ternary \\
symmetry & left/right mirrored & none \\
evidence & \textbf{additive} (linear in the cues) & \textbf{non-additive} (pairwise) \\
joint cue states & \JointStates & \KitchenStates \\
irreducible floor at the default $\tau$ & \FloorJ{} nats & \FloorK{} nats \\
validation checks passed & \NumChecks/\NumChecks & \NumChecks/\NumChecks \\
\bottomrule
\end{tabular}
\end{table}
"""

CURVES = r"""The floor sweep with a single cheap/expensive pair, for reference, since it is the
setting in which the effect was first reported: ranking by total predictive uncertainty correlates
\RoutTotalLo{} with the true gain at the lowest floor and \RoutTotalHi{} at the highest, with a
budgeted skill of \RoutSkillTotalHi{} there --- that is, well below random. Ranking by the
epistemic estimate falls less far over the same sweep (\RoutBaldLo{} to \RoutBaldHi), and both pass
through zero near $\tau = 1.3$. \citet{peale2026flexible} report the same qualitative comparison
against a pointwise-optimal policy on synthetic data; what \S\ref{sec:mechanisms} adds is that
this setting is one row of a table whose other rows do not invert at all.

The correlation-sweep counterpart is in \S\ref{sec:confound}: the epistemic policy's budgeted
skill against random goes \RoutSkillBaldRhoZero{} at $\rho = 0$ to \RoutSkillBaldRhoHigh{} at
$\rho = 0.75$, so the budgeted number and the rank correlation agree about where the estimator
stops helping. Ties in a policy's scores are broken by index order rather than at random, which is
the pessimistic choice for a policy that emits many identical scores --- and that is the case worth
being pessimistic about, since a policy which cannot separate scenes should not be rewarded for it.
"""

REPRO_TEMPLATE = r"""\begin{verbatim}
python3 tests/test_instrument.py                      # %(NumTests)s/%(NumTests)s identities
python3 scripts/validate_instrument.py                # %(NumChecks)s/%(NumChecks)s checks, junction
python3 scripts/validate_instrument.py --world kitchen --tau 0.70
python3 scripts/run_decomposition.py                  # Table 5
python3 scripts/run_routing.py                        # Appendix D
python3 scripts/run_mechanisms.py                     # Table 2
python3 scripts/run_mechanisms.py --world kitchen     # Table 6
python3 scripts/run_prefix.py                         # Table 3
python3 scripts/run_prefix.py --world kitchen --tau 0.70
python3 scripts/run_robustness.py                     # Table 7
python3 scripts/make_figures.py
python3 scripts/audit_numbers.py                      # re-measures every documented number
python3 scripts/make_paper_tables.py                  # regenerates every number in this paper
python3 scripts/audit_paper.py                        # fails on drift or a typed numeral
\end{verbatim}

Results land in \texttt{experiments/results/<name>/} as a \texttt{records.csv} (one row per scored
prediction) and a \texttt{summary.json} (the reduced numbers), so every claim traces to the rows it
came from. Seeds derive from one root through a hash of a label path, so adding an arm to an
experiment cannot change the draws of the arms beside it.
"""
