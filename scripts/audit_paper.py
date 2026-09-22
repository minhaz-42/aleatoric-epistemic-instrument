#!/usr/bin/env python3
"""Refuse to let the paper drift from the results on disk.

`audit_numbers.py` does this for the markdown in `research/`, where a stale number can at least be
corrected in place. A paper is worse: once it is a PDF, a reader cannot re-check anything. So the
paper holds no measured numeral of its own. Every measurement is a macro in
`paper/generated/numbers.tex`, and every table body is a generated file, all written from
`experiments/results/*/summary.json` by `scripts/make_paper_tables.py`.

This script enforces exactly that, in two parts:

  1. it regenerates into a temporary directory and compares, so a committed generated file that no
     longer matches the summaries is a failure rather than a surprise at build time;
  2. it reads `main.tex` and fails on a typed numeral that is not a declared experimental SETTING.

The second check needs the distinction, so here it is. A setting is a number we chose --- a
temperature, a correlation, a prefix length, a confidence level. A measurement is a number the
experiment produced. Settings may be typed, because the prose has to be able to say which knob was
turned; measurements may not, because that is what drifts. Anything with three or more decimal
places is treated as a measurement whatever it looks like, since that is the shape of every
correlation in the paper.

    python3 scripts/audit_paper.py

Exit code 0 if the paper is consistent with the results, 1 otherwise.
"""
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import ROOT

PAPER = os.path.join(ROOT, "paper")
GEN = os.path.join(PAPER, "generated")
SCRIPTS = os.path.join(ROOT, "scripts")

# Experimental settings, which the prose is allowed to name. Add to this list only for a number
# the experiment was GIVEN; a number the experiment RETURNED belongs in numbers.tex.
SETTINGS = {
    "0", "1", "2", "3", "4", "5", "8", "16", "32",       # counts, outcomes, cues, prefix lengths
    "0.35", "0.5", "0.7", "0.75", "0.9", "1.3", "2.0", "3.5",   # tau and rho grid points
    "0.4", "2.5", "1.2", "0.6",                          # tempering and noise scales named in prose
    "95",                                                # the bootstrap confidence level
    "2025", "2026",                                      # years in prose, not in the bibliography
}
MEASURED_DECIMALS = 3   # three or more decimal places is a measurement, no exceptions

SKIP_LINE = re.compile(r"^\s*%|^\s*\\(input|include|usepackage|documentclass|bibliography"
                       r"|bibliographystyle|newcommand|renewcommand|label|ref|cite)")
NUMBER = re.compile(r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)")


def regenerate_and_compare() -> list:
    """Regenerate into a temp dir; report every generated file that differs or is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, PREDICTIVE_EVIDENCE_GEN_DIR=tmp)
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "make_paper_tables.py")],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        if r.returncode != 0:
            return [f"make_paper_tables.py failed: {r.stderr.strip().splitlines()[-1:]}"]
        problems = []
        fresh = sorted(os.listdir(tmp))
        for name in fresh:
            a, b = os.path.join(tmp, name), os.path.join(GEN, name)
            if not os.path.exists(b):
                problems.append(f"{name}: generated file is missing from paper/generated/")
            elif not filecmp.cmp(a, b, shallow=False):
                problems.append(f"{name}: committed copy does not match the results on disk")
        for name in sorted(os.listdir(GEN)):
            if name.endswith(".tex") and name not in fresh:
                problems.append(f"{name}: in paper/generated/ but no longer generated")
        return problems


def typed_numerals(path: str) -> list:
    """Numerals in the prose that are not declared settings."""
    bad, in_verbatim = [], False
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            if "\\begin{verbatim}" in line:
                in_verbatim = True
            if "\\end{verbatim}" in line:
                in_verbatim = False
                continue
            if in_verbatim or SKIP_LINE.match(line):
                continue
            text = line.split("%")[0]
            for tok in NUMBER.findall(text):
                dec = len(tok.split(".")[1]) if "." in tok else 0
                if dec >= MEASURED_DECIMALS or tok not in SETTINGS:
                    bad.append((i, tok, text.strip()[:88]))
    return bad


def main():
    ok = True
    problems = regenerate_and_compare()
    if problems:
        ok = False
        print("generated files are stale -- run scripts/make_paper_tables.py:")
        for p in problems:
            print(f"  {p}")
    else:
        print(f"generated files match the results on disk ({len(os.listdir(GEN))} files)")

    main_tex = os.path.join(PAPER, "main.tex")
    bad = typed_numerals(main_tex)
    if bad:
        ok = False
        print("\ntyped numerals in main.tex that are not declared settings:")
        for line, tok, ctx in bad:
            print(f"  main.tex:{line}  {tok!r}  in: {ctx}")
        print("\nIf it is a measurement, put it in numbers.tex via scripts/make_paper_tables.py.")
        print("If it is a setting the experiment was given, add it to SETTINGS in this script.")
    else:
        print("main.tex contains no typed measurement")

    print("\npaper audit PASSED" if ok else "\npaper audit FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
