#!/usr/bin/env python3
"""Regenerate the paper's numbers, compile it, and report the page count.

One command, because the two steps must not come apart: compiling without regenerating is how a
paper ends up quoting a number no run produced.

    python3 scripts/build_paper.py            # regenerate, compile, report pages
    python3 scripts/build_paper.py --clean    # also remove the LaTeX by-products first
"""
import argparse
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import ROOT

PAPER = os.path.join(ROOT, "paper")
SCRIPTS = os.path.join(ROOT, "scripts")


def run(cmd, cwd, quiet=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(" ".join(cmd) + " FAILED")
        tail = (r.stdout + r.stderr).strip().splitlines()
        print("\n".join(tail[-25:]))
    elif not quiet:
        print(r.stdout)
    return r.returncode


def page_count(log: str) -> int:
    """From the LaTeX log rather than the PDF: modern PDFs put the page tree in a compressed
    object stream, where a byte scan for /Count finds nothing."""
    m = re.search(r"Output written on \S+ \((\d+) pages?", log)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean:
        for pat in ("*.aux", "*.log", "*.out", "*.bbl", "*.blg", "*.fls", "*.fdb_latexmk"):
            for f in glob.glob(os.path.join(PAPER, pat)):
                os.remove(f)

    if run([sys.executable, os.path.join(SCRIPTS, "make_paper_tables.py")], ROOT, quiet=False):
        return 1
    if run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], PAPER):
        return 1

    with open(os.path.join(PAPER, "main.log")) as fh:
        log = fh.read()
    print(f"built paper/main.pdf, {page_count(log)} pages")

    # the body length is what a page limit constrains; the appendix starts after \appendix
    m = re.search(r"\[(\d+)\] \(\./generated/app_worlds", log)
    if m:
        print(f"  body ends on page {m.group(1)} (the appendix starts there)")
    for warn in sorted(set(re.findall(r"(LaTeX Warning: Reference [^\n]+)", log))):
        print("  " + warn)
    for warn in sorted(set(re.findall(r"(LaTeX Warning: Citation [^\n]+)", log))):
        print("  " + warn)
    overfull = len(re.findall(r"Overfull \\hbox", log))
    if overfull:
        print(f"  {overfull} overfull hbox warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
