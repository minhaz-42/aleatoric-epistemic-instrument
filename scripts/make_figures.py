#!/usr/bin/env python3
"""Figures, from the summaries the experiment scripts wrote.

Reads `experiments/results/*/summary.json` and writes to `figures/` and `paper/figures/`. It does
not re-run anything, so a figure can never disagree with the number in the documentation: both come
from the same file. Run the experiments first.

Palette: four categorical slots that pass the all-pairs colour-vision gate together (blue, orange,
aqua, violet). A fifth hue does not pass with any of them, so the fifth series -- the
purely-diagnostic aleatoric policy -- is drawn in neutral grey and dashed, which also matches its
status as a control rather than a proposal. Aqua sits below 3:1 against the surface, so every
series carries a direct label at the line end in addition to the legend, and the underlying CSV is
written beside each figure as the table view.

    python3 scripts/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.io import FIGURES, RESULTS, ROOT, TABLES, read_summary

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GRID = "#e6e5e1"
DIV_POS, DIV_MID, DIV_NEG = "#2a78d6", "#f0efec", "#d03b3b"

POLICY_STYLE = {
    "epistemic_bald":    (BLUE,   "-",  "o", "epistemic (BALD)"),
    "total_uncertainty": (ORANGE, "-",  "s", "total uncertainty"),
    "least_confidence":  (AQUA,   "-",  "^", "least confidence"),
    "margin":            (VIOLET, "-",  "D", "margin"),
    "true_aleatoric":    (MUTED,  "--", "v", "true aleatoric (control)"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 5,
})


def _save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "paper", "figures"), exist_ok=True)
    for d in (FIGURES, os.path.join(ROOT, "paper", "figures")):
        fig.savefig(os.path.join(d, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _table(df, name):
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, name), index=False)
    print(f"  wrote tables/{name}")


# -- figure 1 ---------------------------------------------------------------------------------
def fig_decomposition():
    """The estimated aleatoric term is not a property of the world, though the true one is."""
    s = read_summary("decomposition")
    rows = s["forecasters"]
    short = {"bayes_optimal": "Bayes\noptimal", "tempered[T=0.5]": "tempered\nT=0.5",
             "tempered[T=2]": "tempered\nT=2", "partial[heading,speed]": "partial\nhead+speed",
             "partial[heading]": "partial\nheading",
             "logit_noise[sigma=0.75,rho=0]": "logit noise\nσ=0.75", "marginal": "marginal",
             "uniform": "uniform"}
    names = [short.get(r["forecaster"], r["forecaster"]) for r in rows]
    x = np.arange(len(rows))
    w = 0.2

    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    ax.bar(x - 1.5 * w, [r["aleatoric_true"] for r in rows], w, color=BLUE, label="aleatoric, true H(P*)")
    ax.bar(x - 0.5 * w, [r["aleatoric_hat"] for r in rows], w, color=BLUE, alpha=0.40,
           label="aleatoric, ensemble estimate")
    ax.bar(x + 0.5 * w, [r["epistemic_true"] for r in rows], w, color=ORANGE, label="epistemic, true KL(P*||Q)")
    ax.bar(x + 1.5 * w, [r["epistemic_hat"] for r in rows], w, color=ORANGE, alpha=0.40,
           label="epistemic, ensemble estimate")
    ax.axhline(s["expected_aleatoric_exact"], color=MUTED, ls=":", lw=1.2)
    ax.annotate(f"exact E[H(P*)] = {s['expected_aleatoric_exact']:.3f}",
                (0.32, s["expected_aleatoric_exact"]), xytext=(0, 6),
                textcoords="offset points", fontsize=7.5, color=MUTED, ha="left")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylim(0, 1.52)
    ax.set_ylabel("nats")
    ax.set_title("The true aleatoric term is identical for every forecaster.\n"
                 "Its ensemble estimate tracks the forecaster's confidence instead.", loc="left")
    ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    _save(fig, "fig1_decomposition.png")
    _table(pd.DataFrame(rows), "decomposition.csv")


# -- figure 2 ---------------------------------------------------------------------------------
def fig_confound():
    """V6: the estimate and the quantity it stands in for move in opposite directions."""
    v = read_summary("validation")
    tab = [c for c in v["checks"] if c["id"] == "V6"][0]["detail"]["table"]
    rho = [t["rho"] for t in tab]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(rho, [t["single_member_kl"] for t in tab], color=MUTED, ls="--", marker="v",
            label="one member's error")
    ax.plot(rho, [t["epistemic_true"] for t in tab], color=ORANGE, marker="s",
            label="ensemble's true error  KL(P*||Q̄)")
    ax.plot(rho, [t["epistemic_hat"] for t in tab], color=BLUE, marker="o",
            label="ensemble's estimate  (BALD)")
    for y, c, txt in ((tab[-1]["epistemic_true"], ORANGE, "rises 9.6×"),
                      (tab[-1]["epistemic_hat"], BLUE, "falls to 0")):
        ax.annotate(txt, (rho[-1], y), xytext=(-4, 6 if c == ORANGE else -14),
                    textcoords="offset points", fontsize=7.5, color=c, ha="right")
    ax.set_xlabel("ensemble member correlation  ρ")
    ax.set_ylabel("nats")
    ax.set_title("As ensemble members correlate, the estimate of\n"
                 "epistemic uncertainty inverts against the truth", loc="left")
    ax.grid(lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1)
    _save(fig, "fig2_confound.png")
    _table(pd.DataFrame(tab), "confound_rho_sweep.csv")


# -- figure 3 ---------------------------------------------------------------------------------
def fig_routing():
    """Routing quality against the aleatoric floor, and against ensemble correlation."""
    s = read_summary("routing")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)

    for panel, (ax, (key, xk, xlabel)) in enumerate(zip(axes, [
            ("tau_sweep", "expected_aleatoric", "irreducible floor  E[H(P*)]  (nats)"),
            ("rho_sweep", "rho", "ensemble member correlation  ρ")])):
        rows = s[key]
        xs = [r[xk] for r in rows]
        for pol, (c, ls, mk, lab) in POLICY_STYLE.items():
            ys = [r[f"corr_{pol}"] for r in rows]
            ax.plot(xs, ys, color=c, ls=ls, marker=mk, label=lab,
                    lw=2.0 if pol in ("epistemic_bald", "total_uncertainty") else 1.4,
                    alpha=1.0 if pol in ("epistemic_bald", "total_uncertainty") else 0.75)
        # No end-of-line labels: every series converges at the right edge of the left panel, so
        # direct labels there collide with the marks they name. Identity is carried by the legend,
        # and the relief rule for the low-contrast aqua slot is met by the table view written
        # beside the figure (tables/routing_*.csv).
        ax.axhline(0, color=INK2, lw=1.0)
        if panel == 0:
            ax.annotate("routing at random", (0.02, 0.0), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points", fontsize=7, color=INK2)
        ax.set_xlabel(xlabel); ax.grid(lw=0.6); ax.set_axisbelow(True)
        ax.margins(x=0.10)

    axes[0].set_ylabel("corr(policy score, true escalation gain)")
    axes[0].set_title("Confidence-based routing inverts as the\nirreducible floor rises", loc="left")
    axes[1].set_title("and the epistemic estimate that should fix it\ndegrades as members correlate",
                      loc="left")
    axes[0].legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    _save(fig, "fig3_routing.png")
    _table(pd.DataFrame(s["tau_sweep"]), "routing_tau_sweep.csv")
    _table(pd.DataFrame(s["rho_sweep"]), "routing_rho_sweep.csv")


# -- figure 4 ---------------------------------------------------------------------------------
def fig_mechanisms():
    """Whether confidence routing fails at all depends on why the cheap model is worse."""
    s = read_summary("mechanisms")
    rows = s["rows"]
    mechs = list(dict.fromkeys(r["mechanism"] for r in rows))
    taus = sorted({r["tau"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, pol, title in zip(axes, ["total_uncertainty", "epistemic_bald"],
                              ["ranking by total uncertainty", "ranking by epistemic (BALD)"]):
        M = np.array([[next((r[f"corr_{pol}"] for r in rows
                             if r["mechanism"] == m and r["tau"] == t), np.nan)
                       for t in taus] for m in mechs], dtype=float)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
        im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        # Shade the not-computable cells so they read as "no estimate exists" rather than as a
        # missing measurement: a deterministic forecaster has no ensemble spread at all.
        for i, j in zip(*np.where(np.isnan(M))):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f4f3f0",
                                       edgecolor=SURFACE, hatch="///", lw=0))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "n/a" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color=INK if (np.isnan(v) or abs(v) < 0.55) else "#ffffff")
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f"{t:g}" for t in taus], fontsize=7.5)
        ax.set_yticks(range(len(mechs)))
        ax.set_yticklabels([m.replace("_", " ") for m in mechs] if pol == "total_uncertainty"
                           else [], fontsize=8)
        ax.set_xlabel("world temperature  τ  (higher = larger irreducible floor)")
        ax.set_title(title, loc="left", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02)
    cb.set_label("corr(score, true gain)", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Whether confidence-based routing fails depends on why the cheap model is worse",
                 x=0.005, y=1.06, ha="left", fontsize=10)
    axes[1].annotate("hatched = not computable:\na single deterministic model\nhas no ensemble spread",
                     (0.5, -0.34), xycoords="axes fraction", ha="center", fontsize=7, color=MUTED)
    _save(fig, "fig4_mechanisms.png")
    _table(pd.DataFrame(rows), "mechanisms.csv")


def main():
    print("figures:")
    made = 0
    for fn, need in ((fig_decomposition, "decomposition"), (fig_confound, "validation"),
                     (fig_routing, "routing"), (fig_mechanisms, "mechanisms")):
        if not os.path.exists(os.path.join(RESULTS, need, "summary.json")):
            print(f"  skip {fn.__name__}: run the experiment that writes {need}/ first")
            continue
        fn(); made += 1
    print(f"{made} figures written to figures/ and paper/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
