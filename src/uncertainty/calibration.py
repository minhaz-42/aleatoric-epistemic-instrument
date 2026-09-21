"""Calibration of a forecast distribution, and its relationship to the decomposition.

Calibration is reported alongside the decomposition because the two answer different questions and
are routinely confused. A forecaster can be perfectly calibrated and still carry large epistemic
uncertainty -- the marginal forecaster is the extreme case, being exactly calibrated by construction
while discarding every cue. Calibration constrains the *confidence*; only the KL term constrains
how much a better forecaster could recover.
"""
import numpy as np


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 confidence, equal-width binning.

    Equal-width rather than equal-mass because the comparison across forecasters needs a fixed bin
    grid; equal-mass bins move with the confidence distribution and are not comparable between a
    sharp and a flat forecaster.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    hit = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not sel.any():
            continue
        total += sel.mean() * abs(hit[sel].mean() - conf[sel].mean())
    return float(total)


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """(bin centre, mean confidence, empirical accuracy, count) per occupied bin."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    hit = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            out.append(((lo + hi) / 2, float(conf[sel].mean()), float(hit[sel].mean()), int(sel.sum())))
    return out
