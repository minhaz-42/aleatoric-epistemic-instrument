"""Scoring rules and selective-prediction metrics.

Log loss is the primary metric throughout, for one reason: it is the only one of these that
decomposes exactly into the aleatoric and epistemic terms of equation (3). Brier score and accuracy
are reported beside it as sanity checks and because reviewers expect them, but no claim rests on
them -- a Brier decomposition into the same two terms does not hold exactly, and the project would
have to soften "exactly" to "approximately" everywhere if it led with one.
"""
import numpy as np

from ..uncertainty.calibration import ece, reliability   # re-exported for one import site

__all__ = ["nll", "brier", "accuracy", "ece", "reliability", "risk_coverage", "aurc"]


def nll(probs: np.ndarray, labels: np.ndarray, floor: float = 1e-12) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], floor, None)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score, the squared error against the one-hot outcome."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), np.asarray(labels, dtype=int)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((np.asarray(probs).argmax(axis=1) == np.asarray(labels)).mean())


def risk_coverage(scores: np.ndarray, losses: np.ndarray):
    """Risk as a function of coverage, keeping the LOWEST-scoring cases.

    `scores` is an uncertainty score, so low means confident. At coverage c the forecaster answers
    the c fraction it is most confident about and abstains on the rest; risk is the mean loss over
    what it answered. Returns (coverage, risk), both length N.
    """
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)
    order = np.argsort(scores, kind="stable")
    ordered = losses[order]
    n = len(ordered)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(ordered) / np.arange(1, n + 1)
    return coverage, risk


def aurc(scores: np.ndarray, losses: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is a better uncertainty ranking."""
    coverage, risk = risk_coverage(scores, losses)
    return float(np.trapezoid(risk, coverage) / (coverage[-1] - coverage[0])) if len(coverage) > 1 \
        else float(risk[0])
