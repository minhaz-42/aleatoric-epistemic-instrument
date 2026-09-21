"""Budgeted escalation curves, and the single number that summarises one.

A routing policy is evaluated the way it is deployed: under a fixed budget. At budget b the policy
escalates the b fraction of scenes it scores highest, the rest are answered by the cheap forecaster,
and the curve reports the resulting mean expected loss. Sweeping b from 0 to 1 traces the whole
accuracy-cost frontier, which is the object a deployment actually chooses a point on.

Two conventions, both of which matter for reading the numbers.

Expected loss, not realised loss, is the headline. Realised loss on a sampled outcome is an
unbiased but noisy estimate of the same thing, and at the scene counts used here its Monte Carlo
error is comparable to the effect being measured. Both are computed; `run_routing.py` reports the
expected curve and the realised curve side by side so the agreement can be checked rather than
asserted.

Efficiency is normalised against the oracle *at the same budget*, not against the fully-escalated
loss. A policy at budget b cannot do better than escalating the b scenes with the largest true
gain, so that is the only fair denominator. Normalising against full escalation would flatter every
policy at small b and make budgets incomparable.
"""
import numpy as np
from scipy.stats import spearmanr

DEFAULT_BUDGETS = np.linspace(0.0, 1.0, 21)


def _losses(rows):
    """Per-scene expected loss under each forecaster, from the exact decomposition."""
    a = np.array([r["aleatoric_true"] for r in rows])
    ec = np.array([r["epistemic_true"] for r in rows])
    ee = np.array([r["epistemic_true_expensive"] for r in rows])
    return a + ec, a + ee


def curve(rows, scores, budgets=DEFAULT_BUDGETS, realised: bool = False):
    """Mean loss as a function of escalation budget, for one policy's scores.

    Ties are broken by the stable sort's index order rather than at random. That makes a run
    reproducible, and it is the pessimistic choice for a policy that emits many identical scores --
    which is the case worth being pessimistic about, since a policy that cannot separate scenes
    should not be rewarded for it.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(rows)
    if realised:
        cheap = np.array([r["realised_nll"] for r in rows])
        exp = np.array([r["realised_nll_expensive"] for r in rows])
    else:
        cheap, exp = _losses(rows)
    order = np.argsort(-scores, kind="stable")
    out = np.empty(len(budgets))
    for i, b in enumerate(budgets):
        k = int(round(float(b) * n))
        chosen = np.zeros(n, dtype=bool)
        chosen[order[:k]] = True
        out[i] = np.where(chosen, exp, cheap).mean()
    return np.asarray(budgets, dtype=float), out


def efficiency(rows, scores, budgets=DEFAULT_BUDGETS, realised: bool = False) -> dict:
    """Curve plus its normalised summary against the oracle ranking.

    `realisable` is the fraction of the oracle's achievable gain that the policy captures, budget by
    budget; `auc_efficiency` integrates it over the budget range. Budgets where the oracle itself
    can gain nothing are dropped from the integral rather than scored as 0 or 1, because at those
    budgets every policy is trivially equivalent and including them dilutes the comparison.
    """
    cheap, exp = _losses(rows)
    gains = cheap - exp
    b, loss = curve(rows, scores, budgets, realised=realised)
    _, oracle_loss = curve(rows, gains, budgets, realised=realised)
    base = loss[0] if budgets[0] == 0.0 else cheap.mean()

    captured = base - loss
    available = base - oracle_loss
    live = available > 1e-12
    realisable = np.full_like(captured, np.nan)
    realisable[live] = captured[live] / available[live]
    auc = float(np.trapezoid(realisable[live], b[live]) / (b[live].max() - b[live].min())) \
        if live.sum() > 1 else float("nan")
    return {
        "budgets": b,
        "loss": loss,
        "oracle_loss": oracle_loss,
        "captured": captured,
        "available": available,
        "realisable": realisable,
        "auc_efficiency": auc,
        "loss_all_cheap": float(cheap.mean()),
        "loss_all_expensive": float(exp.mean()),
        "mean_gain": float(gains.mean()),
    }


def gain_correlation(rows, scores) -> float:
    """Spearman correlation between a policy's score and the true escalation gain.

    The primary diagnostic, and easier to read than any budgeted number: it has a clean zero at a
    policy that carries no signal, and a clean sign. A negative value is the case worth naming --
    the policy is not merely uninformative but actively spends the budget on the scenes with the
    least recoverable error, and would be beaten by routing at random.

    Returns nan when the scores are constant, which is a real case rather than an error: a
    forecaster that ignores the prefix emits the same distribution everywhere, so every
    confidence-derived policy scores every scene identically and no ranking exists. That is
    reported as nan rather than as 0, because "carries no signal" and "cannot be ranked at all"
    are different findings.
    """
    scores = np.asarray(scores, dtype=float)
    cheap, exp = _losses(rows)
    if np.ptp(scores) == 0 or np.ptp(cheap - exp) == 0:
        return float("nan")
    return float(spearmanr(scores, cheap - exp).statistic)


def skill(auc_policy: float, auc_random: float) -> float:
    """Budgeted efficiency rescaled so random routing is 0 and the oracle is 1.

    The raw `auc_efficiency` does not have random at zero -- a random policy captures the budget
    fraction of the total gain, which integrates to roughly 0.6 against the oracle rather than to 0.
    That compresses the whole comparison into the top 40% of the range and makes a policy that is
    worse than random still look like it is capturing most of the gain. Every reported comparison
    uses this rescaling; the raw value is kept in the records for traceability.
    """
    denom = 1.0 - auc_random
    return float((auc_policy - auc_random) / denom) if abs(denom) > 1e-12 else float("nan")
