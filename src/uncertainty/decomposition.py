"""Splitting a forecast's uncertainty into the part that is the world's and the part that is the
forecaster's.

Two decompositions live here and the difference between them is the project's subject.

The EXACT one is available only because the world is synthetic. For a forecast Q on a scene whose
true conditional is P*,

    E_{y ~ P*}[-log Q(y)]  =  H(P*)          +  KL(P* || Q)
                              ^ aleatoric       ^ epistemic

Both terms are computed in closed form. H(P*) is a property of the world and the prefix alone: no
forecaster can reduce it, and every forecaster pays it. KL(P* || Q) is a property of the forecaster
alone: it is zero exactly when the forecaster is Bayes-optimal, and it is the entire quantity that a
better model, or a longer prefix, could recover.

The EMPIRICAL one is what a practitioner can actually compute, from an ensemble of M forecasts with
no access to P*. Writing Qbar for the ensemble mean,

    H(Qbar)        =  (1/M) sum_m H(Q_m)   +   [ H(Qbar) - (1/M) sum_m H(Q_m) ]
    ^ total           ^ aleatoric estimate      ^ epistemic estimate, the mutual information

The second decomposition is the standard one -- the bracketed term is BALD -- and it is usually
introduced as though it estimated the first. It does not. The bracketed term measures *disagreement
between the ensemble members*, which equals the forecaster's true error only when the members'
errors are independent. When the members share a bias the term collapses toward zero while
KL(P* || Qbar) stays where it was, and the ensemble reports confidence it has not earned.

This module computes both so that the gap between them can be measured rather than assumed. Which
is the whole reason the world is synthetic: on real video only the second is computable, so the
error in it is invisible.
"""
from dataclasses import dataclass

import numpy as np

from ..forecasting.world import entropy, kl


@dataclass(frozen=True)
class ExactSplit:
    """The closed-form decomposition. Available only in a synthetic world."""
    aleatoric: float      # H(P*), in nats: irreducible given this prefix
    epistemic: float      # KL(P* || Q), in nats: recoverable by a better forecaster
    expected_nll: float   # their sum: the forecaster's expected loss on this scene
    realised_nll: float   # -log Q(y) for the outcome that actually happened

    @property
    def total(self) -> float:
        return self.expected_nll


@dataclass(frozen=True)
class EnsembleSplit:
    """The decomposition a practitioner can compute, from ensemble disagreement alone."""
    total: float             # H(Qbar)
    aleatoric_hat: float     # mean_m H(Q_m)
    epistemic_hat: float     # H(Qbar) - mean_m H(Q_m), the mutual information (BALD)
    mean_prediction: np.ndarray


def exact_split(posterior: np.ndarray, q: np.ndarray, y: int) -> ExactSplit:
    """The closed-form split for one forecast against one known conditional."""
    q = np.asarray(q, dtype=float)
    a = entropy(posterior)
    e = kl(posterior, q)
    return ExactSplit(aleatoric=a, epistemic=e, expected_nll=a + e,
                      realised_nll=float(-np.log(max(q[y], 1e-12))))


def ensemble_split(members: np.ndarray) -> EnsembleSplit:
    """The empirical split from an (M, K) stack of ensemble member forecasts.

    `epistemic_hat` is non-negative by Jensen for any M >= 1, and is exactly zero when every member
    returns the same distribution -- including when they are all identically wrong.
    """
    members = np.atleast_2d(np.asarray(members, dtype=float))
    qbar = members.mean(axis=0)
    total = entropy(qbar)
    aleatoric_hat = float(np.mean([entropy(m) for m in members]))
    return EnsembleSplit(total=total, aleatoric_hat=aleatoric_hat,
                         epistemic_hat=total - aleatoric_hat, mean_prediction=qbar)


def decompose_run(world, scenes, forecaster, n_members: int = 1) -> "list[dict]":
    """Score one forecaster over a list of scenes, returning one row per scene.

    Each row carries both decompositions side by side, plus the ensemble mean's true epistemic
    error. That last column is the reference the empirical estimate is judged against, and it is the
    column that has no counterpart on real video.
    """
    from ..forecasting.task import evidence_of

    rows = []
    for sc in scenes:
        ev = evidence_of(sc, world)
        members = np.stack([forecaster.predict(ev, m) for m in range(n_members)])
        es = ensemble_split(members)
        xs = exact_split(sc.posterior, es.mean_prediction, sc.y)
        rows.append({
            "scene_id": sc.scene_id,
            "forecaster": forecaster.name,
            "n_visible": sc.n_visible,
            "y": sc.y,
            "argmax": int(np.argmax(es.mean_prediction)),
            "correct": int(np.argmax(es.mean_prediction) == sc.y),
            "p_y": float(es.mean_prediction[sc.y]),
            "aleatoric_true": xs.aleatoric,
            "epistemic_true": xs.epistemic,
            "expected_nll": xs.expected_nll,
            "realised_nll": xs.realised_nll,
            "total_hat": es.total,
            "aleatoric_hat": es.aleatoric_hat,
            "epistemic_hat": es.epistemic_hat,
            "bayes_nll": xs.aleatoric,
        })
    return rows
