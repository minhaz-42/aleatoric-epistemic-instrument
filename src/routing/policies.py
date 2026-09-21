"""Escalation policies: given one cheap forecast, decide whether to spend on a better one.

Every policy returns a score per scene, higher meaning "escalate sooner". They differ only in what
they are allowed to look at, and that is the experiment.

The argument the policies are built to test is an identity, not an intuition. Escalating scene s
from a cheap forecaster Q_c to an expensive Q_e changes the expected loss by

    [ H(P*) + KL(P* || Q_c) ]  -  [ H(P*) + KL(P* || Q_e) ]  =  KL(P* || Q_c) - KL(P* || Q_e)

The aleatoric term cancels exactly. It is paid whichever forecaster answers, so it carries no
information about whether escalating is worth it. But H(Q_c), the total predictive uncertainty that
standard confidence-based routing ranks by, contains that term and is often dominated by it. A
confidence-routed system therefore spends its budget partly in proportion to a quantity that is
by construction irrelevant to the decision -- and does so most severely in exactly the domains
where routing is most attractive, because those are the domains with high irreducible uncertainty.

`OracleGain` is the upper bound this predicts, `TotalUncertainty` the standard practice it
indicts, and `EpistemicBALD` the practical estimator that is supposed to close the gap.
"""
import numpy as np

from ..forecasting.world import kl
from ..uncertainty.decomposition import ensemble_split


class Policy:
    name = "base"
    uses_oracle = False

    def score(self, row: dict) -> float:
        raise NotImplementedError

    def scores(self, rows) -> np.ndarray:
        return np.array([self.score(r) for r in rows], dtype=float)


class OracleGain(Policy):
    """Ranks by the true reduction in loss. The upper bound no deployable policy can exceed."""
    name = "oracle_gain"
    uses_oracle = True

    def score(self, row):
        return row["epistemic_true"] - row["epistemic_true_expensive"]


class TotalUncertainty(Policy):
    """Ranks by H(Qbar): total predictive entropy. Standard practice, and the confounded one."""
    name = "total_uncertainty"

    def score(self, row):
        return row["total_hat"]


class EpistemicBALD(Policy):
    """Ranks by the ensemble mutual information. The intended fix, and the one under test."""
    name = "epistemic_bald"

    def score(self, row):
        return row["epistemic_hat"]


class LeastConfidence(Policy):
    """Ranks by 1 - max_k Qbar(k). The most common deployed heuristic."""
    name = "least_confidence"

    def score(self, row):
        return 1.0 - row["max_prob"]


class Margin(Policy):
    """Ranks by the negative gap between the top two outcomes."""
    name = "margin"

    def score(self, row):
        return -(row["max_prob"] - row["second_prob"])


class Random(Policy):
    """Ranks at random. The floor: any policy that fails to beat it has no signal."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def scores(self, rows):
        from ..utils.seeding import rng
        return rng(self.seed, "random_policy").random(len(rows))

    def score(self, row):
        raise NotImplementedError("Random scores a batch at once")


class TrueAleatoric(Policy):
    """Ranks by H(P*) alone: escalate where the world is most uncertain.

    A diagnostic rather than a proposal. It ranks by exactly the term that cancels out of the
    escalation decision, so it isolates how much of `TotalUncertainty`'s behaviour is inherited from
    the irreducible component.

    It does NOT follow that this policy must perform at chance. Cancelling from the decision means
    H(P*) contributes nothing *causally*; it can still be statistically correlated with the gain in
    a particular world, and in this one it is -- corr(H(P*), gain) measures +0.28 at tau=0.35 and
    decays to -0.02 by tau=3.5. So this policy looks useful in worlds with a low aleatoric floor
    and stops working as the floor rises, which is the confound made visible rather than a
    counterexample to it. An earlier revision of this docstring predicted chance performance
    everywhere and was wrong.
    """
    name = "true_aleatoric"
    uses_oracle = True

    def score(self, row):
        return row["aleatoric_true"]


def standard_policies(seed: int = 0) -> list:
    return [OracleGain(), EpistemicBALD(), TotalUncertainty(), LeastConfidence(),
            Margin(), Random(seed), TrueAleatoric()]
