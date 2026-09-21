"""Forecasters whose error is known in closed form before they are run.

These exist to validate the protocol, not to be competitive. A correct protocol must score the
Bayes-optimal forecaster at exactly zero epistemic uncertainty and the uniform forecaster at exactly
log K expected loss; a protocol that does not separate those two is broken, and that check runs
before any vision-language model is loaded.

The family is ordered by what it is designed to expose:

  BayesOptimal      the floor. Epistemic exactly 0. Anything else is a protocol defect.
  Uniform           expected NLL exactly log K, for every scene, in every world.
  Marginal          ignores the prefix. Excess loss equals I(y; z_m) exactly -- check V3.
  PartialEvidence   reads only some of the visible cues. Interpolates between the two above.
  Tempered          the right ranking with the wrong confidence. Isolates calibration from accuracy.
  LogitNoise        the one that matters. Its ensemble correlation `rho` is the confound control.

`LogitNoise` is the instrument's main probe. Its members share a fraction `rho` of their logit
perturbation, so rho=0 gives an ensemble of genuinely independent forecasters and rho=1 gives an
ensemble of clones that are all wrong in the same direction. Mutual-information estimates of
epistemic uncertainty read the spread between members, so they are correct at rho=0 and must
collapse to zero at rho=1 while the true KL stays put. That is not a bug being demonstrated: it is
the exact regime a set of vision-language models sharing pretraining sits in, which is why the
instrument can measure it at all.
"""
import numpy as np
from scipy.special import softmax

from ..forecasting.task import Evidence, Forecaster
from ..utils.seeding import rng


def _posterior_from_evidence(world, ev: Evidence) -> np.ndarray:
    """Rebuild P*(y | z_m) from an Evidence object alone."""
    z = np.zeros(len(ev.cue_names))
    for i, n in enumerate(ev.cue_names):
        if n in ev.visible:
            z[i] = ev.visible[n]
    return world.posterior(z, ev.mask)


class BayesOptimal(Forecaster):
    """Returns P* exactly. KL(P* || Q) = 0 by construction, for every scene."""
    name = "bayes_optimal"
    uses_oracle = True

    def __init__(self, world):
        self.world = world

    def predict(self, ev, member=0):
        return _posterior_from_evidence(self.world, ev)


class Uniform(Forecaster):
    """Uniform over outcomes. Expected NLL is exactly log K everywhere."""
    name = "uniform"

    def __init__(self, K: int):
        self.K = K

    def predict(self, ev, member=0):
        return np.full(self.K, 1.0 / self.K)


class Marginal(Forecaster):
    """P(y) with the prefix ignored. Excess loss equals the mutual information I(y; z_m)."""
    name = "marginal"
    uses_oracle = True

    def __init__(self, world):
        self.world = world
        self._p = world.marginal()

    def predict(self, ev, member=0):
        return self._p.copy()


class PartialEvidence(Forecaster):
    """Bayes-optimal over a subset of the cues, treating the rest as unobserved.

    Not a degraded oracle but a correctly-specified forecaster with a smaller field of view, so its
    excess loss is exactly I(y; z_m) - I(y; z_keep) and stays non-negative by construction.
    """
    uses_oracle = True

    def __init__(self, world, keep):
        self.world = world
        self.keep = tuple(keep)
        self.name = "partial[" + ",".join(self.keep) + "]"
        self._keep_idx = np.array([c.name in self.keep for c in world.spec.cues], dtype=bool)

    def predict(self, ev, member=0):
        z = np.zeros(len(ev.cue_names))
        for i, n in enumerate(ev.cue_names):
            if n in ev.visible:
                z[i] = ev.visible[n]
        return self.world.posterior(z, ev.mask & self._keep_idx)


class Tempered(Forecaster):
    """P* raised to 1/temperature and renormalised: the right ranking, the wrong confidence.

    temperature < 1 is overconfident, > 1 underconfident, = 1 the oracle. The accuracy of the
    argmax is untouched, which is what makes this the clean separator between being right and
    knowing how right you are.
    """
    uses_oracle = True

    def __init__(self, world, temperature: float):
        self.world = world
        self.temperature = float(temperature)
        self.name = f"tempered[T={temperature:g}]"

    def predict(self, ev, member=0):
        p = _posterior_from_evidence(self.world, ev)
        return softmax(np.log(np.clip(p, 1e-12, None)) / self.temperature)


class LogitNoise(Forecaster):
    """P* perturbed in logit space, with a tunable correlation between ensemble members.

    The perturbation for member m of scene s is

        eps = sigma * ( sqrt(rho) * shared(s) + sqrt(1 - rho) * private(s, m) )

    so any two distinct members correlate at exactly `rho` and every member has the same marginal
    noise scale `sigma` regardless of rho. Holding the marginal scale fixed while sweeping rho is
    what makes the confound control interpretable: the true error of a single member does not change
    with rho, only the ensemble's ability to see it does.
    """
    stochastic = True
    uses_oracle = True

    def __init__(self, world, sigma: float, rho: float = 0.0, seed: int = 20260921):
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        self.world = world
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.seed = int(seed)
        self.name = f"logit_noise[sigma={sigma:g},rho={rho:g}]"

    def _eps(self, scene_id: int, member: int, K: int) -> np.ndarray:
        shared = rng(self.seed, "shared", str(scene_id)).normal(size=K)
        private = rng(self.seed, "private", str(scene_id), str(member)).normal(size=K)
        return self.sigma * (np.sqrt(self.rho) * shared + np.sqrt(1.0 - self.rho) * private)

    def predict(self, ev, member=0):
        p = _posterior_from_evidence(self.world, ev)
        logits = np.log(np.clip(p, 1e-12, None)) + self._eps(ev.scene_id, member, len(p))
        return softmax(logits)


class CueDropout(Forecaster):
    """An ensemble whose members each read a random subset of the visible cues.

    This is the fair test of the mutual-information estimator, and it is the one configuration in
    which that estimator should genuinely work. Members here disagree *because* the evidence is
    incomplete, which is exactly the disagreement BALD is supposed to detect, and their errors are
    independent by construction rather than shared. Every other ensemble in this instrument either
    has correlated members or is a single deterministic forecaster for which the estimate is
    identically zero -- neither gives the method a chance.

    Each cue survives independently with probability `keep_p` per member, so a member is a correctly
    specified forecaster over a smaller field of view rather than a corrupted one, and its excess
    loss stays non-negative for the same reason `PartialEvidence`'s does.
    """
    stochastic = True
    uses_oracle = True

    def __init__(self, world, keep_p: float = 0.6, seed: int = 20260921):
        self.world = world
        self.keep_p = float(keep_p)
        self.seed = int(seed)
        self.name = f"cue_dropout[p={keep_p:g}]"

    def predict(self, ev, member=0):
        z = np.zeros(len(ev.cue_names))
        for i, n in enumerate(ev.cue_names):
            if n in ev.visible:
                z[i] = ev.visible[n]
        keep = rng(self.seed, "dropout", str(ev.scene_id), str(member)).random(len(z)) < self.keep_p
        return self.world.posterior(z, ev.mask & keep)


def standard_suite(world) -> list:
    """The forecasters every validation run scores, in the order the reports list them."""
    K = len(world.spec.outcomes)
    return [
        BayesOptimal(world),
        Tempered(world, 0.5),
        Tempered(world, 2.0),
        PartialEvidence(world, ("heading", "speed")),
        PartialEvidence(world, ("heading",)),
        LogitNoise(world, 0.75, rho=0.0),
        Marginal(world),
        Uniform(K),
    ]
