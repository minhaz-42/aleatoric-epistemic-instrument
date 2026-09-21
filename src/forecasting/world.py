"""EvidenceWorld: a video world whose predictive distribution is known in closed form.

Why a synthetic world at all. The question this project asks is whether a forecaster's uncertainty
is irreducible (the clip does not determine the future) or reducible (the forecaster cannot read the
clip). On real video that distinction is unobservable: nobody knows the true conditional
distribution of the future given a clip, so "irreducible" cannot be measured, only assumed. The
instrument therefore starts in a world where the conditional is computable exactly, validates every
estimator against it, and only then is pointed at a model whose behaviour is unknown.

The construction, and why each piece is what it is.

A scene carries `d` latent cues. Each cue takes a value from a small finite alphabet with a known
prior, and the cues are independent. Finiteness and independence are not cosmetic: together they
make the marginalisation below an exact finite sum rather than an approximation, which is the entire
point of the world.

The outcome is drawn from a temperature-softened linear model of the cues,

    P(y | z) = softmax(W z + b) / tau)                                                   (1)

so `tau` alone sets how far the *full* cue vector determines the future. At tau -> 0 the world is
deterministic given z; at large tau it is uniform regardless of z.

A prefix renders only some cues. Which ones is recorded by an observability mask `m`, drawn
independently of `z` -- a cue can be occluded, out of frame, or not yet have happened. An observer
who sees the prefix perfectly still sees only `z_m`, so the best it can do is the posterior

    P*(y | z_m) = sum_{z_h} prod_{i in hidden} p_i(z_i) * P(y | z_m, z_h)                (2)

which is exact because the alphabet is finite and the cues are independent. `P*` is the object the
whole project is about. Two consequences are used everywhere downstream, and both are identities
rather than estimates:

    E_{y ~ P*}[-log Q(y)]  =  H(P*)  +  KL(P* || Q)                                      (3)

The expected log loss of ANY forecaster Q splits exactly into an irreducible term that depends only
on the world and the mask, and a reducible term that depends only on the forecaster. H(P*) is what
this project means by aleatoric uncertainty and KL(P* || Q) is what it means by epistemic
uncertainty. They are defined per scene, not per dataset.

Note what (3) does NOT say. It is a statement about a forecaster's *expected* loss under the true
conditional. It does not license reading the split off a single sampled outcome, and the protocol
never does: it averages over sampled outcomes and reports the Monte Carlo error.

The mask is drawn independently of `z`, which matters for (2): it lets the hidden cues keep their
prior instead of acquiring a mask-induced posterior. A world where occlusion depends on the scene
would break that step, and `visibility_depends_on_scene` is asserted False wherever (2) is used.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class Cue:
    """One latent factor of a scene, with a finite alphabet and a known prior.

    `visibility` is the probability that a prefix renders this cue at all. It is a property of the
    cue rather than of the scene, which is what keeps the mask independent of `z`.
    """
    name: str
    values: tuple
    prior: tuple
    visibility: float
    description: str

    def __post_init__(self):
        if len(self.values) != len(self.prior):
            raise ValueError(f"cue {self.name}: {len(self.values)} values but {len(self.prior)} priors")
        if abs(sum(self.prior) - 1.0) > 1e-12:
            raise ValueError(f"cue {self.name}: prior sums to {sum(self.prior)}, not 1")
        if min(self.prior) < 0:
            raise ValueError(f"cue {self.name}: negative prior mass")
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError(f"cue {self.name}: visibility {self.visibility} outside [0, 1]")


@dataclass(frozen=True)
class WorldSpec:
    name: str
    cues: tuple
    outcomes: tuple
    W: np.ndarray                      # (K, d) cue weights on the outcome logits
    b: np.ndarray                      # (K,)   outcome bias
    tau: float                         # temperature: the only knob on the aleatoric floor
    visibility_depends_on_scene: bool = False

    def __post_init__(self):
        K, d = len(self.outcomes), len(self.cues)
        if self.W.shape != (K, d):
            raise ValueError(f"W is {self.W.shape}, expected {(K, d)}")
        if self.b.shape != (K,):
            raise ValueError(f"b is {self.b.shape}, expected {(K,)}")
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")


@dataclass(frozen=True)
class Scene:
    """One sampled scene: its latent cues, what the prefix reveals, and the realised outcome."""
    scene_id: int
    z: np.ndarray                      # (d,) realised cue values
    mask: np.ndarray                   # (d,) bool, True where the prefix reveals the cue
    y: int                             # realised outcome, drawn from P(y | z) with the FULL z
    posterior: np.ndarray              # (K,) P*(y | z_m), equation (2)
    aleatoric: float                   # H(P*) in nats, the irreducible floor for this scene

    @property
    def n_visible(self) -> int:
        return int(self.mask.sum())


class World:
    """A `WorldSpec` plus the exact inference it supports.

    All of the exact quantities are computed from one enumeration of the joint cue alphabet, built
    once in the constructor. With the default spec that is 81 states, so exactness costs nothing.
    """

    def __init__(self, spec: WorldSpec):
        self.spec = spec
        self.d = len(spec.cues)
        self.K = len(spec.outcomes)

        # Enumerate the joint cue alphabet: states (S, d), prior (S,).
        grids = np.meshgrid(*[np.asarray(c.values, dtype=float) for c in spec.cues], indexing="ij")
        self.states = np.stack([g.ravel() for g in grids], axis=1)            # (S, d)
        priors = np.meshgrid(*[np.asarray(c.prior, dtype=float) for c in spec.cues], indexing="ij")
        self.state_prior = np.prod(np.stack([p.ravel() for p in priors], axis=1), axis=1)  # (S,)
        self.state_prior = self.state_prior / self.state_prior.sum()

        # P(y | z) for every state, equation (1). Computed once; everything else indexes into it.
        logits = self.states @ spec.W.T + spec.b                              # (S, K)
        self.state_conditional = softmax(logits / spec.tau, axis=1)           # (S, K)

    # -- exact inference -------------------------------------------------------------------

    def conditional(self, z: np.ndarray) -> np.ndarray:
        """P(y | z) for a fully specified cue vector, equation (1)."""
        logits = (np.asarray(z, dtype=float) @ self.spec.W.T + self.spec.b) / self.spec.tau
        return softmax(logits)

    def posterior(self, z: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """P*(y | z_m), equation (2): the Bayes-optimal forecast from a prefix.

        Exact, by summing over every joint state consistent with the visible cues and weighting by
        the prior. Because the mask is independent of `z`, the hidden cues keep their prior.
        """
        if self.spec.visibility_depends_on_scene:
            raise AssertionError(
                "posterior() marginalises hidden cues under their prior, which is only valid when "
                "the observability mask is independent of the scene"
            )
        z = np.asarray(z, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return self.marginal()
        consistent = np.all(np.isclose(self.states[:, mask], z[mask]), axis=1)   # (S,)
        w = self.state_prior[consistent]
        if w.sum() <= 0:
            raise ValueError("no joint state is consistent with the visible cues")
        w = w / w.sum()
        return w @ self.state_conditional[consistent]

    def marginal(self) -> np.ndarray:
        """P(y) with nothing observed: the prior-predictive outcome distribution."""
        return self.state_prior @ self.state_conditional

    # -- sampling --------------------------------------------------------------------------

    def sample_mask(self, rng: np.random.Generator) -> np.ndarray:
        """Draw an observability mask. Independent of `z`, which equation (2) relies on."""
        return rng.random(self.d) < np.array([c.visibility for c in self.spec.cues])

    def sample_scene(self, rng: np.random.Generator, scene_id: int,
                     mask: Optional[np.ndarray] = None) -> Scene:
        """Draw cues from their priors, draw a mask, then draw the outcome from the FULL cue vector.

        The outcome is drawn from P(y | z), not from P*(y | z_m). That is what makes the world
        honest: the future really is generated by cues the observer may not see, and the observer's
        conditional distribution given what it does see is exactly P*.
        """
        idx = [rng.choice(len(c.values), p=np.asarray(c.prior)) for c in self.spec.cues]
        z = np.array([c.values[i] for c, i in zip(self.spec.cues, idx)], dtype=float)
        m = self.sample_mask(rng) if mask is None else np.asarray(mask, dtype=bool)
        y = int(rng.choice(self.K, p=self.conditional(z)))
        post = self.posterior(z, m)
        return Scene(scene_id=scene_id, z=z, mask=m, y=y, posterior=post,
                     aleatoric=float(entropy(post)))

    def sample_scenes(self, rng: np.random.Generator, n: int, start: int = 0) -> list:
        return [self.sample_scene(rng, start + i) for i in range(n)]

    # -- world-level summaries -------------------------------------------------------------

    def expected_aleatoric(self) -> float:
        """E_m E_{z_m}[H(P*)]: the mean irreducible entropy over the world's own mask distribution.

        Exact. Enumerates every mask pattern weighted by its probability, and for each, every
        visible-cue assignment weighted by its prior.
        """
        total = 0.0
        for mi in range(1 << self.d):
            mask = np.array([(mi >> i) & 1 for i in range(self.d)], dtype=bool)
            p_mask = np.prod([c.visibility if mask[i] else 1.0 - c.visibility
                              for i, c in enumerate(self.spec.cues)])
            if p_mask <= 0:
                continue
            for s in range(len(self.states)):
                post = self.posterior(self.states[s], mask)
                total += p_mask * self.state_prior[s] * entropy(post)
        return float(total)

    def expected_partial_excess(self, keep_idx: np.ndarray) -> float:
        """E[ KL( P*(.|z_m) || P*(.|z_{m and keep}) ) ]: the exact expected excess loss of a
        forecaster that reads only the cues in `keep`.

        Exact, by enumerating every mask pattern and every joint cue state. This is the closed-form
        reference that check V4 compares the measured ladder against. Note that the inequality it
        certifies -- more cues cannot hurt -- holds for this EXPECTATION and not scene by scene: for
        an individual scene a forecaster reading fewer cues can land closer to the full posterior by
        chance, and about 2% of scenes do. V4 therefore checks the mean, which is the quantity the
        data-processing inequality actually constrains.
        """
        keep_idx = np.asarray(keep_idx, dtype=bool)
        total = 0.0
        for mi in range(1 << self.d):
            mask = np.array([(mi >> i) & 1 for i in range(self.d)], dtype=bool)
            p_mask = np.prod([c.visibility if mask[i] else 1.0 - c.visibility
                              for i, c in enumerate(self.spec.cues)])
            if p_mask <= 0:
                continue
            for s in range(len(self.states)):
                z = self.states[s]
                total += p_mask * self.state_prior[s] * kl(self.posterior(z, mask),
                                                           self.posterior(z, mask & keep_idx))
        return float(total)

    def mutual_information(self, mask: np.ndarray) -> float:
        """I(y; z_m) under a fixed mask: how much the visible cues tell you about the outcome.

        Exact. This is the quantity a prior-only forecaster gives up, and validation check V3
        asserts that its excess loss equals this number.
        """
        mask = np.asarray(mask, dtype=bool)
        h_marginal = entropy(self.marginal())
        h_cond = 0.0
        for s in range(len(self.states)):
            h_cond += self.state_prior[s] * entropy(self.posterior(self.states[s], mask))
        return float(h_marginal - h_cond)


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats, with 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def kl(p: np.ndarray, q: np.ndarray, floor: float = 1e-12) -> float:
    """KL(p || q) in nats. `q` is floored because a forecaster is allowed to put zero mass on an
    outcome and the protocol must still return a finite, comparable number."""
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), floor, None)
    nz = p > 0
    return float((p[nz] * (np.log(p[nz]) - np.log(q[nz]))).sum())
