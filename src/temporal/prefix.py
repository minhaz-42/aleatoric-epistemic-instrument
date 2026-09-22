"""Prefix length as the second lever on predictive evidence.

The forecast is made from a prefix: the frames up to time t, with the outcome resolved at t + h. How
much evidence the prefix carries is not fixed -- a longer prefix has had more chances to reveal a
cue that was occluded, out of frame, or had not happened yet. This module makes that explicit,
because it turns "how much irreducible uncertainty is there" from a constant of the world into
something the system can act on.

That matters for routing. The usual framing offers one response to an uncertain forecast: spend more
on the model. A prefix-aware system has a second: spend more on the evidence. The identity in
`routing/policies.py` says the aleatoric term cancels when the choice is between two models on the
same prefix -- but it does NOT cancel when the choice is between two prefix lengths, because a
longer prefix changes P* itself. Those are different decisions driven by different quantities, and
conflating them is the same error as conflating the two uncertainty terms in the first place.

The reveal model is deliberately the simplest one that keeps equation (2) exact: cue i is revealed
by a prefix of length L with probability

    v_i(L) = v_i * (1 - exp(-L / L_i))

where v_i is the cue's asymptotic visibility and L_i its reveal time constant. This depends on L
alone, never on the scene, so the mask stays independent of z and the marginalisation in
`World.posterior` remains valid. A model in which occlusion depended on the cue values would break
that, and `visibility_depends_on_scene` exists to make such a world refuse to be marginalised.
"""
from dataclasses import dataclass

import numpy as np

from ..forecasting.world import Cue, Scene, World, WorldSpec, entropy

# Reveal time constants in frames, ordered as the junction world's cues. Heading is apparent
# immediately; the signal and the blocker take time to come into view.
JUNCTION_REVEAL = {"heading": 1.0, "speed": 2.5, "signal": 6.0, "blocker": 8.0}

DEFAULT_PREFIX_FRAMES = 8
DEFAULT_HORIZON_FRAMES = 12


@dataclass(frozen=True)
class PrefixSpec:
    """How much video the forecaster sees, and how far ahead it must predict."""
    n_frames: int = DEFAULT_PREFIX_FRAMES
    horizon: int = DEFAULT_HORIZON_FRAMES
    fps: float = 8.0

    @property
    def seconds(self) -> float:
        return self.n_frames / self.fps

    @property
    def horizon_seconds(self) -> float:
        return self.horizon / self.fps


def visibility_at(cue: Cue, n_frames: int, reveal=None) -> float:
    """The probability that a prefix of `n_frames` reveals this cue."""
    tc = (reveal or JUNCTION_REVEAL).get(cue.name, 1.0)
    if n_frames <= 0:
        return 0.0
    return float(cue.visibility * (1.0 - np.exp(-float(n_frames) / tc)))


def world_at_prefix(base: World, n_frames: int, reveal=None) -> World:
    """The same world viewed through a prefix of a given length.

    Only the visibilities change: the cues, the weights and the temperature are untouched, so any
    difference between two prefix lengths is attributable to evidence alone.
    """
    spec = base.spec
    cues = tuple(Cue(c.name, c.values, c.prior, visibility_at(c, n_frames, reveal), c.description)
                 for c in spec.cues)
    return World(WorldSpec(name=f"{spec.name}|prefix={n_frames}", cues=cues, outcomes=spec.outcomes,
                           W=spec.W, b=spec.b, tau=spec.tau,
                           visibility_depends_on_scene=spec.visibility_depends_on_scene))


def prefix_sweep(base: World, lengths=(1, 2, 4, 8, 16, 32), reveal=None) -> dict:
    """Expected aleatoric floor as a function of prefix length. Exact, not sampled."""
    return {int(L): world_at_prefix(base, L, reveal).expected_aleatoric() for L in lengths}


def nested_scene(base: World, rng, scene_id: int, lengths, reveal=None) -> dict:
    """One scene observed at several prefix lengths, with NESTED masks.

    A longer prefix must reveal a superset of what a shorter one revealed -- evidence does not
    disappear as you keep watching. Independent mask draws per length would violate that and would
    make a prefix-extension gain occasionally negative for a reason that is an artefact of the
    sampler rather than a property of the world.

    The construction is one uniform draw `u_i` per cue, held fixed across lengths, with cue `i`
    visible at length L iff `u_i < v_i(L)`. Because `v_i` is increasing in L, nesting is automatic
    and no rejection is needed. The mask still depends only on L and the draw, never on `z`, so
    equation (2) stays valid at every length.

    Returns {L: Scene}, all sharing one cue vector and one outcome.
    """
    spec = base.spec
    idx = [rng.choice(len(c.values), p=np.asarray(c.prior)) for c in spec.cues]
    z = np.array([c.values[i] for c, i in zip(spec.cues, idx)], dtype=float)
    u = rng.random(len(spec.cues))
    y = int(rng.choice(len(spec.outcomes), p=base.conditional(z)))

    out = {}
    for L in lengths:
        w = world_at_prefix(base, L, reveal)
        mask = np.array([u[i] < w.spec.cues[i].visibility for i in range(len(spec.cues))])
        post = w.posterior(z, mask)
        out[int(L)] = (w, Scene(scene_id=scene_id, z=z, mask=mask, y=y, posterior=post,
                                aleatoric=float(entropy(post))))
    return out
