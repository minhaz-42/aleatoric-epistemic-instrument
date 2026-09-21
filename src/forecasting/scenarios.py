"""The concrete worlds the instrument runs on.

`junction` is the default and the only one used for headline numbers. An agent approaches a
four-way junction and the forecast is which exit it takes. The four cues were chosen so that each
one is a different *kind* of evidence, and so that they differ in how often a prefix shows them:

  heading   the agent's own orientation. Almost always visible, and the strongest single cue.
  speed     how fast it is moving. Usually visible, weak on its own, and it interacts with blocker.
  signal    the junction's light. Frequently out of frame in a short prefix.
  blocker   an obstruction on one branch. Frequently occluded by the agent itself.

That spread is deliberate. If every cue were equally visible the observability mask would carry no
information and every scene would sit at the same aleatoric floor, which is precisely the
degenerate case the routing experiment needs to avoid.

Cue weights are fixed by hand rather than fitted. They are a property of the instrument, not a
result, and they are held constant across every experiment so that a change in a measured number is
attributable to the forecaster or the protocol rather than to the world.
"""
import numpy as np

from .world import Cue, World, WorldSpec

# Ordered as the columns of W.
JUNCTION_CUES = (
    Cue("heading", (-1.0, 0.0, 1.0), (1 / 3, 1 / 3, 1 / 3), 0.95,
        "Agent orientation on approach: -1 left of centre, 0 straight, +1 right of centre."),
    Cue("speed", (-1.0, 0.0, 1.0), (0.3, 0.4, 0.3), 0.85,
        "Approach speed: -1 slow, 0 moderate, +1 fast. Fast approaches rarely turn sharply."),
    Cue("signal", (-1.0, 0.0, 1.0), (0.35, 0.30, 0.35), 0.55,
        "Junction signal: -1 stop, 0 none, +1 go. Often outside a short prefix's field of view."),
    Cue("blocker", (-1.0, 0.0, 1.0), (0.25, 0.50, 0.25), 0.40,
        "Obstruction: -1 blocks the left branch, 0 none, +1 blocks the right. Often self-occluded."),
)

JUNCTION_OUTCOMES = ("left", "straight", "right", "stop")

#            heading  speed  signal  blocker
JUNCTION_W = np.array([
    [ 1.70, -0.55,  0.25,  -1.30],   # left      turning left needs heading, is hurt by a left blocker
    [ 0.00,  0.85,  0.60,   0.00],   # straight  favoured by speed and a go signal
    [-1.70, -0.55,  0.25,   1.30],   # right     mirror of left
    [ 0.00, -0.70, -1.40,   0.30],   # stop      driven by the signal, which is the least visible cue
], dtype=float)

JUNCTION_B = np.array([0.0, 0.25, 0.0, -0.10], dtype=float)


def junction_world(tau: float = 0.90, name: str = "junction") -> World:
    """The default world. `tau` is the only knob on how much of the future the cues determine."""
    return World(WorldSpec(name=f"{name}@tau={tau:g}", cues=JUNCTION_CUES,
                           outcomes=JUNCTION_OUTCOMES, W=JUNCTION_W, b=JUNCTION_B, tau=tau))


def fully_observed_world(tau: float = 0.90) -> World:
    """The junction with every cue always visible.

    A control, not a scenario. With no hidden cues the posterior of equation (2) collapses to the
    conditional of equation (1), so any discrepancy between the two code paths shows up here.
    """
    cues = tuple(Cue(c.name, c.values, c.prior, 1.0, c.description) for c in JUNCTION_CUES)
    return World(WorldSpec(name=f"junction-full@tau={tau:g}", cues=cues, outcomes=JUNCTION_OUTCOMES,
                           W=JUNCTION_W, b=JUNCTION_B, tau=tau))


def blind_world(tau: float = 0.90) -> World:
    """The junction with every cue always hidden.

    The other control. Every posterior equals the marginal, so the aleatoric floor equals H(P(y))
    and no forecaster can beat a constant. Routing must gain exactly nothing here.
    """
    cues = tuple(Cue(c.name, c.values, c.prior, 0.0, c.description) for c in JUNCTION_CUES)
    return World(WorldSpec(name=f"junction-blind@tau={tau:g}", cues=cues, outcomes=JUNCTION_OUTCOMES,
                           W=JUNCTION_W, b=JUNCTION_B, tau=tau))


# The temperature sweep used by the routing experiment. Each entry is a world identical to the
# default except in how much the cues determine the outcome, which is what isolates the effect of
# the aleatoric floor on routing.
TAU_SWEEP = (0.35, 0.50, 0.70, 0.90, 1.30, 2.00, 3.50)


def registry() -> dict:
    return {
        "junction": junction_world,
        "junction_full": fully_observed_world,
        "junction_blind": blind_world,
    }
