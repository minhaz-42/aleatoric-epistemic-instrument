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


# ---------------------------------------------------------------------------------------------
# The second world. It exists to answer one question: are the findings structural, or artefacts of
# the junction's particular shape? It differs from `junction` on every axis that could plausibly
# matter, so a result that holds in both is not a property of either.
#
#   outcomes        3, not 4
#   cues            5, not 4
#   alphabets       mixed binary and ternary, not uniformly ternary
#   symmetry        none. `junction` is left/right mirrored, which creates exact ties (see D6)
#   visibility      flatter (0.90 down to 0.35), so the mask is less dominated by one cue
#   evidence        NON-ADDITIVE. `junction` is linear in the cues; here a pair of cues can mean
#                   something neither means alone, so losing one cue can be catastrophic rather
#                   than gradual. This is the axis most likely to break the mechanism map.
#
# An egocentric kitchen clip: someone is at a counter and the forecast is the next manipulation.
# ---------------------------------------------------------------------------------------------

KITCHEN_CUES = (
    Cue("grasp", (0.0, 1.0), (0.40, 0.60), 0.90,
        "Is an object held. Binary, and almost always apparent from the hand."),
    Cue("tool", (-1.0, 0.0, 1.0), (0.35, 0.30, 0.35), 0.70,
        "Which implement is in view: -1 blade, 0 none, +1 spoon."),
    Cue("vessel", (0.0, 1.0), (0.45, 0.55), 0.60,
        "Is a container present on the surface. Binary."),
    Cue("gaze", (-1.0, 0.0, 1.0), (0.30, 0.40, 0.30), 0.45,
        "Where attention is directed: -1 board, 0 ahead, +1 vessel."),
    Cue("surface", (-1.0, 0.0, 1.0), (0.30, 0.40, 0.30), 0.35,
        "State of the work surface: -1 cluttered, 0 clear, +1 prepared. Often out of frame."),
)

KITCHEN_OUTCOMES = ("pour", "cut", "stir")

#                     grasp  tool  vessel  gaze  surface
KITCHEN_W = np.array([
    [ 0.35,  0.55,  0.80,  0.60,  0.15],   # pour
    [ 0.50, -0.95,  -0.40, -0.65,  0.45],  # cut
    [ 0.45,  0.30,  0.35,  0.10, -0.30],   # stir
], dtype=float)

KITCHEN_B = np.array([0.05, 0.00, -0.05], dtype=float)


def _kitchen_interactions() -> np.ndarray:
    """Pairwise cue interactions, one matrix per outcome.

    These are what make the evidence non-additive. Pouring needs a vessel AND a spoon-like tool;
    either alone says little. Cutting needs a grasp AND a blade. Stirring is deliberately left
    almost additive, so the world contains both kinds of outcome and the contrast is internal
    rather than between worlds.
    """
    K, d = len(KITCHEN_OUTCOMES), len(KITCHEN_CUES)
    V = np.zeros((K, d, d))
    gi, ti, vi, ga, su = 0, 1, 2, 3, 4
    V[0, ti, vi] = V[0, vi, ti] = 0.75      # pour: tool x vessel
    V[0, ga, vi] = V[0, vi, ga] = 0.45      # pour: gaze at the vessel
    V[1, gi, ti] = V[1, ti, gi] = -0.95     # cut:  grasp x blade (tool = -1)
    V[1, ga, su] = V[1, su, ga] = -0.40     # cut:  looking at a cluttered board
    V[2, gi, vi] = V[2, vi, gi] = 0.25      # stir: mild
    return V


KITCHEN_REVEAL = {"grasp": 1.5, "tool": 3.0, "vessel": 4.0, "gaze": 5.0, "surface": 9.0}


def kitchen_world(tau: float = 0.70, name: str = "kitchen") -> World:
    """The second world. Non-additive evidence, 3 outcomes, 5 cues, no symmetry."""
    return World(WorldSpec(name=f"{name}@tau={tau:g}", cues=KITCHEN_CUES,
                           outcomes=KITCHEN_OUTCOMES, W=KITCHEN_W, b=KITCHEN_B, tau=tau,
                           V=_kitchen_interactions()))


def registry() -> dict:
    return {
        "junction": junction_world,
        "junction_full": fully_observed_world,
        "junction_blind": blind_world,
        "kitchen": kitchen_world,
    }
