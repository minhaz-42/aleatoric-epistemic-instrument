"""What a forecaster is shown, and what it must return.

`Evidence` is the only thing a forecaster receives. It carries the values of the visible cues and
nothing else -- not the hidden cues, not the realised outcome, not the posterior. That is structural
rather than a convention: a forecaster cannot read a hidden cue because the object does not contain
one.

Diagnostic forecasters legitimately need the world (the Bayes-optimal one is defined by it). Those
declare `uses_oracle = True` and the protocol asserts the flag, so a forecaster that is meant to
stand in for a model cannot quietly consult the answer. This mirrors the same guard in the
`uses_chain` flag of the audio-editing instrument.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Evidence:
    """What a prefix reveals. Contains no hidden cue and no outcome."""
    scene_id: int
    cue_names: tuple
    visible: dict          # cue name -> value, visible cues only
    mask: np.ndarray       # (d,) bool, kept so a forecaster can know *that* a cue is missing

    @property
    def n_visible(self) -> int:
        return len(self.visible)

    def describe(self) -> str:
        """Human- and prompt-readable rendering, used by the VLM arm's text channel."""
        if not self.visible:
            return "No cue is visible in this prefix."
        return "; ".join(f"{k}={self.visible[k]:+.0f}" for k in self.cue_names if k in self.visible)


def evidence_of(scene, world) -> Evidence:
    names = tuple(c.name for c in world.spec.cues)
    vis = {n: float(scene.z[i]) for i, n in enumerate(names) if scene.mask[i]}
    return Evidence(scene_id=scene.scene_id, cue_names=names, visible=vis, mask=scene.mask.copy())


class Forecaster:
    """One forecast: evidence in, a distribution over outcomes out.

    `member` indexes an ensemble draw. A deterministic forecaster ignores it; a stochastic one must
    be a deterministic function of (scene_id, member) so that a run is reproducible and two
    experiments that request member 3 get the same member 3.
    """
    name = "base"
    uses_oracle = False
    stochastic = False

    def predict(self, ev: Evidence, member: int = 0) -> np.ndarray:
        raise NotImplementedError

    def predict_many(self, evidences, member: int = 0) -> np.ndarray:
        """(N, K) stack. Overridden where a vectorised path exists."""
        return np.stack([self.predict(e, member) for e in evidences])
