"""The clip abstraction the VLM arm consumes, and where real video would enter.

`SyntheticClip` wraps a rendered scene and is the only implementation that exists. `Clip` is the
interface a real-video corpus would have to satisfy, and it is written down here rather than left
implicit because the honest statement of this project's scope is that the real-video arm has not
been run. Anything a real corpus would need that the synthetic path does not -- decoding, frame
rate normalisation, a ground-truth outcome annotation, and above all a defensible estimate of P*
that does not exist -- shows up as an unimplemented method rather than as an oversight.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


class Clip:
    """A prefix of video plus the question asked about its continuation."""
    clip_id: str
    outcomes: tuple

    def frames(self) -> np.ndarray:
        raise NotImplementedError

    def question(self) -> str:
        raise NotImplementedError

    def truth(self) -> Optional[int]:
        raise NotImplementedError


@dataclass
class SyntheticClip(Clip):
    scene: object
    world: object
    n_frames: int = 8
    _cache: Optional[np.ndarray] = None

    @property
    def clip_id(self) -> str:
        return f"scene-{self.scene.scene_id:06d}"

    @property
    def outcomes(self) -> tuple:
        return self.world.spec.outcomes

    def frames(self) -> np.ndarray:
        if self._cache is None:
            from .render import render_prefix
            self._cache = render_prefix(self.scene, self.world, self.n_frames)
        return self._cache

    def question(self) -> str:
        return ("An agent is approaching a four-way junction from the south. "
                "Which of these happens next?")

    def truth(self) -> int:
        return self.scene.y

    def posterior(self) -> np.ndarray:
        """The Bayes-optimal answer. Never shown to a forecaster; used only for scoring."""
        return self.scene.posterior
