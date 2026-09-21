"""A vision-language model behind the same `Forecaster` interface as the synthetic baselines.

Wiring the model into the same interface is what makes the comparison meaningful: the identical
protocol, metrics and routing code that measured the closed-form baselines measures the model, so a
difference in the numbers is a difference in the forecaster and not in how it was scored.

The one thing that does NOT carry over is the reference. Every exact quantity in this project is
defined against P*, the true conditional, and P* exists only because the world is synthetic. Run on
the synthetic world, this class is fully scoreable -- the model sees rendered frames, and its
epistemic error against the known P* is exactly computable. That is the experiment worth running,
and it is the one this file is for. Run on real video, only the empirical decomposition is
available, and the project's whole point is that the empirical decomposition is the part that cannot
be trusted without a reference.

`uses_oracle` is False, and that is enforced rather than declared: the model receives rendered
frames and the outcome names, and no code path here touches `scene.posterior` or `scene.z`.

Never run. See STATUS.md section 5.
"""
import numpy as np

from ..forecasting.task import Forecaster
from ..video.clip import SyntheticClip
from ..video.render import contact_sheet
from . import cache, prompt as P


class VLMForecaster(Forecaster):
    uses_oracle = False
    stochastic = True

    def __init__(self, backend, world, elicitation: str = P.VERBALISED,
                 n_frames: int = 8, n_votes: int = 8, temperature: float = 1.0,
                 use_cache: bool = True):
        self.backend = backend
        self.world = world
        self.elicitation = elicitation
        self.n_frames = n_frames
        self.n_votes = n_votes
        self.temperature = temperature
        self.use_cache = use_cache
        self.name = f"vlm[{backend.name},{elicitation}]"
        self.failures = []

    def _ask(self, text_prompt, img_b64, raw, member):
        k = cache.key(self.backend.name, self.elicitation, self.temperature, member,
                      text_prompt, raw)
        if self.use_cache:
            hit = cache.get(k)
            if hit is not None:
                return hit["text"], True
        r = self.backend.complete(P.SYSTEM, text_prompt, img_b64, self.temperature)
        if r.refused:
            raise RuntimeError(f"model refused: {r.meta}")
        if self.use_cache:
            cache.put(k, {"text": r.text, "model": r.model,
                          "input_tokens": r.input_tokens, "output_tokens": r.output_tokens})
        return r.text, False

    def predict_scene(self, scene, member: int = 0) -> np.ndarray:
        """Forecast one scene. Takes the Scene rather than an Evidence because it needs the frames,
        and the frames are rendered through the mask, so no hidden cue reaches the model."""
        clip = SyntheticClip(scene=scene, world=self.world, n_frames=self.n_frames)
        img_b64, raw = P.encode_sheet(contact_sheet(clip.frames()))
        text_prompt = P.build(clip, self.elicitation)
        outcomes = clip.outcomes

        if self.elicitation == P.VERBALISED:
            text, _ = self._ask(text_prompt, img_b64, raw, member)
            try:
                return P.parse_verbalised(text, outcomes)
            except Exception as exc:
                self.failures.append({"scene_id": scene.scene_id, "member": member,
                                      "error": str(exc), "text": text[:300]})
                raise

        texts = []
        for v in range(self.n_votes):
            t, _ = self._ask(text_prompt, img_b64, raw, member * self.n_votes + v)
            texts.append(t)
        p, unparsed = P.parse_sampled(texts, outcomes)
        if unparsed:
            self.failures.append({"scene_id": scene.scene_id, "member": member,
                                  "error": f"{unparsed}/{len(texts)} votes unparseable", "text": ""})
        return p

    def predict(self, ev, member: int = 0):
        raise NotImplementedError(
            "VLMForecaster needs the rendered scene, not an Evidence object. Call predict_scene(). "
            "The Evidence path is deliberately absent so a model can never be handed cue values "
            "as text when the experiment is about reading them from pixels."
        )
