"""Model backends for the VLM arm.

Two backends: the Anthropic Messages API, and any OpenAI-compatible gateway for open-weight models.
Both return plain response text; parsing into a distribution is `prompt.py`'s job.

One API constraint shapes the experiment rather than merely the code. On the current Anthropic
models -- Opus 5, Sonnet 5, Opus 4.8 and 4.7 -- the `temperature` parameter has been removed and a
request that sends one is rejected with a 400. Decoding randomness is therefore not under the
caller's control on those models, which matters here because the SAMPLED elicitation mode forms an
ensemble out of repeated calls and was relying on temperature to set how much members differ.

This is worth stating plainly rather than routing around, because it strengthens the synthetic
result instead of weakening it. An ensemble whose members cannot be pushed apart is an ensemble
whose members are highly correlated, and `baselines/predictors.py` shows in closed form what happens
to a mutual-information estimate of epistemic uncertainty as that correlation goes to one: it
collapses to zero while the true error stays put. The API change pushes real deployments further
into the regime the instrument predicts is broken, so the prediction becomes more load-bearing, not
less.

`TEMPERATURE_SUPPORTED` records where the parameter is still accepted. Every run records which
models it used and whether their diversity was controllable, because a sampled-mode number means
something different in each case.

Nothing in this file has been run. No API key is configured and no request has been issued.
"""
import os
from dataclasses import dataclass, field

# Models that still accept `temperature`. Sending it to anything else returns a 400, so the
# elicitation mode's meaning has to be read alongside this table.
TEMPERATURE_SUPPORTED = {
    "claude-haiku-4-5": True,
    "claude-opus-4-6": True,
    "claude-sonnet-4-6": True,
    "claude-opus-5": False,
    "claude-sonnet-5": False,
    "claude-opus-4-8": False,
    "claude-opus-4-7": False,
}

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 256          # a probability object or one word; nothing here needs more


@dataclass
class Response:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    refused: bool = False
    meta: dict = field(default_factory=dict)


class Backend:
    name = "base"

    def complete(self, system: str, prompt: str, image_b64: str = None,
                 temperature: float = None) -> Response:
        raise NotImplementedError

    def accepts_temperature(self) -> bool:
        return False


class AnthropicBackend(Backend):
    """Anthropic Messages API, with the contact sheet sent as a base64 PNG image block."""

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low"):
        import anthropic
        self.model = model
        self.effort = effort
        self.name = f"anthropic:{model}"
        self._client = anthropic.Anthropic()

    def accepts_temperature(self) -> bool:
        return TEMPERATURE_SUPPORTED.get(self.model, False)

    def complete(self, system, prompt, image_b64=None, temperature=None):
        content = []
        if image_b64 is not None:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": image_b64}})
        content.append({"type": "text", "text": prompt})

        kwargs = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": content}],
            # Low effort: this is a short classification, and the forecast wanted is the model's
            # immediate read of the frames rather than the product of a long deliberation.
            output_config={"effort": self.effort},
        )
        if temperature is not None and self.accepts_temperature():
            kwargs["temperature"] = temperature

        msg = self._client.messages.create(**kwargs)
        if msg.stop_reason == "refusal":
            return Response(text="", model=self.model, refused=True,
                            meta={"stop_details": str(getattr(msg, "stop_details", None))})
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Response(text=text, model=self.model,
                        input_tokens=msg.usage.input_tokens,
                        output_tokens=msg.usage.output_tokens)


class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible gateway, for open-weight VLMs.

    Present so the ensemble arm can mix model families. An ensemble drawn from one family shares
    pretraining and is the correlated case by construction, which is the regime the instrument
    predicts breaks the estimator -- so a cross-family ensemble is the one comparison that could
    show the estimator working, and the arm is not worth running without it.
    """

    def __init__(self, model: str, base_url: str = None, api_key: str = None):
        from openai import OpenAI
        self.model = model
        self.name = f"openai-compat:{model}"
        self._client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_COMPAT_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "not-needed"),
        )

    def accepts_temperature(self) -> bool:
        return True

    def complete(self, system, prompt, image_b64=None, temperature=None):
        parts = []
        if image_b64 is not None:
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
        parts.append({"type": "text", "text": prompt})
        r = self._client.chat.completions.create(
            model=self.model, max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": parts}],
            **({"temperature": temperature} if temperature is not None else {}),
        )
        u = getattr(r, "usage", None)
        return Response(text=r.choices[0].message.content or "", model=self.model,
                        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(u, "completion_tokens", 0) or 0)
