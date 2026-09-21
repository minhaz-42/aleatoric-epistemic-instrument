"""Turning a clip into a question, and a response into a distribution.

Two elicitation modes, because they measure different things and the project needs both.

VERBALISED asks for explicit probabilities in one call. It is cheap -- one call per ensemble member
-- and it is what a deployed system would do. What it reports is the model's *stated* confidence,
which is a claim about itself and need not agree with its behaviour.

SAMPLED asks for a single answer and repeats the call at temperature 1, forming the distribution
from the vote counts. It is M times more expensive and it measures the model's *behavioural*
distribution.

The distinction is load-bearing here rather than incidental. The ensemble decomposition reads
epistemic uncertainty off disagreement between members, so what counts as a member has to be pinned
down before the number means anything. Under SAMPLED, repeated calls to one model differ only by
decoding noise, which is not the epistemic uncertainty the decomposition is about -- it is closer to
a re-draw from the same forecaster. Under VERBALISED with several distinct models, members differ by
model identity, which is nearer the intended meaning but introduces exactly the shared-pretraining
correlation that the `rho` sweep predicts will collapse the estimate. Neither mode is the right
answer on its own, and the point of running both is that the synthetic result says what to expect
from each.

No prompt in this file has been run against a model. See STATUS.md section 5.
"""
import base64
import io
import json

import numpy as np

VERBALISED = "verbalised"
SAMPLED = "sampled"

SYSTEM = (
    "You are forecasting the immediate future of a short top-down video of a road junction. "
    "You will be shown the frames of a prefix, in order, tiled left-to-right then top-to-bottom. "
    "Answer only from what the frames show. Parts of the scene may be occluded by dark grey "
    "blocks; an occluder means the information is unavailable, not that nothing is there."
)

_VERBALISED_INSTRUCTION = """The agent is the yellow disc approaching from the south.

Which of the following happens next?
{options}

Some of what determines the answer may not be visible. If so, your probabilities should be spread
out: that is the correct answer, not a hedge.

Reply with only a JSON object mapping every option to a probability, summing to 1. No other text.
Example: {{"left": 0.1, "straight": 0.6, "right": 0.2, "stop": 0.1}}"""

_SAMPLED_INSTRUCTION = """The agent is the yellow disc approaching from the south.

Which of the following happens next?
{options}

Reply with exactly one of the option words and nothing else."""


def build(clip, elicitation: str = VERBALISED) -> str:
    options = "\n".join(f"- {o}" for o in clip.outcomes)
    template = _VERBALISED_INSTRUCTION if elicitation == VERBALISED else _SAMPLED_INSTRUCTION
    return template.format(options=options)


def encode_sheet(sheet: np.ndarray) -> tuple:
    """(base64 png, raw bytes) for a contact sheet, for the image part of the request."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    buf = io.BytesIO()
    mpimg.imsave(buf, np.clip(sheet, 0, 1), format="png")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), raw


def parse_verbalised(text: str, outcomes, floor: float = 1e-6) -> np.ndarray:
    """Parse a probability object. Raises rather than guessing if it cannot be read.

    A silently-defaulted uniform on a parse failure would be indistinguishable from a model that
    genuinely reported uniform, and would bias the aleatoric estimate upward exactly where the model
    was least cooperative. Failures must surface as failures and be counted in the run's summary.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    obj = json.loads(text[start:end + 1])
    p = np.array([float(obj.get(o, 0.0)) for o in outcomes], dtype=float)
    if np.any(p < 0):
        raise ValueError(f"negative probability in {obj!r}")
    if p.sum() <= 0:
        raise ValueError(f"probabilities sum to zero in {obj!r}")
    return np.clip(p / p.sum(), floor, None) / np.clip(p / p.sum(), floor, None).sum()


def parse_sampled(texts, outcomes, floor: float = 1e-6) -> np.ndarray:
    """Vote counts over repeated single-answer calls, Laplace-smoothed.

    Smoothing is not cosmetic: an unsmoothed vote distribution puts exact zeros on unvoted outcomes,
    and log loss against an outcome with zero mass is infinite. The smoothing constant is reported
    with every result that uses this mode, because it sets a floor on the measurable confidence and
    therefore caps how sharp a sampled forecaster can appear.
    """
    counts = np.zeros(len(outcomes), dtype=float)
    lower = [o.lower() for o in outcomes]
    unparsed = 0
    for t in texts:
        s = t.strip().lower().strip(".\"' ")
        hit = [i for i, o in enumerate(lower) if o == s] or \
              [i for i, o in enumerate(lower) if o in s]
        if len(hit) == 1:
            counts[hit[0]] += 1
        else:
            unparsed += 1
    if counts.sum() == 0:
        raise ValueError(f"no parseable answer in {len(texts)} samples")
    p = (counts + 1.0) / (counts.sum() + len(outcomes))
    return p, unparsed
