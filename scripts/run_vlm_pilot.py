#!/usr/bin/env python3
"""The VLM arm. NEVER RUN -- see STATUS.md section 5.

This scores a vision-language model with exactly the protocol that scored the closed-form
baselines, on rendered prefixes from the same world. Because the world is synthetic, the model's
epistemic error against the true conditional is computable, which is the one thing that cannot be
done on real video and the whole reason to point the instrument here first.

It refuses to run without `--confirm`. The refusal is deliberate: this is the only part of the
project that spends money, and an accidental invocation of the full design would cost real money
for a run nobody asked for. `--estimate` prints what a run would cost and exits.

    python3 scripts/run_vlm_pilot.py --estimate
    python3 scripts/run_vlm_pilot.py --confirm --scenes 40 --members 3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import accuracy, brier, ece, nll
from src.forecasting.scenarios import junction_world
from src.forecasting.world import entropy, kl
from src.temporal.splits import split_scenes
from src.uncertainty.decomposition import ensemble_split
from src.utils.io import write_records, write_summary
from src.utils.seeding import rng
from src.vlm import prompt as P
from src.vlm.client import DEFAULT_MODEL, TEMPERATURE_SUPPORTED, AnthropicBackend
from src.vlm.predictor import VLMForecaster

# Input tokens are dominated by one contact sheet; output is a short JSON object. Rough, and
# labelled as such wherever it is printed.
TOKENS_IN, TOKENS_OUT = 1600, 60
PRICE = {"claude-opus-5": (5.00, 25.00), "claude-sonnet-5": (2.00, 10.00),
         "claude-haiku-4-5": (1.00, 5.00)}


def estimate(model, scenes, members, votes, elicitation):
    calls = scenes * members * (votes if elicitation == P.SAMPLED else 1)
    pin, pout = PRICE.get(model, (5.00, 25.00))
    usd = calls * (TOKENS_IN / 1e6 * pin + TOKENS_OUT / 1e6 * pout)
    return calls, usd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--votes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--elicitation", default=P.VERBALISED, choices=[P.VERBALISED, P.SAMPLED])
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    calls, usd = estimate(args.model, args.scenes, args.members, args.votes, args.elicitation)
    temp_ok = TEMPERATURE_SUPPORTED.get(args.model, False)
    print(f"model        {args.model}")
    print(f"elicitation  {args.elicitation}")
    print(f"scenes       {args.scenes} x {args.members} members"
          + (f" x {args.votes} votes" if args.elicitation == P.SAMPLED else ""))
    print(f"calls        {calls}")
    print(f"cost         ~${usd:.2f}  (rough: {TOKENS_IN} in / {TOKENS_OUT} out per call)")
    print(f"temperature  {'controllable' if temp_ok else 'NOT accepted by this model (400)'}")
    if not temp_ok and args.elicitation == P.SAMPLED:
        print("\n  WARNING: sampled elicitation forms its ensemble from repeated calls, and this")
        print("  model rejects `temperature`. Member diversity is therefore not under your")
        print("  control, which is the high-rho regime that V6 shows collapses the estimate.")
    if args.estimate:
        return 0
    if not args.confirm:
        print("\nRefusing to run without --confirm. This is the only part of the project that")
        print("spends money. Re-run with --confirm once the number above is acceptable.")
        return 1

    world = junction_world(args.tau)
    scenes = split_scenes(world.sample_scenes(rng(args.seed, "vlm"), args.scenes * 3))["test"]
    scenes = scenes[:args.scenes]
    backend = AnthropicBackend(args.model)
    f = VLMForecaster(backend, world, elicitation=args.elicitation,
                      n_frames=args.frames, n_votes=args.votes)

    rows = []
    for i, sc in enumerate(scenes, 1):
        try:
            members = np.stack([f.predict_scene(sc, m) for m in range(args.members)])
        except Exception as exc:                  # a parse or refusal failure is data, not a crash
            print(f"  [{i}/{len(scenes)}] scene {sc.scene_id}: FAILED  {exc}")
            continue
        es = ensemble_split(members)
        rows.append({
            "scene_id": sc.scene_id, "n_visible": sc.n_visible, "y": sc.y,
            "correct": int(np.argmax(es.mean_prediction) == sc.y),
            "aleatoric_true": entropy(sc.posterior),
            "epistemic_true": kl(sc.posterior, es.mean_prediction),
            "total_hat": es.total, "aleatoric_hat": es.aleatoric_hat,
            "epistemic_hat": es.epistemic_hat,
            **{f"q{k}": float(v) for k, v in enumerate(es.mean_prediction)},
        })
        print(f"  [{i}/{len(scenes)}] scene {sc.scene_id}  "
              f"H(P*)={rows[-1]['aleatoric_true']:.3f}  KL={rows[-1]['epistemic_true']:.3f}")

    if not rows:
        print("no scene completed; nothing written")
        return 1
    frame = pd.DataFrame(rows)
    probs = frame[[f"q{k}" for k in range(len(world.spec.outcomes))]].to_numpy()
    labels = frame["y"].to_numpy()
    write_records("vlm_pilot", frame)
    write_summary("vlm_pilot", {
        "model": args.model, "elicitation": args.elicitation, "n_scenes": len(rows),
        "n_requested": len(scenes), "n_members": args.members, "seed": args.seed,
        "temperature_controllable": temp_ok,
        "expected_aleatoric_exact": world.expected_aleatoric(),
        "aleatoric_true": float(frame["aleatoric_true"].mean()),
        "aleatoric_hat": float(frame["aleatoric_hat"].mean()),
        "epistemic_true": float(frame["epistemic_true"].mean()),
        "epistemic_hat": float(frame["epistemic_hat"].mean()),
        "nll": nll(probs, labels), "brier": brier(probs, labels),
        "accuracy": accuracy(probs, labels), "ece": ece(probs, labels),
        "failures": f.failures,
    })
    print(f"\n{len(rows)}/{len(scenes)} scenes scored, {len(f.failures)} parse failures")
    print(f"true epistemic KL(P*||Q) = {frame['epistemic_true'].mean():.4f} nats")
    print(f"estimate       BALD      = {frame['epistemic_hat'].mean():.4f} nats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
