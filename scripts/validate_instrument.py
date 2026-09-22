#!/usr/bin/env python3
"""Run V1-V7 against forecasters whose behaviour is known in closed form.

This is the gate. It runs before any experiment and before any model is loaded, and every check has
an answer that was fixed by the mathematics rather than by a previous run. Writes
`experiments/results/validation/summary.json`.

    python3 scripts/validate_instrument.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.protocol import run_all
from src.forecasting.scenarios import junction_world, kitchen_world
from src.utils.io import write_summary
from src.utils.seeding import rng

SEED = 20260921


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=2000)
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--world", default="junction", choices=["junction", "kitchen"],
                    help="The checks are world-independent by construction; running them in a "
                         "second world is how that claim is tested rather than asserted.")
    args = ap.parse_args()

    world = {"junction": junction_world, "kitchen": kitchen_world}[args.world](args.tau)
    scenes = world.sample_scenes(rng(args.seed, "validation", args.world), args.scenes)

    print(f"world   : {world.spec.name}")
    print(f"scenes  : {len(scenes)}")
    print(f"E[H(P*)]: {world.expected_aleatoric():.4f} nats   (log K = "
          f"{__import__('numpy').log(len(world.spec.outcomes)):.4f})\n")

    checks = run_all(world, scenes)
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.id}  {c.name}")
        print(f"       expected: {c.expected}")
        print(f"       measured: {c.measured}")
    n_pass = sum(c.passed for c in checks)
    print(f"\n{n_pass}/{len(checks)} checks passed")

    write_summary("validation" if args.world == "junction" else f"validation_{args.world}", {
        "world": world.spec.name, "world_key": args.world,
        "n_scenes": len(scenes), "seed": args.seed,
        "expected_aleatoric": world.expected_aleatoric(),
        "checks": [{"id": c.id, "name": c.name, "expected": c.expected,
                    "measured": c.measured, "passed": c.passed, "detail": c.detail}
                   for c in checks],
        "n_passed": n_pass, "n_checks": len(checks),
    })
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
