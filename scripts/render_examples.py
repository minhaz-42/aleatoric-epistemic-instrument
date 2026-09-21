#!/usr/bin/env python3
"""Render example prefixes, one per observability pattern, plus the hidden-cue leak check.

The contact sheets are what the VLM arm would send. The leak check is the important part: it takes
scenes that differ only in a cue the prefix hides and asserts the rendered frames are bitwise
identical. If that ever fails, every aleatoric floor measured against rendered video is wrong,
because the forecaster would have had access to evidence the closed form assumed it did not.

    python3 scripts/render_examples.py
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forecasting.scenarios import junction_world
from src.utils.io import ROOT
from src.utils.seeding import rng
from src.video.render import contact_sheet, render_prefix, save_png

OUT = os.path.join(ROOT, "figures", "examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    world = junction_world(0.90)
    scenes = world.sample_scenes(rng(args.seed, "examples"), 600)
    names = [c.name for c in world.spec.cues]
    os.makedirs(OUT, exist_ok=True)

    seen, made = set(), 0
    for s in scenes:
        key = tuple(s.mask.astype(int))
        if key in seen:
            continue
        seen.add(key)
        vis = "-".join(n for n, m in zip(names, s.mask) if m) or "none"
        path = os.path.join(OUT, f"mask_{''.join(str(k) for k in key)}_{vis}.png")
        save_png(contact_sheet(render_prefix(s, world, args.frames)), path)
        made += 1
        print(f"  {os.path.basename(path):58s} H(P*)={s.aleatoric:.3f} nats  "
              f"outcome={world.spec.outcomes[s.y]}")
    print(f"\n{made} of {1 << world.d} observability patterns rendered to figures/examples/")

    print("\nleak check: scenes differing only in a hidden cue must render identically")
    pairs = 0
    for s in scenes[:250]:
        for i in np.where(~s.mask)[0]:
            for v in world.spec.cues[i].values:
                if v == s.z[i]:
                    continue
                alt = s.z.copy(); alt[i] = v
                d = float(np.abs(render_prefix(s, world, args.frames)
                                 - render_prefix(replace(s, z=alt), world, args.frames)).max())
                if d != 0.0:
                    print(f"  LEAK: cue {names[i]} changed the frames by {d:.3e}")
                    return 1
                pairs += 1
    print(f"  {pairs} pairs checked, max pixel difference exactly 0.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
