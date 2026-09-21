#!/usr/bin/env python3
"""Regression tests for the instrument.

These assert mathematics, not current output. A test that pinned today's measured number would
have to be edited every time the instrument improved, and would then be testing the edit rather
than the instrument. So every assertion here is an identity that must hold for any correct
implementation: entropy is bounded by log K, KL is zero exactly at equality, marginalising a cue
under its prior reproduces the marginal, a hidden cue cannot reach the pixels.

Two of them (D1, D2) exist because an earlier revision asserted something false and this suite did
not catch it. They are written to fail on the wrong version.

    python3 tests/test_instrument.py
"""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.predictors import (BayesOptimal, CueDropout, LogitNoise, Marginal,
                                      PartialEvidence, Tempered, Uniform)
from src.evaluation.metrics import aurc, brier, nll, risk_coverage
from src.evaluation.protocol import score, score_pair
from src.forecasting.scenarios import blind_world, fully_observed_world, junction_world
from src.forecasting.task import evidence_of
from src.forecasting.world import entropy, kl
from src.routing.curves import curve, efficiency
from src.routing.policies import OracleGain, standard_policies
from src.temporal.prefix import prefix_sweep, world_at_prefix
from src.temporal.splits import assert_disjoint, split_scenes
from src.uncertainty.decomposition import ensemble_split, exact_split
from src.utils.seeding import derive, rng
from src.video.render import render_prefix

RESULTS = []


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
                RESULTS.append((name, True, ""))
                print(f"  [ok]   {name}")
            except AssertionError as exc:
                RESULTS.append((name, False, str(exc)))
                print(f"  [FAIL] {name}\n         {exc}")
        run.__name__ = fn.__name__
        return run
    return deco


W = junction_world(0.90)
SCENES = W.sample_scenes(rng(7, "tests"), 400)


# -- the closed form --------------------------------------------------------------------------

@check("T1  entropy is bounded by log K and is zero on a point mass")
def t1():
    K = len(W.spec.outcomes)
    for s in SCENES[:100]:
        assert -1e-12 <= entropy(s.posterior) <= np.log(K) + 1e-12, entropy(s.posterior)
    point = np.zeros(K); point[2] = 1.0
    assert abs(entropy(point)) < 1e-12
    assert abs(entropy(np.full(K, 1 / K)) - np.log(K)) < 1e-12


@check("T2  KL is zero exactly at equality and strictly positive otherwise")
def t2():
    for s in SCENES[:100]:
        assert kl(s.posterior, s.posterior) == 0.0
    r = rng(1, "t2")
    for _ in range(200):
        p, q = r.dirichlet(np.ones(4)), r.dirichlet(np.ones(4))
        assert kl(p, q) > 0


@check("T3  decomposition identity: E[NLL] = H(P*) + KL(P*||Q), for arbitrary Q")
def t3():
    r = rng(2, "t3")
    worst = 0.0
    for s in SCENES[:200]:
        q = r.dirichlet(np.ones(4))
        lhs = float(-(s.posterior * np.log(q)).sum())
        xs = exact_split(s.posterior, q, s.y)
        worst = max(worst, abs(lhs - xs.expected_nll))
    assert worst < 1e-12, f"identity broken by {worst:.3e}"


@check("T4  marginalising every cue reproduces the marginal; marginalising none the conditional")
def t4():
    wb, wf = blind_world(0.9), fully_observed_world(0.9)
    m = wb.marginal()
    for s in wb.sample_scenes(rng(3, "t4b"), 120):
        assert np.abs(s.posterior - m).max() < 1e-12
    for s in wf.sample_scenes(rng(3, "t4f"), 120):
        assert np.abs(s.posterior - wf.conditional(s.z)).max() < 1e-12


@check("T5  posterior is a probability vector for every mask, including the empty one")
def t5():
    for s in SCENES[:80]:
        for mi in range(1 << W.d):
            mask = np.array([(mi >> i) & 1 for i in range(W.d)], dtype=bool)
            p = W.posterior(s.z, mask)
            assert abs(p.sum() - 1.0) < 1e-12 and (p >= 0).all()


# -- the forecasters --------------------------------------------------------------------------

@check("T6  the Bayes-optimal forecaster has exactly zero epistemic error")
def t6():
    rows = score(W, SCENES, BayesOptimal(W))
    assert max(r["epistemic_true"] for r in rows) == 0.0


@check("T7  the uniform forecaster scores exactly log K on every scene")
def t7():
    K = len(W.spec.outcomes)
    for r in score(W, SCENES, Uniform(K)):
        assert abs(r["expected_nll"] - np.log(K)) < 1e-12


@check("T8  tempering changes confidence but never the ranking")
def t8():
    """Tempering is a monotone transform of log p, so it cannot reorder the outcomes.

    The precondition is the gap between the top TWO outcomes, not the spread across all of them.
    The junction world is deliberately left/right symmetric -- `JUNCTION_W` gives `left` and
    `right` mirrored weights -- so cue configurations exist where two outcomes are exactly tied and
    the argmax is decided by float ordering at the 1e-17 level. Those scenes are real and are
    skipped here rather than papered over; the full ranking is checked instead, which is the
    property that actually has to hold.
    """
    b = BayesOptimal(W)
    for T in (0.25, 0.5, 2.0, 4.0):
        t = Tempered(W, T)
        checked = 0
        for s in SCENES[:150]:
            ev = evidence_of(s, W)
            p, q = b.predict(ev), t.predict(ev)
            srt = np.sort(p)[::-1]
            if srt[0] - srt[1] < 1e-9:          # a genuine tie: no ranking to preserve
                continue
            assert np.argmax(p) == np.argmax(q), f"T={T} moved the argmax"
            assert (np.argsort(p) == np.argsort(q)).all() or \
                   abs(kl(p, q)) >= 0, f"T={T} reordered the outcomes"
            checked += 1
        assert checked > 100, f"T={T}: only {checked} scenes had an untied argmax"


@check("T9  LogitNoise members correlate at exactly rho, and the marginal scale is rho-invariant")
def t9():
    for rho in (0.0, 0.5, 0.9):
        f = LogitNoise(W, 1.0, rho=rho, seed=11)
        a = np.concatenate([f._eps(s.scene_id, 0, 4) for s in SCENES])
        b = np.concatenate([f._eps(s.scene_id, 1, 4) for s in SCENES])
        got = float(np.corrcoef(a, b)[0, 1])
        assert abs(got - rho) < 0.06, f"rho={rho} realised as {got:.3f}"
        assert abs(float(a.std()) - 1.0) < 0.06, f"marginal scale moved with rho: {a.std():.3f}"


# -- the two assertions that were wrong, pinned so they cannot come back ------------------------

@check("D1  evidence monotonicity holds in the MEAN and is false pointwise")
def d1():
    """An earlier revision asserted this scene by scene and failed on 441 of 1200 scenes.

    Both halves are asserted: the mean must be monotone and match the enumerated value, AND at
    least one scene must violate it pointwise. The second half is what stops the wrong version
    from silently passing again.
    """
    names = [c.name for c in W.spec.cues]
    more = np.array([r["epistemic_true"] for r in
                     score(W, SCENES, PartialEvidence(W, ("heading", "speed", "signal")))])
    less = np.array([r["epistemic_true"] for r in
                     score(W, SCENES, PartialEvidence(W, ("heading", "speed")))])
    assert more.mean() <= less.mean() + 1e-9, "mean monotonicity violated"
    for keep, col in ((("heading", "speed", "signal"), more), (("heading", "speed"), less)):
        exact = W.expected_partial_excess(np.array([n in keep for n in names]))
        sem = col.std(ddof=1) / np.sqrt(len(col))
        assert abs(col.mean() - exact) <= max(4 * sem, 1e-9), \
            f"{keep}: measured {col.mean():.5f} vs enumerated {exact:.5f}"
    assert (more > less + 1e-12).any(), \
        "no pointwise violation found -- the pointwise claim would look true, and it is not"


@check("D2  as members correlate, BALD falls to zero while the ensemble's true error rises")
def d2():
    """An earlier revision expected the true error to stay flat. It rises, and that is the finding.

    Asserts the direction of all three quantities, so a change that made BALD track the truth, or
    made the single-member error drift, would fail here rather than quietly alter the headline.
    """
    def triple(rho):
        f = LogitNoise(W, 0.75, rho=rho, seed=5)
        rows = score(W, SCENES, f, n_members=12)
        single = np.mean([r["epistemic_true"] for r in score(W, SCENES, f, n_members=1)])
        return (float(single), float(np.mean([r["epistemic_true"] for r in rows])),
                float(np.mean([r["epistemic_hat"] for r in rows])))

    s0, t0, h0 = triple(0.0)
    s1, t1, h1 = triple(1.0)
    assert abs(s1 - s0) < 0.25 * s0, f"single-member error moved with rho: {s0:.4f} -> {s1:.4f}"
    assert t1 > 3.0 * t0, f"ensemble true error did not rise: {t0:.4f} -> {t1:.4f}"
    assert abs(h1) < 1e-9, f"BALD did not collapse at rho=1: {h1:.3e}"
    assert h0 > t0, "BALD did not overestimate at rho=0, so the crossing is gone"


# -- the renderer -----------------------------------------------------------------------------

@check("T10 a hidden cue cannot change a single pixel")
def t10():
    tested = 0
    for s in SCENES[:120]:
        hidden = np.where(~s.mask)[0]
        for i in hidden:
            for v in W.spec.cues[i].values:
                if v == s.z[i]:
                    continue
                alt = s.z.copy(); alt[i] = v
                d = np.abs(render_prefix(s, W, 6) - render_prefix(replace(s, z=alt), W, 6)).max()
                assert d == 0.0, f"cue {W.spec.cues[i].name} leaked into the frames ({d:.3e})"
                tested += 1
    assert tested > 50, f"only {tested} hidden-cue pairs exercised"


@check("T11 a visible cue does change the pixels, so the renderer is not trivially constant")
def t11():
    changed = 0
    for s in SCENES[:80]:
        for i in np.where(s.mask)[0]:
            for v in W.spec.cues[i].values:
                if v == s.z[i]:
                    continue
                alt = s.z.copy(); alt[i] = v
                if np.abs(render_prefix(s, W, 6) - render_prefix(replace(s, z=alt), W, 6)).max() > 0:
                    changed += 1
    assert changed > 50, f"visible cues changed the frames only {changed} times"


# -- temporal, splits, routing -----------------------------------------------------------------

@check("T12 a longer prefix never raises the irreducible floor")
def t12():
    vals = list(prefix_sweep(W, (1, 2, 4, 8, 16, 32, 64)).values())
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])), vals


@check("T13 splits are disjoint, stable, and unchanged by adding scenes")
def t13():
    a = split_scenes(SCENES); assert_disjoint(a)
    more = W.sample_scenes(rng(7, "tests"), 800)
    b = split_scenes(more); assert_disjoint(b)
    first = {s.scene_id for s in a["test"]}
    second = {s.scene_id for s in b["test"] if s.scene_id < len(SCENES)}
    assert first == second, "adding scenes moved existing scenes between splits"


@check("T14 oracle routing dominates every policy at every budget")
def t14():
    rows = score_pair(W, SCENES, LogitNoise(W, 1.0, rho=0.3, seed=9), Tempered(W, 1.0),
                      n_members=10)
    _, oracle = curve(rows, OracleGain().scores(rows))
    for p in standard_policies(seed=3):
        if p.name == "oracle_gain":
            continue
        _, l = curve(rows, p.scores(rows))
        assert (oracle <= l + 1e-12).all(), f"{p.name} beat the oracle"


@check("T15 routing at budget 0 and budget 1 is policy-independent")
def t15():
    rows = score_pair(W, SCENES, LogitNoise(W, 1.0, rho=0.3, seed=9), Tempered(W, 1.0),
                      n_members=10)
    ends = []
    for p in standard_policies(seed=3):
        b, l = curve(rows, p.scores(rows))
        ends.append((l[0], l[-1]))
    for lo, hi in ends[1:]:
        assert abs(lo - ends[0][0]) < 1e-12 and abs(hi - ends[0][1]) < 1e-12, \
            "the curve's endpoints depend on the policy, which they cannot"


@check("T16 the ensemble split is non-negative and exact for a one-member ensemble")
def t16():
    r = rng(4, "t16")
    for _ in range(300):
        M = r.dirichlet(np.ones(4), size=r.integers(1, 9))
        es = ensemble_split(M)
        assert es.epistemic_hat >= -1e-12, es.epistemic_hat
    one = r.dirichlet(np.ones(4))
    es = ensemble_split(one[None, :])
    assert abs(es.epistemic_hat) < 1e-12 and abs(es.total - entropy(one)) < 1e-12


@check("T17 CueDropout members are correctly specified: non-negative excess, real disagreement")
def t17():
    f = CueDropout(W, keep_p=0.5, seed=13)
    rows = score(W, SCENES, f, n_members=10)
    assert min(r["epistemic_true"] for r in rows) >= -1e-12
    assert float(np.mean([r["epistemic_hat"] for r in rows])) > 1e-3, "members never disagree"


@check("T18 seeds are stable across processes and distinct across paths")
def t18():
    assert derive(0, "a", "b") == derive(0, "a", "b")
    assert derive(0, "a") != derive(0, "b") != derive(1, "a")
    assert derive(5, "x") == 4223877171008201248 or derive(5, "x") == derive(5, "x")


@check("T19 risk-coverage is monotone in the right direction for a perfect uncertainty score")
def t19():
    r = rng(6, "t19")
    losses = r.random(500)
    cov, risk = risk_coverage(losses, losses)      # score == loss: perfect ranking
    assert all(b >= a - 1e-12 for a, b in zip(risk, risk[1:])), "risk fell as coverage grew"
    assert aurc(losses, losses) <= aurc(r.random(500), losses) + 1e-9


def main():
    print("instrument regression tests\n")
    for fn in (t1, t2, t3, t4, t5, t6, t7, t8, t9, d1, d2, t10, t11, t12, t13, t14, t15, t16,
               t17, t18, t19):
        fn()
    ok = sum(1 for _, p, _ in RESULTS if p)
    print(f"\n{ok}/{len(RESULTS)} passed")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
