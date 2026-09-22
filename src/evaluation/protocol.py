"""The scoring protocol, and the checks that validate it.

Everything a forecaster is judged by flows through `score` and `score_pair`, so the baselines and a
vision-language model are measured by identical code. The checks V1-V7 below have answers that are
known before they run, which is what makes them checks rather than results: each one asserts a
quantity the closed form fixes, and a protocol that fails any of them is reporting something other
than what it claims.
"""
from dataclasses import dataclass, field

import numpy as np

from ..forecasting.task import evidence_of
from ..forecasting.world import entropy, kl
from ..uncertainty.decomposition import ensemble_split, exact_split


def score(world, scenes, forecaster, n_members: int = 1) -> list:
    """One row per scene for one forecaster."""
    rows = []
    for sc in scenes:
        ev = evidence_of(sc, world)
        members = np.stack([forecaster.predict(ev, m) for m in range(n_members)])
        es = ensemble_split(members)
        xs = exact_split(sc.posterior, es.mean_prediction, sc.y)
        srt = np.sort(es.mean_prediction)[::-1]
        rows.append({
            "scene_id": sc.scene_id, "forecaster": forecaster.name, "n_visible": sc.n_visible,
            "y": sc.y, "correct": int(np.argmax(es.mean_prediction) == sc.y),
            "max_prob": float(srt[0]), "second_prob": float(srt[1]),
            "aleatoric_true": xs.aleatoric, "epistemic_true": xs.epistemic,
            "expected_nll": xs.expected_nll, "realised_nll": xs.realised_nll,
            "total_hat": es.total, "aleatoric_hat": es.aleatoric_hat,
            "epistemic_hat": es.epistemic_hat,
            "_q": es.mean_prediction,
        })
    return rows


def score_pair(world, scenes, cheap, expensive, n_members: int = 1) -> list:
    """Rows for a cheap/expensive pair, carrying both forecasters' true epistemic error.

    The routing policies need the expensive forecaster's error to compute the achievable gain, but
    only `OracleGain` is allowed to read it -- every other policy sees a strict subset of the row.
    The subset each policy touches is the whole experimental manipulation, so it is enforced by
    which keys a policy's `score` reads rather than by comment.
    """
    a = score(world, scenes, cheap, n_members)
    b = score(world, scenes, expensive, n_members)
    for ra, rb in zip(a, b):
        assert ra["scene_id"] == rb["scene_id"]
        ra["epistemic_true_expensive"] = rb["epistemic_true"]
        ra["realised_nll_expensive"] = rb["realised_nll"]
        ra["forecaster_expensive"] = rb["forecaster"]
    return a


# ---------------------------------------------------------------------------------------------
# Validation checks. Each returns a Check whose `expected` was known before the code ran.
# ---------------------------------------------------------------------------------------------

@dataclass
class Check:
    id: str
    name: str
    expected: str
    measured: str
    passed: bool
    detail: dict = field(default_factory=dict)


def v1_oracle_floor(world, scenes) -> Check:
    """The Bayes-optimal forecaster must carry exactly zero epistemic uncertainty."""
    from ..baselines.predictors import BayesOptimal
    rows = score(world, scenes, BayesOptimal(world))
    worst = max(r["epistemic_true"] for r in rows)
    mean_nll = float(np.mean([r["expected_nll"] for r in rows]))
    mean_alea = float(np.mean([r["aleatoric_true"] for r in rows]))
    return Check("V1", "Bayes-optimal forecaster has zero epistemic uncertainty",
                 "max KL(P*||Q) = 0 exactly", f"max = {worst:.3e}",
                 worst < 1e-12,
                 {"max_epistemic": worst, "mean_expected_nll": mean_nll,
                  "mean_aleatoric": mean_alea,
                  "nll_equals_aleatoric": abs(mean_nll - mean_alea) < 1e-12})


def v2_uniform_reference(world, scenes) -> Check:
    """The uniform forecaster must score exactly log K, scene by scene."""
    from ..baselines.predictors import Uniform
    K = len(world.spec.outcomes)
    rows = score(world, scenes, Uniform(K))
    dev = max(abs(r["expected_nll"] - np.log(K)) for r in rows)
    return Check("V2", "Uniform forecaster scores log K on every scene",
                 f"expected NLL = log {K} = {np.log(K):.6f} exactly", f"max deviation = {dev:.3e}",
                 dev < 1e-12, {"max_deviation": dev, "log_K": float(np.log(K))})


def v3_marginal_excess_is_information(world, scenes) -> Check:
    """The prefix-ignoring forecaster's excess loss must equal the mutual information.

    Exactly, in expectation: E[KL(P* || P(y))] = H(P(y)) - E[H(P*)]. The comparison is against the
    closed-form expectation, so the tolerance is Monte Carlo error over the sampled scenes and not
    a fudge factor -- it is reported beside the number.
    """
    from ..baselines.predictors import Marginal
    rows = score(world, scenes, Marginal(world))
    measured = float(np.mean([r["epistemic_true"] for r in rows]))
    sem = float(np.std([r["epistemic_true"] for r in rows], ddof=1) / np.sqrt(len(rows)))
    exact = entropy(world.marginal()) - world.expected_aleatoric()
    return Check("V3", "Marginal forecaster's excess loss equals I(y; z_m)",
                 f"I(y; z_m) = {exact:.6f} nats (exact)",
                 f"{measured:.6f} +/- {3 * sem:.6f} (3 sem)",
                 abs(measured - exact) < max(3 * sem, 1e-9),
                 {"exact": exact, "measured": measured, "sem": sem})


def v4_evidence_monotone(world, scenes) -> Check:
    """More cues cannot increase expected epistemic error, and each rung must hit its closed form.

    The check is on the MEAN, not scene by scene. The data-processing inequality constrains
    E[KL(P* || P*_S)], and that expectation is what `World.expected_partial_excess` computes
    exactly. Pointwise it is simply false: on an individual scene a forecaster reading fewer cues
    can land closer to the full posterior by chance, and about 2% of scenes here do. An earlier
    revision of this check asserted the pointwise version and failed on 441 of 1200 scenes; the
    assertion was wrong, not the instrument.

    Two things are checked together, and the second is the stronger one: the ladder is monotone,
    and every rung matches a number computed by enumeration rather than by sampling.
    """
    from ..baselines.predictors import BayesOptimal, Marginal, PartialEvidence
    # Cue names come from the world, so the ladder is the same experiment in any world rather than
    # one wired to the junction's vocabulary. Rungs drop cues from the least visible upward, which
    # is the order in which a real prefix loses them.
    names = [c.name for c in world.spec.cues]
    order = sorted(names, key=lambda n: -dict((c.name, c.visibility) for c in world.spec.cues)[n])
    keeps = [tuple(order[:k]) for k in range(len(order), 0, -1)] + [()]
    ladder = [(BayesOptimal(world), tuple(order))]
    for keep in keeps[1:-1]:
        ladder.append((PartialEvidence(world, keep), keep))
    ladder.append((Marginal(world), ()))

    rungs, ok = [], True
    for f, keep in ladder:
        vals = np.array([r["epistemic_true"] for r in score(world, scenes, f)])
        exact = world.expected_partial_excess(np.array([n in keep for n in names]))
        sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        hit = abs(float(vals.mean()) - exact) <= max(3 * sem, 1e-9)
        ok &= hit
        rungs.append({"forecaster": f.name, "keep": list(keep), "measured": float(vals.mean()),
                      "exact": exact, "sem": sem, "matches_closed_form": bool(hit)})

    means = [r["measured"] for r in rungs]
    monotone = all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1))
    return Check("V4", "Expected epistemic error is monotone in the cues used, and matches closed form",
                 "ladder non-decreasing; every rung within 3 sem of its enumerated value",
                 f"monotone={monotone}; "
                 f"{sum(r['matches_closed_form'] for r in rungs)}/{len(rungs)} rungs match",
                 bool(monotone and ok), {"rungs": rungs, "monotone": bool(monotone)})


def v5_prefix_monotone(world, lengths=(1, 2, 4, 8, 16, 32)) -> Check:
    """A longer prefix cannot raise the irreducible floor. Exact, computed by enumeration."""
    from ..temporal.prefix import prefix_sweep
    sweep = prefix_sweep(world, lengths)
    vals = [sweep[int(L)] for L in lengths]
    diffs = np.diff(vals)
    return Check("V5", "Aleatoric floor is non-increasing in prefix length",
                 "every successive difference <= 0",
                 f"max increase = {float(diffs.max()):.3e}",
                 bool((diffs <= 1e-12).all()),
                 {"sweep": {int(k): float(v) for k, v in sweep.items()}})


def v6_ensemble_confound(world, scenes, sigma=0.75, rhos=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0),
                         n_members=12) -> Check:
    """THE confound control, and the project's central measurement.

    Three quantities are tracked as the ensemble's members go from independent to identical, at a
    noise scale that holds each member's own error fixed:

      single      KL(P* || Q_m) for one member. Must stay flat: rho changes how members relate to
                  each other, not how wrong any one of them is. This is the control that makes the
                  other two readable.
      true        KL(P* || Qbar) for the ensemble mean, which is what actually answers. Must RISE,
                  because averaging only cancels error that the members do not share.
      BALD        H(Qbar) - mean_m H(Q_m), the estimate. Must FALL to exactly zero, because it
                  measures disagreement and identical members do not disagree.

    The prediction is therefore not that the estimate degrades but that the estimate and the
    quantity it stands in for move in OPPOSITE directions, and cross. An earlier revision of this
    check expected the true error to stay flat while the estimate fell; that was wrong, and the
    truth is the stronger claim.
    """
    from ..baselines.predictors import LogitNoise
    table = []
    for rho in rhos:
        f = LogitNoise(world, sigma, rho=rho)
        rows = score(world, scenes, f, n_members=n_members)
        single = float(np.mean([r["epistemic_true"] for r in score(world, scenes, f, n_members=1)]))
        t = float(np.mean([r["epistemic_true"] for r in rows]))
        h = float(np.mean([r["epistemic_hat"] for r in rows]))
        table.append({"rho": float(rho), "single_member_kl": single, "epistemic_true": t,
                      "epistemic_hat": h, "ratio_hat_over_true": h / t if t > 0 else float("nan")})

    lo, hi = table[0], table[-1]
    single_flat = abs(hi["single_member_kl"] - lo["single_member_kl"]) < 0.25 * lo["single_member_kl"]
    true_rises = hi["epistemic_true"] > 3.0 * lo["epistemic_true"]
    bald_collapses = abs(hi["epistemic_hat"]) < 1e-9
    bald_monotone = all(table[i]["epistemic_hat"] >= table[i + 1]["epistemic_hat"] - 1e-12
                        for i in range(len(table) - 1))
    true_monotone = all(table[i]["epistemic_true"] <= table[i + 1]["epistemic_true"] + 1e-12
                        for i in range(len(table) - 1))
    return Check("V6", "As members correlate, the estimate and the true error move in opposite directions",
                 "single-member error flat; KL(P*||Qbar) rises monotonically; BALD falls monotonically to 0",
                 f"single {lo['single_member_kl']:.4f}->{hi['single_member_kl']:.4f}; "
                 f"true {lo['epistemic_true']:.4f}->{hi['epistemic_true']:.4f} "
                 f"({hi['epistemic_true'] / max(lo['epistemic_true'], 1e-12):.1f}x); "
                 f"BALD {lo['epistemic_hat']:.4f}->{hi['epistemic_hat']:.2e}",
                 bool(single_flat and true_rises and bald_collapses
                      and bald_monotone and true_monotone),
                 {"table": table, "sigma": sigma, "n_members": n_members,
                  "single_flat": bool(single_flat), "true_rises": bool(true_rises),
                  "bald_collapses": bool(bald_collapses),
                  "bald_monotone": bool(bald_monotone), "true_monotone": bool(true_monotone)})


def v7_routing_dominance(world, scenes, n_members=12) -> Check:
    """The oracle policy must dominate every other policy at every budget.

    A ranking cannot beat ranking by the true gain, so any policy that does indicates a defect in
    the curve rather than a good policy. Checked at every budget, not on the integral, because an
    integral can hide a crossing.
    """
    from ..baselines.predictors import LogitNoise, Tempered
    from ..routing.curves import DEFAULT_BUDGETS, curve
    from ..routing.policies import standard_policies
    cheap = LogitNoise(world, 1.0, rho=0.3)
    expensive = Tempered(world, 1.0)                 # exactly Bayes-optimal: the best available
    rows = score_pair(world, scenes, cheap, expensive, n_members=n_members)
    _, oracle = curve(rows, [p for p in standard_policies() if p.name == "oracle_gain"][0].scores(rows))
    worst, offender = 0.0, None
    for p in standard_policies():
        if p.name == "oracle_gain":
            continue
        _, l = curve(rows, p.scores(rows))
        d = float((oracle - l).max())                # oracle loss must never exceed another policy's
        if d > worst:
            worst, offender = d, p.name
    return Check("V7", "Oracle routing dominates every policy at every budget",
                 "no policy achieves lower loss than the oracle at any budget",
                 f"largest violation = {worst:.3e}" + (f" by {offender}" if offender else ""),
                 worst < 1e-12, {"worst_violation": worst, "offender": offender})


def run_all(world, scenes) -> list:
    return [v1_oracle_floor(world, scenes), v2_uniform_reference(world, scenes),
            v3_marginal_excess_is_information(world, scenes), v4_evidence_monotone(world, scenes),
            v5_prefix_monotone(world), v6_ensemble_confound(world, scenes),
            v7_routing_dominance(world, scenes)]
