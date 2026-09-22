"""Why the cheap forecaster is worse -- the axis the mechanism map varies.

`run_routing.py` sweeps the irreducible floor with one cheap/expensive pair and finds that ranking
by total predictive uncertainty inverts. Taken alone that invites the reading that the floor is the
whole story. It is not: whether confidence routing inverts depends on the relationship between the
cheap forecaster's confidence and its error, and that relationship is a property of WHY the cheap
forecaster is worse.

The registry lives here rather than in the script because two scripts need exactly the same
definitions -- `run_mechanisms.py`, which reports one seed, and `run_robustness.py`, which re-runs
the same settings across seeds. Two copies of a mechanism definition would be two experiments that
look like one.

Every mechanism is defined structurally (cues ordered by visibility, not by the junction world's
vocabulary) so the same experiment runs unchanged in any world.
"""
SEED = 20260921


def by_visibility(w) -> list:
    """Cue names, most visible first."""
    return [c.name for c in sorted(w.spec.cues, key=lambda c: -c.visibility)]


def mechanisms(seed: int = SEED) -> dict:
    """name -> (description, world -> (cheap, expensive)).

    `seed` offsets every stochastic forecaster's noise stream, so a caller sweeping seeds varies
    the forecasters as well as the scenes. At the default it reproduces the published run exactly.
    """
    from ..baselines.predictors import CueDropout, LogitNoise, PartialEvidence, Tempered

    return {
        "missing_evidence": (
            "cheap sees only the most-visible cue; expensive also sees the next two",
            lambda w: (PartialEvidence(w, tuple(by_visibility(w)[:1])),
                       PartialEvidence(w, tuple(by_visibility(w)[:3])))),
        "injected_noise": (
            "both read every visible cue, cheap with 3.4x the logit noise",
            lambda w: (LogitNoise(w, 1.20, rho=0.3, seed=seed),
                       LogitNoise(w, 0.35, rho=0.3, seed=seed + 1))),
        "overconfidence": (
            "cheap has the right ranking at T=0.4; expensive is Bayes-optimal",
            lambda w: (Tempered(w, 0.4), Tempered(w, 1.0))),
        "underconfidence": (
            "cheap has the right ranking at T=2.5; expensive is Bayes-optimal",
            lambda w: (Tempered(w, 2.5), Tempered(w, 1.0))),
        "evidence_ensemble": (
            "cheap is an ensemble that drops cues at random; expensive keeps them all. "
            "The fair test of the mutual-information estimator: members disagree because evidence "
            "is missing, and their errors are independent by construction",
            lambda w: (CueDropout(w, keep_p=0.5),
                       PartialEvidence(w, tuple(c.name for c in w.spec.cues)))),
        "mixed": (
            "cheap is noisy; expensive instead has more evidence",
            lambda w: (LogitNoise(w, 1.00, rho=0.3, seed=seed + 2),
                       PartialEvidence(w, tuple(by_visibility(w)[:3])))),
    }
