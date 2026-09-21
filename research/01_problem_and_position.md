# Problem formulation

## 1. The question

A forecaster is shown a prefix of video and asked what happens next. It reports a distribution over
outcomes and, one way or another, a level of uncertainty. The question this project asks is what
that uncertainty is *about*:

> When a model is uncertain about what happens next in a video, is that because the evidence in the
> clip does not determine the future, or because the model cannot read the evidence?

The distinction is not terminological. Only the second kind is worth spending on. A clip whose
continuation is genuinely undetermined by what it shows will defeat every forecaster equally, and
escalating it to a larger model, or running an ensemble over it, buys nothing. A clip the model
simply misreads is recoverable. Any system that decides where to spend compute — routing,
escalation, deferral, active labelling — is implicitly making this distinction, and the standard
practice of ranking by predictive confidence does not make it at all.

## 2. Formal statement

A scene carries latent cues `z`. The outcome is drawn from `P(y | z)`. A prefix reveals a subset of
the cues, recorded by an observability mask `m` drawn independently of `z`, so an observer sees
`z_m` and no more. The best any forecaster can do from that prefix is the posterior

```
P*(y | z_m)  =  sum_{z_h}  P(z_h) · P(y | z_m, z_h)                                  (1)
```

For any forecaster `Q`, expected log loss splits exactly:

```
E_{y ~ P*} [ -log Q(y) ]   =   H(P*)        +   KL(P* || Q)                          (2)
                               ^ aleatoric      ^ epistemic
```

This is an identity, not a modelling assumption, and it is the axis the whole project turns on:

- **`H(P*)` is a property of the world and the prefix.** No forecaster can reduce it and every
  forecaster pays it. It is the same number for a state-of-the-art model and for a coin flip.
- **`KL(P* || Q)` is a property of the forecaster.** It is zero exactly when the forecaster is
  Bayes-optimal, and it is the entire quantity a better model, or a longer prefix, could recover.

Both are defined **per scene**, not per dataset. That matters: a dataset-level "aleatoric
uncertainty" averages over prefixes that differ enormously in how much they reveal, and the
averaging is what makes the quantity look unactionable.

## 3. Why this forces a synthetic world

`P*` is not observable on real video. Nobody knows the true conditional distribution of the future
given a dashcam clip, so `H(P*)` cannot be measured there — only assumed. Every empirical claim
about aleatoric uncertainty on real video is therefore a claim about an estimator whose target is
unavailable for comparison.

So the instrument starts somewhere `P*` is computable in closed form, validates every estimator
against it, and only then points at a model whose behaviour is unknown. Concretely: finite cue
alphabets and independent cues make the marginalisation in (1) an exact finite sum rather than an
approximation. The full construction is `research/04_instrument.md`.

The cost of this choice is stated plainly: **nothing here is evidence about real video yet.** What
it is evidence about is the *estimators* — and since those estimators are the ones deployed on real
video, showing that they fail in a world where the truth is known is informative about what they
are doing where it is not.

## 4. What the escalation decision actually depends on

Escalating one scene from a cheap forecaster `Q_c` to an expensive `Q_e` changes expected loss by

```
[ H(P*) + KL(P*||Q_c) ]  −  [ H(P*) + KL(P*||Q_e) ]  =  KL(P*||Q_c) − KL(P*||Q_e)    (3)
```

**The aleatoric term cancels exactly.** It is paid whichever forecaster answers, so it carries no
information about whether escalating is worth it.

But `H(Q_c)` — the total predictive entropy that confidence-based routing ranks by — *contains*
that term, and is often dominated by it. A confidence-routed system spends part of its budget in
proportion to a quantity that is by construction irrelevant to the decision, and does so most
severely in the domains where routing is most attractive, because those are the domains with high
irreducible uncertainty.

Two cautions, both of which the measurements bore out and neither of which was obvious in advance:

1. **Cancelling from the decision is not the same as being uncorrelated with it.** `H(P*)` has no
   causal role in (3), but it can still be *statistically* correlated with the gain in a particular
   world. It is here: `corr(H(P*), gain)` measures +0.28 at τ=0.35 and decays to −0.02 by τ=3.5. So
   a policy that ranks purely by the irreducible floor looks useful in a low-floor world. That is
   the confound being visible, not a counterexample to it.
2. **Whether confidence routing fails at all depends on why the cheap model is worse.** If its
   deficit is missing evidence, its uncertainty is honest and confidence routing degrades gently. If
   its deficit is miscalibration, its entropy is low exactly where it is most wrong and confidence
   routing inverts. This is measured in `research/04_instrument.md` §5 and is the main qualification
   on the headline.

## 5. The estimator that is supposed to fix this

The standard fix is to estimate the epistemic term from ensemble disagreement. For an ensemble of
`M` forecasts with mean `Q̄`:

```
H(Q̄)  =  (1/M) Σ_m H(Q_m)   +   [ H(Q̄) − (1/M) Σ_m H(Q_m) ]                         (4)
^ total   ^ aleatoric estimate    ^ epistemic estimate (the mutual information; BALD)
```

(4) is usually introduced as though it estimated (2). It does not. The bracketed term measures
**disagreement between the members**, which equals the forecaster's error only when the members'
errors are independent. The project's central measurement is what happens when they are not — which
is the regime an ensemble of similarly-pretrained vision-language models sits in by construction.

## 6. Scope, and what is not claimed

- No vision-language model has been run. Not one API call has been made. The VLM arm is written,
  wired and costed, and it is waiting on the decision in `research/03_research_decision.md`.
- No real video corpus has been downloaded. `data/` is empty by design.
- No literature verification has been performed. See `research/02_novelty_analysis.md`, which
  states this at the top and treats it as the project's largest open risk.
- The results are about estimators in a world where the truth is known. Their bearing on real video
  is an argument, made in §3, not a measurement.
