# Novelty analysis

**Revision 2026-09-22. The verification pass has run, it changed the answer, and every reference
has since been re-read from its own arXiv record before entering the paper.**

An earlier revision of this document opened by recording that no literature had been checked and
that nothing could be written up until that changed. It has now been checked. Three of the
project's seven claims are **refuted** by two 2026 papers, and the project's position is materially
weaker than it looked before the search. That is the verification pass doing its job.

Marker convention: `[V]` read from the paper's own text, `[A]` abstract read directly, `[S]` search
summary only, `[U]` unverified recollection. Nothing marked `[S]` or `[U]` may enter a paper.

## 1. Verdicts

| id | claim | verdict |
|---|---|---|
| A | The aleatoric/epistemic decomposition of expected log loss | **known**, as expected. Instrumentation, claimed as nothing. |
| B | The mutual-information (BALD) estimate measures ensemble disagreement, not forecaster error, and the two come apart | **REFUTED** `[V]` — Young 2026 (2606.12646) prove the measure and the definition are extensionally inconsistent, and show disagreement collapsing to zero *beneath a positive truth*. |
| C | The estimated aleatoric term tracks forecaster confidence rather than the world | **WOUNDED** `[A]` — implied by the same paper's result that the deployed estimate tracks the training procedure. Not independently established, but not safe to claim. |
| D | The aleatoric term cancels from the escalation decision, so confidence routing spends budget on an irrelevant quantity | **REFUTED** `[V]` — Peale et al. 2026 (2605.07805) state it in the introduction and build a router on exactly this decomposition. |
| E | Confidence routing inverts as the irreducible floor rises | **REFUTED as a phenomenon** `[V]` — Peale et al. Figure 1 shows total-uncertainty routing underperforming a pointwise-optimal policy on synthetic data. The *below-random* measurement and the floor sweep are ours, but the effect is theirs. |
| F | Whether it inverts depends on **why** the cheap model is worse | **PARTIALLY ANTICIPATED** `[V]` — Peale et al. "lay out different regimes … and describe when routing based on reducible uncertainty yields the most benefit," with their axis being the *correlation* between reducible and irreducible uncertainty. Our axis is the *cause* of the deficit. Adjacent and overlapping, not identical. |
| G | Prefix length as a second lever, with the irreducible term cancelling for model choice but **not** for prefix choice | **SURVIVES, narrowly** `[S]` — the video-anticipation literature knows the future gets more uncertain with a longer prediction horizon; the observation-prefix framing and the non-cancellation result were not found. Now measured (§4). |

## 2. The two papers that did the damage

### Robin Young, *Epistemic Uncertainty Is Not the Reducible Kind*, arXiv:2606.12646, 10 June 2026 `[V]`

Abstract, verbatim: *"The standard taxonomy of predictive uncertainty defines epistemic uncertainty
as the part removable by collecting more data, while the standard measure identifies it with a
mutual-information term. We prove the definition and the measure are extensionally inconsistent."*
And: *"Ensemble disagreement, the deployed epistemic estimate, tracks the training procedure rather
than the epistemic term. It collapses to zero beneath a positive truth under consistent training."*

This is claim B, proved, with synthetic closed-form constructions (a two-region Bernoulli, a
Gaussian-process amplitude construction, overparametrised linear regression). It also removes the
idea that a synthetic instrument with a known `P*` is itself a contribution.

**What it does not do**, checked directly against the paper: it does not parameterise or sweep
ensemble member **correlation** — collapse is driven by convergence to the Bayes predictor under
consistent training, and members are assumed independently seeded. It runs **no** experiment on
routing, cascades, deferral or selective prediction; the only mention is a remark in the Discussion
that *"Every acquisition loop that routes high-epistemic inputs to more deployment-distribution
sampling spends budget that Corollary 3 proves is worth at most zero."* It does not analyse
aleatoric uncertainty's effect on any decision procedure.

### Peale, Devic, Gopalan, Wieder, Gollakota, *Flexible Routing via Uncertainty Decomposition*, arXiv:2605.07805, 8 May 2026 `[V]`

Stanford / USC / Apple. From the introduction: *"Confidence-based methods often fail to account for
irreducible (aleatoric) uncertainty (such as ambiguous or noisy utterances), needlessly routing to
the oracle even when it would also perform poorly."* Their router decomposes total uncertainty into
irreducible and reducible components via higher-order predictors and *"predict[s] with the weak
model when uncertainty is low, route[s] to the oracle when reducible uncertainty is high, and
abstain[s] when irreducible uncertainty is high."* Figure 1 compares *"Route by Total Uncertainty"*
against a pointwise-optimal policy on synthetic data.

That is claims D and E. It also states the regime condition that overlaps claim F: *"we expect to
see a benefit when reducible uncertainty is not too correlated with irreducible uncertainty."*

**What it does not do**: it assumes *"multiple independent annotations per input"* and a
higher-order calibration guarantee. It does not use ensembles, does not sweep ensemble correlation,
does not compare against a random-routing baseline, does not vary the irreducible floor as a
controlled variable, and does not distinguish failure by the cause of the weak model's deficit.

Also noted there, and useful: *"most LLM routing papers rarely explicitly consider epistemic and
aleatoric uncertainty decompositions, and simply optimize the pareto-frontier of certain
performance-cost curves."*

## 3. What actually survives

Honestly, and it is much less than before the search:

1. **The estimator you actually have, rather than the one that is assumed.** Peale et al. need
   multiple independent annotations per input and a higher-order calibration guarantee. Young
   assumes independently seeded members. Neither covers the practical substitute — an ensemble of
   similarly-pretrained models — and our ρ sweep shows precisely what that substitute does: at ρ=0
   the estimate overstates the ensemble's error by 8.1×, and at ρ=1 it reports zero for an ensemble
   carrying 0.149 nats of recoverable error, with the estimate and the truth crossing near ρ=0.5.
   This reproduces in both worlds (10.0× and 9.5× rises in the true error).
   This is a complementary negative result about a regime both papers assume away.
2. **The mechanism map by deficit cause** (claim F), which is a different axis from Peale et al.'s
   correlation condition, and measured against an explicit random baseline: benign under missing
   evidence (+0.158 → +0.043), catastrophic under miscalibration (−0.443 → −0.992), and
   floor-invariant when the estimate is well-posed (+0.557 → +0.663, and +0.819 → +0.843 in the
   second world).
3. **The prefix lever** (claim G), §4 below. The only measurement here that is specific to video.

## 4. Claim G, now measured

`scripts/run_prefix.py`. Two responses to an uncertain forecast — spend on the model, or spend on
the evidence — and the irreducible term cancels from the first but not the second.

**92–97% of the prefix-extension gain is exactly ΔH(P*)**, the term that cancels entirely from model
escalation. That holds in both worlds at every prefix length, so the two decisions are driven by
almost disjoint quantities.

Whether the two correlations have *opposite signs* depends on the world, and the honest version is
the narrower one:

| prefix | junction: model / prefix | kitchen (non-additive): model / prefix |
|---|---|---|
| 1 → 2 | **−0.189** / **+0.151** | **−0.490** / **+0.059** |
| 2 → 4 | +0.001 / +0.135 | −0.346 / +0.052 |
| 4 → 8 | +0.084 / +0.117 | −0.220 / +0.083 |
| 8 → 16 | +0.127 / +0.070 | −0.133 / +0.060 |

In the non-additive world the opposition holds throughout; in the additive one only at the shortest
prefix. Short prefixes with interacting cues are the regime video forecasting operates in, and an
image classifier has no second decision to serve at all.

## 5. Bibliography

Re-verified 2026-09-22, one record at a time, before any of it was allowed into `paper/refs.bib`.
Two entries were **wrong in this table** and are corrected below; the citation keys in the paper
carry the corrected forms.

| ref | status | note |
|---|---|---|
| arXiv:2606.12646, Robin Young, *Epistemic Uncertainty Is Not the Reducible Kind*, 10 Jun 2026 | `[V]` abstract + HTML body read; abstract re-read 09-22 and matches §2 verbatim | refutes B. stat.ML, cross-listed cs.IT and cs.LG; **no venue listed** |
| arXiv:2605.07805, **Charlotte Peale, Siddartha Devic, Parikshit Gopalan, Udi Wieder, Aravind Gollakota**, *Flexible Routing via Uncertainty Decomposition*, 8 May 2026 | `[V]` pp. 1–4 read from the PDF; abstract re-read 09-22 | refutes D and E; anticipates part of F. cs.LG |
| arXiv:2605.06350, **Dylan Bouchard**, *Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades*, 7 May 2026 | `[A]` abstract read 09-22 | cost-quality frontier, piecewise concavity and shadow prices; does **not** touch the decomposition. Also finds a pre-generation router beating the best cascade policy, for a structural-cost reason rather than a signal reason |
| arXiv:2603.02719, **L. Julián Lechuga López, Farah E. Shamout, Tim G. J. Rudner**, *An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification*, 3 Mar 2026 | `[V]` targeted read; abstract re-read 09-22 | selective prediction degrading under class-dependent miscalibration — "high uncertainty to correct predictions and low uncertainty to incorrect ones". **No random baseline is measured and the floor is not varied** |
| arXiv:2409.02628, **Andreas Kirsch**, *(Implicit) Ensembles of Ensembles: Epistemic Uncertainty Collapse in Large Models*, 4 Sep 2024 | `[A]` record read 09-22. **This table previously gave neither the author nor the full title** | collapse as **model complexity** rises, explained by *implicit ensembling*. A different cause from member correlation, and no routing. Wounds §3.1, does not kill it |
| arXiv:2403.14715, **Guoxuan Xia, Olivier Laurent, Gianni Franchi, Christos-Savvas Bouganis**, *Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It*, **ICLR 2025** | `[V]` targeted read; record read 09-22. **This table previously gave a shortened title and no venue** | LS suppresses the max logit more where the prediction is likely correct, degrading the uncertainty ranking. **No random baseline, no controlled aleatoric variation, no ablation separating miscalibration from evidence deficiency** |
| arXiv:2412.18808, **Gustaf Ahdritz, Aravind Gollakota, Parikshit Gopalan, Charlotte Peale, Udi Wieder**, *Provable Uncertainty Decomposition via Higher-Order Calibration*, **ICLR 2025** | `[A]` record read 09-22, **upgraded from `[S]`** | the decomposition Peale et al. build on. Needs *k-snapshots* — k independent conditional labels per point — which is the assumption our ρ sweep is about not having. Four of its five authors are Peale et al.'s |
| **Yazan Abu Farha, Juergen Gall**, *Uncertainty-Aware Anticipation of Activities*, arXiv:1908.09540, **ICCV 2019 workshop (HBU)** | `[A]` record read 09-22, **upgraded from `[S]`** | video anticipation under ambiguity by sampling future sequences; the axis is the prediction **horizon**, not the observation prefix |

## 6. The two flagged reads, now done

**arXiv:2409.02628, *Epistemic Uncertainty Collapse in Large Models*** `[A]`. Reports epistemic
uncertainty collapsing as **model complexity** rises, with *implicit ensembling* offered as the
explanation, in both explicit ensembles and wider single models. This is a real wound to §3.1 —
"the deployed epistemic estimate collapses" is now documented from two independent directions — but
the cause is complexity, not member correlation, and there is no routing, no correlation sweep and
no comparison against a known ground-truth epistemic error. §3.1 survives in narrowed form: what
remains ours is the ρ-parameterised link from member correlation to *routing* performance against a
known truth, not the collapse itself.

**arXiv:2403.14715, *Why Label Smoothing Degrades Selective Classification*** `[V]`. Label smoothing
degrades selective classification because "the regularization gradient on max logits decreases as
error probability increases", exacerbating over- and underconfidence and degrading the uncertainty
ranking. This is the miscalibration half of the mechanism map, already known in direction. It does
**not** compare against random rejection, does not vary aleatoric uncertainty as a controlled
variable, and does not separate miscalibration from evidence deficiency by ablation — which is
exactly what §3.2 does.

**Nothing is outstanding.** The last item — the author list and venue of arXiv:2606.12646 — is
now captured: Robin Young, submitted 10 June 2026, stat.ML with cross-lists to cs.IT and cs.LG, no
venue listed. A single-author preprint with no venue is worth re-checking for a published version
before citing, but it does not change any verdict.

## 7. What the pre-paper re-verification changed (2026-09-22)

Before writing the paper, every entry in §5 was re-read from its own arXiv record, because the
project's rule — nothing marked `[S]` or `[U]` may enter a paper — has to be enforced at the point
the citation is actually made rather than at the point it is planned.

It found two defects in §5 itself, both of the kind that would have produced a wrong citation:
arXiv:2409.02628 was listed with no author and a shortened title (it is Kirsch, *(Implicit)
Ensembles of Ensembles*), and arXiv:2403.14715 was listed with a shortened title and no venue (it
is Xia et al., ICLR 2025). Two `[S]` entries were upgraded to `[A]` with identifiers and venues so
they could be cited at all, and one of them mattered more than expected: Ahdritz et al. require
*k-snapshots*, k independent conditional labels per input, which is precisely the assumption §3.1
is about not having. That sharpened the paper's positioning rather than changing any verdict.

**No verdict in §1 moved.** The two refuting abstracts match what §2 quotes, checked word for word.
