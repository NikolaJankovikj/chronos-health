# Phase 1 — Methodology and Discussion

This document explains why each Phase-1 design decision was made, what was tried first, and what went wrong along the way. The numbers live in `results/phase1_results.md`. This document is for the thesis methods and discussion chapters.

---

## 1. Dataset and target signal

**Why WESAD.** Phase 1 needs a clean stress label, high-quality ECG, and extra sensor channels. WESAD provides all three. Its stressor, the Trier Social Stress Test (TSST: public speech plus mental arithmetic), is the most reproducible acute psychosocial stressor in the literature. The 15 subjects follow one of two protocol orders: 7 stress-first and 8 amusement-first. Each labelled condition is separated from the next by unlabelled transition periods.

**Why RMSSD.** RMSSD reflects parasympathetic (vagal) activity, the branch that withdraws acutely under stress. It is also stable on short windows. SDNN and LF/HF need 5-minute windows, and LF/HF is distorted by breathing. Heart rate is kept as a covariate rather than the target because it mixes sympathetic activation with vagal withdrawal.

**Why a 60-s window at 1 Hz.** At about 65 bpm, a 60-s window holds about 65 intervals, above the ~30 needed for a stable RMSSD. Shorter windows are noisy, and longer ones blur condition boundaries.

**Preprocessing.**
1. NeuroKit2 cleans the 700 Hz chest ECG and detects R-peaks.
2. RR intervals outside 300–2000 ms (30–200 bpm) are rejected.
3. Intervals deviating more than 20% from a 5-beat local median are rejected (Malik criterion).
4. Rejected intervals are set to NaN, not deleted. Successive differences are therefore never taken across a rejected beat. Deleting them would join non-adjacent intervals and create spurious differences.

Rejection rates were 0.0–3.0% per subject.

**Sanity check (T1.7).** 10 of 15 subjects have lower RMSSD under stress than at baseline. The other 5 (S6, S7, S9, S10, S15) go the other way. Individual variability in autonomic stress responses is well documented, so these subjects are kept and analysed separately.

---

## 2. Covariates

All covariates come from the chest RespiBAN and are pooled to 1 Hz:
- heart rate
- accelerometer magnitude
- EDA
- respiration rate (0.1–0.5 Hz band-pass, then peak detection)
- skin temperature

They were included for methodological continuity with Phase 2, where physical activity is the main confound of RMSSD. One property of WESAD decides what they can do here. Every condition is seated except the TSST, and every covariate responds to the TSST:
- EDA, heart rate and skin temperature respond to stress itself (sympathetic arousal, peripheral vasoconstriction).
- Respiration and movement respond to the speaking task.

So in WESAD no covariate varies for non-stress reasons inside the reference context. That property determines the covariate results (§5).

---

## 3. Evaluation design

### 3.1 First attempt: rolling context (AUROC ≈ 0.47)
The natural streaming design forecasts each 30-s window from the preceding 5 minutes. Every method scored near chance. Stress begins 17–44 minutes after the baseline ends, so by the time a stress window is scored the rolling context already contains depressed RMSSD. Forecasting "more of the same" then produces no anomaly. Forecast-residual detectors detect **changes**. WESAD's labels mark a sustained state, and the transition into it happens in unlabelled time.

### 3.2 Final design: personal-baseline reference
The question becomes: *"Given five minutes of this person's rest, is this 30-s window anomalous?"*
- **Context:** the last 300 s of the first two-thirds of the baseline.
- **Held-out baseline windows:** start 60 s after the context ends. That gap equals the RMSSD window, so no beat is shared with the context.
- **Stress windows:** the scored positives.
- **Windows:** 30 s, non-overlapping, and tiled inside time-contiguous label segments so no window mixes conditions.

A wearable that calibrates on a rest period and then monitors for deviations would answer exactly this question.

### 3.3 Controls added so the primary AUROC can be trusted
- **Time-distance control.** Held-out baseline windows are minutes from the context, while stress windows are 17–44 minutes away. Amusement and meditation windows are just as distant but not stressful. Reporting stress-vs-(amusement + meditation) AUROC tests whether the detector responds to stress or merely to elapsed time and any change. It is above chance (0.69–0.79 for the univariate models) but well below the primary AUROC. So the detector picks up deviations from rest, and stress deviates the most.
- **HRV-dependence control.** For covariate models, the observed RMSSD is replaced by the context mean and the AUROC is recomputed. A detector that still separates stress is not detecting an HRV anomaly.
- **Covariate range diagnostic.** For each covariate, the share of stress-period values that fall outside the context's range.

---

## 4. Scoring

| Attempt | Problem |
|---|---|
| Fraction of observations outside the forecast [q10, q90] band | An 80% band already yields ~20% exceedance at rest; stress and baseline scores were indistinguishable |
| One-sided normalised deviation (only drops count) | Inverted AUROC for the 5 atypical subjects |
| **Two-sided normalised deviation** (final) | Direction-agnostic; works for both response types |

The final score is the window mean of |observed − point forecast| / σ̂. σ̂ comes from each model's own predictive uncertainty, converted to standard-deviation units so that the scores of different models are comparable:
- **SMA:** context std
- **ARIMA(X):** forecast standard error
- **Chronos-2:** (q90 − q10) / 2.563

The cost of the two-sided score is specificity: it rewards any departure from rest, which the time-distance control quantifies (§3.3).

---

## 5. Models and what the results say about them

**Primary comparison: univariate, identical inputs.**
- **SMA:** context mean and std.
- **ARIMA(2,0,2):** fitted per subject on the context.
- **Chronos-2-small:** zero-shot.

All three reach a median AUROC of about 0.96 and are practically equivalent. Chronos-2 is not meaningfully better than SMA (upper 95% bound +0.010), and ARIMA is better by a negligible 0.004. This is expected. With a fixed reference, each model's prediction reduces to an estimate of the person's resting level and spread, and five minutes of rest determine that well for any method. Zero-shot pretraining does not help with an estimation problem this simple.

**Secondary comparison: covariates.** Both covariate models get the covariates' history and their concurrent values in the scored window (Chronos-2 via `future_df`, ARIMAX as forecast exogenous inputs).
- **Chronos-2** gets significantly worse (−0.235 AUROC). During stress the covariates lie far outside the context's range (EDA and skin temperature 100% outside, EDA shifted by a median 42 context SDs). Chronos-2 handles unfamiliar covariates by widening its predictive interval, which shrinks the standardised residual exactly in the stress windows. With RMSSD held flat, its AUROC is 0.065: the covariates alone rank stress windows as *less* anomalous than baseline windows.
- **ARIMAX** reaches a median of 1.000, but the HRV-dependence control leaves it at 1.000. Its linear fit on near-constant covariates is barely identified, and extrapolating it to stress-level EDA and HR yields forecasts unrelated to RMSSD. It has become an accidental multimodal stress detector, which also explains why it alone separates stress from amusement and meditation (0.968). This says nothing about whether covariates help HRV anomaly detection.

**Conclusion on covariates.** WESAD cannot test the thesis's covariate claim. A model can only learn to "explain away" RMSSD changes if its context shows RMSSD responding to non-stress covariate changes, and WESAD's seated reference contains none. What WESAD does show is the failure modes of naive covariate use: Chronos-2 becomes uncertain, and ARIMAX extrapolates.

### Information parity
Every model gets the same context. Both covariate models get the same concurrent covariates. No model receives future target values. An earlier version gave Chronos-2 covariate history only, and its ARIMAX score ignored the covariates entirely. The original "no covariate effect" result was therefore an artefact; see §7.

---

## 6. Statistics

- **Unit of inference:** the subject (n = 15). Windows within a subject overlap through the 60-s RMSSD and are not independent.
- **Comparisons:** a paired Wilcoxon signed-rank test on per-subject AUROC. Zero differences are dropped, as the test requires; these are mostly subjects where both methods score 1.000, and the remaining count is reported.
- **Families and multiplicity:** two families with a Holm correction within each.
  - Primary: Chronos-2 vs SMA, Chronos-2 vs ARIMA.
  - Secondary: the covariate effect for each model family.
- **Effect sizes:** mean paired difference with a 95% bootstrap CI (subjects resampled), plus the matched-pairs rank-biserial r.
- **Decision rule:** "better" needs p_Holm < 0.05. A CI inside ±0.05 AUROC (the conventional minimal important difference) means "equivalent". A one-sided bound inside the margin means "not meaningfully better". A non-significant result is never read as "no difference" unless the CI supports it.

---

## 7. Decision and defect log

| # | Step | Consequence |
|---|---|---|
| 1 | RMSSD target, 60-s window | Physiologically clean separation for typical subjects |
| 2 | Malik filtering; rejected beats initially *deleted* | Later found to create differences across gaps; changed to NaN marking (S7 stress RMSSD 72 → 64 ms) |
| 3 | Rolling-context evaluation | AUROC ≈ 0.47, explained by the sustained-state labels (§3.1) |
| 4 | Personal-baseline reference | Now the primary design |
| 5 | Band exceedance → one-sided → two-sided score | Final two-sided score (§4) |
| 6 | ARIMAX long-range `get_forecast` unstable; replaced by the in-sample mean | The replacement silently made ARIMAX ignore its covariates |
| 7 | Code review (2026-11-09) found: covariates never reached the scored window; `id()`-keyed ARIMAX cache; respiration filter cutoff 2.45 Hz instead of 0.5 Hz; "Cliff's δ" was a sign statistic; no guard between context and held-out baseline; SMA used 60 s of context vs 300 s for the other models; windows could straddle gaps; no multiplicity correction or CIs | All fixed, with a test each in `tests/test_pipeline.py`. The respiration fix had a negligible effect (~0.3 breaths/min). The covariate fix reversed the covariate conclusion |
| 8 | Controls added: time distance, HRV dependence, covariate range | Turned "ARIMAX + covariates = 1.000" from an apparent win into a diagnosed artefact |

---

## 8. What Phase 1 establishes

1. The pipeline is sound: artifact rates are under 3%, 10 of 15 subjects show the expected RMSSD drop, and every defect found in review is fixed and tested.
2. Against a personal resting reference, stress produces detectable RMSSD deviations (median AUROC ≈ 0.96) for any univariate method. The detection is of deviation from rest, not specifically of stress.
3. Zero-shot Chronos-2 is a viable scorer but brings no advantage over SMA or ARIMA on this task.
4. The covariate claim cannot be tested on WESAD. Naive covariate use hurts Chronos-2, and its apparent benefit for ARIMAX comes from the covariates, not from HRV.

**For Phase 2 (MMASH):** the covariate claim has to be tested where the reference context contains non-stress covariate variation (activity, sleep, meals). Sensitivity must be reported alongside the false-positive rate, because "suppressed" alarms may simply be inflated uncertainty. Stress markers (HR, EDA) should not be used as covariates for an HRV-residual detector.

**What Phase 2 then found** (`results/phase2_results.md`): with a 30-min free-living context, covariates were neutral for Chronos-2 rather than harmful, which is consistent with the out-of-range diagnosis above. However, they did not help either:
- Detection of an unexplained RMSSD drop was weak for every model (AUROC 0.61–0.66 at a 30% drop).
- Activity produced almost no false alarms to suppress in the first place.
- Chronos-2 again showed no practical advantage over the simple baselines.
