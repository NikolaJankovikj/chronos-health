# Phase 2 Results — MMASH: Context-Aware Anomaly Scoring in Free-Living Data

**Dataset:** MMASH v1.0.0 (PhysioNet, ODbL), 22 healthy adults, ~24 h each: Polar H7 RR intervals and an ActiGraph wGT3X-BT, plus sleep, an activity log, questionnaires and saliva.
**Analysed:** 20 subjects (exclusions in §1)
**Final run:** 2026-11-09
**Reproduce** (needs `DataPaper/` from `MMASH.zip`, gitignored): `uv run python scripts/02_preprocess_mmash.py && uv run python scripts/04_run_phase2.py && uv run python scripts/05_plot_phase2.py`
**Tables:** `results/phase2_{per_subject,summary,comparisons,depth_sweep,exploratory,window_counts}.csv` · **Figures:** `results/figures/fig6–fig9`

---

## 0. Answer in one paragraph

The Phase 2 claim is that in free-living data, covariates (movement, posture, sleep, logged exercise, eating, stimulants) let Chronos-2 avoid false alarms during activity while still flagging RMSSD deviations the covariates do not explain. **The data do not support it.**
- **Detection:** an unexplained 30% RMSSD drop is detected with a median per-subject AUROC of only 0.61–0.66 by every model, from the context mean (SMA) to Chronos-2 with covariates.
- **Covariates do not help Chronos-2:** the paired difference is −0.006, 95% CI −0.013 to +0.002 (equivalent).
- **Chronos-2 vs the classical model:** Chronos-2 with covariates beats persistence with covariates by 0.040 AUROC (p_Holm < 0.001), but the whole CI lies inside the ±0.05 margin, and it does not beat SMA.
- **There is little activity false alarm to suppress:** active windows score barely higher than sedentary ones even without covariates (median AUROC 0.50–0.55), and adding covariates changes this in no model family.
- **Exploratory correlations:** none with daily stress or cortisol change survives the 12 tests run.

---

## 1. Data and preprocessing

| Step | Choice |
|---|---|
| Activity codes (T2.1) | Confirmed from the PhysioNet documentation: 1 sleeping, 2 laying down, 3 sitting, 4 light movement, 5 medium, 6 heavy, 7 eating, 8 small screen, 9 large screen, 10 caffeine, 11 smoking, 12 alcohol. Code 0 is undocumented and ignored. |
| Day encoding | Users 8 and 9 encode day 2 as `-29` (a month-rollover bug); mapped to 2 |
| RR gaps | Every user has 2,200–2,700 recording gaps per day, mostly 5–10 s. A NaN is inserted before each beat that follows a gap, so no successive difference spans a gap. |
| Artifacts | The same Malik-style filter as Phase 1 (300–2000 ms; >20% from the 5-beat median → NaN) |
| RMSSD | 60-s trailing window at 1 Hz; valid only with ≥ 20 successive differences. 82–100% of seconds are valid per eligible user. |
| Covariates (1 Hz) | Movement: 60-s trailing vector magnitude and steps/min. Posture: fraction of the trailing minute lying and standing. In bed (`sleep.csv`). Logged exercise (codes 5–6), eating (7), stimulants (10–12). |
| Not a covariate | HR. It is derived from the same heartbeats and carries the stress response itself (Phase 1 §4.3–4.4). |
| Exclusions (fixed before scoring) | user_4: 18.6% RR artifacts (> 10%), RMSSD implausible during activity. user_11: no sleep record, so the in-bed covariate would be wrong. |

---

## 2. Protocol

**The problem.** MMASH has no stress labels. Sensitivity is therefore measured semi-synthetically with a paired design, and false alarms are measured on the real data.

| Element | Choice |
|---|---|
| Positions | Every 300 s after the first 30 min; kept if ≥ 50% of the context and ≥ 50% of the scored window have a valid RMSSD. 8–41 active, 125–196 sedentary and 29–102 sleep windows per subject. |
| Context | The 30 min before position s (long enough to contain activity transitions) |
| Forecast | From s to s + 90 s, with the covariates for that span given as known-future values (they are measured at the same time) |
| Scored window | [s + 60, s + 90) s. Its 60-s RMSSD contains no beat from the context (the same guard as Phase 1). |
| Negatives | The real scored windows |
| Positives | The same windows with RMSSD × (1 − d), d = 30%. No scored value contains a pre-s beat, so this is exactly what scaling every successive difference after s by (1 − d) produces: an abrupt vagal withdrawal at s that the covariates do not explain. Context, covariates and forecast are identical, so only the drop differs. |
| Score | Mean over the window of \|observed − point forecast\| / σ̂ (as in Phase 1) |
| Primary metric | Per-subject AUROC, injected vs real windows (all states) |
| Secondary metric | Activity false alarms: per-subject AUROC of real active windows vs real sedentary-awake windows. 0.5 means activity is not flagged. A window is *active* if it averages ≥ 60 steps/min or overlaps logged exercise; *sleep* if more than half of it is in bed. |
| Descriptive | Injection AUROC within each state, and a depth sweep d ∈ {10…50}% |
| Exploratory | Spearman ρ between each model's mean sedentary score and the Daily Stress Inventory, and the cortisol change (wake-up − bedtime). Uncorrected. |
| Inference | As Phase 1: paired Wilcoxon, Holm within each family, subject-bootstrap 95% CIs, ±0.05 AUROC equivalence margin. n = 20 subjects. |

**Models: three families, each with and without covariates.** All see the same context and future covariates.

| Family | Without covariates | With covariates |
|---|---|---|
| Static | SMA: context mean and std | OLS: RMSSD on the covariates over the context (covariates constant in the context are dropped); σ = residual std |
| Dynamic, classical | Persistence: last RMSSD; σ = RMS of the context's k-step changes at the scored leads | Persistence + covariates: last RMSSD + β·(covariate change); β from OLS of k-step RMSSD changes on k-step covariate changes |
| Foundation model | Chronos-2-small, zero-shot | Chronos-2-small + covariates (history + `future_df`) |

**Planned comparisons.**
- Primary (injection AUROC): Chronos-2 + cov vs Chronos-2, and Chronos-2 + cov vs Persistence + cov.
- Secondary (activity false alarms): the covariate effect within each family.

---

## 3. Results

### 3.1 Summary (median over 20 subjects; fig6, fig8)

| Model | Injection AUROC | Sedentary | Active | Sleep | Activity false alarms | Sleep vs sedentary |
|---|---|---|---|---|---|---|
| SMA | 0.660 | 0.656 | 0.631 | 0.677 | 0.546 | 0.488 |
| OLS + cov | 0.645 | 0.645 | 0.588 | 0.689 | 0.533 | 0.480 |
| Persistence | 0.621 | 0.608 | 0.603 | 0.694 | 0.505 | 0.467 |
| Persistence + cov | 0.613 | 0.607 | 0.582 | 0.677 | 0.525 | 0.475 |
| Chronos-2 | 0.645 | 0.636 | 0.585 | 0.751 | 0.528 | 0.493 |
| **Chronos-2 + cov** | **0.654** | 0.632 | 0.611 | 0.719 | 0.511 | 0.506 |

The Sedentary, Active and Sleep columns are the injection AUROC within that state.

### 3.2 Planned comparisons (A − B; fig9)

| Family | A − B | Mean Δ [95% CI] | p (Holm) | r | Reading |
|---|---|---|---|---|---|
| Primary: injection | Chronos-2 + cov − Chronos-2 | −0.006 [−0.013, +0.002] | 0.143 | −0.38 | Equivalent: covariates neither help nor hurt |
| Primary: injection | Chronos-2 + cov − Persistence + cov | +0.040 [+0.032, +0.047] | < 0.001 | +1.00 | Chronos-2 better, but within the margin |
| Secondary: activity alarms | Chronos-2 + cov − Chronos-2 | −0.014 [−0.037, +0.007] | 0.737 | −0.24 | No significant difference |
| Secondary: activity alarms | Persistence + cov − Persistence | +0.023 [+0.005, +0.040] | 0.115 | +0.56 | No significant difference |
| Secondary: activity alarms | OLS + cov − SMA | −0.010 [−0.049, +0.029] | 0.737 | −0.12 | No significant difference |

Unplanned, descriptive only: Chronos-2 + cov − SMA = +0.006 [−0.009, +0.020]; Chronos-2 − Persistence = +0.036 [+0.028, +0.043]; Chronos-2 − SMA = +0.012 [−0.001, +0.025].

### 3.3 Detection vs drop size (median injection AUROC; fig7)

| Drop | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|
| SMA | 0.514 | 0.572 | 0.660 | 0.742 | 0.814 |
| OLS + cov | 0.519 | 0.572 | 0.645 | 0.725 | 0.794 |
| Persistence | 0.512 | 0.560 | 0.621 | 0.689 | 0.749 |
| Persistence + cov | 0.513 | 0.559 | 0.613 | 0.673 | 0.733 |
| Chronos-2 | 0.512 | 0.568 | 0.645 | 0.735 | 0.806 |
| Chronos-2 + cov | 0.515 | 0.573 | 0.654 | 0.731 | 0.799 |

For scale, the TSST lowered RMSSD by about 47% in WESAD's typical subjects (53 → 28 ms).

### 3.4 Exploratory associations (Spearman ρ, uncorrected; n = 20, cortisol n = 19)

| Model | ρ with Daily Stress | ρ with cortisol change |
|---|---|---|
| SMA | −0.09 (p 0.72) | −0.21 (p 0.39) |
| OLS + cov | −0.11 (p 0.64) | 0.00 (p 0.99) |
| Persistence | −0.15 (p 0.52) | −0.10 (p 0.69) |
| Persistence + cov | −0.15 (p 0.52) | −0.07 (p 0.78) |
| Chronos-2 | 0.19 (p 0.42) | 0.46 (p 0.046) |
| Chronos-2 + cov | 0.29 (p 0.21) | 0.32 (p 0.19) |

One of 12 tests is nominally below 0.05, which is about what chance predicts (0.6 expected). It does not survive any multiplicity correction and is not a finding.

---

## 4. Interpretation

1. **Detecting an unexplained HRV drop in free-living data is hard for every model.** At 60–90 s after onset, a 30% vagal withdrawal is separated from ordinary windows with an AUROC of about 0.65. Even a 50% drop, larger than the average TSST response, reaches only about 0.80. Minute-scale RMSSD variability in daily life is comparable in size to stress-sized drops. This limits the residual approach itself, not any particular forecaster.
2. **The foundation model brings no practical advantage.** Chronos-2 beats persistence (0.036–0.040, consistently across subjects), but by less than the equivalence margin. It does not beat the context mean (SMA), which has the highest median. Phase 1 found the same.
3. **Covariates do not help, in any family.** For Chronos-2 the effect is bounded to within ±0.013 AUROC. Unlike Phase 1, covariates also do not *hurt* Chronos-2: the 30-min context contains activity variation, so the covariate values in the forecast span are rarely unfamiliar. Only 2–3% of movement and posture values fall outside the context's range, against 13–100% for the WESAD covariates. That supports the Phase 1 diagnosis that the harm came from covariates outside the context's range.
4. **The confound Phase 2 was designed around is small here.** Even the context-blind models barely flag activity (median AUROC 0.50–0.55 for active vs sedentary), because a rolling 30-min context already adapts to the person's current level and spread. There was little false-alarm burden for covariates to remove.
5. **Thesis claim status.** The Phase 2 hypothesis, that covariates suppress activity false positives while preserving sensitivity, is not supported. What the data do support: zero-shot Chronos-2 is a viable but not superior residual scorer, and naive covariate conditioning is neutral when the context covers the covariate range.

---

## 5. Limitations

1. **The positives are semi-synthetic:** a flat multiplicative drop starting abruptly, with no accompanying change in covariates or HR. Real stress responses ramp, recover, and co-occur with posture and activity changes.
2. **There is no ground-truth stress label.** The daily questionnaire and two cortisol samples per person cannot validate window-level scores.
3. **The activity log is self-reported.** The active stratum also uses a steps threshold (≥ 60 steps/min) chosen a priori.
4. **Settings were not tuned:** 30-min context, 30-s window, 60-s guard and 300-s stride. Longer windows could raise sensitivity at the cost of latency.
5. **Only `chronos-2-small` was run.** ARIMAX (in the original plan) was replaced by the persistence family, which is the classical dynamic baseline that can be fitted per window. Per-window SARIMAX fits over 20 × 270 windows were not computationally feasible on CPU. Phase 1 showed ARIMA ≈ SMA ≈ Chronos-2 anyway.
6. **n = 20.** CIs are reported so that nulls are bounded rather than merely non-significant.

---

## 6. Protocol revision log

The first full run used a scored window immediately after the context (no guard), and only SMA and OLS as classical baselines. It gave Chronos-2 a median injection AUROC of 0.957 against 0.654 for SMA. That gap was an artefact:
- **Shared beats:** the first scored RMSSD values were computed mostly from context beats, so any model tracking the most recent level could anticipate them almost perfectly.
- **An impossible drop:** an instantaneous 30% drop in such values cannot happen at the beat level.
- **No dynamic classical baseline:** neither classical baseline could exploit recency, so the comparison was not like-for-like.

The protocol was revised before interpreting results: add a 60-s guard so the injected drop is exactly beat-consistent, and add the persistence family. All results above come from the revised protocol. The first run's tables are not reported as findings.
