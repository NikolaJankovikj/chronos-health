# Phase 1 Results — WESAD: Stress Detection from HRV Forecast Residuals

**Dataset:** WESAD, 15 subjects (S2–S17; S1 and S12 were excluded by the dataset authors for sensor malfunction)
**Final run:** 2026-11-09 — supersedes every earlier Phase-1 number (see §6)
**Reproduce** (needs the raw WESAD release in `WESAD/`, gitignored): `uv run python scripts/01_preprocess_wesad.py && uv run python scripts/03_run_phase1.py && uv run python scripts/04_plot_phase1.py`
**Tables:** `results/phase1_{per_subject,summary,comparisons,covariate_ood,hrv_dependence}.csv` · **Figures:** `results/figures/`

---

## 0. Answer in one paragraph

Given five minutes of a person's resting baseline, all three univariate detectors (SMA, ARIMA, zero-shot Chronos-2) flag TSST stress windows with a median per-subject AUROC of about 0.96. **Chronos-2 is not better than either baseline.** Against SMA the 95% CI of the paired difference rules out any Chronos-2 advantage larger than +0.010 AUROC. Against ARIMA it is worse on every non-tied subject (p_Holm = 0.031), but by only 0.004 AUROC, which is negligible. **Adding covariates makes Chronos-2 clearly worse** (−0.235 AUROC, 95% CI −0.363 to −0.130, p_Holm < 0.001). During stress the covariates sit far outside anything the rest context shows, and Chronos-2 reacts by widening its forecast interval. ARIMAX with covariates scores a median AUROC of 1.000, but a control shows this comes entirely from the covariates rather than from HRV. On WESAD the covariate question cannot be answered: the seated rest context has no non-stress covariate variation to learn from.

---

## 1. Protocol (fixed before the final run)

| Element | Choice | Why |
|---|---|---|
| Target | Rolling RMSSD, 60 s window, 1 Hz, from Malik-filtered chest-ECG RR intervals (rejected intervals are NaN, never differenced across) | Parasympathetic marker, stable on 60 s |
| Reference context | Last 300 s of the first 2/3 of the subject's baseline (label 1) | Personal resting reference; the rest of the baseline is held out |
| Guard | 60 s between the context end and the first held-out baseline window | Equals the RMSSD window, so no beat is shared with the context |
| Windows | 30 s, non-overlapping, tiled inside each time-contiguous label segment | One label per window, no window spans a gap |
| Primary task | Stress (label 2) vs held-out baseline (label 1) | The thesis question |
| Control task | Stress vs amusement + meditation (labels 3, 4) | Non-stress windows that, like stress, lie far in time from the context |
| Score | Mean over the window of \|observed − point forecast\| / σ̂ (two-sided) | Same formula for every model; two-sided because 5 subjects' RMSSD rises under stress |
| σ̂ | SMA: context std · ARIMA(X): forecast s.e. · Chronos-2: (q90 − q10)/2.563 | Puts every score in σ units |
| Metric | Per-subject AUROC (AUPRC also reported) | Threshold-free, per person |
| Inference | Paired Wilcoxon signed-rank on per-subject AUROC; Holm correction within each family; mean paired difference with 95% subject-bootstrap CI (10,000 resamples); matched-pairs rank-biserial r | n = 15 subjects is the unit of inference, not windows |
| Decision rule | Better = p_Holm < 0.05. A 95% CI inside ±0.05 AUROC = equivalent; a one-sided bound inside the margin = "not meaningfully better" | 0.05 is the conventional minimal important AUROC difference |

**Models.** All receive the identical context.

| Family | Model | Inputs |
|---|---|---|
| Primary (univariate) | SMA: mean and std of the context | Context RMSSD |
| | ARIMA(2,0,2), fitted per subject on the context, stationarity enforced | Context RMSSD |
| | Chronos-2-small (`autogluon/chronos-2-small`, 28M), zero-shot, CPU | Context RMSSD |
| Secondary (covariates) | ARIMAX(2,0,2) | + HR, ACC magnitude, EDA, respiration rate, skin temp: history **and** the scored window's concurrent values |
| | Chronos-2-small | Same covariates (`df` history + `future_df`) |

Forecasting models score each window as if it immediately followed the context. That is the right question for a personal-reference design: "what would this person's resting RMSSD look like?" Stress begins 17–44 min (median 37) after the context ends. Seven subjects ran the stress-first protocol order (S2, S3, S6, S9, S11, S14, S16) and eight ran amusement-first (S4, S5, S7, S8, S10, S13, S15, S17).

**Windows per subject:** 10–11 held-out baseline, 20–24 stress, 36–38 amusement + meditation.

---

## 2. Data checks

- **Artifact rate** (share of RR intervals rejected): 0.0–3.0% per subject (S2 3.0%, S3 2.3%, S7 1.5%, S17 1.1%, all others < 1%). Every subject is under the 5% threshold of task T1.6.
- **Central biomarker assumption (T1.7):** mean stress RMSSD is below mean baseline RMSSD in 10 of 15 subjects. The 5 atypical subjects are S6, S7, S9, S10 and S15. All subjects are retained.

---

## 3. Results

### 3.1 Summary (median over 15 subjects)

| Method | AUROC, stress vs baseline [IQR] | AUPRC | AUROC, stress vs amusement + meditation |
|---|---|---|---|
| SMA | 0.955 [0.859, 1.000] | 0.981 | 0.785 |
| ARIMA | 0.968 [0.883, 1.000] | 0.987 | 0.704 |
| **Chronos-2** | **0.959** [0.878, 1.000] | 0.983 | 0.688 |
| ARIMAX + covariates | 1.000 [0.947, 1.000] | 1.000 | 0.968 |
| Chronos-2 + covariates | 0.660 [0.357, 0.895] | 0.811 | 0.560 |

All 15 subjects produced valid scores for every method.

### 3.2 Per-subject AUROC (stress vs held-out baseline)

| Subject | SMA | ARIMA | Chronos-2 | ARIMAX + cov | Chronos-2 + cov |
|---|---|---|---|---|---|
| S2 | 0.955 | 1.000 | 1.000 | 0.640 | 0.660 |
| S3 | 0.952 | 0.933 | 0.929 | 0.957 | 0.524 |
| S4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.876 |
| S5 | 1.000 | 0.944 | 0.926 | 1.000 | 0.835 |
| S6 * | 0.645 | 0.619 | 0.606 | 0.727 | 0.364 |
| S7 * | 0.887 | 0.926 | 0.922 | 1.000 | 0.649 |
| S8 | 0.923 | 0.968 | 0.959 | 0.936 | 0.877 |
| S9 * | 0.602 | 0.602 | 0.597 | 1.000 | 0.351 |
| S10 * | 0.682 | 0.367 | 0.367 | 1.000 | 0.167 |
| S11 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| S13 | 0.831 | 0.839 | 0.835 | 0.864 | 0.302 |
| S14 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| S15 * | 0.977 | 0.982 | 0.982 | 0.995 | 0.914 |
| S16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| S17 | 1.000 | 1.000 | 1.000 | 1.000 | 0.117 |

\* Atypical subject (stress RMSSD ≥ baseline RMSSD). Medians for typical / atypical subjects: SMA 1.000 / 0.682, ARIMA 1.000 / 0.619, Chronos-2 1.000 / 0.606, ARIMAX + cov 1.000 / 1.000, Chronos-2 + cov 0.856 / 0.364 (fig3).

### 3.3 Planned comparisons (A − B in per-subject AUROC; fig4)

| Family | A − B | Mean Δ [95% CI] | p (Holm) | r | Non-tied n | Verdict |
|---|---|---|---|---|---|---|
| Primary | Chronos-2 − SMA | −0.022 [−0.070, +0.010] | 0.695 | −0.16 | 10 | Chronos-2 not meaningfully better |
| Primary | Chronos-2 − ARIMA | −0.004 [−0.007, −0.001] | 0.031 | −1.00 | 7 | ARIMA better, negligible |
| Secondary | Chronos-2 + cov − Chronos-2 | −0.235 [−0.363, −0.130] | < 0.001 | −1.00 | 13 | Covariates hurt |
| Secondary | ARIMAX + cov − ARIMA | +0.063 [−0.037, +0.177] | 0.131 | +0.56 | 10 | ARIMA not meaningfully better (see §4.3: the gain is not HRV) |

"Non-tied n" counts the subjects with a non-zero difference. Wilcoxon drops ties, which here are mostly subjects where both methods score 1.000.

### 3.4 Covariate controls

**How far the covariates move during stress, relative to the rest context** (median over subjects):

| Covariate | Stress values outside the context's [min, max] | Mean shift, in context SDs |
|---|---|---|
| EDA | 100% | 42.3 |
| Skin temperature | 100% | 6.7 |
| Respiration rate | 46% | 1.4 |
| Heart rate | 32% | 3.0 |
| ACC magnitude | 13% | 4.6 |

**Does the detector depend on HRV?** Each window's observed RMSSD is replaced by the context mean and the AUROC is recomputed. A detector that still separates stress is using only the covariates.

| Model | Median AUROC, RMSSD held flat |
|---|---|
| ARIMAX + covariates | 1.000 (unchanged: the detection comes entirely from the covariates) |
| Chronos-2 + covariates | 0.065 (inverted: the covariates alone push stress-window scores *below* baseline windows) |

### 3.5 Success criteria (PLAN.md §5.5)

| Criterion | Result |
|---|---|
| 1. Chronos-2 median AUROC > 0.70 | **PASS** (0.959) |
| 2. Chronos-2 beats SMA | **FAIL**: not meaningfully better (upper CI +0.010) |
| 3. Chronos-2 beats ARIMA | **FAIL**: ARIMA marginally better |
| 4. Covariates help Chronos-2 | **FAIL**: covariates significantly hurt |

PLAN.md §5.5 also allowed "or effect size > small" as a pass. That lenient reading gives the same outcome, because every effect points against Chronos-2 (r = −0.16, −1.00, −1.00).

---

## 4. Interpretation

### 4.1 Univariate: the foundation model adds nothing here
With a fixed resting reference, each univariate model's prediction for a window is essentially an estimate of that person's resting RMSSD level and spread. Five minutes of rest pins that down well for all three, so they end up equivalent in practice: a 28M-parameter zero-shot model matches the context mean and std. For the 10 typical subjects the stress drop is large and every method reaches a median of 1.000. The 5 atypical subjects set the ceiling for all methods equally (medians 0.61–0.68).

### 4.2 The detector is not stress-specific
Against amusement and meditation, which are also far in time from the context, the univariate medians fall to 0.69–0.79. The two-sided score flags any departure from rest. Stress departs further than amusement or meditation, so the score is better than chance, but it is a "deviation from personal rest" detector, not a stress detector. Because the control windows are as far from the context as the stress windows are, they also show that the primary AUROC is not just an artefact of time elapsed since the context.

### 4.3 Covariates cannot be evaluated on WESAD
A covariate model can only "explain" an RMSSD change if its context shows how RMSSD moves with that covariate. In WESAD the context is five minutes of seated rest, so the covariates barely vary, and during the TSST every covariate moves because of the stressor itself (EDA, HR, temperature) or the speaking task (respiration, movement), all far outside the context's range (§3.4). The two models fail in opposite ways:

- **Chronos-2** responds to covariates it has not seen by widening its predictive interval. That shrinks the standardized residual exactly in the stress windows (flat-RMSSD AUROC 0.065), so detection gets worse.
- **ARIMAX** extrapolates a near-unidentifiable linear fit (covariates that are nearly constant in the context) and produces forecasts unrelated to RMSSD. Its detection survives with RMSSD held flat, so it is acting as an accidental multimodal stress detector driven by EDA and HR. That also explains its high stress-vs-non-stress AUROC (0.968). It is not evidence that covariates help HRV anomaly detection.

The defensible conclusion is that WESAD cannot test the thesis's covariate claim, and that naive covariate use is harmful (Chronos-2) or misleading (ARIMAX). Testing the claim requires data whose reference context contains non-stress covariate variation, which is the purpose of Phase 2.

### 4.4 What this means for Phase 2
- Chronos-2's response to unfamiliar covariates is to become uncertain. In free-living data an exercise bout outside the context's range would therefore push scores down. That looks like "false-positive suppression", but it suppresses true positives just as much. Phase 2 must report sensitivity alongside the false-positive rate, and its context must contain activity variation (a longer context).
- HR and EDA should not be covariates for an HRV-residual detector: they carry the stress signal themselves (§3.4 and §4.3).

*Checked in Phase 2 (`results/phase2_results.md`):* with a 30-min free-living context, only 2–3% of covariate values fell outside the context's range, and covariates were neutral for Chronos-2. Both sensitivity and false alarms were reported as recommended.

---

## 5. Limitations

1. **n = 15.** Ceiling effects leave 5–8 tied subjects per primary comparison, which limits power. The CIs bound the possible effects regardless.
2. **Windows within a subject are dependent** (overlapping 60-s RMSSD). Inference is therefore done at the subject level only.
3. **A 300-s reference context** (from PLAN.md) was not tuned. Longer contexts were not tested.
4. **A posture change during the TSST** (standing) would lower RMSSD independently of psychological stress. ACC magnitude cannot capture posture (gravity has the same norm in every orientation). This was not verified.
5. **Only `chronos-2-small` was run.** The 120M model was not evaluated.
6. **The atypical response in 5 subjects is unexplained.**

---

## 6. Changes from earlier Phase-1 write-ups

Earlier versions of this document reported SMA 0.979, ARIMAX 0.981, Chronos-2 0.960 with and without covariates, and "no significant differences". Those numbers came from code with the following defects, all fixed (`tests/test_pipeline.py` covers each):

| Defect | Effect on earlier conclusions |
|---|---|
| Covariates never reached the scored window (Chronos-2 got them as history only; the ARIMAX score ignored them) | "Covariates make no difference" was true by construction |
| RMSSD differenced across rejected beats | Shifted stress RMSSD in some subjects (S7: 72 → 64 ms), moving several AUROCs |
| ARIMAX fit cache keyed on `id()` could reuse another subject's fit | Possible cross-subject contamination |
| Respiration band-pass upper cutoff 2.45 Hz instead of 0.5 Hz | Negligible on WESAD (≈0.3 breaths/min) |
| "Cliff's δ" was a sign statistic; no multiplicity correction; no CIs | Effect sizes mislabelled; "no difference" claimed from non-significance |
| No guard between context and held-out baseline | First held-out windows shared beats with the context |
| SMA used the last 60 s of context while the other models used 300 s | Unequal information |
| Windows could straddle label gaps | Mixed-condition windows possible |
| Protocol order assumed fixed ("Baseline → Stress → …") | Wrong for 8 of 15 subjects (amusement-first) |
