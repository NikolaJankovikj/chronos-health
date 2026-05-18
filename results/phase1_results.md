# Phase 1 Results — WESAD: Psychological Stress Detection via HRV Anomaly Scoring

**Dataset:** WESAD (Wearable Stress and Affect Detection)  
**Subjects:** 15 (S2–S17, excluding S1 and S12 — sensor malfunction)  
**Date completed:** 2026-05-04

---

## 1. Experimental Setup

### 1.1 Dataset Overview

WESAD is a controlled lab study. Each subject wore a chest device (RespiBAN, 700 Hz) and a wrist device (Empatica E4). The session included four conditions:

| Condition | Label | Mean duration |
|---|---|---|
| Baseline (sitting, resting) | 1 | ~19 min |
| Stress (TSST — public speaking + mental arithmetic) | 2 | ~10 min |
| Amusement (funny video clips) | 3 | ~6 min |
| Meditation | 4 | ~13 min (split) |

Label 0 = transient (excluded from evaluation).

The protocol order is always: Baseline → Stress → Meditation 1 → Amusement → Meditation 2, with ~13-minute unlabelled transition gaps between conditions.

### 1.2 Target Signal: RMSSD

Rolling RMSSD (root mean square of successive RR-interval differences) was computed at 1 Hz using a 60-second sliding window over Malik-filtered RR intervals derived from NeuroKit2 R-peak detection on the chest ECG (700 Hz).

- RMSSD is a parasympathetic HRV metric. It typically **decreases** under psychological stress due to autonomic nervous system activation.
- Of the 15 subjects, **10 show the expected pattern** (baseline RMSSD > stress RMSSD); **5 subjects show an atypical pattern** (stress RMSSD ≥ baseline RMSSD). This is consistent with the published WESAD results and reflects genuine inter-individual variability in HRV stress response.

### 1.3 Covariate Signals

All covariates computed at 1 Hz from the chest RespiBAN:

| Covariate | Source | Processing |
|---|---|---|
| `hr_bpm` | RR intervals | Mean HR per second |
| `acc_mag` | 3-axis accelerometer | L2 norm, mean-pooled |
| `eda` | Electrodermal activity | Mean-pooled |
| `resp_rate` | Respiration belt | Bandpass (0.1–0.5 Hz) + peak detection @ 10 Hz, 60s rolling |
| `temp` | Skin temperature | Mean-pooled |

### 1.4 Evaluation Design — Personal Baseline Context

**Key design decision:** a naive rolling-window evaluation was initially tried but produced AUROC ≈ 0.47 (near chance) for all methods. Investigation revealed why: the WESAD protocol has a 13-minute unlabelled gap between the baseline and stress conditions. By the time any stress window is evaluated, a rolling context window (≤5 min) falls entirely within the stress period — the model's recent history already includes depressed RMSSD, so no anomaly is detected.

The correct evaluation for WESAD uses a **personal-baseline context**:

- **Context:** last 300 seconds (5 min) drawn from the first 2/3 of each subject's baseline period (pure label=1, no data leakage).
- **Evaluation windows:** drawn from (a) held-out late baseline (last 1/3 of baseline) and (b) the full stress period. Each window is 30 seconds, stride 30 seconds (non-overlapping).
- **Question answered:** *"Given 5 minutes of this person's resting baseline, is this 30-second window anomalous?"*

This design is clinically meaningful: a deployed wearable system would use an initial rest period to establish the user's personal HRV baseline before monitoring for deviations.

**Window counts per subject:** 32–37 total windows (12–13 baseline + 20–21 stress per subject).

### 1.5 Anomaly Score

All four methods output a scalar anomaly score per 30-second window using a **two-sided absolute normalised deviation** from the context baseline distribution:

```
score = mean( |expected - observed| / scale )
```

Where `expected` and `scale` are model-specific:

| Method | Expected | Scale |
|---|---|---|
| SMA | Rolling mean of context RMSSD (last 60 s) | Rolling std |
| ARIMAX | Mean of SARIMAX(2,0,2) fitted values over context | Residual std √scale |
| Chronos2_noCV | Median forecast (q0.5) — no covariates | IQR (q0.9 − q0.1) |
| Chronos2_CV | Median forecast (q0.5) — with all 5 covariates | IQR (q0.9 − q0.1) |

Two-sided absolute deviation (rather than one-sided) is used because ~1/3 of subjects show atypical upward stress responses; one-sided scoring would produce inverted (< 0.5) AUROC for those subjects.

### 1.6 Models

| Model | Type | Training required | Covariates |
|---|---|---|---|
| SMA | Simple Moving Average | None (analytic) | No |
| ARIMAX | SARIMAX(2,0,2) | Per-subject fit on context window | Yes |
| Chronos2_noCV | `autogluon/chronos-2-small` (28M params) | None — zero-shot | No |
| Chronos2_CV | `autogluon/chronos-2-small` (28M params) | None — zero-shot | Yes (past-only) |

**Note on ARIMAX implementation:** The initial ARIMAX design called `get_forecast(steps=H, exog=exog_test)` where `exog_test` contained the actual covariate values from the evaluation window — giving it access to future sensor readings that Chronos-2 does not have. This would have been a genuine information asymmetry. However, long-range SARIMAX extrapolation proved unstable (forecasts diverge when the evaluation window is 13+ minutes from the context end, as in the WESAD baseline→stress gap), so the implementation was changed. ARIMAX now uses only the baseline mean and residual standard deviation from the fitted model (`mu = fittedvalues.mean()`, `sigma = sqrt(model.scale)`), with no access to future covariate values. Chronos-2 similarly uses only past covariates from the context window. Both methods have equivalent information access in the final implementation.

**Evaluation protocol:** Leave-one-subject-out (LOSO). Chronos-2 is inherently zero-shot — no subject data is ever used for training. ARIMAX is fitted independently per subject on their own context window and evaluated on held-out windows from the same subject.

---

## 2. Results

### 2.1 Overall AUROC and AUPRC (all 15 subjects)

| Method | Median AUROC | IQR | Median AUPRC |
|---|---|---|---|
| **SMA** | **0.979** | [0.833, 1.000] | **0.988** |
| **ARIMAX** | **0.981** | [0.862, 1.000] | **0.989** |
| Chronos2_noCV | 0.960 | [0.829, 1.000] | 0.980 |
| Chronos2_CV | 0.960 | [0.833, 1.000] | 0.980 |

### 2.2 Per-Subject AUROC

| Subject | SMA | ARIMAX | Chronos2_noCV | Chronos2_CV |
|---|---|---|---|---|
| S2 | 0.979 | 0.938 | **1.000** | **1.000** |
| S3 | 0.893 | **0.984** | 0.921 | 0.921 |
| S4 | **1.000** | **1.000** | **1.000** | **1.000** |
| S5 | 0.916 | **1.000** | 0.857 | 0.872 |
| S6 * | 0.707 | 0.619 | 0.593 | 0.593 |
| S7 * | **0.982** | 0.971 | 0.960 | 0.960 |
| S8 | 0.883 | 0.917 | 0.958 | **0.958** |
| S9 * | 0.322 | 0.637 | **0.667** | 0.659 |
| S10 * | 0.513 | **0.808** | 0.487 | 0.474 |
| S11 | **1.000** | **1.000** | **1.000** | **1.000** |
| S13 | 0.783 | 0.776 | **0.801** | 0.794 |
| S14 | **1.000** | **1.000** | **1.000** | **1.000** |
| S15 * | 0.981 | 0.981 | **0.985** | **0.985** |
| S16 | **1.000** | **1.000** | **1.000** | **1.000** |
| S17 | **1.000** | **1.000** | **1.000** | **1.000** |

\* Atypical subjects (stress RMSSD ≥ baseline RMSSD). Bold = highest AUROC in row.

### 2.3 Results Stratified by Subject Type

**Typical subjects (n=10, baseline RMSSD > stress RMSSD):**

| Method | Median AUROC | IQR |
|---|---|---|
| SMA | 0.990 | [0.899, 1.000] |
| ARIMAX | 1.000 | [0.949, 1.000] |
| Chronos2_noCV | 1.000 | [0.930, 1.000] |
| Chronos2_CV | 1.000 | [0.930, 1.000] |

**Atypical subjects (n=5, stress RMSSD ≥ baseline RMSSD):**

| Method | Median AUROC | IQR |
|---|---|---|
| SMA | 0.707 | [0.513, 0.981] |
| **ARIMAX** | **0.808** | [0.637, 0.971] |
| Chronos2_noCV | 0.667 | [0.593, 0.960] |
| Chronos2_CV | 0.659 | [0.593, 0.960] |

### 2.4 Statistical Comparisons (Paired Wilcoxon Signed-Rank, n=15)

| Comparison | Δ Median AUROC | p-value | Cliff's δ |
|---|---|---|---|
| Chronos2_CV vs SMA | −0.019 | 0.878 | +0.13 |
| Chronos2_CV vs ARIMAX | −0.021 | 0.508 | 0.00 |
| Chronos2_CV vs Chronos2_noCV | 0.000 | 0.500 | −0.20 |
| Chronos2_noCV vs SMA | −0.019 | 0.799 | +0.13 |

No comparison reaches statistical significance. All p-values are well above α = 0.05.

### 2.5 Phase-1 Success Criteria

| Criterion | Result |
|---|---|
| Chronos2_CV AUROC > 0.70 (median) | ✅ PASS (0.960) |
| Chronos2_CV > SMA (median) | ❌ FAIL (0.960 < 0.979) |
| Chronos2_CV > ARIMAX (median) | ❌ FAIL (0.960 < 0.981) |
| Covariates improve performance (CV > noCV) | ❌ FAIL (no difference) |

---

## 3. Interpretation

### 3.1 Main finding

All four methods achieve high AUROC (~0.96–0.98 median) when evaluated against each subject's personal resting baseline. No method is statistically significantly better than any other (all p > 0.05, N = 15). The simplest baseline (SMA) performs on par with the 28M-parameter foundation model.

This is a **ceiling effect**, not a sign that the methods are equally powerful in general. It reflects three compounding factors:

1. **Strong personal-baseline signal.** For the 10 typical subjects, mean baseline RMSSD (~53 ms) is far above mean stress RMSSD (~28 ms) — a separation of nearly 4 standard deviations. Even a single-number comparison (SMA) resolves this trivially.

2. **Atypical subjects dominate variance.** The IQR is wide ([0.83, 1.00]) because 5 subjects show an inverted stress response. None of the four methods resolve these subjects well; ARIMAX has a slight edge on them (median 0.808 vs ~0.67 for others), likely because the fitted noise model captures covariate structure even when raw RMSSD direction is reversed.

3. **WESAD has no physical activity variation.** All conditions are sedentary. Activity covariates (ACC, RESP rate) carry essentially no information for discriminating stress from baseline. This explains why Chronos2_CV = Chronos2_noCV — the key covariate advantage of Chronos-2 over its predecessors simply has no signal to work with here.

### 3.2 Why naive rolling-window evaluation fails on WESAD

The ~13-minute gap between the baseline and stress conditions, combined with a rolling context window, means that by the time stress windows are evaluated the model's entire context is within the stress period. The "recent normal" is already depressed RMSSD. The model predicts low RMSSD, observes low RMSSD, and generates no anomaly signal. This produces AUROC ≈ 0.47 — indistinguishable from random. The personal-baseline context design eliminates this artefact and is the appropriate evaluation for this dataset.

### 3.3 Implication for thesis framing

The Phase-1 finding establishes a clear motivation for Phase 2:

> *In a controlled lab setting with a known clean baseline, any method from a 3-line SMA to a 28M-parameter foundation model can detect psychological stress from HRV with high accuracy. The challenge that distinguishes Chronos-2's covariate capability from simpler methods is not lab-based psychological stress detection — it is the free-living scenario where physical activity continuously confounds HRV interpretation. Phase 2 (MMASH) tests whether context-aware foundation models suppress false anomaly flags during exercise, sleep, and eating, while preserving sensitivity to unexplained HRV deviations.*

---

## 4. Limitations

1. **N = 15 subjects** — underpowered for detecting small effect sizes between methods. With N=15 and paired tests, the minimum detectable Cliff's δ at power=0.8 is approximately 0.5.
2. **Single lab session per subject** — no longitudinal stability assessment.
3. **WESAD stress is exclusively psychological (TSST)** — no physical stress condition, so Phase-1 cannot validate the physical-vs-psychological discrimination claim.
4. **5 atypical subjects** — the reason stress RMSSD ≥ baseline RMSSD is unknown (possible pre-existing stress during baseline, ECG quality, individual autonomic reactivity profiles). These subjects are included in all analyses but flagged as a limitation.
5. **ARIMAX's marginal lead on atypical subjects** — After the long-range extrapolation fix, ARIMAX no longer uses future covariate values. Its edge on atypical subjects (median AUROC 0.808 vs ~0.67 for Chronos-2) is more plausibly attributed to a better-calibrated noise estimate: SARIMAX fits the autocorrelation structure of the full 300-second baseline and produces a tighter MLE residual standard deviation (`sqrt(model.scale)`) than SMA's rolling empirical std over the last 60 seconds. For atypical subjects where the anomaly is an *upward* RMSSD deviation, a tighter sigma amplifies the normalised score and improves discrimination. This remains a post-hoc explanation at N=5 and should not be over-interpreted.

---

## 5. Artefacts and Decisions Log

| Issue | Resolution |
|---|---|
| First pass used rolling context → AUROC ≈ 0.47 (near chance) | Replaced with personal-baseline context |
| One-sided anomaly score → AUROC < 0.5 for atypical subjects | Switched to two-sided absolute deviation |
| ARIMAX long-range `get_forecast()` produced unstable extrapolations (e.g., S2 ARIMAX = 0.083) | Replaced with mean/std of fitted values over context window |
| ARIMAX `conf_int()` returned `numpy.ndarray` not `DataFrame` in installed statsmodels version | Fixed with `np.asarray()` indexing |
| 5 subjects show stress RMSSD > baseline RMSSD (S6, S7, S9, S10, S15) | Expected inter-individual variability; documented in results, all subjects retained |
| RMSSD warmup NaN (first 60 s of session) | Forward-filled covariates; RMSSD NaN propagates naturally to context quality check |

---

## 6. Files

| File | Description |
|---|---|
| `data/wesad_processed/S*.parquet` | Per-subject 1 Hz DataFrame (rmssd, hr_bpm, acc_mag, eda, resp_rate, temp, label) |
| `data/phase1_results.csv` | Per-subject per-method AUROC, AUPRC, n_windows |
| `scripts/01_preprocess_wesad.py` | Preprocessing pipeline (T1.4–T1.8) |
| `scripts/03_run_phase1.py` | LOSO evaluation (T1.9–T1.13) |
| `src/thesis/eval/harness.py` | Personal-baseline context evaluation harness |
| `src/thesis/models/{sma,arimax,chronos2}.py` | Scorer implementations |
