# Thesis Plan: Context-Aware HRV Anomaly Detection with Chronos-2

**Working title:** *Context-Aware HRV Anomaly Detection with Chronos-2: A Two-Phase Evaluation of Zero-Shot Foundation Models for Wearable Stress Monitoring*

## 1. Thesis claim

Two-phase evaluation:

- **Phase 1 (WESAD)** — *Detection validation:* Chronos-2 with physiological covariates produces forecast residuals that detect psychologically-induced HRV anomalies in controlled lab data, outperforming ARIMAX and SMA baselines, zero-shot, leave-one-subject-out.
- **Phase 2 (MMASH)** — *Contextual attribution:* In free-living data, Chronos-2 with activity + sleep + ingestion covariates suppresses false-positive anomalies during exercise/sleep/meals; the residual (unexplained HRV deviation) is interpreted as a *candidate* psychological-stress signal, validated circumstantially via correlation with self-reported daily stress and cortisol AUC.

**Status (2026-11-09):** both phases are complete. Phase 1: stress is detectable (median AUROC ≈ 0.96), but Chronos-2 is not better than SMA or ARIMA. Phase 2: the covariate claim is not supported. See `results/phase1_results.md` and `results/phase2_results.md`.

iOS / on-device deployment is a deferred Phase 3, out of scope for this plan.

## 2. Constraints and decisions locked in

- Compute: M1 MacBook Air, **CPU only**.
- Primary model: `autogluon/chronos-2-small` (28M params); compare against full `amazon/chronos-2` (120M) only as a ceiling reference if compute allows.
- Target series: **rolling RMSSD** at 1 Hz, computed over a 60 s sliding window of artifact-corrected RR intervals.
- Covariates (Phase 1): HR (1 Hz mean), ACC magnitude, EDA, RESP rate, TEMP — all downsampled to 1 Hz. Covariate models get them as history **and** as concurrent (known-future) values for the scored window; the primary comparison is univariate (see `results/phase1_results.md`).
- Covariates (Phase 2): 60-s trailing Vector Magnitude and steps/min, inclinometer lying/standing fractions, `is_sleeping` from sleep.csv, and Activity.csv flags for exercise (codes 5–6), eating (7) and stimulants (10–12), all at 1 Hz. HR is **not** a covariate: it carries the stress response itself (Phase 1 finding).
- Baselines: **SMA** (sanity floor, univariate), **ARIMAX** (with covariates, headline classical baseline). LSTM forecaster deferred.
- Evaluation: leave-one-subject-out; AUROC/AUPRC of anomaly score vs. labels; paired Wilcoxon signed-rank for method comparisons; report effect sizes.
- Anomaly score: fraction of observed values outside [q10, q90] forecast band, **and** standardized residual = (observed − median forecast) / forecast IQR. Report both.

## 3. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | CPU inference too slow on MMASH | Batch forecasts; 60 s stride; benchmark on WESAD first |
| 2 | RR-interval artifacts → garbage RMSSD | NeuroKit2 / hrv-analysis artifact correction step (mandatory) |
| 3 | Food/caffeine/alcohol confound the "stress residual" | Include full Activity.csv code as a categorical covariate |
| 4 | MMASH activity coding scheme unverified | First task: find and document activity codes from MMASH paper |
| 5 | Small N (15 / 22) limits statistical power | Paired tests + effect sizes; document in limitations |
| 6 | "Residual ≠ stress" interpretation overreach | Frame residual as *candidate* signal; cortisol/questionnaire are circumstantial validation only |

## 4. Project structure (to be scaffolded next)

```
project/
├── PLAN.md                     # this file
├── pyproject.toml              # dependencies
├── README.md
├── WESAD/                      # raw, gitignored
├── DataPaper/                  # raw MMASH, gitignored
├── data/                       # processed parquet outputs (gitignored)
│   ├── wesad_processed/
│   └── mmash_processed/
├── src/
│   ├── thesis/
│   │   ├── __init__.py
│   │   ├── io/                 # data loaders per dataset
│   │   │   ├── wesad.py
│   │   │   └── mmash.py
│   │   ├── hrv/                # R-peak detection, artifact correction, RMSSD
│   │   │   └── rmssd.py
│   │   ├── covariates/         # downsample, align, encode categorical
│   │   │   ├── wesad.py
│   │   │   └── mmash.py
│   │   ├── models/             # forecasting wrappers
│   │   │   ├── chronos2.py
│   │   │   ├── arimax.py
│   │   │   └── sma.py
│   │   ├── anomaly/            # residual → score → threshold
│   │   │   └── scoring.py
│   │   └── eval/               # metrics, LOSO harness, statistical tests
│   │       └── harness.py
│   └── scripts/                # entry points for each experiment
│       ├── 01_preprocess_wesad.py
│       ├── 02_preprocess_mmash.py
│       ├── 03_run_phase1.py
│       └── 04_run_phase2.py
├── notebooks/                  # exploratory, not for results
└── tests/                      # pytest, for non-trivial logic only
```

## 5. Phase 1 — WESAD detection (concrete task list)

Each task has a verification step. Do not advance until the verification passes.

### 5.1 Setup
- [ ] **T1.1** Add deps to `pyproject.toml`: `chronos-forecasting`, `neurokit2`, `hrv-analysis`, `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `pyarrow`, `tqdm`. Verify: `uv sync` (or `pip install -e .`) completes without errors.
- [ ] **T1.2** Scaffold `src/` directory tree above with empty `__init__.py` files. Verify: `python -c "import thesis"` runs.
- [ ] **T1.3** CPU smoke test for Chronos-2-small: load model, run one synthetic forecast, time it. Verify: completes in <5s on M1 CPU.

### 5.2 WESAD preprocessing
- [ ] **T1.4** Loader: read `SX.pkl`, extract chest signals (ECG, EDA, RESP, TEMP, ACC) and labels at 700 Hz. Verify: shapes match docs; labels in {0,1,2,3,4} only (drop 5/6/7).
- [ ] **T1.5** R-peak detection (NeuroKit2) on chest ECG → RR interval series. Verify: heart rate plot looks physiologically plausible (40–180 bpm range) on at least one subject, manually.
- [ ] **T1.6** RR-interval artifact correction (`signal_fixpeaks` or Malik filter). Verify: <5% of intervals flagged on quality subjects; document % flagged per subject.
- [ ] **T1.7** RMSSD computation @ 1 Hz with 60 s sliding window. Verify: RMSSD during baseline > RMSSD during stress on majority of subjects (qualitative sanity check, this is the central biomarker assumption).
- [ ] **T1.8** Covariate construction: downsample ACC magnitude / EDA / RESP rate / TEMP to 1 Hz, align with RMSSD. Save per subject as parquet under `data/wesad_processed/`. Verify: all series same length, no NaNs after gap-filling, timestamps aligned.

### 5.3 Models
- [ ] **T1.9** SMA baseline: forecast RMSSD as rolling mean of last N=60 samples; anomaly score = (observed − rolling mean) / rolling std. Verify: produces a finite score for every test window.
- [ ] **T1.10** ARIMAX baseline: per-subject `statsmodels.tsa.SARIMAX` fit on first half of available data with covariates as exog; forecast on second half; standardized residual as anomaly score. Verify: model converges for all 15 subjects (if not, log and inspect).
- [ ] **T1.11** Chronos-2-small wrapper: `predict_df`-based forecaster with sliding context (e.g., 5 min) and short horizon (e.g., 30 s); past-only covariates. Verify: produces probabilistic forecast (q10/q50/q90) for one subject end-to-end.

### 5.4 Evaluation harness
- [ ] **T1.12** LOSO harness: for each subject, run all 3 models, collect per-window anomaly scores aligned with stress labels (binary: 2=stress vs 1=baseline; ignore 3/4 for primary metric). Verify: produces per-subject AUROC/AUPRC for each model.
- [ ] **T1.13** Statistical comparison: paired Wilcoxon signed-rank across subjects on AUROC differences; report median, IQR, p-values, Cliff's delta. Verify: produces a clean results table.

### 5.5 Phase-1 success criteria
Phase 1 is "done" when:
1. Chronos-2-small with covariates achieves AUROC > 0.7 (median across LOSO subjects) on stress vs baseline windows.
2. Chronos-2-small with covariates beats ARIMAX (paired test, p<0.05 OR effect size > small) and beats SMA decisively.
3. Ablation (Chronos-2 univariate vs. with covariates) shows covariates help — quantifies the value of Chronos-2's headline feature.

If the model **fails** these criteria, that is also a publishable result (negative result on foundation models for this task), but the thesis framing changes — discuss before proceeding to Phase 2.

## 6. Phase 2 — MMASH contextual attribution (done)

Final protocol, results and revision log: `results/phase2_results.md`. Summary of tasks:

- [x] T2.1: Activity codes confirmed from the PhysioNet documentation (`src/thesis/covariates/mmash.py`).
- [x] T2.2: Loader, including the day-2 encoding bug in users 8 and 9 (`src/thesis/io/mmash.py`).
- [x] T2.3: RR artifact filter, gap-aware successive differences, RMSSD @ 1 Hz with ≥ 20 differences per window.
- [x] T2.4: Covariates on the same 1 Hz grid (`scripts/02_preprocess_mmash.py`).
- [x] T2.5: Models: SMA / OLS + cov, persistence / persistence + cov, Chronos-2 / Chronos-2 + cov. ARIMAX was replaced by the persistence family, because per-window SARIMAX fits were infeasible on CPU.
- [x] T2.6: Sensitivity measured by paired injection of an unexplained RMSSD drop (no stress labels exist). Activity false alarms measured as AUROC of active vs sedentary real windows.
- [x] T2.7: Injection AUROC reported within sleep, active and sedentary windows.
- [x] T2.8: Exploratory Spearman correlations with Daily Stress and cortisol change: none survive.

## 7. Out of scope

- iOS / CoreML deployment and on-device latency benchmarking
- Fine-tuning Chronos-2 (zero-shot is the thesis claim)
- LSTM forecaster baseline (revisit only if ARIMAX is implausibly weak)
- Frequency-domain HRV (LF/HF) — too window-hungry for available data
- Multi-dataset transfer / domain adaptation experiments

## 8. Open questions to revisit during execution

- Optimal Chronos-2 context length and prediction horizon (start with 5 min / 30 s; tune empirically).
- Whether to include questionnaire-based subject-level features (BIS/BAS, STAI trait) as static covariates — Chronos-2 doesn't natively support static features, so likely no.
- Whether full `amazon/chronos-2` (120M) is worth running as a ceiling reference — depends on Phase-1 compute budget.
