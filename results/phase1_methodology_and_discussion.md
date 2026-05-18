# Phase 1 — Methodology Journey and Discussion

This document explains **every major decision made during Phase 1**, why each decision was made, what we tried before that decision, and how the results reflect each choice. It is intended to serve as a detailed methods and discussion narrative for the thesis.

---

## 1. Why WESAD and Why HRV?

### The starting point

The thesis asks whether a zero-shot foundation model (Chronos-2) can detect physiologically anomalous HRV — and whether providing contextual covariates (movement, temperature, respiration) makes the detection meaningfully better than simpler approaches.

Before any modelling can begin, a dataset must provide:
1. A clean, interpretable physiological label (stress vs. not)
2. High-quality ECG or RR-interval data
3. Additional sensor channels for contextual covariates

WESAD satisfies all three. Its stress condition — the **Trier Social Stress Test (TSST)**, a validated laboratory stressor involving public speaking and mental arithmetic — is one of the most reliably reproducible acute psychological stress protocols in the literature. Having clean condition labels with known timestamps makes it the right choice for a controlled first-phase validation.

### Why RMSSD as the target series

HRV has several metrics. The choice of which one to use as the model's target is non-trivial:

- **SDNN** (standard deviation of RR intervals) reflects overall autonomic variability. It requires at least 5-minute windows for reliable estimates and captures both sympathetic and parasympathetic activity.
- **LF/HF ratio** (frequency domain) is even more window-hungry (5-minute minimum), methodologically contested, and sensitive to breathing rate artefacts.
- **RMSSD** (root mean square of successive RR differences) reflects specifically the **parasympathetic branch** of the autonomic nervous system — the branch that acutely withdraws under psychological and physical stress. It is stable over 30–60 second windows, making it appropriate for 1 Hz time-series analysis.

We chose RMSSD because it is the most direct metric for the stress-withdrawal effect we are trying to detect. Heart rate (HR) was included as a covariate rather than the target because it conflates sympathetic activation with parasympathetic withdrawal — a person with high HR could be exercising (sympathetic) or stressed (both), and HR alone does not carry the same directional specificity as RMSSD.

### Why 60-second rolling windows at 1 Hz

The 60-second window represents a physiologically principled choice:
- At a resting HR of ~65 bpm, a 60-second window contains ~65 RR intervals — enough for a statistically stable RMSSD estimate (the minimum recommended is ~30 beats).
- Shorter windows (e.g., 10–20 s) produce highly noisy RMSSD values that would make the forecasting task harder without adding information.
- Longer windows (e.g., 5 min = 300 s) blur the temporal boundary between conditions and are standard for frequency-domain HRV, not RMSSD.

The result of this choice can be seen in the processed data: S2's RMSSD values range from 28–208 ms with a mean of ~62 ms during baseline and ~39 ms during stress — a clear, biologically plausible separation.

---

## 2. Preprocessing: R-Peak Detection and Artifact Correction

### Why R-peak detection matters

WESAD's pkl files contain ECG sampled at 700 Hz, not pre-computed RR intervals. To get RMSSD, you need the times between R-peaks (the sharp spikes in the ECG waveform). Missed peaks, false detections, or motion artifacts produce RR interval sequences with outliers (e.g., an interval of 2.5 s because a beat was missed) that completely corrupt RMSSD — one spurious outlier in a 60-beat window can double the RMSSD value.

We used **NeuroKit2's `ecg_peaks`** function for R-peak detection. NeuroKit2's algorithm applies bandpass filtering before detection to suppress motion artifacts and high-frequency noise, which is important given that subjects are instrumented with a chest belt that moves slightly during breathing.

### The Malik artifact filter

After R-peak detection, we apply a two-stage artifact filter:
1. **Physiological range filter**: remove any RR interval outside [300 ms, 2000 ms] (equivalent to HR outside [30, 200] bpm). An interval below 300 ms almost certainly represents a double-detection; above 2000 ms represents a missed beat.
2. **Malik criterion**: remove any interval that deviates >20% from a local 5-beat rolling median. This catches subtler artefacts where the interval is technically plausible but locally inconsistent.

In practice, S2 had ~0% of beats removed — clean ECG. A few subjects with noted sensor placement issues (per their readme files) had slightly higher removal rates. This is important to check because unusually high removal rates (>5%) can indicate segments where RMSSD is unreliable and should be treated with caution in interpretation.

### Verification (T1.7 sanity check)

After computing RMSSD, the first sanity check was: **does baseline RMSSD > stress RMSSD for the majority of subjects?** This is the central biomarker assumption of the entire thesis. The result: 10/15 subjects confirmed (67%). The 5 atypical subjects (S6, S7, S9, S10, S15) have stress RMSSD ≥ baseline RMSSD — a genuine finding about individual variability, not a preprocessing bug. This has important consequences for the evaluation design (see Section 4).

---

## 3. Covariate Construction

### Why these five covariates

The five covariates — HR, ACC magnitude, EDA, respiration rate, skin temperature — were selected because they represent distinct physiological systems that are known to covary with stress:

- **HR**: a faster heart rate often accompanies psychological stress (sympathetic activation). Including it alongside RMSSD lets the model reason about the HR–HRV tradeoff: low RMSSD with high HR is more anomalous than low RMSSD with low HR.
- **ACC magnitude**: in free-living (Phase 2), physical activity is the primary confound for low RMSSD. In WESAD, all conditions are sedentary, so ACC carries little information — but it is included for methodological consistency across phases.
- **EDA (electrodermal activity)**: sweat gland activity driven by sympathetic arousal. EDA rises sharply during psychological stress and is widely used as an objective stress marker in affective computing. Including it gives the model a direct signal of arousal.
- **Respiration rate**: breathing affects HRV directly through respiratory sinus arrhythmia (RSA). Slow, deep breathing raises RMSSD; rapid or irregular breathing suppresses it. Including respiration rate allows the model to distinguish stress-related RMSSD suppression from RMSSD changes driven purely by respiratory patterns.
- **Skin temperature**: a slower-responding peripheral vasoconstriction marker. Less directly related to acute stress than EDA, but included as a thermal context signal.

All five are mean-pooled from 700 Hz to 1 Hz and forward-filled to handle the occasional NaN during the first 60-second RMSSD warmup window.

### What the WESAD covariates actually tell us

An important constraint: because WESAD subjects are sedentary throughout, ACC magnitude varies almost not at all between conditions. This means **in WESAD, covariates carry almost no additional information beyond RMSSD itself**. EDA does rise during TSST (the intended stress signal), but its rise is contemporaneous with the RMSSD drop — it does not *precede* or *explain away* the RMSSD change in any mechanistic sense for a forecaster trained on general time series.

The consequence of this constraint on results is discussed in Section 6.

---

## 4. The Evaluation Design: From Failure to the Personal-Baseline Approach

### First attempt: rolling context window (AUROC ≈ 0.47)

The initially implemented evaluation was the most natural one for streaming anomaly detection: at each timestep, use the **previous 300 seconds as context** to forecast the next 30 seconds, and compare the observation to the forecast. This is how the system would work in deployment.

The result: **all four methods produced AUROC ≈ 0.47** — essentially random. The per-subject score distributions showed stress windows and baseline windows with nearly identical mean scores.

The diagnosis required looking at the WESAD protocol timeline:

```
Session timeline for a typical subject:
Minute 7    → Baseline begins      (label = 1)
Minute 26   → Baseline ends
Minute 27–39 → Transition gap     (label = 0)
Minute 39   → TSST stress begins   (label = 2)
Minute 50   → TSST ends
```

The **13-minute unlabelled gap** between the baseline and stress periods is the problem. By the time any stress window arrives for evaluation, the rolling 5-minute context window falls entirely within the stress period itself — the model's recent history already shows depressed RMSSD. Because the model predicts "more of the same," it predicts low RMSSD and the observation is also low: no anomaly detected.

This is not a model failure; it is a dataset structure issue. In a real deployment without protocol labels, a wearable device would face exactly this problem whenever a stressor has been active for longer than the context window. This is why Phase 2 (MMASH, free-living) is necessary to test the realistic scenario.

### Why the personal-baseline context is the right evaluation for WESAD

The rolling-window approach failed because it asks the wrong question for WESAD: it asks "is this window different from the last 5 minutes?" when the last 5 minutes are all stress. The right question for WESAD is: "is this window different from this person's resting state?"

The **personal-baseline context** evaluation answers exactly that question:

1. Take the last 300 seconds from the first 2/3 of each subject's labeled baseline period — this becomes the fixed context.
2. Hold out the remaining 1/3 of baseline for evaluation (prevents data leakage — the model never sees these windows during context construction).
3. Evaluate all windows drawn from (a) held-out late-baseline and (b) the full stress period.

This means every evaluation window is compared against the same reference: "what does this person's HRV look like at rest?" Stress windows that deviate strongly from that reference get high anomaly scores; held-out baseline windows that look similar to the reference get low scores.

This design is clinically meaningful because a real-world wearable system does exactly this: it calibrates against the user's personal resting baseline (e.g., during sleep or a morning rest period) before monitoring daytime HRV for deviations. It is also the most honest evaluation of the thesis's central claim about per-person anomaly detection.

**The 2/3 context / 1/3 holdout split** was chosen to:
- Give enough baseline data for a stable context window (typically ~12 minutes for context, leaving ~6 minutes for evaluation).
- Prevent data leakage: the context and the evaluated baseline windows are strictly separated.

---

## 5. Scoring Design: From Band Exceedance to Absolute Deviation

### First attempt: band exceedance (symmetric, 80% CI)

The initial anomaly score was **band exceedance**: the fraction of observed RMSSD values in a 30-second window that fall outside the forecasted [q10, q90] interval. This is a natural choice for probabilistic forecasters like Chronos-2, which explicitly outputs quantile bands.

The problem emerged when analysing score distributions after the rolling-window experiment: stress windows and baseline windows had nearly identical mean band exceedance scores (0.493 vs 0.492 for SMA on S2). The reason: with an 80% confidence interval, approximately 20% of baseline windows already exceed the band by definition (false positive rate built into the threshold). Stress windows exceeded the band only marginally more often, because the RMSSD drop during stress was moderate relative to the context distribution spread.

### Second attempt: one-sided normalised deviation

The more appropriate score for stress-induced HRV suppression is a **one-sided score**: how far **below** the expected value is the observation? Stress causes RMSSD to drop, not to rise. The score was changed to:

```
score = mean(max(0, expected − observed) / scale)
```

This correctly assigned higher scores to stress windows (observed RMSSD below baseline expectation) and lower scores to baseline windows (observed RMSSD near expectation). The AUROC for S2 SMA immediately improved from 0.502 to ~0.54.

However, this introduced a new problem: the **5 atypical subjects** (S6, S7, S9, S10, S15) who show stress RMSSD ≥ baseline RMSSD. For these subjects, the score `max(0, expected − observed)` is always near zero during stress (because observed > expected), producing AUROC < 0.5 — inverted, worse than random.

### Final approach: two-sided absolute deviation

The correct generalisation is **absolute normalised deviation**:

```
score = mean(|expected − observed| / scale)
```

This detects anomalies regardless of direction — whether RMSSD drops (typical) or rises (atypical) relative to the baseline expectation. For the 10 typical subjects, the scores are dominated by the downward deviation during stress; for the 5 atypical subjects, the upward deviation is detected instead.

This choice is scientifically principled: an anomaly detector should not assume the *direction* of HRV change under stress — individual variability means both directions occur in practice. The important signal is *deviation from personal baseline*, whatever direction that takes.

**Why this still makes sense for Phase 2 (MMASH):** in the free-living setting, we want to detect both exercise-induced and stress-induced deviations. Exercise causes RMSSD to drop; some recovery states cause it to rise above normal. An absolute score handles all these cases correctly.

---

## 6. Why Methods Perform Comparably — The Ceiling Effect Explained

After implementing the personal-baseline context with absolute deviation scoring, all four methods achieved median AUROC of 0.96–0.98. Statistically, no method is significantly better than any other (all p > 0.05, N=15).

This result is explained by three reinforcing factors:

**Factor 1: The signal-to-noise ratio is very high for typical subjects.**
For the 10 typical subjects, mean baseline RMSSD is ~53 ms and mean stress RMSSD is ~28 ms — a difference of ~25 ms against a within-condition standard deviation of roughly 6–10 ms. This gives a separation of ~3–4 standard deviations. Any method that correctly models the baseline mean and variance will achieve near-perfect discrimination at this signal strength. Even a 3-line SMA achieves AUROC=0.990 (median) on typical subjects.

**Factor 2: Chronos-2's covariate advantage has no signal to work with in WESAD.**
Chronos-2's key differentiator over Chronos-1 is native support for past-only and known-future covariates. The underlying hypothesis of the thesis is that these covariates — especially movement — allow the model to "explain away" expected HRV changes and focus the anomaly score on unexplained deviations. In WESAD, all subjects are sedentary throughout all conditions. ACC barely moves. Respiration rate changes slightly, EDA rises during TSST — but none of these creates a situation where the model needs covariates to *avoid* flagging a non-stress HRV change as an anomaly. The covariate pathway is simply not exercised by this dataset. This is exactly why Chronos2_CV ≈ Chronos2_noCV across all subjects.

**Factor 3: The binding constraint for all methods is inter-individual variability, not model architecture.**
The 5 atypical subjects limit every method's performance equally. For these subjects, no amount of model sophistication recovers the discriminability because the underlying signal (stress RMSSD > baseline RMSSD) is in the opposite direction from what would be expected. ARIMAX shows a slight edge on atypical subjects (median AUROC 0.808 vs ~0.67 for others) — likely because its fitted time-series model captures covariate structure that SMA and Chronos-2 do not exploit in the same way when the RMSSD direction is reversed. But this edge is not statistically significant at N=5.

**What this ceiling effect tells us and does not tell us:**
- It tells us: all methods correctly learn the personal resting HRV baseline and detect deviations from it, when the signal is strong.
- It does not tell us: which method handles *weak or ambiguous signals* better. That question can only be answered in Phase 2, where the confound (physical activity) creates exactly the ambiguous signal that requires covariate context to resolve.

---

## 7. Method-Specific Findings

### SMA (Simple Moving Average)

SMA uses the mean and standard deviation of the last 60 seconds of context RMSSD as its entire model. Despite its simplicity, it achieves median AUROC=0.979 — near ceiling on typical subjects. This is both a reassuring sanity check (the signal is real and detectable) and a challenge for the thesis: any more complex method needs to beat this to justify the added complexity.

The implication for the thesis: SMA serves as a meaningful "lowest possible bar" that all other methods must clear. On WESAD with a clean personal baseline, none of them clear it by a statistically significant margin.

### ARIMAX — SARIMAX(2,0,2)

ARIMAX was implemented with a per-subject fit on the baseline context window. An important design issue emerged during development: calling `get_forecast(steps=H)` for evaluation windows that are 13+ minutes after the context end produces unstable extrapolations. Long-range ARIMA forecasts revert to the long-run mean, but the uncertainty bands grow so wide that the normalised deviation score becomes uninformative (S2 ARIMAX = 0.083 before this was fixed).

The fix was to use the in-sample fitted values' mean and residual standard deviation (`sqrt(model.scale)`) as the baseline distribution parameters — effectively making ARIMAX behave like a statistically principled SMA that fits the noise structure more carefully. After this fix, ARIMAX achieved median AUROC=0.981 and notably performed best among the four methods on atypical subjects.

**A note on the ARIMAX implementation:** The original design passed future covariate values (`exog_test`) to `get_forecast()`, giving ARIMAX access to sensor readings from inside the evaluation window — an information advantage Chronos-2 does not have. However, this approach was abandoned because SARIMAX long-range extrapolation is unstable over the 13-minute baseline-to-stress gap in WESAD (forecasts diverged, producing AUROC as low as 0.083 on S2). The final implementation replaces the forecast with the baseline context mean and MLE residual standard deviation — no future covariates are used. Both ARIMAX and Chronos-2 have equivalent information access in the current code.

### Chronos-2 (no covariates vs. with covariates)

Both Chronos-2 variants achieve median AUROC=0.960. The covariate ablation (CV vs noCV) shows no meaningful difference (p=0.500, Cliff's δ=−0.20). This null result on the ablation is not a failure — it is scientifically correct given WESAD's sedentary conditions. The ablation confirms that the covariate pathway in Chronos-2 is not harmful (it doesn't hurt) but simply has no signal to work with in this dataset.

**Chronos-2 vs SMA and ARIMAX:** Chronos-2 achieves slightly lower median AUROC than both SMA (0.960 vs 0.979) and ARIMAX (0.960 vs 0.981). For SMA, this likely reflects that Chronos-2's probabilistic IQR is wider than SMA's empirical rolling standard deviation — a wider uncertainty band means the normalised deviation score is smaller for the same absolute RMSSD difference, reducing discrimination at the ceiling. For ARIMAX, its MLE residual standard deviation (fitted over 300 seconds with autocorrelation accounted for) is tighter still, giving the sharpest normalisation of the three — hence its marginal lead on atypical subjects where the anomaly is a modest upward deviation. None of these differences are statistically significant at N=15. Chronos-2's advantage over both is expected to emerge in Phase 2, where non-stationarity (activity transitions) creates a scenario that neither SMA nor ARIMAX can handle with a fixed baseline mean.

---

## 8. What Phase 1 Proves and What It Does Not

### What Phase 1 proves

1. **The preprocessing pipeline is correct.** RMSSD values are physiologically plausible, R-peak detection quality is high, and the T1.7 sanity check (baseline > stress RMSSD in majority of subjects) is satisfied.
2. **Personal-baseline deviation is detectable by all methods.** When comparing against a clean resting baseline, all four methods achieve high AUROC (~0.96–0.98) at detecting psychological stress.
3. **Chronos-2 is a viable anomaly scorer.** It achieves AUROC=0.960 with zero training on any WESAD subject, confirming the zero-shot claim.
4. **On WESAD, no method statistically dominates.** The ceiling effect prevents differentiation. This is an important null result that must be stated honestly.

### What Phase 1 does not prove

1. **Covariates help with stress detection** — they do not, on WESAD (sedentary). This absence of benefit is dataset-structural, not a model failure.
2. **Chronos-2 outperforms classical methods** — it does not, on this dataset.
3. **The detection generalises to free-living conditions** — WESAD is a controlled lab study. The 13-minute gap, the known protocol order, and the sedentary conditions make it non-representative of deployment.

### The motivating argument for Phase 2

The Phase 1 results sharpen the Phase 2 research question. In a controlled lab with a known clean baseline, simple approaches work as well as foundation models. The unique value proposition of Chronos-2's covariate capability can only emerge in a scenario where:
- The context window does not automatically represent the "normal" state (free-living has continuous activity transitions).
- Physical activity creates genuine confounds (low RMSSD during exercise is normal, not anomalous — a context-blind model would flag it incorrectly).
- No clean "baseline period" is pre-labelled (the model must distinguish activity-induced HRV changes from stress-induced ones in real time).

Phase 2 (MMASH) provides exactly this scenario. The evaluation shifts from "detect psychological stress against a clean baseline" to "suppress false positives during exercise, eating, and sleep — while leaving unexplained RMSSD deviations flagged." That is where the covariate pathway is expected to provide meaningful advantage.

---

## 9. Timeline of Design Decisions and Their Consequences

| Step | Decision | Why | Consequence on results |
|---|---|---|---|
| 1 | RMSSD as target, 60s window | Parasympathetically specific, stable on short windows | Physiologically meaningful, clean separation for typical subjects |
| 2 | Malik artifact filter | Prevents single outlier beats from corrupting RMSSD | <1% beats removed, clean RMSSD traces |
| 3 | All 5 covariates | Cover all known physiological confounds for Phase 2 | No impact on WESAD (sedentary); essential for Phase 2 |
| 4 | Rolling-window evaluation | Seemed natural for streaming detection | AUROC ≈ 0.47 — near chance; revealed 13-min protocol gap |
| 5 | Personal-baseline context | The correct question for WESAD given protocol structure | AUROC jumps to 0.96–0.98 |
| 6 | Band exceedance score | Natural for probabilistic forecasters | Stress/baseline scores indistinguishable (≈0.493 vs 0.492) |
| 7 | One-sided deviation score | Stress causes RMSSD to drop | Fixed typical subjects; broke atypical subjects (AUROC < 0.5) |
| 8 | Two-sided absolute deviation | Direction-agnostic; handles individual variability | All subjects valid; atypical subjects recover to 0.67–0.98 |
| 9 | ARIMAX long-range forecast fix | `get_forecast()` unstable at 13+ min extrapolation; original design also passed future covariates (`exog_test`) — an information advantage not available to Chronos-2 | Replaced with baseline mean + MLE residual std; no future covariates used; S2 ARIMAX recovered from 0.083 to 0.938; both methods now have equivalent information access |

---

## 10. Summary

Phase 1 is a rigorous, honest validation of the core HRV anomaly-detection pipeline on controlled data. Its key contribution to the thesis is establishing that:

- The Chronos-2 foundation model can detect personal-baseline HRV deviations at zero shot, with AUROC comparable to classical methods.
- On WESAD, the ceiling imposed by the strong stress signal means no method differentiates meaningfully.
- The covariate advantage of Chronos-2 is not activated by WESAD's sedentary conditions.
- These findings collectively justify Phase 2: only in free-living data, where physical activity continuously perturbs HRV and no clean baseline is pre-labelled, does the context-awareness of Chronos-2 have the opportunity to demonstrate meaningful advantage over simpler approaches.

The full results, statistical tests, and data files are documented in `results/phase1_results.md` and `data/phase1_results.csv`.
