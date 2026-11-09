"""Phase-2 evaluation protocol on MMASH (free-living, 24 h per subject).

There are no stress labels, so sensitivity is measured with a paired,
semi-synthetic design.  At every sampled position s the models see the 30-min
context before s and the covariates from s on, and forecast RMSSD up to
s + GUARD_S + HORIZON_S.  The scored window is the last HORIZON_S seconds, so
its 60-s RMSSD shares no beat with the context.  The real window is scored as
it is (negative) and again with its RMSSD lowered by INJECTION_DEPTH
(positive).  Because no scored RMSSD value contains a pre-s beat, lowering it
is exactly what scaling every successive difference after s would produce:
an *unexplained* vagal drop starting at s, with the covariates untouched.
Both versions share context and forecast, so only the injected drop differs.

  * primary   = per-subject AUROC, injected vs real windows (all states)
  * secondary = activity false alarms: AUROC of real active windows vs real
                sedentary-awake windows (0.5 = activity is not flagged)
  * strata    = injection AUROC within sleep / active / sedentary windows
  * sweep     = injection AUROC over INJECTION_SWEEP depths
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from thesis.covariates.mmash import COVARIATES

TARGET = "rmssd"
CONTEXT_S = 1800  # 30 min: long enough to contain activity transitions
HORIZON_S = 30
GUARD_S = 60  # = RMSSD window: scored values contain no beat from the context
FORECAST_S = GUARD_S + HORIZON_S
STRIDE_S = 300
MIN_VALID_FRAC = 0.5  # of seconds with a valid RMSSD, in both context and window
INJECTION_DEPTH = 0.30
INJECTION_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5)
ACTIVE_STEPS_PER_MIN = 60  # walking cadence
MAX_ARTIFACT_FRAC = 0.10
SLEEP, ACTIVE, SEDENTARY = "sleep", "active", "sedentary"
MAJORITY = 0.5  # a window counts as sleep when most of its seconds are in bed

# forecaster(contexts, futures) -> (point, sigma), both shaped (n_windows, FORECAST_S);
# futures[i] holds the covariates for the FORECAST_S steps after contexts[i].
Forecaster = Callable[
    [list[pd.DataFrame], list[pd.DataFrame]], tuple[np.ndarray, np.ndarray]
]


def exclusion_reason(df: pd.DataFrame) -> str | None:
    if df.attrs["artifact_frac"] > MAX_ARTIFACT_FRAC:
        return f"RR artifacts {df.attrs['artifact_frac']:.1%} > {MAX_ARTIFACT_FRAC:.0%}"
    if df["sleeping"].sum() == 0:
        return "no sleep annotation (the sleep covariate would be wrong)"
    return None


def window_starts(df: pd.DataFrame) -> np.ndarray:
    """Positions with a full context and enough valid RMSSD in context and scored window."""
    valid = df[TARGET].notna().to_numpy()
    starts = np.arange(CONTEXT_S, len(df) - FORECAST_S + 1, STRIDE_S)
    ok = [
        valid[s - CONTEXT_S : s].mean() >= MIN_VALID_FRAC
        and valid[s + GUARD_S : s + FORECAST_S].mean() >= MIN_VALID_FRAC
        for s in starts
    ]
    return starts[np.array(ok, dtype=bool)]


def stratum(window: pd.DataFrame) -> str:
    if window["sleeping"].mean() > MAJORITY:
        return SLEEP
    if window["steps"].mean() >= ACTIVE_STEPS_PER_MIN or window["exercise"].any():
        return ACTIVE
    return SEDENTARY


def deviation_scores(
    observed: np.ndarray, point: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Mean |observed − point| / σ per window (rows), ignoring missing seconds."""
    return np.nanmean(np.abs(observed - point) / sigma, axis=1)


def _model_scores(observed: np.ndarray, point: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    """Real-window scores and, per depth, scores with the window's RMSSD lowered."""
    scores = {"score": deviation_scores(observed, point, sigma)}
    for depth in INJECTION_SWEEP:
        scores[f"score_inj_{depth}"] = deviation_scores(observed * (1 - depth), point, sigma)
    return scores


def score_subject(df: pd.DataFrame, forecasters: dict[str, Forecaster]) -> pd.DataFrame:
    """One row per (window, model): stratum, real score, and injected scores per depth."""
    df = df.reset_index(drop=True)
    starts = window_starts(df)
    contexts = [df.iloc[s - CONTEXT_S : s] for s in starts]
    futures = [df.iloc[s : s + FORECAST_S] for s in starts]
    scored = [f.iloc[GUARD_S:] for f in futures]
    observed = np.stack([w[TARGET].to_numpy(dtype=float) for w in scored])
    strata = [stratum(w) for w in scored]
    frames = []
    for name, forecaster in forecasters.items():
        point, sigma = forecaster(contexts, futures)
        scores = _model_scores(observed, point[:, GUARD_S:], sigma[:, GUARD_S:])
        frames.append(pd.DataFrame({"start": starts, "stratum": strata, "model": name, **scores}))
    return pd.concat(frames, ignore_index=True)


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def subject_metrics(windows: pd.DataFrame) -> dict[str, float]:
    """Metrics for one subject and one model."""
    inj = f"score_inj_{INJECTION_DEPTH}"
    by = {s: windows[windows["stratum"] == s] for s in (SLEEP, ACTIVE, SEDENTARY)}
    out = {
        "auroc_injection": _auroc(windows[inj].values, windows["score"].values),
        "fa_active": _auroc(by[ACTIVE]["score"].values, by[SEDENTARY]["score"].values),
        "fa_sleep": _auroc(by[SLEEP]["score"].values, by[SEDENTARY]["score"].values),
        "mean_sedentary_score": by[SEDENTARY]["score"].mean(),
    }
    for s, w in by.items():
        out[f"auroc_injection_{s}"] = _auroc(w[inj].values, w["score"].values)
        out[f"n_{s}"] = len(w)
    for depth in INJECTION_SWEEP:
        out[f"auroc_depth_{depth}"] = _auroc(
            windows[f"score_inj_{depth}"].values, windows["score"].values
        )
    return out


def _stack(fits: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (point vector, scalar σ) → (point, σ) arrays shaped (n, FORECAST_S)."""
    point = np.stack([p for p, _ in fits])
    return point, np.repeat(np.array([[sg] for _, sg in fits]), point.shape[1], axis=1)


def _sma_forecaster(contexts: list[pd.DataFrame], futures: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    from thesis.models import sma

    fits = []
    for c, f in zip(contexts, futures, strict=True):
        mu, sigma = sma.forecast(c)
        fits.append((np.full(len(f), mu), sigma))
    return _stack(fits)


def _linreg_forecaster(cols: list[str]) -> Forecaster:
    from thesis.models import linreg

    return lambda contexts, futures: _stack(
        [linreg.forecast(c, f, cols) for c, f in zip(contexts, futures, strict=True)])


def _persistence_forecaster(cols: list[str]) -> Forecaster:
    from thesis.models import persistence

    leads = np.arange(GUARD_S + 1, FORECAST_S + 1)  # the leads that are scored
    return lambda contexts, futures: _stack(
        [persistence.forecast(c, f, cols, leads) for c, f in zip(contexts, futures, strict=True)])


def _chronos_forecaster(pipeline, cols: list[str]) -> Forecaster:
    from thesis.models import chronos2

    return lambda contexts, futures: chronos2.forecast_batch(pipeline, contexts, futures, FORECAST_S, cols)


def phase2_forecasters(pipeline) -> dict[str, Forecaster]:
    """Three model families, each without and with covariates."""
    return {
        "SMA": _sma_forecaster,
        "LinReg_cov": _linreg_forecaster(COVARIATES),
        "Persistence": _persistence_forecaster([]),
        "Persistence_cov": _persistence_forecaster(COVARIATES),
        "Chronos2": _chronos_forecaster(pipeline, []),
        "Chronos2_cov": _chronos_forecaster(pipeline, COVARIATES),
    }
