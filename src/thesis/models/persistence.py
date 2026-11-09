from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "rmssd"
MIN_STD = 1e-9


def forecast(
    context_df: pd.DataFrame, future_df: pd.DataFrame, covariate_cols: list[str], leads: np.ndarray
) -> tuple[np.ndarray, float]:
    """Persistence forecast, optionally corrected by covariate changes.

    point(step) = last RMSSD + β · (x(step) − x(last)), with β fitted by OLS (no
    intercept) of k-step RMSSD changes on k-step covariate changes over the
    context, pooled over k in `leads`.  σ is the RMS of those fitted k-step
    errors.  Without covariates this is naive persistence with its empirical
    error at the scored leads.
    """
    y = context_df[TARGET].to_numpy(dtype=float)
    varying = [c for c in covariate_cols if context_df[c].std() > MIN_STD]
    x = context_df[varying].to_numpy(dtype=float)
    dy = np.concatenate([y[k:] - y[:-k] for k in leads])
    dx = np.concatenate([x[k:] - x[:-k] for k in leads])
    ok = np.isfinite(dy)
    beta = np.linalg.lstsq(dx[ok], dy[ok], rcond=None)[0] if varying else np.zeros(0)
    sigma = float(np.sqrt(np.mean((dy[ok] - dx[ok] @ beta) ** 2)))
    last = np.flatnonzero(np.isfinite(y))[-1]
    point = y[last] + (future_df[varying].to_numpy(dtype=float) - x[last]) @ beta
    return point, sigma
