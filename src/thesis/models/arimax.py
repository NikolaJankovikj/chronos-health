from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

TARGET = "rmssd"
ORDER  = (2, 0, 2)


def make_scorer(covariate_cols: list[str]) -> callable:
    """Return an ARIMAX scorer.

    Fits SARIMAX(2,0,2) on the context window once and reuses the result
    for all subsequent calls with the same context.  The scorer caches the
    fitted model so the per-window cost is just `get_forecast`, not a full fit.

    NOTE: ARIMAX receives future covariate values for the horizon (realistic —
    sensor readings are always available).  This gives ARIMAX a small information
    advantage over Chronos-2 (past-only covariates).  Document in thesis.
    """
    _cache: dict = {}   # keyed on id(context_df) to avoid refitting same context

    def scorer(context_df: pd.DataFrame, observed_df: pd.DataFrame) -> float:
        ctx_id = id(context_df)

        if ctx_id not in _cache:
            y = context_df[TARGET].values.astype(float)
            exog = context_df[covariate_cols].values.astype(float) if covariate_cols else None

            # Interpolate NaN in target
            nan_mask = np.isnan(y)
            if nan_mask.mean() > 0.3:
                _cache[ctx_id] = None
            else:
                if nan_mask.any():
                    idx = np.where(~nan_mask)[0]
                    y = np.interp(np.arange(len(y)), idx, y[idx])
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        model  = SARIMAX(y, exog=exog, order=ORDER,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)
                        result = model.fit(disp=False, maxiter=100)
                    _cache[ctx_id] = result
                except Exception:
                    _cache[ctx_id] = None

        result = _cache.get(ctx_id)
        if result is None:
            return float("nan")

        # Use the model's in-sample fitted residual std as the noise scale —
        # more reliable than long-range get_forecast() which becomes unstable
        # when evaluation windows are far from the context end (e.g., 13-min
        # gap between WESAD baseline and stress conditions).
        try:
            sigma = float(np.sqrt(result.scale)) if result.scale > 0 else float("nan")
            if np.isnan(sigma) or sigma < 1e-6:
                return float("nan")
            mu = float(result.fittedvalues.mean())
        except Exception:
            return float("nan")

        obs   = observed_df[TARGET].values.astype(float)
        valid = ~np.isnan(obs)
        if valid.sum() == 0:
            return float("nan")
        # Absolute deviation from baseline mean, normalised by residual std.
        return float(np.mean(np.abs(mu - obs[valid]) / sigma))

    return scorer
