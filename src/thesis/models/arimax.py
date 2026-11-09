from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

TARGET = "rmssd"
ORDER  = (2, 0, 2)
MAX_NAN_FRAC = 0.3


def _fit(context_df: pd.DataFrame, covariate_cols: list[str]):
    """Fit SARIMAX on the context; None if the target is too sparse or the fit fails."""
    y = context_df[TARGET].values.astype(float)
    nan_mask = np.isnan(y)
    if nan_mask.mean() > MAX_NAN_FRAC:
        return None
    if nan_mask.any():
        idx = np.where(~nan_mask)[0]
        y = np.interp(np.arange(len(y)), idx, y[idx])
    exog = context_df[covariate_cols].values.astype(float) if covariate_cols else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            return SARIMAX(y, exog=exog, order=ORDER).fit(disp=False, maxiter=100)
    except (np.linalg.LinAlgError, ValueError):  # degenerate context, e.g. constant covariate
        return None


def make_scorer(covariate_cols: list[str]) -> callable:
    """Return an ARIMAX scorer: mean |observed − forecast| / forecast s.e.

    The model is fitted once per context.  The forecast is driven by the scored
    window's own (concurrent) covariates, the same information Chronos-2 gets
    via `future_df`, and treats the window as immediately following the context.
    """
    # Keeping a reference to the context makes the `is` check sound; an id()-keyed
    # cache can hand one subject's fit to the next once the old frame is freed.
    last: dict = {"context": None, "result": None}

    def scorer(context_df: pd.DataFrame, observed_df: pd.DataFrame) -> float:
        if last["context"] is not context_df:
            last["context"] = context_df
            last["result"] = _fit(context_df, covariate_cols)
        result = last["result"]
        if result is None:
            return float("nan")

        exog = observed_df[covariate_cols].values.astype(float) if covariate_cols else None
        fc = result.get_forecast(steps=len(observed_df), exog=exog)
        mean = np.asarray(fc.predicted_mean)
        se = np.asarray(fc.se_mean)

        obs = observed_df[TARGET].values.astype(float)
        valid = ~np.isnan(obs) & (se > 1e-6)
        if valid.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs(obs[valid] - mean[valid]) / se[valid]))

    return scorer
