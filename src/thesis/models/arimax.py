from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

TARGET = "rmssd"
ORDER = (2, 0, 2)


def make_scorer(covariate_cols: list[str]) -> callable:
    """Return an ARIMAX scorer: per-window SARIMAX fit with band-exceedance score.

    NOTE: ARIMAX receives future covariate values for the forecast horizon
    (since they come from sensor readings).  This gives ARIMAX an information
    advantage over Chronos-2 (which uses only past covariates) — document in thesis.
    """
    def scorer(context_df: pd.DataFrame, observed_df: pd.DataFrame) -> float:
        y_train = context_df[TARGET].values.astype(float)
        exog_train = context_df[covariate_cols].values.astype(float) if covariate_cols else None
        exog_test = observed_df[covariate_cols].values.astype(float) if covariate_cols else None

        if np.isnan(y_train).mean() > 0.3:
            return float("nan")

        # Forward-fill NaN in training target
        mask = np.isnan(y_train)
        if mask.any():
            idx = np.where(~mask)[0]
            if len(idx) == 0:
                return float("nan")
            y_train = np.interp(np.arange(len(y_train)), idx, y_train[idx])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                model = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=100)

            horizon = len(observed_df)
            fc = result.get_forecast(steps=horizon, exog=exog_test)
            # conf_int returns a numpy array in this statsmodels version
            ci = np.asarray(fc.conf_int(alpha=0.2))
            pred_mean = np.asarray(fc.predicted_mean)
            half_band = (ci[:, 1] - ci[:, 0]) / 2.0 + 1e-6

        except Exception:
            return float("nan")

        obs = observed_df[TARGET].values.astype(float)
        valid = ~np.isnan(obs)
        if valid.sum() == 0:
            return float("nan")
        # One-sided: how far BELOW the ARIMAX forecast is the observed RMSSD?
        return float(np.mean(np.maximum(0.0, (pred_mean[valid] - obs[valid]) / half_band[valid])))

    return scorer
