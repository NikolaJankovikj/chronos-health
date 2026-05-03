from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "rmssd"


def make_scorer(stats_window: int = 60) -> callable:
    """Return an SMA scorer: pseudo-band-exceedance using rolling Gaussian stats.

    Uses the last `stats_window` seconds of context to estimate mu/sigma, then
    computes the fraction of observed values outside (mu ± 1.645σ) as the
    anomaly score.  Covariate columns are ignored (univariate baseline).
    """
    def scorer(context_df: pd.DataFrame, observed_df: pd.DataFrame) -> float:
        rmssd = context_df[TARGET].dropna().values
        if len(rmssd) < stats_window:
            return float("nan")
        window = rmssd[-stats_window:]
        mu = window.mean()
        sigma = window.std()
        if sigma < 1e-6:
            return float("nan")
        obs = observed_df[TARGET].dropna().values
        if len(obs) == 0:
            return float("nan")
        # One-sided: how far BELOW the recent mean is the observed RMSSD?
        # Stress causes HRV to drop, so flag downward deviations.
        return float(np.mean(np.maximum(0.0, (mu - obs) / sigma)))

    return scorer
