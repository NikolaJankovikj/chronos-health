from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "rmssd"


def forecast(context_df: pd.DataFrame) -> tuple[float, float]:
    """(μ, σ): mean and std of the context RMSSD, ignoring missing seconds."""
    rmssd = context_df[TARGET].dropna().values
    if len(rmssd) == 0:
        return float("nan"), float("nan")
    return float(rmssd.mean()), float(rmssd.std())


def make_scorer() -> callable:
    """Return an SMA scorer: mean |observed − μ| / σ over the scored window.

    μ, σ come from the whole context (the same context every other model
    sees).  Univariate — covariate columns are ignored.
    """
    def scorer(context_df: pd.DataFrame, observed_df: pd.DataFrame) -> float:
        mu, sigma = forecast(context_df)
        obs = observed_df[TARGET].dropna().values
        if not sigma >= 1e-6 or len(obs) == 0:
            return float("nan")
        # Two-sided: ~1/3 of subjects show RMSSD rising, not falling, under stress.
        return float(np.mean(np.abs(mu - obs) / sigma))

    return scorer
