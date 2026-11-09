from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "rmssd"


def make_scorer() -> callable:
    """Return an SMA scorer: mean |observed − μ| / σ over the scored window.

    μ, σ are the mean and std of the whole context (the same context every
    other model sees).  Univariate — covariate columns are ignored.
    """
    def scorer(context_df: pd.DataFrame, observed_df: pd.DataFrame) -> float:
        rmssd = context_df[TARGET].dropna().values
        sigma = rmssd.std() if len(rmssd) else 0.0
        obs = observed_df[TARGET].dropna().values
        if sigma < 1e-6 or len(obs) == 0:
            return float("nan")
        # Two-sided: ~1/3 of subjects show RMSSD rising, not falling, under stress.
        return float(np.mean(np.abs(rmssd.mean() - obs) / sigma))

    return scorer
