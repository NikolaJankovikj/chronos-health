from __future__ import annotations

import numpy as np
import pandas as pd


# Chronos-2 predict_df output column names (verified in T1.3 smoke test)
COL_PRED = "predictions"  # median / q50
COL_Q10 = "0.1"
COL_Q50 = "0.5"
COL_Q90 = "0.9"


def residual_score(
    observed: pd.Series,
    q10: pd.Series,
    q50: pd.Series,
    q90: pd.Series,
) -> pd.Series:
    """(observed - q50) / (q90 - q10); sign indicates direction of deviation."""
    iqr = (q90 - q10).replace(0, np.nan)
    return (observed - q50) / iqr


def band_exceedance(
    observed: pd.Series,
    q10: pd.Series,
    q90: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling fraction of timesteps where observed falls outside [q10, q90]."""
    outside = ((observed < q10) | (observed > q90)).astype(float)
    return outside.rolling(window=window, min_periods=1).mean()
