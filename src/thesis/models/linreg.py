from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "rmssd"
MIN_STD = 1e-9


def forecast(context_df: pd.DataFrame, future_df: pd.DataFrame, covariate_cols: list[str]) -> tuple[np.ndarray, float]:
    """OLS of RMSSD on the covariates over the context; return (prediction for
    each future step, residual σ).

    Covariates that never vary in the context are dropped: the context carries
    no information about their effect, and keeping them only adds an arbitrary,
    unidentified coefficient.  With no varying covariate this is the SMA.
    """
    ctx = context_df[context_df[TARGET].notna()]
    varying = [c for c in covariate_cols if ctx[c].std() > MIN_STD]
    design = np.column_stack([np.ones(len(ctx)), ctx[varying].to_numpy(dtype=float)])
    y = ctx[TARGET].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    dof = max(len(y) - design.shape[1], 1)
    sigma = float(np.sqrt(np.sum((y - design @ beta) ** 2) / dof))
    future = np.column_stack([np.ones(len(future_df)), future_df[varying].to_numpy(dtype=float)])
    return future @ beta, sigma
