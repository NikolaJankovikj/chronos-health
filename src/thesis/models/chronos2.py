from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MODEL = "autogluon/chronos-2-small"
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
TARGET = "rmssd"

# Column names from predict_df output (verified in T1.3 smoke test)
COL_Q10  = "0.1"
COL_Q50  = "0.5"
COL_Q90  = "0.9"

_EPOCH = pd.Timestamp("2000-01-01")
# q90 − q10 of a Gaussian is 2·1.2816σ; dividing by this puts scores in σ units,
# the same units as the SMA (context std) and ARIMAX (forecast s.e.) scores.
_Q10_Q90_PER_SIGMA = 2.5631


def load_pipeline(model_id: str = DEFAULT_MODEL):
    """Load Chronos-2 pipeline on CPU."""
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")


def _make_chronos_df(df: pd.DataFrame, item_id: str, offset_s: int = 0) -> pd.DataFrame:
    """Add item_id and synthetic 1 Hz timestamps (starting `offset_s` after epoch)."""
    out = df.reset_index(drop=True).copy()
    out["item_id"] = item_id
    out["timestamp"] = pd.date_range(_EPOCH + pd.Timedelta(seconds=offset_s),
                                     periods=len(df), freq="1s")
    return out


def forecast_batch(
    pipeline,
    contexts: list[pd.DataFrame],
    futures: list[pd.DataFrame] | None,
    horizon: int,
    covariate_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast many windows in one call; return (q50, σ̂), each shaped (n, horizon).

    contexts[i] holds the target and covariate history; futures[i] (when
    covariates are used) holds the covariates for the `horizon` steps that
    immediately follow it, passed as known-future values.
    """
    ids = [f"w{i:06d}" for i in range(len(contexts))]
    ctx = pd.concat([_make_chronos_df(c[[TARGET] + covariate_cols], k) for c, k in zip(contexts, ids, strict=True)])
    fut = None
    if covariate_cols:
        fut = pd.concat([_make_chronos_df(f[covariate_cols], k, offset_s=len(c))
                         for c, f, k in zip(contexts, futures, ids, strict=True)])
    pred = pipeline.predict_df(
        ctx,
        future_df=fut,
        prediction_length=horizon,
        quantile_levels=DEFAULT_QUANTILES,
        id_column="item_id",
        timestamp_column="timestamp",
        target=TARGET,
    ).sort_values(["item_id", "timestamp"])
    q50 = pred[COL_Q50].to_numpy().reshape(len(ids), horizon)
    sigma = (pred[COL_Q90] - pred[COL_Q10]).to_numpy().reshape(len(ids), horizon) / _Q10_Q90_PER_SIGMA + 1e-6
    return q50, sigma


def make_scorer(pipeline, covariate_cols: list[str]) -> callable:
    """Return a Chronos-2 scorer: mean |observed − q50| / σ̂, σ̂ = (q90 − q10) / 2.5631.

    Covariates enter twice: their context history (past) and the scored
    window's own values (`future_df`, known-future).  The window is placed
    immediately after the context, matching the ARIMAX scorer.
    """
    def scorer(context_df: pd.DataFrame, observed_df: pd.DataFrame) -> float:
        q50, sigma = forecast_batch(pipeline, [context_df], [observed_df], len(observed_df), covariate_cols)
        obs = observed_df[TARGET].values.astype(float)
        valid = ~np.isnan(obs)
        if valid.sum() == 0:
            return float("nan")
        # Two-sided: ~1/3 of subjects show RMSSD rising, not falling, under stress.
        return float(np.mean(np.abs(q50[0][valid] - obs[valid]) / sigma[0][valid]))

    return scorer
