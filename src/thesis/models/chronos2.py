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


def load_pipeline(model_id: str = DEFAULT_MODEL):
    """Load Chronos-2 pipeline on CPU."""
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")


def _make_chronos_df(df: pd.DataFrame, item_id: str) -> pd.DataFrame:
    """Add item_id and synthetic 1 Hz timestamps required by predict_df."""
    out = df.reset_index(drop=True).copy()
    out["item_id"] = item_id
    out["timestamp"] = pd.date_range("2000-01-01", periods=len(df), freq="1s")
    return out


def make_scorer(pipeline, covariate_cols: list[str]) -> callable:
    """Return a Chronos-2 scorer using past-only covariates (realistic deployment mode).

    The scorer does NOT provide future covariate values — unlike ARIMAX, it uses
    only historical context.  Document this asymmetry in the thesis.
    """
    # Warm up JIT once at scorer creation time
    _dummy = pd.DataFrame({
        "item_id": "warmup",
        "timestamp": pd.date_range("2000-01-01", periods=10, freq="1s"),
        TARGET: np.random.default_rng(0).standard_normal(10) + 50,
    })
    pipeline.predict_df(
        _dummy, prediction_length=5,
        quantile_levels=DEFAULT_QUANTILES,
        id_column="item_id", timestamp_column="timestamp", target=TARGET,
    )

    def scorer(context_df: pd.DataFrame, observed_df: pd.DataFrame) -> float:
        ctx = _make_chronos_df(context_df[[TARGET] + covariate_cols], item_id="s0")

        try:
            pred = pipeline.predict_df(
                ctx,
                prediction_length=len(observed_df),
                quantile_levels=DEFAULT_QUANTILES,
                id_column="item_id",
                timestamp_column="timestamp",
                target=TARGET,
            )
        except Exception:
            return float("nan")

        q10  = pred[COL_Q10].values
        q50  = pred[COL_Q50].values
        q90  = pred[COL_Q90].values
        iqr  = q90 - q10 + 1e-6
        obs  = observed_df[TARGET].values.astype(float)
        valid = ~np.isnan(obs)
        if valid.sum() == 0:
            return float("nan")
        # Two-sided absolute deviation: handles both HRV suppression and
        # atypical upward stress responses across subjects.
        return float(np.mean(np.abs(q50[valid] - obs[valid]) / iqr[valid]))

    return scorer
