from __future__ import annotations

import pandas as pd

DEFAULT_MODEL = "autogluon/chronos-2-small"
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


def load_pipeline(model_id: str = DEFAULT_MODEL):
    """Load Chronos-2 pipeline on CPU."""
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")


def forecast(
    pipeline,
    context_df: pd.DataFrame,
    prediction_length: int,
    target: str,
    id_col: str = "item_id",
    timestamp_col: str = "timestamp",
    quantile_levels: list[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Forecast one window; return DataFrame with quantile columns."""
    return pipeline.predict_df(
        context_df,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        id_column=id_col,
        timestamp_column=timestamp_col,
        target=target,
    )
