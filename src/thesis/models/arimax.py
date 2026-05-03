from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_and_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    exog_cols: list[str],
    order: tuple[int, int, int] = (2, 0, 2),
) -> pd.Series:
    """Fit SARIMAX on train split; return point forecasts aligned to test index."""
    model = SARIMAX(
        train[target],
        exog=train[exog_cols] if exog_cols else None,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    forecasts = result.forecast(steps=len(test), exog=test[exog_cols] if exog_cols else None)
    return pd.Series(forecasts.values, index=test.index)
