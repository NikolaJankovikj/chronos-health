from __future__ import annotations

import pandas as pd


def forecast(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling-mean forecast; shifted by 1 step to avoid look-ahead."""
    return series.shift(1).rolling(window=window, min_periods=1).mean()
