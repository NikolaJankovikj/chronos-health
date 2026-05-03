from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLING_RATE = 700


def build(signals: dict, label_array: np.ndarray) -> pd.DataFrame:
    """Downsample WESAD chest signals to 1 Hz and return an aligned DataFrame.

    Columns: acc_mag, eda, resp_rate, temp, hr_bpm, label.
    All values are mean-pooled over non-overlapping 700-sample windows.
    """
    ...
