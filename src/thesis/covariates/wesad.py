from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from thesis.io.wesad import SAMPLING_RATE as SR
from thesis.io.wesad import VALID_LABELS


def _pool_mean(arr_1d: np.ndarray, n_seconds: int) -> np.ndarray:
    """Mean-pool a 700 Hz 1-D array to 1 Hz."""
    return arr_1d[: n_seconds * SR].reshape(n_seconds, SR).mean(axis=1)


def _pool_label(labels: np.ndarray, n_seconds: int) -> np.ndarray:
    """Mode-pool labels to 1 Hz; windows with mixed or no valid labels → 0."""
    windows = labels[: n_seconds * SR].reshape(n_seconds, SR)
    result = np.zeros(n_seconds, dtype=np.int32)
    for i, w in enumerate(windows):
        valid = w[np.isin(w, list(VALID_LABELS))]
        if len(valid) > 0:
            unique = np.unique(valid)
            if len(unique) == 1:
                result[i] = unique[0]
    return result


def _resp_rate_1hz(resp_700hz: np.ndarray, n_seconds: int) -> np.ndarray:
    """Estimate respiration rate (breaths/min) at 1 Hz via bandpass + peak detection."""
    target_sr = 10
    factor = SR // target_sr
    n_blocks = (n_seconds * SR) // factor
    resp_low = resp_700hz[: n_blocks * factor].reshape(n_blocks, factor).mean(axis=1)

    nyq = target_sr / 2.0
    b, a = butter(4, [0.1 / nyq, 0.5 / nyq], btype="band")
    resp_filt = filtfilt(b, a, resp_low)

    # Detect inhalation peaks (min 2 s apart = max 30 breaths/min)
    peaks, _ = find_peaks(resp_filt, distance=target_sr * 2)
    peak_times = peaks / float(target_sr)

    rate_1hz = np.full(n_seconds, np.nan)
    for i in range(n_seconds):
        t_end = float(i + 1)
        in_window = peak_times[(peak_times >= t_end - 60.0) & (peak_times <= t_end)]
        if len(in_window) >= 2:
            rate_1hz[i] = 60.0 / np.mean(np.diff(in_window))

    return rate_1hz


def build(
    signals: dict[str, np.ndarray],
    rmssd_1hz: np.ndarray,
    hr_1hz: np.ndarray,
) -> pd.DataFrame:
    """Assemble a 1 Hz DataFrame of RMSSD target + all covariates + label.

    Columns: rmssd, hr_bpm, acc_mag, eda, resp_rate, temp, label.
    """
    n_seconds = len(signals["label"]) // SR

    acc_mag = np.linalg.norm(signals["acc"][: n_seconds * SR], axis=1)

    return pd.DataFrame(
        {
            "rmssd":     rmssd_1hz[:n_seconds],
            "hr_bpm":    hr_1hz[:n_seconds],
            "acc_mag":   _pool_mean(acc_mag, n_seconds),
            "eda":       _pool_mean(signals["eda"], n_seconds),
            "resp_rate": _resp_rate_1hz(signals["resp"], n_seconds),
            "temp":      _pool_mean(signals["temp"], n_seconds),
            "label":     _pool_label(signals["label"], n_seconds),
        }
    )
