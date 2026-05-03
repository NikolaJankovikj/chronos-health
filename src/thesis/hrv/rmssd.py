from __future__ import annotations

import bisect

import numpy as np
from scipy.ndimage import median_filter


def extract_rr(ecg: np.ndarray, sampling_rate: int = 700) -> tuple[np.ndarray, np.ndarray]:
    """Detect R-peaks, apply Malik artifact filter, return (rr_ms, peak_times_s).

    peak_times_s[i] is the time (seconds from recording start) of the later beat
    in pair i; rr_ms[i] = peak_times_s[i] - peak_times_s[i-1] (in milliseconds).
    """
    import neurokit2 as nk

    ecg_clean = nk.ecg_clean(ecg, sampling_rate=sampling_rate)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
    peaks = info["ECG_R_Peaks"]

    rr_ms = np.diff(peaks).astype(np.float64) / sampling_rate * 1000.0
    peak_times_s = peaks[1:].astype(np.float64) / sampling_rate

    # Malik filter — step 1: physiological range
    mask = (rr_ms >= 300.0) & (rr_ms <= 2000.0)
    rr_ms = rr_ms[mask]
    peak_times_s = peak_times_s[mask]

    # Malik filter — step 2: reject intervals deviating >20% from local 5-beat median
    if len(rr_ms) >= 5:
        local_med = median_filter(rr_ms, size=5, mode="nearest")
        mask2 = np.abs(rr_ms - local_med) / local_med < 0.20
        rr_ms = rr_ms[mask2]
        peak_times_s = peak_times_s[mask2]

    return rr_ms, peak_times_s


def rmssd_series(
    rr_ms: np.ndarray,
    peak_times_s: np.ndarray,
    n_seconds: int,
    window_s: int = 60,
) -> np.ndarray:
    """Rolling RMSSD (ms) at 1 Hz using an O(n) sliding-window approach.

    Returns array of length n_seconds; NaN where fewer than 2 beats in the window.
    Index i corresponds to the interval [i, i+1) seconds from recording start.
    """
    results = np.full(n_seconds, np.nan)
    ts = peak_times_s  # sorted float seconds

    for i in range(n_seconds):
        t_end = float(i + 1)
        t_start = t_end - window_s
        left = bisect.bisect_left(ts, t_start)
        right = bisect.bisect_right(ts, t_end)
        rr_win = rr_ms[left:right]
        if len(rr_win) >= 2:
            diffs = np.diff(rr_win)
            results[i] = float(np.sqrt(np.mean(diffs ** 2)))

    return results


def hr_series(
    rr_ms: np.ndarray,
    peak_times_s: np.ndarray,
    n_seconds: int,
) -> np.ndarray:
    """Instantaneous HR (bpm) at 1 Hz: mean of 60000/rr for beats in each second."""
    results = np.full(n_seconds, np.nan)
    ts = peak_times_s

    for i in range(n_seconds):
        t_end = float(i + 1)
        left = bisect.bisect_left(ts, t_end - 1.0)
        right = bisect.bisect_right(ts, t_end)
        rr_win = rr_ms[left:right]
        if len(rr_win) >= 1:
            results[i] = float(np.mean(60000.0 / rr_win))

    return results
