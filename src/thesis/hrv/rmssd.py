from __future__ import annotations

import bisect

import numpy as np
from scipy.ndimage import median_filter


def filter_rr(rr_ms: np.ndarray) -> np.ndarray:
    """Malik-style artifact filter; rejected intervals become NaN.

    NaN (rather than deletion) keeps positions, so downstream successive
    differences are never taken across a rejected beat.
    """
    out = rr_ms.astype(np.float64)
    out[(out < 300.0) | (out > 2000.0)] = np.nan  # outside 30–200 bpm

    in_range = ~np.isnan(out)
    if in_range.sum() >= 5:
        vals = out[in_range]
        local_med = median_filter(vals, size=5, mode="nearest")
        vals[np.abs(vals - local_med) / local_med >= 0.20] = np.nan
        out[in_range] = vals
    return out


def extract_rr(ecg: np.ndarray, sampling_rate: int = 700) -> tuple[np.ndarray, np.ndarray]:
    """Detect R-peaks and filter artifacts; return (rr_ms, peak_times_s).

    peak_times_s[i] is the time (seconds from recording start) of the later beat
    in pair i. rr_ms[i] is NaN where the interval was rejected as an artifact.
    """
    import neurokit2 as nk

    ecg_clean = nk.ecg_clean(ecg, sampling_rate=sampling_rate)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
    peaks = info["ECG_R_Peaks"]

    rr_ms = np.diff(peaks).astype(np.float64) / sampling_rate * 1000.0
    peak_times_s = peaks[1:].astype(np.float64) / sampling_rate
    return filter_rr(rr_ms), peak_times_s


def rmssd_series(
    rr_ms: np.ndarray,
    peak_times_s: np.ndarray,
    n_seconds: int,
    window_s: int = 60,
    min_diffs: int = 1,
) -> np.ndarray:
    """Rolling RMSSD (ms) at 1 Hz over a `window_s` trailing window.

    Returns array of length n_seconds; NaN where the window holds fewer than
    `min_diffs` successive differences between valid, adjacent intervals.
    Index i corresponds to the interval [i, i+1) seconds from recording start.
    """
    results = np.full(n_seconds, np.nan)
    ts = peak_times_s  # sorted float seconds

    for i in range(n_seconds):
        t_end = float(i + 1)
        t_start = t_end - window_s
        left = bisect.bisect_left(ts, t_start)
        right = bisect.bisect_right(ts, t_end)
        diffs = np.diff(rr_ms[left:right])
        diffs = diffs[~np.isnan(diffs)]
        if len(diffs) >= min_diffs:
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
        rr_win = rr_win[~np.isnan(rr_win)]
        if len(rr_win) >= 1:
            results[i] = float(np.mean(60000.0 / rr_win))

    return results
