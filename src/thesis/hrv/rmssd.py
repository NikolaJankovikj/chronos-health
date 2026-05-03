from __future__ import annotations

import numpy as np
import pandas as pd


def detect_rpeaks(ecg: np.ndarray, sampling_rate: int = 700) -> np.ndarray:
    """Return R-peak sample indices from a raw ECG signal via NeuroKit2."""
    import neurokit2 as nk
    _, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
    return info["ECG_R_Peaks"]


def correct_rr(rr_s: np.ndarray) -> np.ndarray:
    """Remove RR-interval artifacts; return corrected array in seconds.

    Applies Malik criterion: reject intervals outside [0.3, 2.0] s and
    those differing >20% from the local median.
    """
    import neurokit2 as nk
    rr_ms = rr_s * 1000.0
    rr_corrected, _ = nk.signal_fixpeaks(rr_ms, iterative=True, show=False)
    return rr_corrected / 1000.0


def rmssd_series(
    rr_s: np.ndarray,
    timestamps: pd.DatetimeIndex,
    window_s: int = 60,
) -> pd.Series:
    """Compute rolling RMSSD at 1 Hz from an irregular RR interval series.

    Resamples RR intervals onto a 1 Hz grid using nearest-neighbour
    interpolation, then slides a window_s-second window to compute RMSSD.
    """
    rr_series = pd.Series(rr_s, index=timestamps)
    grid = pd.date_range(timestamps[0].ceil("s"), timestamps[-1].floor("s"), freq="1s")
    rr_resampled = rr_series.reindex(rr_series.index.union(grid)).interpolate("time").reindex(grid)

    def _rmssd(window: pd.Series) -> float:
        diffs = np.diff(window.dropna().values)
        return float(np.sqrt(np.mean(diffs**2))) if len(diffs) >= 2 else np.nan

    return rr_resampled.rolling(window=window_s, min_periods=window_s // 2).apply(_rmssd, raw=False)
