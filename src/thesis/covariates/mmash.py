"""MMASH → 1 Hz frame: RMSSD target, context covariates, and window strata.

Activity codes (MMASH documentation, PhysioNet v1.0.0): 1 sleeping, 2 laying down,
3 sitting, 4 light movement, 5 medium (fast walk, bike), 6 heavy (gym, running),
7 eating, 8 small screen, 9 large screen, 10 caffeinated drink, 11 smoking,
12 alcohol.  Code 0 is undocumented and ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.hrv.rmssd import filter_rr, rmssd_series
from thesis.io.mmash import seconds

EXERCISE_CODES = (5, 6)
EATING_CODES = (7,)
STIMULANT_CODES = (10, 11, 12)  # caffeine, smoking, alcohol
# Matches the RMSSD window, so covariates describe the same minute as the target.
SMOOTH_S = 60
# Timestamps have 1 s resolution: a jump larger than the interval itself plus
# this slack means beats were dropped before this one.
GAP_SLACK_S = 2.0
# A 60-s RMSSD needs enough successive differences to be stable (~1/3 of a minute at rest).
MIN_DIFFS = 20
MAX_LOG_INTERVAL_S = (
    12 * 3600
)  # longer log intervals are entry errors (e.g. End before Start)

COVARIATES = [
    "vm",
    "steps",
    "lying",
    "standing",
    "sleeping",
    "exercise",
    "eating",
    "stimulant",
]


def rr_with_gap_breaks(rr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(rr_ms, beat_time_s) with artifacts as NaN and a NaN inserted before every
    beat that follows a recording gap, so no successive difference spans a gap."""
    t = seconds(rr["day"], rr["time"])
    ibi_s = rr["ibi_s"].to_numpy(dtype=float)
    rr_ms = filter_rr(ibi_s * 1000.0)
    gap = np.flatnonzero(np.diff(t) > ibi_s[1:] + GAP_SLACK_S) + 1
    return np.insert(rr_ms, gap, np.nan), np.insert(t, gap, t[gap])


def _intervals(starts: np.ndarray, ends: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """1 where a grid second falls in any [start, end) interval; midnight-crossing
    intervals wrap, implausibly long ones are dropped."""
    ends = np.where(ends < starts, ends + 86_400, ends)
    keep = (ends - starts) <= MAX_LOG_INTERVAL_S
    flag = np.zeros(len(grid))
    for s, e in zip(starts[keep], ends[keep], strict=True):
        flag[(grid >= s) & (grid < e)] = 1.0
    return flag


def _log_flag(
    activity: pd.DataFrame, codes: tuple[int, ...], grid: np.ndarray
) -> np.ndarray:
    rows = activity[activity["Activity"].isin(codes)].dropna(subset=["Start", "End"])
    if rows.empty:
        return np.zeros(len(grid))
    return _intervals(
        seconds(rows["Day"], rows["Start"]), seconds(rows["Day"], rows["End"]), grid
    )


def _sleep_flag(sleep: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    if sleep.empty:
        return np.zeros(len(grid))
    starts = seconds(sleep["In Bed Date"], sleep["In Bed Time"])
    ends = seconds(sleep["Out Bed Date"], sleep["Out Bed Time"])
    return _intervals(starts, ends, grid)


def _actigraph(actigraph: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Trailing-minute movement and posture; seconds without data are skipped, and
    minutes without any data carry the last value forward."""
    ag = actigraph.assign(t=seconds(actigraph["day"], actigraph["time"])).set_index("t")
    ag = ag[~ag.index.duplicated()].reindex(grid)
    cols = ["Vector Magnitude", "Steps", "Inclinometer Lying", "Inclinometer Standing"]
    out = ag[cols].rolling(SMOOTH_S, min_periods=1).mean()
    out.columns = ["vm", "steps", "lying", "standing"]
    out["steps"] *= 60.0  # steps per minute
    return out.ffill().bfill()


def build(subject: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, float]:
    """1 Hz frame indexed by seconds since day-1 midnight, and the RR artifact rate."""
    rr_ms, beat_t = rr_with_gap_breaks(subject["rr"])
    grid = np.arange(int(beat_t.min()), int(beat_t.max()) + 1)
    rmssd = rmssd_series(
        rr_ms, beat_t - grid[0], len(grid), window_s=SMOOTH_S, min_diffs=MIN_DIFFS
    )
    df = _actigraph(subject["actigraph"], grid)
    df.insert(0, "rmssd", rmssd)
    df["sleeping"] = _sleep_flag(subject["sleep"], grid)
    activity = subject["activity"]
    df["exercise"] = _log_flag(activity, EXERCISE_CODES, grid)
    df["eating"] = _log_flag(activity, EATING_CODES, grid)
    df["stimulant"] = _log_flag(activity, STIMULANT_CODES, grid)
    df.index.name = "t"
    return df, float(
        np.isnan(filter_rr(subject["rr"]["ibi_s"].to_numpy() * 1000.0)).mean()
    )
