import numpy as np
import pandas as pd

from thesis.covariates import mmash as cov
from thesis.eval import phase2 as p2
from thesis.io.mmash import seconds
from thesis.models import linreg, persistence


def test_seconds_fixes_day_rollover_and_accepts_hh_mm():
    t = seconds(pd.Series([1, -29, 2]), pd.Series(["10:00", "00:00:30", "00:00:30"]))
    assert t.tolist() == [36_000, 86_430, 86_430]


def test_rr_gap_breaks_adjacency_but_keeps_intervals():
    rr = pd.DataFrame({
        "ibi_s": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        "day": 1,
        "time": [
            "10:00:00",
            "10:00:01",
            "10:00:02",
            "10:00:12",
            "10:00:13",
            "10:00:14",
        ],
    })
    rr_ms, t = cov.rr_with_gap_breaks(rr)
    assert len(rr_ms) == 7 and np.isnan(
        rr_ms[3]
    )  # break inserted before the beat after the gap
    assert np.isfinite(rr_ms[4]) and t[3] == t[4]


def test_log_intervals_wrap_midnight_and_drop_entry_errors():
    grid = np.arange(0, 2 * 86_400, 600)
    starts, ends = (
        np.array([23 * 3600, 12 * 3600 + 600]),
        np.array([1 * 3600, 10 * 3600 + 2100]),
    )
    flag = cov._intervals(starts, ends, grid)
    assert flag[grid == 23 * 3600 + 1800][0] == 1.0  # 23:00 → 01:00 wraps
    assert flag[grid == 86_400 + 1800][0] == 1.0
    assert (
        flag[(grid > 2 * 3600) & (grid < 22 * 3600)].sum() == 0
    )  # 12:10 → 10:35 (22 h) dropped


def test_linreg_ignores_covariates_constant_in_context():
    ctx = pd.DataFrame({
        "rmssd": [10.0, 12.0, 14.0, 16.0],
        "vm": [0.0, 1.0, 2.0, 3.0],
        "sleeping": 0.0,
    })
    fut = pd.DataFrame({"vm": [4.0], "sleeping": [1.0]})
    point, sigma = linreg.forecast(ctx, fut, ["vm", "sleeping"])
    assert (
        np.isclose(point[0], 18.0) and sigma < 1e-6
    )  # sleeping never varied, so it is ignored


def test_window_starts_require_valid_context_and_scored_window():
    n = p2.CONTEXT_S + 3 * p2.STRIDE_S
    rmssd = np.ones(n)
    second = p2.CONTEXT_S + p2.STRIDE_S
    rmssd[second : second + p2.GUARD_S] = np.nan  # guard gap may be missing
    rmssd[second + p2.GUARD_S : second + p2.FORECAST_S] = np.nan  # scored window may not
    starts = p2.window_starts(pd.DataFrame({"rmssd": rmssd}))
    assert starts.tolist() == [p2.CONTEXT_S, p2.CONTEXT_S + 2 * p2.STRIDE_S]


def test_persistence_uses_last_value_and_covariate_changes():
    x = np.sin(np.arange(200) / 7.0)
    ctx = pd.DataFrame({"rmssd": 30 + 2 * x, "vm": x})
    fut = pd.DataFrame({"vm": [x[-1] + 1.0]})
    point, sigma = persistence.forecast(ctx, fut, ["vm"], np.array([1, 2, 3]))
    assert np.isclose(point[0], ctx["rmssd"].iloc[-1] + 2.0) and sigma < 1e-9
    point, sigma = persistence.forecast(ctx, fut, [], np.array([1]))
    assert point[0] == ctx["rmssd"].iloc[-1]
    assert np.isclose(sigma, np.sqrt(np.mean(np.diff(ctx["rmssd"]) ** 2)))


def test_injection_raises_score_only_through_the_target():
    observed = np.full((1, 4), 50.0)
    point, sigma = np.full((1, 4), 50.0), np.full((1, 4), 5.0)
    assert p2.deviation_scores(observed, point, sigma)[0] == 0.0
    assert np.isclose(p2.deviation_scores(observed * 0.7, point, sigma)[0], 3.0)


def test_stratum_precedence():
    base = {"sleeping": [0.0] * 4, "steps": [0.0] * 4, "exercise": [0.0] * 4}
    assert p2.stratum(pd.DataFrame(base)) == p2.SEDENTARY
    assert p2.stratum(pd.DataFrame({**base, "exercise": [0, 0, 0, 1.0]})) == p2.ACTIVE
    assert (
        p2.stratum(pd.DataFrame({**base, "steps": [100.0] * 4, "sleeping": [1.0] * 4}))
        == p2.SLEEP
    )
