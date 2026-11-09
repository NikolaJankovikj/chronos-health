import numpy as np
import pandas as pd

from thesis.covariates.wesad import _resp_rate_1hz
from thesis.eval import harness as h
from thesis.hrv.rmssd import filter_rr, rmssd_series
from thesis.models import arimax, chronos2


def test_filter_rr_rejects_missed_beat_and_out_of_range():
    rr = np.array([800, 810, 1600, 805, 800, 250, 795, 800], dtype=float)
    out = filter_rr(rr)
    assert np.isnan(out[[2, 5]]).all()
    assert not np.isnan(np.delete(out, [2, 5])).any()


def test_rmssd_skips_differences_across_rejected_beats():
    rr = np.array([800, 800, np.nan, 1000, 1000])
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Deleting the NaN would yield a spurious 200 ms difference (RMSSD ≈ 115).
    assert rmssd_series(rr, times, n_seconds=6, window_s=60)[-1] == 0.0


def test_resp_rate_ignores_noise_above_breathing_band():
    n_seconds = 180
    t = np.arange(n_seconds * 700) / 700
    resp = np.sin(2 * np.pi * 0.25 * t) + 0.6 * np.sin(2 * np.pi * 1.5 * t)
    rate = _resp_rate_1hz(resp, n_seconds)
    assert abs(np.nanmedian(rate[90:]) - 15.0) < 1.0


def _arimax_frames(rng, slope):
    x = rng.standard_normal(330)
    y = 50 + slope * x + rng.standard_normal(330)
    df = pd.DataFrame({"rmssd": y, "acc_mag": x})
    return df.iloc[:300], df.iloc[300:]


def test_arimax_refits_when_context_changes():
    rng = np.random.default_rng(0)
    ctx_a, obs = _arimax_frames(rng, slope=5.0)
    ctx_b, _ = _arimax_frames(rng, slope=-5.0)
    shared = arimax.make_scorer(["acc_mag"])
    shared(ctx_a, obs)
    assert shared(ctx_b, obs) == arimax.make_scorer(["acc_mag"])(ctx_b, obs)


def test_arimax_window_covariates_expose_hidden_shift():
    # RMSSD tracks the covariate; a +5 shift is invisible against the covariate-
    # driven spread unless the forecast follows the window's own covariate values.
    rng = np.random.default_rng(1)
    ctx, obs = _arimax_frames(rng, slope=10.0)
    shifted = obs.assign(rmssd=obs["rmssd"] + 5.0)
    assert arimax.make_scorer(["acc_mag"])(ctx, shifted) > 3.0
    assert arimax.make_scorer([])(ctx, shifted) < 1.5


class _RecordingPipeline:
    def __init__(self):
        self.future_df = "unset"

    def predict_df(self, df, future_df=None, prediction_length=None, **_):
        self.future_df = future_df
        return pd.DataFrame({
            "0.1": np.zeros(prediction_length),
            "0.5": np.ones(prediction_length),
            "0.9": np.full(prediction_length, 2.0),
        })


def test_chronos_passes_window_covariates_as_known_future():
    ctx = pd.DataFrame({"rmssd": np.ones(10), "acc_mag": np.zeros(10)})
    obs = pd.DataFrame({"rmssd": np.ones(3), "acc_mag": [7.0, 8.0, 9.0]})
    pipe = _RecordingPipeline()
    chronos2.make_scorer(pipe, ["acc_mag"])(ctx, obs)
    assert pipe.future_df["acc_mag"].tolist() == [7.0, 8.0, 9.0]
    assert pipe.future_df["timestamp"].iloc[0] == pd.Timestamp("2000-01-01 00:00:10")

    chronos2.make_scorer(pipe, [])(ctx, obs)
    assert pipe.future_df is None


def test_compare_methods_rank_biserial_drops_ties():
    a = pd.Series([1.0, 2.0, 3.0, 0.0, 5.0])
    b = pd.Series([0.0, 0.0, 0.0, 0.5, 5.0])
    cmp = h.compare_methods(a, b)
    # |diffs| 1, 2, 3, 0.5 → ranks 2, 3, 4, 1; (9 − 1) / 10
    assert (cmp["n"], cmp["n_nonzero"]) == (5, 4)
    assert np.isclose(cmp["rank_biserial"], 0.8)
    assert cmp["ci_low"] <= cmp["mean_diff"] <= cmp["ci_high"]


def test_holm_keeps_nan_out_of_the_family():
    adj = h.holm([0.01, float("nan"), 0.04])
    assert np.isclose(adj[0], 0.02) and np.isclose(adj[2], 0.04) and np.isnan(adj[1])


def test_verdict():
    tight = {"mean_diff": 0.01, "ci_low": -0.02, "ci_high": 0.03}
    a_capped = {"mean_diff": -0.02, "ci_low": -0.07, "ci_high": 0.01}
    wide = {"mean_diff": -0.1, "ci_low": -0.2, "ci_high": 0.1}
    assert h.verdict(tight, p_adj=0.5) == "equivalent"
    assert h.verdict(tight, p_adj=0.01) == "A better (negligible)"
    assert h.verdict(a_capped, p_adj=0.5) == "A not meaningfully better"
    assert h.verdict(wide, p_adj=0.5) == "inconclusive"
    assert h.verdict(wide, p_adj=0.01) == "B better"


def test_windows_skip_runs_that_span_a_gap():
    idx = np.r_[np.arange(0, 45), np.arange(100, 160)]
    starts = [w[0] for w in h.windows(idx)]
    assert starts == [0, 100, 130]  # nothing straddles 44 → 100; each run tiled from its start


def test_held_out_baseline_starts_after_guard():
    labels = np.r_[np.zeros(10), np.ones(900), np.full(100, 2)].astype(int)
    df = pd.DataFrame({"rmssd": 1.0, "label": labels})
    ctx, held_out = h.split_baseline(df)
    assert len(ctx) == h.CONTEXT_S
    assert held_out[0] - ctx.index[-1] == h.GUARD_S + 1
    assert held_out[-1] == 909
