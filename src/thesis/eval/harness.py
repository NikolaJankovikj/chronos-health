from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

TARGET = "rmssd"
COVARIATE_COLS = ["hr_bpm", "acc_mag", "eda", "resp_rate", "temp"]

HORIZON_S = 30   # prediction / scoring window length
STRIDE_S  = 30   # non-overlapping horizons

# Fraction of the baseline period used as context (the rest is held out for evaluation)
BASELINE_CONTEXT_FRAC = 0.67


def _ffill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in COVARIATE_COLS:
        if col in out.columns:
            out[col] = out[col].ffill().bfill()
    return out


def _build_baseline_context(df: pd.DataFrame, context_s: int) -> pd.DataFrame | None:
    """Extract the last `context_s` seconds from the first 2/3 of the baseline period.

    Using only the first 2/3 keeps the last 1/3 available as held-out baseline
    evaluation windows — preventing data leakage.
    """
    baseline_idx = np.where(df["label"].values == 1)[0]
    if len(baseline_idx) == 0:
        return None
    split = int(len(baseline_idx) * BASELINE_CONTEXT_FRAC)
    if split < context_s:
        # Baseline too short — use all of first 2/3
        ctx_idx = baseline_idx[:split]
    else:
        ctx_idx = baseline_idx[split - context_s : split]
    if len(ctx_idx) < 30:
        return None
    return df.iloc[ctx_idx[0] : ctx_idx[-1] + 1]


def score_subject(
    df: pd.DataFrame,
    scorer: Callable[[pd.DataFrame, pd.DataFrame], float],
    context_s: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Personal-baseline context evaluation — the correct approach for WESAD.

    Context is always the subject's own resting baseline (first 2/3 of the
    baseline period).  Evaluation windows are drawn from:
      - held-out late-baseline (label=1)  → ground-truth label 0
      - full stress period (label=2)       → ground-truth label 1

    This eliminates the cross-condition contamination that plagues rolling-window
    evaluation on the WESAD protocol (13-minute gap between baseline and stress
    means rolling context is always within-stress by evaluation time).
    """
    df = _ffill(df)
    context_df = _build_baseline_context(df, context_s)
    if context_df is None:
        return np.array([]), np.array([])

    scores, labels = [], []

    # ── Held-out baseline windows (last 1/3 of baseline period) ─────────────
    baseline_idx = np.where(df["label"].values == 1)[0]
    split        = int(len(baseline_idx) * BASELINE_CONTEXT_FRAC)
    eval_base    = baseline_idx[split:]

    for i in range(0, len(eval_base) - HORIZON_S + 1, STRIDE_S):
        chunk = eval_base[i : i + HORIZON_S]
        if len(chunk) < HORIZON_S:
            break
        obs = df.iloc[chunk[0] : chunk[-1] + 1]
        s = scorer(context_df, obs)
        if not np.isnan(s):
            scores.append(s)
            labels.append(0)

    # ── Stress windows ────────────────────────────────────────────────────────
    stress_idx = np.where(df["label"].values == 2)[0]
    for i in range(0, len(stress_idx) - HORIZON_S + 1, STRIDE_S):
        chunk = stress_idx[i : i + HORIZON_S]
        if len(chunk) < HORIZON_S:
            break
        obs = df.iloc[chunk[0] : chunk[-1] + 1]
        s = scorer(context_df, obs)
        if not np.isnan(s):
            scores.append(s)
            labels.append(1)

    return np.array(scores), np.array(labels)


def subject_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """AUROC and AUPRC for one subject; NaN if fewer than 2 classes present."""
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n_windows": len(scores)}
    return {
        "auroc":     roc_auc_score(labels, scores),
        "auprc":     average_precision_score(labels, scores),
        "n_windows": len(scores),
    }


def loso_evaluate(
    subject_dfs: dict[str, pd.DataFrame],
    scorer: Callable[[pd.DataFrame, pd.DataFrame], float],
    context_s: int = 300,
) -> pd.DataFrame:
    """Run personal-baseline-context evaluation across all subjects."""
    rows = {}
    for sid, df in subject_dfs.items():
        s, l = score_subject(df, scorer, context_s=context_s)
        rows[sid] = subject_metrics(s, l)
    return pd.DataFrame(rows).T


def compare_methods(
    results_a: pd.Series,
    results_b: pd.Series,
    method_a: str = "A",
    method_b: str = "B",
) -> dict:
    """Paired Wilcoxon signed-rank test + Cliff's delta on AUROC."""
    diff = (results_a - results_b).dropna()
    if len(diff) < 3:
        return {"method_a": method_a, "method_b": method_b,
                "p_value": float("nan"), "cliffs_delta": float("nan"),
                "median_a": results_a.median(), "median_b": results_b.median(),
                "n": len(diff)}
    stat, p = wilcoxon(diff)
    return {
        "method_a":      method_a,
        "method_b":      method_b,
        "median_a":      results_a.median(),
        "median_b":      results_b.median(),
        "wilcoxon_stat": stat,
        "p_value":       p,
        "cliffs_delta":  float(np.sum(np.sign(diff))) / len(diff),
        "n":             len(diff),
    }
