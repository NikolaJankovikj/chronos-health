from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

TARGET = "rmssd"
COVARIATE_COLS = ["hr_bpm", "acc_mag", "eda", "resp_rate", "temp"]

# Evaluation windows
CONTEXT_S = 300    # 5 min of historical context
HORIZON_S = 30     # 30 s prediction / scoring window
STRIDE_S  = 30     # non-overlapping horizons


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill covariate NaNs; drop the first CONTEXT_S rows (RMSSD warmup)."""
    df = df.copy()
    for col in COVARIATE_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    return df.iloc[CONTEXT_S:].reset_index(drop=True)


def score_subject(
    df: pd.DataFrame,
    scorer: Callable[[pd.DataFrame, pd.DataFrame], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Slide windows over one subject's DataFrame; return (scores, labels).

    Label encoding: 1 = stress (WESAD label 2), 0 = baseline (WESAD label 1).
    Windows where the horizon majority is neither are skipped.
    """
    df = _prep(df)
    n = len(df)
    scores, labels = [], []

    for t in range(0, n - HORIZON_S - CONTEXT_S + 1, STRIDE_S):
        context_end = t + CONTEXT_S
        observed_end = context_end + HORIZON_S
        if observed_end > n:
            break

        context_df = df.iloc[t:context_end]
        observed_df = df.iloc[context_end:observed_end]

        # Skip windows with >50% missing RMSSD in context
        if context_df[TARGET].isna().mean() > 0.5:
            continue

        horizon_labels = observed_df["label"].values
        stress_frac = (horizon_labels == 2).mean()
        base_frac   = (horizon_labels == 1).mean()
        if stress_frac >= 0.5:
            label = 1
        elif base_frac >= 0.5:
            label = 0
        else:
            continue  # skip mixed / meditation / amusement windows

        score = scorer(context_df, observed_df)
        if not np.isnan(score):
            scores.append(score)
            labels.append(label)

    return np.array(scores), np.array(labels)


def subject_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """AUROC and AUPRC for one subject; NaN if fewer than 2 classes."""
    if len(np.unique(labels)) < 2 or len(scores) == 0:
        return {"auroc": float("nan"), "auprc": float("nan"), "n_windows": len(scores)}
    return {
        "auroc":     roc_auc_score(labels, scores),
        "auprc":     average_precision_score(labels, scores),
        "n_windows": len(scores),
    }


def loso_evaluate(
    subject_dfs: dict[str, pd.DataFrame],
    scorer: Callable[[pd.DataFrame, pd.DataFrame], float],
) -> pd.DataFrame:
    """Run LOSO evaluation; return DataFrame with one row per subject."""
    rows = {}
    for sid, df in subject_dfs.items():
        scores, labels = score_subject(df, scorer)
        rows[sid] = subject_metrics(scores, labels)
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
        return {"method_a": method_a, "method_b": method_b, "p_value": float("nan"),
                "cliffs_delta": float("nan"), "median_a": results_a.median(),
                "median_b": results_b.median(), "n": len(diff)}
    stat, p = wilcoxon(diff)
    cliffs_d = float(np.sum(np.sign(diff))) / len(diff)
    return {
        "method_a":     method_a,
        "method_b":     method_b,
        "median_a":     results_a.median(),
        "median_b":     results_b.median(),
        "wilcoxon_stat": stat,
        "p_value":      p,
        "cliffs_delta": cliffs_d,
        "n":            len(diff),
    }
