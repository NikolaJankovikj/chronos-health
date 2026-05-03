from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_subject(scores: pd.Series, labels: pd.Series) -> dict[str, float]:
    """Compute AUROC and AUPRC for one subject; return as dict."""
    mask = labels.notna() & scores.notna()
    y_true = labels[mask].values
    y_score = scores[mask].values
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
    }


def loso_results(per_subject: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Aggregate per-subject dicts into a results DataFrame."""
    return pd.DataFrame(per_subject).T


def compare_methods(
    results_a: pd.Series,
    results_b: pd.Series,
    method_a: str = "A",
    method_b: str = "B",
) -> dict:
    """Paired Wilcoxon signed-rank test + Cliff's delta between two AUROC series."""
    diff = results_a - results_b
    stat, p = wilcoxon(diff.dropna())
    n = len(diff.dropna())
    cliffs_d = float(np.sum(np.sign(diff.dropna()))) / n
    return {
        "method_a": method_a,
        "method_b": method_b,
        "median_a": results_a.median(),
        "median_b": results_b.median(),
        "wilcoxon_stat": stat,
        "p_value": p,
        "cliffs_delta": cliffs_d,
        "n": n,
    }
