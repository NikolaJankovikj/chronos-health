"""Phase-1 evaluation protocol: personal-baseline reference, per-subject AUROC.

Protocol (fixed before the final run, see results/phase1_results.md §1):
  * context   = last CONTEXT_S s of the first 2/3 of the subject's baseline
  * held-out  = rest of the baseline, starting GUARD_S s after the context
  * windows   = HORIZON_S s, non-overlapping, contiguous in time, one label each
  * primary   = AUROC, stress vs held-out baseline
  * secondary = AUROC, stress vs amusement + meditation (non-stress conditions
                that, like stress, lie far in time from the context)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

Scorer = Callable[[pd.DataFrame, pd.DataFrame], float]

TARGET = "rmssd"
COVARIATE_COLS = ["hr_bpm", "acc_mag", "eda", "resp_rate", "temp"]

BASELINE, STRESS, AMUSEMENT, MEDITATION = 1, 2, 3, 4
NON_STRESS = (AMUSEMENT, MEDITATION)

CONTEXT_S = 300
HORIZON_S = 30
STRIDE_S = 30
# Equal to the RMSSD window, so no beat is shared by context and held-out baseline.
GUARD_S = 60
BASELINE_CONTEXT_FRAC = 0.67
MIN_CONTEXT_S = 30

ALPHA = 0.05
# Smallest AUROC difference treated as meaningful; a 95% CI inside ±margin
# means "equivalent".  0.05 is the conventional minimal important AUROC gap.
EQUIVALENCE_MARGIN = 0.05
N_BOOTSTRAP = 10_000
MIN_PAIRS = 3  # fewest non-tied subjects for which a signed-rank test is reported


def _ffill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in COVARIATE_COLS:
        if col in out.columns:
            out[col] = out[col].ffill().bfill()
    return out


def split_baseline(
    df: pd.DataFrame, context_s: int = CONTEXT_S
) -> tuple[pd.DataFrame | None, np.ndarray]:
    """Return (context frame, held-out baseline positions); context None if too short."""
    baseline_idx = np.where(df["label"].values == BASELINE)[0]
    split = int(len(baseline_idx) * BASELINE_CONTEXT_FRAC)
    ctx_idx = baseline_idx[max(0, split - context_s) : split]
    if len(ctx_idx) < MIN_CONTEXT_S:
        return None, np.array([], dtype=int)
    return df.iloc[ctx_idx], baseline_idx[split + GUARD_S :]


def windows(idx: np.ndarray) -> Iterator[np.ndarray]:
    """Non-overlapping HORIZON_S windows tiled within each time-contiguous run of `idx`."""
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    for run in runs:
        for i in range(0, len(run) - HORIZON_S + 1, STRIDE_S):
            yield run[i : i + HORIZON_S]


def score_subject(
    df: pd.DataFrame, scorer: Scorer, context_s: int = CONTEXT_S
) -> tuple[np.ndarray, np.ndarray]:
    """Score every evaluation window; return (scores, condition labels)."""
    df = _ffill(df)
    context_df, held_out = split_baseline(df, context_s)
    if context_df is None:
        return np.array([]), np.array([])

    labels = df["label"].values
    groups = {BASELINE: held_out}
    groups.update({lab: np.where(labels == lab)[0] for lab in (STRESS, *NON_STRESS)})

    scores, window_labels = [], []
    for lab, idx in groups.items():
        for chunk in windows(idx):
            s = scorer(context_df, df.iloc[chunk])
            if not np.isnan(s):
                scores.append(s)
                window_labels.append(lab)
    return np.array(scores), np.array(window_labels)


def _auc(
    scores: np.ndarray, labels: np.ndarray, negatives: tuple[int, ...]
) -> tuple[float, float]:
    """(AUROC, AUPRC) of stress vs `negatives`; NaN if a class is missing."""
    mask = np.isin(labels, (STRESS, *negatives))
    y = labels[mask] == STRESS
    if y.all() or not y.any():
        return float("nan"), float("nan")
    return roc_auc_score(y, scores[mask]), average_precision_score(y, scores[mask])


def subject_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    auroc, auprc = _auc(scores, labels, (BASELINE,))
    auroc_ns, _ = _auc(scores, labels, NON_STRESS)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "auroc_vs_nonstress": auroc_ns,
        "n_windows": len(scores),
    }


def evaluate_subjects(
    subject_dfs: dict[str, pd.DataFrame], scorer: Scorer
) -> pd.DataFrame:
    """Run the protocol independently per subject (zero-shot / per-subject fit)."""
    return pd.DataFrame({
        sid: subject_metrics(*score_subject(df, scorer))
        for sid, df in subject_dfs.items()
    }).T


def _bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    """95% percentile CI of the mean, resampling subjects (fixed seed)."""
    rng = np.random.default_rng(0)
    boots = rng.choice(diff, size=(N_BOOTSTRAP, len(diff))).mean(axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(low), float(high)


def _signed_rank(nonzero: np.ndarray) -> tuple[float, float]:
    """(Wilcoxon p, matched-pairs rank-biserial r); NaN below MIN_PAIRS."""
    if len(nonzero) < MIN_PAIRS:
        return float("nan"), float("nan")
    ranks = rankdata(np.abs(nonzero))
    r = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    return float(wilcoxon(nonzero).pvalue), float(r)


def compare_methods(results_a: pd.Series, results_b: pd.Series) -> dict[str, float]:
    """Paired comparison of per-subject AUROC (A − B).

    mean_diff with a subject-bootstrap 95% CI; Wilcoxon signed-rank p (zero
    differences dropped, as Wilcoxon does); matched-pairs rank-biserial r.
    """
    diff = (results_a - results_b).dropna().values
    nonzero = diff[diff != 0]
    nan = float("nan")
    mean_diff, (ci_low, ci_high) = (diff.mean(), _bootstrap_ci(diff)) if len(diff) else (nan, (nan, nan))
    p_value, rank_biserial = _signed_rank(nonzero)
    return {
        "mean_diff": float(mean_diff), "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p_value, "rank_biserial": rank_biserial,
        "n": len(diff), "n_nonzero": len(nonzero),
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values; NaN entries stay NaN and do not count toward m."""
    p = np.asarray(p_values, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        adj[ok] = multipletests(p[ok], method="holm")[1]
    return adj.tolist()


def verdict(cmp: dict[str, float], p_adj: float) -> str:
    """Read a comparison against α (Holm-adjusted p) and the ±EQUIVALENCE_MARGIN.

    Significant → which method is better ("negligible" if the CI sits inside the
    margin).  Otherwise the CI can still rule out a meaningful advantage: both
    directions → "equivalent"; one direction → "A/B not meaningfully better".
    """
    m = EQUIVALENCE_MARGIN
    a_capped, b_capped = cmp["ci_high"] < m, cmp["ci_low"] > -m
    if p_adj < ALPHA:
        winner = "A better" if cmp["mean_diff"] > 0 else "B better"
        return f"{winner} (negligible)" if a_capped and b_capped else winner
    if a_capped and b_capped:
        return "equivalent"
    if a_capped:
        return "A not meaningfully better"
    if b_capped:
        return "B not meaningfully better"
    return "inconclusive"


def compare_family(table: pd.DataFrame, pairs: list[tuple[str, str]], family: str) -> pd.DataFrame:
    """Paired comparisons of per-subject metric columns, Holm-corrected within the family."""
    rows = [{"family": family, "a": a, "b": b, **compare_methods(table[a], table[b])} for a, b in pairs]
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"].tolist())
    out["verdict"] = [verdict(r, r["p_holm"]) for r in out.to_dict("records")]
    return out


def phase1_scorers(pipeline) -> dict[str, Scorer]:
    """Primary (univariate) scorers first, then the covariate variants."""
    from thesis.models import arimax, chronos2, sma

    return {
        "SMA": sma.make_scorer(),
        "ARIMA": arimax.make_scorer([]),
        "Chronos2": chronos2.make_scorer(pipeline, []),
        "ARIMAX_cov": arimax.make_scorer(COVARIATE_COLS),
        "Chronos2_cov": chronos2.make_scorer(pipeline, COVARIATE_COLS),
    }
