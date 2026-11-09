"""T2.5–T2.8 — Phase-2 evaluation on MMASH (protocol: thesis.eval.phase2).

Primary (Holm family): does Chronos-2 + covariates detect an injected,
covariate-unexplained RMSSD drop better than Chronos-2 alone and than the
matched classical model (persistence + covariates)?
Secondary (Holm family): do covariates reduce activity false alarms, within
each model family (static, persistence, Chronos-2)?
Exploratory: per-subject mean sedentary score vs Daily Stress Inventory and
cortisol (wake-up minus bedtime).

Run with:
    uv run python scripts/04_run_phase2.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from thesis.eval import harness as h
from thesis.eval import phase2 as p2
from thesis.io.mmash import load_subject
from thesis.models import chronos2

PROCESSED_DIR = Path("data/mmash_processed")
MMASH_ROOT = Path("DataPaper")
OUT_DIR = Path("results")
WINDOWS_PATH = Path("data/phase2_windows.parquet")

MODELS = ["SMA", "LinReg_cov", "Persistence", "Persistence_cov", "Chronos2", "Chronos2_cov"]
PRIMARY = [("Chronos2_cov", "Chronos2"), ("Chronos2_cov", "Persistence_cov")]
SECONDARY = [("Chronos2_cov", "Chronos2"), ("Persistence_cov", "Persistence"), ("LinReg_cov", "SMA")]


def load_eligible() -> dict[str, pd.DataFrame]:
    paths = sorted(
        PROCESSED_DIR.glob("user_*.parquet"), key=lambda p: int(p.stem.split("_")[1])
    )
    eligible = {}
    for path in paths:
        df = pd.read_parquet(path)
        reason = p2.exclusion_reason(df)
        if reason:
            print(f"  excluded {path.stem}: {reason}")
        else:
            eligible[path.stem] = df
    print(f"{len(eligible)} subjects eligible")
    return eligible


def per_subject_metrics(windows: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"subject": sid, "model": model, **p2.subject_metrics(w)}
        for (sid, model), w in windows.groupby(["subject", "model"])
    ]
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "auroc_injection",
        "auroc_injection_sedentary",
        "auroc_injection_active",
        "auroc_injection_sleep",
        "fa_active",
        "fa_sleep",
    ]
    return metrics.groupby("model")[cols].median().reindex(MODELS)


def secondary_direction(comparisons: pd.DataFrame) -> pd.DataFrame:
    """For false alarms lower is the desired direction, so state it plainly."""
    out = comparisons.copy()
    significant = out["p_holm"] < h.ALPHA
    out["verdict"] = np.where(
        ~significant,
        "no significant difference",
        np.where(
            out["mean_diff"] < 0,
            "fewer activity alarms in A",
            "more activity alarms in A",
        ),
    )
    return out


def subject_outcomes(subject_ids: list[str]) -> pd.DataFrame:
    """Daily Stress Inventory and cortisol change (wake-up − bedtime) per subject."""
    rows = {}
    for sid in subject_ids:
        subj = load_subject(MMASH_ROOT, int(sid.split("_")[1]))
        saliva = subj["saliva"]
        cortisol = saliva.set_index("SAMPLES")["Cortisol NORM"] if len(saliva) else pd.Series(dtype=float)
        rows[sid] = {"daily_stress": subj["questionnaire"]["Daily_stress"].iloc[0],
                     "cortisol_change": cortisol.get("wake up", np.nan) - cortisol.get("before sleep", np.nan)}
    return pd.DataFrame(rows).T.astype(float)


def exploratory(metrics: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ of each model's mean sedentary score with each outcome (uncorrected)."""
    outcomes = subject_outcomes(list(metrics["subject"].unique()))
    rows = []
    for model, m in metrics.groupby("model"):
        joined = m.set_index("subject")[["mean_sedentary_score"]].join(outcomes)
        for outcome in outcomes.columns:
            pair = joined[["mean_sedentary_score", outcome]].dropna()
            rho, p = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append({"model": model, "outcome": outcome, "spearman_rho": rho, "p_uncorrected": p, "n": len(pair)})
    return pd.DataFrame(rows)


def build_tables(windows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = per_subject_metrics(windows)
    wide = {col: metrics.pivot(index="subject", columns="model", values=col) for col in ("auroc_injection", "fa_active")}
    primary = h.compare_family(wide["auroc_injection"], PRIMARY, "primary: injection AUROC")
    secondary = secondary_direction(h.compare_family(wide["fa_active"], SECONDARY, "secondary: activity false alarms"))
    sweep_cols = [f"auroc_depth_{d}" for d in p2.INJECTION_SWEEP]
    return {
        "per_subject": metrics,
        "summary": summarize(metrics),
        "comparisons": pd.concat([primary, secondary], ignore_index=True),
        "depth_sweep": metrics.groupby("model")[sweep_cols].median().reindex(MODELS),
        "exploratory": exploratory(metrics),
        "window_counts": windows[windows["model"] == "SMA"].groupby(["subject", "stratum"]).size().unstack(fill_value=0),
    }


def report(tables: dict[str, pd.DataFrame]) -> None:
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        for name in ("window_counts", "summary", "comparisons", "depth_sweep", "exploratory"):
            print(f"\n=== {name} ===\n{tables[name]}")
    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"phase2_{name}.csv", index=name in ("summary", "depth_sweep", "window_counts"))
    print(f"\nTables saved → {OUT_DIR}/phase2_*.csv; windows → {WINDOWS_PATH}")


def main() -> None:
    subjects = load_eligible()
    forecasters = p2.phase2_forecasters(chronos2.load_pipeline())
    scored = [p2.score_subject(df, forecasters).assign(subject=sid) for sid, df in tqdm(subjects.items(), desc="subjects")]
    windows = pd.concat(scored, ignore_index=True)
    windows.to_parquet(WINDOWS_PATH)
    report(build_tables(windows))


if __name__ == "__main__":
    main()
