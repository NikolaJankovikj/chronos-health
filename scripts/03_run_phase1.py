"""T1.9–T1.13 — Phase-1 LOSO evaluation on WESAD.

Runs four scorers and prints a comparison table:
  1. SMA            — univariate, no fit, context-blind floor baseline
  2. ARIMAX         — SARIMAX(2,0,2) with covariates per window (information advantage: knows future covariates)
  3. Chronos-2 (no covariates) — foundation model, univariate, zero-shot ablation
  4. Chronos-2 (covariates)    — foundation model, context-aware, zero-shot (past-only covariates)

Run with:
    uv run python scripts/03_run_phase1.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesis.eval.harness import COVARIATE_COLS, compare_methods, loso_evaluate
from thesis.models import arimax, chronos2, sma

PROCESSED_DIR = Path("data/wesad_processed")
RESULTS_PATH  = Path("data/phase1_results.csv")


def load_subjects(processed_dir: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for path in sorted(processed_dir.glob("S*.parquet")):
        dfs[path.stem] = pd.read_parquet(path)
    print(f"Loaded {len(dfs)} subjects from {processed_dir}")
    return dfs


def run_scorer(name: str, scorer, subject_dfs: dict) -> pd.DataFrame:
    print(f"\n  [{name}]")
    results = loso_evaluate(subject_dfs, scorer)
    med_auroc = results["auroc"].dropna().median()
    med_auprc = results["auprc"].dropna().median()
    valid = results["auroc"].notna().sum()
    print(f"    median AUROC={med_auroc:.3f}  AUPRC={med_auprc:.3f}  ({valid}/{len(results)} subjects valid)")
    return results


def main() -> None:
    subject_dfs = load_subjects(PROCESSED_DIR)

    print("\n=== Loading Chronos-2-small pipeline ===")
    pipeline = chronos2.load_pipeline()
    print("  Pipeline ready.")

    print("\n=== Running LOSO evaluation ===")

    scorers = {
        "SMA":             sma.make_scorer(stats_window=60),
        "ARIMAX":          arimax.make_scorer(covariate_cols=COVARIATE_COLS),
        "Chronos2_noCV":   chronos2.make_scorer(pipeline, covariate_cols=[]),
        "Chronos2_CV":     chronos2.make_scorer(pipeline, covariate_cols=COVARIATE_COLS),
    }

    all_results: dict[str, pd.DataFrame] = {}
    for name, scorer in tqdm(scorers.items(), desc="scorers"):
        all_results[name] = run_scorer(name, scorer, subject_dfs)

    # ── Results table ────────────────────────────────────────────────────────
    print("\n=== Per-subject AUROC ===")
    auroc_df = pd.DataFrame({k: v["auroc"] for k, v in all_results.items()})
    print(auroc_df.round(3).to_string())

    print("\n=== Summary (median ± IQR across subjects) ===")
    for name, res in all_results.items():
        a = res["auroc"].dropna()
        p = res["auprc"].dropna()
        print(f"  {name:<20s}  AUROC {a.median():.3f} [{a.quantile(.25):.3f}–{a.quantile(.75):.3f}]"
              f"  AUPRC {p.median():.3f}")

    # ── Statistical comparisons ───────────────────────────────────────────────
    print("\n=== Pairwise comparisons (Wilcoxon signed-rank + Cliff's δ) ===")
    pairs = [
        ("Chronos2_CV",   "SMA",           "Chronos2(CV) vs SMA"),
        ("Chronos2_CV",   "ARIMAX",         "Chronos2(CV) vs ARIMAX"),
        ("Chronos2_CV",   "Chronos2_noCV",  "Chronos2(CV) vs Chronos2(noCV)  [covariate ablation]"),
        ("Chronos2_noCV", "SMA",            "Chronos2(noCV) vs SMA"),
    ]
    for a, b, label in pairs:
        cmp = compare_methods(all_results[a]["auroc"], all_results[b]["auroc"],
                              method_a=a, method_b=b)
        direction = ">" if cmp["median_a"] > cmp["median_b"] else "<"
        sig = "✓" if (not np.isnan(cmp["p_value"]) and cmp["p_value"] < 0.05) else " "
        print(f"  {sig} {label}")
        print(f"      {a}={cmp['median_a']:.3f}  {direction}  {b}={cmp['median_b']:.3f}"
              f"  p={cmp['p_value']:.3f}  δ={cmp['cliffs_delta']:.2f}  n={cmp['n']}")

    # ── Save ─────────────────────────────────────────────────────────────────
    rows = []
    for name, res in all_results.items():
        for sid, row in res.iterrows():
            rows.append({"subject": sid, "method": name, **row.to_dict()})
    out = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved → {RESULTS_PATH}")

    # ── Phase-1 success criteria ─────────────────────────────────────────────
    cv_auroc = all_results["Chronos2_CV"]["auroc"].dropna().median()
    print("\n=== Phase-1 success criteria ===")
    print(f"  1. Chronos2_CV AUROC > 0.70:  {'PASS' if cv_auroc > 0.70 else 'FAIL'}  ({cv_auroc:.3f})")
    cmp_sma   = compare_methods(all_results["Chronos2_CV"]["auroc"], all_results["SMA"]["auroc"])
    cmp_arx   = compare_methods(all_results["Chronos2_CV"]["auroc"], all_results["ARIMAX"]["auroc"])
    cmp_abl   = compare_methods(all_results["Chronos2_CV"]["auroc"], all_results["Chronos2_noCV"]["auroc"])
    cv_gt_sma = all_results["Chronos2_CV"]["auroc"].dropna().median() > all_results["SMA"]["auroc"].dropna().median()
    cv_gt_arx = all_results["Chronos2_CV"]["auroc"].dropna().median() > all_results["ARIMAX"]["auroc"].dropna().median()
    cv_gt_abl = all_results["Chronos2_CV"]["auroc"].dropna().median() > all_results["Chronos2_noCV"]["auroc"].dropna().median()
    print(f"  2. Chronos2_CV > SMA (median): {'PASS' if cv_gt_sma else 'FAIL'}")
    print(f"  3. Chronos2_CV > ARIMAX (median): {'PASS' if cv_gt_arx else 'FAIL'}")
    print(f"  4. Covariates help (CV > noCV): {'PASS' if cv_gt_abl else 'FAIL'}")


if __name__ == "__main__":
    main()
