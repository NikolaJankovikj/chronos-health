"""T1.9–T1.13 — Phase-1 evaluation on WESAD (protocol: thesis.eval.harness).

Primary comparison (univariate, identical input — the context RMSSD only):
  SMA, ARIMA(2,0,2), Chronos-2-small zero-shot.
Secondary (covariate effect): ARIMAX and Chronos-2 given all five covariates,
as history and as concurrent (known-future) values, plus a diagnostic of how
far those covariates move outside the range the context shows.

Run with:
    uv run python scripts/03_run_phase1.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from thesis.eval import harness as h
from thesis.models import chronos2

PROCESSED_DIR = Path("data/wesad_processed")
OUT_DIR = Path("results")

PRIMARY = [("Chronos2", "SMA"), ("Chronos2", "ARIMA")]
SECONDARY = [("Chronos2_cov", "Chronos2"), ("ARIMAX_cov", "ARIMA")]
TARGET_AUROC = 0.70  # PLAN.md §5.5 criterion 1


def load_subjects() -> dict[str, pd.DataFrame]:
    paths = sorted(PROCESSED_DIR.glob("S*.parquet"), key=lambda p: int(p.stem[1:]))
    return {p.stem: pd.read_parquet(p) for p in paths}


def summarize(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        a = res["auroc"].astype(float)
        rows[name] = {
            "n_subjects": int(a.notna().sum()),
            "auroc_median": a.median(),
            "auroc_q25": a.quantile(0.25),
            "auroc_q75": a.quantile(0.75),
            "auprc_median": res["auprc"].astype(float).median(),
            "auroc_vs_nonstress_median": res["auroc_vs_nonstress"]
            .astype(float)
            .median(),
        }
    return pd.DataFrame(rows).T


def compare_family(
    auroc: pd.DataFrame, pairs: list[tuple[str, str]], family: str
) -> pd.DataFrame:
    """Paired comparisons with Holm correction inside the family."""
    rows = [
        {"family": family, "a": a, "b": b, **h.compare_methods(auroc[a], auroc[b])}
        for a, b in pairs
    ]
    df = pd.DataFrame(rows)
    df["p_holm"] = h.holm(df["p_value"].tolist())
    df["verdict"] = [h.verdict(r, r["p_holm"]) for r in df.to_dict("records")]
    return df


def covariate_ood(subject_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per covariate: share of stress-period values outside the context's [min, max],
    and the stress-period mean shift in context-std units (median over subjects)."""
    rows = []
    for sid, df in subject_dfs.items():
        df = h._ffill(df)
        ctx, _ = h.split_baseline(df)
        stress = df.loc[df["label"] == h.STRESS]
        for col in h.COVARIATE_COLS:
            outside = (stress[col] < ctx[col].min()) | (stress[col] > ctx[col].max())
            shift = abs(stress[col].mean() - ctx[col].mean()) / ctx[col].std()
            rows.append({
                "subject": sid,
                "covariate": col,
                "frac_outside": outside.mean(),
                "shift_sd": shift,
            })
    return (
        pd.DataFrame(rows).groupby("covariate")[["frac_outside", "shift_sd"]].median()
    )


def print_criteria(
    summary: pd.DataFrame, primary: pd.DataFrame, secondary: pd.DataFrame
) -> None:
    """PLAN.md §5.5, judged on Holm-adjusted tests (superiority needs p_holm < α)."""
    med = summary.loc["Chronos2", "auroc_median"]
    print("\n=== Phase-1 success criteria ===")
    print(
        f"  1. Chronos2 median AUROC > {TARGET_AUROC}: {'PASS' if med > TARGET_AUROC else 'FAIL'} ({med:.3f})"
    )
    for i, row in enumerate(
        [*primary.to_dict("records"), secondary.to_dict("records")[0]], start=2
    ):
        ok = row["verdict"].startswith("A better")
        print(
            f"  {i}. {row['a']} better than {row['b']}: {'PASS' if ok else 'FAIL'} ({row['verdict']})"
        )


def _with_flat_rmssd(scorer: h.Scorer) -> h.Scorer:
    def flat(context_df: pd.DataFrame, observed_df: pd.DataFrame) -> float:
        return scorer(context_df, observed_df.assign(rmssd=context_df["rmssd"].mean()))

    return flat


def hrv_dependence(subject_dfs: dict[str, pd.DataFrame], scorers: dict[str, h.Scorer]) -> pd.DataFrame:
    """Median AUROC of each covariate model when the observed RMSSD is replaced by the
    context mean.  Staying high means the detector separates stress from the covariates
    alone, i.e. it is not detecting an HRV anomaly."""
    rows = {}
    for name in ("ARIMAX_cov", "Chronos2_cov"):
        flat = h.evaluate_subjects(subject_dfs, _with_flat_rmssd(scorers[name]))
        rows[name] = {"auroc_flat_rmssd_median": flat["auroc"].astype(float).median()}
    return pd.DataFrame(rows).T


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"phase1_{name}.csv", index=name in ("summary", "covariate_ood", "hrv_dependence"))
    print(f"\nTables saved → {OUT_DIR}/phase1_*.csv")


def print_tables(tables: dict[str, pd.DataFrame]) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        for name in ("auroc", "summary", "comparisons", "covariate_ood", "hrv_dependence"):
            print(f"\n=== {name} ===\n{tables[name]}")


def main() -> None:
    subject_dfs = load_subjects()
    print(f"Loaded {len(subject_dfs)} subjects")
    scorers = h.phase1_scorers(chronos2.load_pipeline())
    results = {name: h.evaluate_subjects(subject_dfs, s) for name, s in tqdm(scorers.items(), desc="scorers")}

    auroc = pd.DataFrame({k: v["auroc"].astype(float) for k, v in results.items()})
    summary = summarize(results)
    primary = compare_family(auroc, PRIMARY, "primary")
    secondary = compare_family(auroc, SECONDARY, "secondary")
    tables = {
        "auroc": auroc,
        "summary": summary,
        "comparisons": pd.concat([primary, secondary], ignore_index=True),
        "covariate_ood": covariate_ood(subject_dfs),
        "hrv_dependence": hrv_dependence(subject_dfs, scorers),
    }
    print_tables(tables)
    print_criteria(summary, primary, secondary)
    tables["per_subject"] = pd.concat(results, names=["method", "subject"]).reset_index()
    save_tables({k: v for k, v in tables.items() if k != "auroc"})


if __name__ == "__main__":
    main()
