"""Generate Phase-1 result figures and save to results/figures/.

Produces five publication-quality plots:
  fig1_auroc_summary.png        — Box plot of AUROC distribution per method
  fig2_per_subject_heatmap.png  — Heatmap of per-subject AUROC for all methods
  fig3_stratified_bar.png       — Typical vs atypical subject comparison
  fig4_auprc_vs_auroc.png       — Scatter: AUPRC vs AUROC coloured by method
  fig5_score_distributions.png  — Stress vs baseline score separation on S2

Run with:
    uv run python scripts/04_plot_phase1.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_CSV  = Path("data/phase1_results.csv")
PROCESSED    = Path("data/wesad_processed")
OUT_DIR      = Path("results/figures")

METHOD_ORDER  = ["SMA", "ARIMAX", "Chronos2_noCV", "Chronos2_CV"]
METHOD_LABELS = {
    "SMA":           "SMA",
    "ARIMAX":        "ARIMAX",
    "Chronos2_noCV": "Chronos-2\n(no covariates)",
    "Chronos2_CV":   "Chronos-2\n(with covariates)",
}
METHOD_COLORS = {
    "SMA":           "#4C72B0",
    "ARIMAX":        "#DD8452",
    "Chronos2_noCV": "#55A868",
    "Chronos2_CV":   "#C44E52",
}

ATYPICAL = {"S6", "S7", "S9", "S10", "S15"}

SUBJECT_ORDER = [
    "S2","S3","S4","S5","S6","S7","S8","S9",
    "S10","S11","S13","S14","S15","S16","S17",
]

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":     150,
})


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df   = pd.read_csv(RESULTS_CSV)
    auroc = df.pivot(index="subject", columns="method", values="auroc")[METHOD_ORDER]
    auroc = auroc.reindex(SUBJECT_ORDER)
    return df, auroc


# ── Figure 1: Box plot — AUROC distribution per method ───────────────────────
def fig1_auroc_boxplot(auroc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [auroc[m].dropna().values for m in METHOD_ORDER]
    bp   = ax.boxplot(
        data, patch_artist=True, notch=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=5, alpha=0.6),
    )
    for patch, method in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.75)

    # Individual subject points (jittered)
    rng = np.random.default_rng(42)
    for i, (method, vals) in enumerate(zip(METHOD_ORDER, data), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color=METHOD_COLORS[method], s=30, zorder=3, alpha=0.8,
        )

    ax.set_xticks(range(1, len(METHOD_ORDER) + 1))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("AUROC")
    ax.set_title("Phase 1 — AUROC Distribution per Method (N=15 subjects)")
    ax.set_ylim(0.1, 1.08)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.axhline(0.7, color="grey", linestyle=":",  linewidth=0.8, label="Target (0.7)")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_auroc_summary.png")
    plt.close(fig)
    print("  Saved fig1_auroc_summary.png")


# ── Figure 2: Per-subject heatmap ────────────────────────────────────────────
def fig2_heatmap(auroc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 8))

    mat   = auroc.values.astype(float)
    im    = ax.imshow(mat, cmap="RdYlGn", vmin=0, aspect="auto")
    plt.colorbar(im, ax=ax, label="AUROC", fraction=0.03, pad=0.02)

    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(SUBJECT_ORDER)))
    ylabels = []
    for s in SUBJECT_ORDER:
        label = f"{s} *" if s in ATYPICAL else s
        ylabels.append(label)
    ax.set_yticklabels(ylabels)

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.45 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    # Mark atypical subjects with a bracket on y-axis
    ax.set_title("Per-subject AUROC  (* = atypical stress response)", fontsize=11)
    ax.tick_params(axis="y", which="both", length=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_per_subject_heatmap.png")
    plt.close(fig)
    print("  Saved fig2_per_subject_heatmap.png")


# ── Figure 3: Stratified bar chart — typical vs atypical ─────────────────────
def fig3_stratified(auroc: pd.DataFrame) -> None:
    typical  = [s for s in SUBJECT_ORDER if s not in ATYPICAL]
    atypical = [s for s in SUBJECT_ORDER if s in ATYPICAL]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    titles = [
        f"Typical subjects (n={len(typical)})\nBaseline RMSSD > Stress RMSSD",
        f"Atypical subjects (n={len(atypical)})\nStress RMSSD ≥ Baseline RMSSD",
    ]
    groups = [typical, atypical]

    for ax, group, title in zip(axes, groups, titles):
        medians = [auroc.loc[group, m].median() for m in METHOD_ORDER]
        q25     = [auroc.loc[group, m].quantile(0.25) for m in METHOD_ORDER]
        q75     = [auroc.loc[group, m].quantile(0.75) for m in METHOD_ORDER]
        yerr_lo = np.array(medians) - np.array(q25)
        yerr_hi = np.array(q75) - np.array(medians)

        x = np.arange(len(METHOD_ORDER))
        bars = ax.bar(
            x, medians,
            yerr=[yerr_lo, yerr_hi],
            color=[METHOD_COLORS[m] for m in METHOD_ORDER],
            alpha=0.8, capsize=5, error_kw=dict(linewidth=1.5),
        )
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Median AUROC (error bars = IQR)")
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)

        for bar, val in zip(bars, medians):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.04,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle("Phase 1 — AUROC by Subject Type", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_stratified_bar.png")
    plt.close(fig)
    print("  Saved fig3_stratified_bar.png")


# ── Figure 4: AUPRC vs AUROC scatter ─────────────────────────────────────────
def fig4_auprc_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        ax.scatter(
            sub["auroc"], sub["auprc"],
            label=METHOD_LABELS[method].replace("\n", " "),
            color=METHOD_COLORS[method],
            s=55, alpha=0.8, zorder=3,
        )

    ax.set_xlabel("AUROC")
    ax.set_ylabel("AUPRC")
    ax.set_title("Phase 1 — AUPRC vs AUROC per Subject per Method")
    ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(0.1, 1.05)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_auprc_vs_auroc.png")
    plt.close(fig)
    print("  Saved fig4_auprc_vs_auroc.png")


# ── Figure 5: Score distributions — stress vs baseline on S2 ─────────────────
def fig5_score_distributions() -> None:
    """Re-run scorers on S2 and plot raw score distributions per condition."""
    from thesis.eval.harness import score_subject, COVARIATE_COLS
    from thesis.models import sma, arimax, chronos2

    df_s2 = pd.read_parquet(PROCESSED / "S2.parquet")
    pipeline = chronos2.load_pipeline()

    scorers = {
        "SMA":           sma.make_scorer(),
        "ARIMAX":        arimax.make_scorer(covariate_cols=COVARIATE_COLS),
        "Chronos-2\n(no covariates)": chronos2.make_scorer(pipeline, covariate_cols=[]),
        "Chronos-2\n(with covariates)": chronos2.make_scorer(pipeline, covariate_cols=COVARIATE_COLS),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    color_stress   = "#C44E52"
    color_baseline = "#4C72B0"

    for ax, (name, scorer) in zip(axes, scorers.items()):
        scores, labels = score_subject(df_s2, scorer)
        stress_scores   = scores[labels == 1]
        baseline_scores = scores[labels == 0]

        bins = np.linspace(
            min(scores.min(), 0),
            max(scores.max(), 1),
            25,
        )
        ax.hist(baseline_scores, bins=bins, color=color_baseline,
                alpha=0.65, label=f"Baseline (n={len(baseline_scores)})")
        ax.hist(stress_scores,   bins=bins, color=color_stress,
                alpha=0.65, label=f"Stress (n={len(stress_scores)})")

        auroc_val = df_s2.pipe(lambda _: None)
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) == 2:
            auroc_val = roc_auc_score(labels, scores)
            ax.set_title(f"{name}\n(S2, AUROC={auroc_val:.3f})", fontsize=9)
        else:
            ax.set_title(name, fontsize=9)

        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Phase 1 — Anomaly Score Distributions: Stress vs Baseline (Subject S2)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_score_distributions.png")
    plt.close(fig)
    print("  Saved fig5_score_distributions.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, auroc = load()
    print("Generating Phase-1 figures ...\n")

    fig1_auroc_boxplot(auroc)
    fig2_heatmap(auroc)
    fig3_stratified(auroc)
    fig4_auprc_scatter(df)
    fig5_score_distributions()

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
