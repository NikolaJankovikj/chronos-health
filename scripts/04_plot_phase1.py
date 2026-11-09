"""Generate Phase-1 figures from results/phase1_*.csv into results/figures/.

  fig1_auroc_summary.png        — per-subject AUROC per method: primary and non-stress control
  fig2_per_subject_heatmap.png  — per-subject primary AUROC
  fig3_stratified_bar.png       — typical vs atypical subjects
  fig4_paired_differences.png   — planned comparisons: mean ΔAUROC, 95% CI, equivalence band
  fig5_score_distributions.png  — window scores by condition for one subject (re-runs models)

Run with:
    uv run python scripts/04_plot_phase1.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.eval import harness as h

RESULTS = Path("results")
PROCESSED = Path("data/wesad_processed")
OUT_DIR = RESULTS / "figures"
EXAMPLE_SUBJECT = "S2"
DARK_CELL_DISTANCE = 0.35  # |AUROC − 0.5| beyond which heatmap cells get white text

METHODS = ["SMA", "ARIMA", "Chronos2", "ARIMAX_cov", "Chronos2_cov"]
LABELS = {
    "SMA": "SMA",
    "ARIMA": "ARIMA",
    "Chronos2": "Chronos-2",
    "ARIMAX_cov": "ARIMAX\n(+covariates)",
    "Chronos2_cov": "Chronos-2\n(+covariates)",
}
# Fixed categorical slots 1–5 of the validated default palette (CVD-checked).
COLORS = {
    "Chronos2": "#2a78d6",
    "ARIMA": "#eb6834",
    "SMA": "#1baf7a",
    "Chronos2_cov": "#eda100",
    "ARIMAX_cov": "#e87ba4",
}
# Conditions in fig5; line style is the secondary encoding.
CONDITIONS = {
    h.BASELINE: ("Held-out baseline", "#6250d6", "-"),
    h.STRESS: ("Stress", "#e34948", "--"),
    h.AMUSEMENT: ("Amusement + meditation", "#8a8a85", ":"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})


def subject_order() -> list[str]:
    return sorted(
        {p.stem for p in PROCESSED.glob("S*.parquet")}, key=lambda s: int(s[1:])
    )


def atypical_subjects() -> set[str]:
    """Subjects whose mean stress RMSSD is not below their mean baseline RMSSD."""
    out = set()
    for sid in subject_order():
        df = pd.read_parquet(PROCESSED / f"{sid}.parquet")
        means = df.groupby("label")["rmssd"].mean()
        if means[h.STRESS] >= means[h.BASELINE]:
            out.add(sid)
    return out


def metric_table(per_subject: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = per_subject.pivot(index="subject", columns="method", values=metric)
    return table.reindex(index=subject_order(), columns=METHODS).astype(float)


def _box(ax, table: pd.DataFrame, title: str) -> None:
    per_method = [table[m].dropna().values for m in METHODS]
    bp = ax.boxplot(
        per_method,
        patch_artist=True,
        widths=0.5,
        showfliers=False,  # every subject is drawn as a jittered point
        medianprops={"color": "#1a1a19", "linewidth": 1.5},
    )
    rng = np.random.default_rng(0)
    for i, (m, vals) in enumerate(zip(METHODS, per_method, strict=True), start=1):
        bp["boxes"][i - 1].set(facecolor=COLORS[m], alpha=0.35, edgecolor=COLORS[m])
        ax.scatter(
            i + rng.uniform(-0.12, 0.12, len(vals)),
            vals,
            s=18,
            color=COLORS[m],
            zorder=3,
        )
    ax.axhline(0.5, color="#8a8a85", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, len(METHODS) + 1), [LABELS[m] for m in METHODS], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10)


def fig1_summary(per_subject: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    _box(
        axes[0],
        metric_table(per_subject, "auroc"),
        "Primary: stress vs held-out baseline",
    )
    _box(
        axes[1],
        metric_table(per_subject, "auroc_vs_nonstress"),
        "Control: stress vs amusement + meditation",
    )
    axes[0].set_ylabel("Per-subject AUROC (dashed = chance)")
    fig.suptitle(
        f"Phase 1 — AUROC per method (N = {len(subject_order())} subjects)",
        fontweight="bold",
    )
    _save(fig, "fig1_auroc_summary.png")


def _annotate_cells(ax, values: np.ndarray) -> None:
    for (i, j), val in np.ndenumerate(values):
        if np.isnan(val):
            continue
        color = "white" if abs(val - 0.5) > DARK_CELL_DISTANCE else "#1a1a19"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)


def _draw_comparison(ax, y: int, r: pd.Series) -> None:
    """One forest-plot row: mean difference, CI whiskers, Holm p and verdict."""
    xerr = [[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]]
    ax.errorbar(r["mean_diff"], y, xerr=xerr, fmt="o", color=COLORS[r["a"]],
                markersize=8, capsize=4, linewidth=2)
    ax.text(max(r["ci_high"], h.EQUIVALENCE_MARGIN) + 0.01, y,
            f"p(Holm) = {r['p_holm']:.3f} · {r['verdict']}", va="center", fontsize=8)


def fig2_heatmap(per_subject: pd.DataFrame) -> None:
    table = metric_table(per_subject, "auroc")
    atypical = atypical_subjects()
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(
        table.values, cmap="PuOr", vmin=0, vmax=1, aspect="auto"
    )  # diverging at chance
    fig.colorbar(im, ax=ax, label="AUROC", fraction=0.04, pad=0.02)
    ax.set_xticks(
        range(len(METHODS)),
        [LABELS[m] for m in METHODS],
        rotation=20,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(
        range(len(table)), [f"{s} *" if s in atypical else s for s in table.index]
    )
    _annotate_cells(ax, table.values)
    ax.grid(False)
    ax.set_title(
        "Per-subject AUROC, stress vs held-out baseline (* = atypical)", fontsize=10
    )
    _save(fig, "fig2_per_subject_heatmap.png")


def fig3_stratified(per_subject: pd.DataFrame) -> None:
    table = metric_table(per_subject, "auroc")
    atypical = atypical_subjects()
    groups = {
        "Typical (stress RMSSD < baseline)": [
            s for s in table.index if s not in atypical
        ],
        "Atypical (stress RMSSD ≥ baseline)": [s for s in table.index if s in atypical],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (title, subjects) in zip(axes, groups.items(), strict=True):
        sub = table.loc[subjects]
        med, q25, q75 = sub.median(), sub.quantile(0.25), sub.quantile(0.75)
        x = np.arange(len(METHODS))
        ax.bar(
            x,
            med,
            width=0.6,
            color=[COLORS[m] for m in METHODS],
            yerr=[med - q25, q75 - med],
            capsize=4,
            error_kw={"linewidth": 1},
        )
        for xi, v, top in zip(x, med, q75, strict=True):
            ax.text(xi, top + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x, [LABELS[m] for m in METHODS], fontsize=8)
        ax.set_title(f"{title}, n = {len(subjects)}", fontsize=10)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("Median AUROC (bars = IQR)")
    _save(fig, "fig3_stratified_bar.png")


def fig4_paired_differences(comparisons: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rows = comparisons.iloc[::-1].reset_index(drop=True)
    ax.axvspan(
        -h.EQUIVALENCE_MARGIN,
        h.EQUIVALENCE_MARGIN,
        color="#e6e5e0",
        zorder=0,
        label=f"Equivalence margin ±{h.EQUIVALENCE_MARGIN}",
    )
    ax.axvline(0, color="#8a8a85", linewidth=0.8)
    for i, r in rows.iterrows():
        _draw_comparison(ax, i, r)
    ax.set_yticks(
        range(len(rows)),
        [
            f"{LABELS[a]} − {LABELS[b]}".replace("\n", " ")
            for a, b in zip(rows["a"], rows["b"], strict=True)
        ],
        fontsize=8,
    )
    ax.set_xlabel("Mean paired difference in AUROC (95% bootstrap CI over subjects)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(min(-0.4, rows["ci_low"].min() - 0.02), 0.8)  # right side leaves room for the verdict text
    _save(fig, "fig4_paired_differences.png")


def fig5_score_distributions() -> None:
    from thesis.models import chronos2

    df = pd.read_parquet(PROCESSED / f"{EXAMPLE_SUBJECT}.parquet")
    scorers = h.phase1_scorers(chronos2.load_pipeline())
    fig, axes = plt.subplots(1, len(scorers), figsize=(16, 3.8))
    for ax, (name, scorer) in zip(axes, scorers.items(), strict=True):
        scores, labels = h.score_subject(df, scorer)
        labels = np.where(np.isin(labels, h.NON_STRESS), h.AMUSEMENT, labels)
        bins = np.linspace(0, np.percentile(scores, 99), 25)
        for lab, (text, color, style) in CONDITIONS.items():
            ax.hist(
                scores[labels == lab],
                bins=bins,
                histtype="step",
                linewidth=2,
                color=color,
                linestyle=style,
                label=f"{text} (n={np.sum(labels == lab)})",
            )
        ax.set_title(LABELS[name].replace("\n", " "), fontsize=9)
        ax.set_xlabel("Anomaly score (σ units)")
    axes[0].set_ylabel("Windows")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"Window anomaly scores by condition — subject {EXAMPLE_SUBJECT}",
        fontweight="bold",
    )
    _save(fig, "fig5_score_distributions.png")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name)
    plt.close(fig)
    print(f"  saved {name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_subject = pd.read_csv(RESULTS / "phase1_per_subject.csv")
    comparisons = pd.read_csv(RESULTS / "phase1_comparisons.csv")
    fig1_summary(per_subject)
    fig2_heatmap(per_subject)
    fig3_stratified(per_subject)
    fig4_paired_differences(comparisons)
    fig5_score_distributions()


if __name__ == "__main__":
    main()
