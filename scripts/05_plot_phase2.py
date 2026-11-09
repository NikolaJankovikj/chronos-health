"""Generate Phase-2 figures from results/phase2_*.csv into results/figures/.

  fig6_phase2_injection_auroc.png  — per-subject AUROC for the injected drop, by model
  fig7_phase2_depth_sweep.png      — median AUROC vs injected depth
  fig8_phase2_activity_alarms.png  — activity false-alarm AUROC (0.5 = not flagged)
  fig9_phase2_comparisons.png      — planned comparisons with 95% CIs

Run with:
    uv run python scripts/05_plot_phase2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.eval import harness as h
from thesis.eval import phase2 as p2

RESULTS = Path("results")
OUT_DIR = RESULTS / "figures"

MODELS = [
    "SMA",
    "LinReg_cov",
    "Persistence",
    "Persistence_cov",
    "Chronos2",
    "Chronos2_cov",
]
LABELS = {
    "SMA": "SMA",
    "LinReg_cov": "OLS\n+cov",
    "Persistence": "Persistence",
    "Persistence_cov": "Persistence\n+cov",
    "Chronos2": "Chronos-2",
    "Chronos2_cov": "Chronos-2\n+cov",
}
# Fixed categorical slots 1–6 of the validated default palette (CVD-checked).
COLORS = {
    "Chronos2_cov": "#2a78d6",
    "Persistence_cov": "#eb6834",
    "LinReg_cov": "#1baf7a",
    "Chronos2": "#eda100",
    "Persistence": "#e87ba4",
    "SMA": "#008300",
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


def _box(ax, table: pd.DataFrame, ylabel: str) -> None:
    per_model = [table[m].dropna().values for m in MODELS]
    bp = ax.boxplot(
        per_model,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops={"color": "#1a1a19", "linewidth": 1.5},
    )
    rng = np.random.default_rng(0)
    for i, (m, vals) in enumerate(zip(MODELS, per_model, strict=True), start=1):
        bp["boxes"][i - 1].set(facecolor=COLORS[m], alpha=0.35, edgecolor=COLORS[m])
        ax.scatter(
            i + rng.uniform(-0.12, 0.12, len(vals)),
            vals,
            s=16,
            color=COLORS[m],
            zorder=3,
        )
    ax.axhline(0.5, color="#8a8a85", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, len(MODELS) + 1), [LABELS[m] for m in MODELS], fontsize=8)
    ax.set_ylabel(ylabel)


def fig6_injection(per_subject: pd.DataFrame) -> None:
    table = per_subject.pivot(
        index="subject", columns="model", values="auroc_injection"
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _box(
        ax, table, f"Per-subject AUROC, {p2.INJECTION_DEPTH:.0%} injected drop vs real"
    )
    ax.set_ylim(0.3, 1.02)
    ax.set_title(
        f"Phase 2 — detecting an unexplained RMSSD drop (N = {len(table)})",
        fontweight="bold",
    )
    _save(fig, "fig6_phase2_injection_auroc.png")


def fig7_sweep(sweep: pd.DataFrame) -> None:
    depths = np.array(p2.INJECTION_SWEEP) * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in MODELS:
        style = "-" if m.endswith("_cov") else "--"
        ax.plot(depths, sweep.loc[m].values, style, color=COLORS[m], linewidth=2, marker="o",
                markersize=5, label=LABELS[m].replace("\n", " "))
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Injected RMSSD drop (%)")
    ax.set_ylabel("Median per-subject AUROC")
    ax.set_title("Detection vs drop size (solid = with covariates)", fontsize=10)
    _save(fig, "fig7_phase2_depth_sweep.png")


def fig8_activity(per_subject: pd.DataFrame) -> None:
    table = per_subject.pivot(index="subject", columns="model", values="fa_active")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _box(ax, table, "AUROC, active vs sedentary real windows")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Activity false alarms (0.5 = activity not flagged; above = flagged)",
        fontsize=10,
    )
    _save(fig, "fig8_phase2_activity_alarms.png")


def _comparison_label(row: pd.Series) -> str:
    metric = "injection" if row["family"].startswith("primary") else "activity alarms"
    return f"{metric}: {LABELS[row['a']]} − {LABELS[row['b']]}".replace("\n", " ")


def _draw_comparison(ax, y: int, r: pd.Series) -> None:
    xerr = [[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]]
    ax.errorbar(r["mean_diff"], y, xerr=xerr, fmt="o", color=COLORS[r["a"]], markersize=7, capsize=4)
    ax.text(0.1, y, f"p(Holm) = {r['p_holm']:.3f} · {r['verdict']}", va="center", fontsize=8)


def fig9_comparisons(comparisons: pd.DataFrame) -> None:
    rows = comparisons.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axvspan(-h.EQUIVALENCE_MARGIN, h.EQUIVALENCE_MARGIN, color="#e6e5e0", zorder=0,
               label=f"±{h.EQUIVALENCE_MARGIN} AUROC")
    ax.axvline(0, color="#8a8a85", linewidth=0.8)
    for i, r in rows.iterrows():
        _draw_comparison(ax, i, r)
    ax.set_yticks(range(len(rows)), [_comparison_label(r) for _, r in rows.iterrows()], fontsize=8)
    ax.set_xlim(-0.1, 0.45)
    ax.set_xlabel("Mean paired difference (A − B), 95% bootstrap CI over subjects")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "fig9_phase2_comparisons.png")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name)
    plt.close(fig)
    print(f"  saved {name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_subject = pd.read_csv(RESULTS / "phase2_per_subject.csv")
    fig6_injection(per_subject)
    fig7_sweep(pd.read_csv(RESULTS / "phase2_depth_sweep.csv", index_col=0))
    fig8_activity(per_subject)
    fig9_comparisons(pd.read_csv(RESULTS / "phase2_comparisons.csv"))


if __name__ == "__main__":
    main()
