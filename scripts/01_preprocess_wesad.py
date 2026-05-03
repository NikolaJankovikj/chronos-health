"""T1.4–T1.8 — Preprocess all WESAD subjects → data/wesad_processed/S{id}.parquet.

Run with:
    uv run python scripts/01_preprocess_wesad.py

Verification (T1.7): prints mean RMSSD per condition for each subject.
Expected: baseline RMSSD > stress RMSSD for the majority of subjects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesis.covariates.wesad import build as build_covariates
from thesis.hrv.rmssd import extract_rr, hr_series, rmssd_series
from thesis.io.wesad import SUBJECT_IDS, load_subject

WESAD_ROOT = Path("WESAD")
OUT_DIR = Path("data/wesad_processed")
WINDOW_S = 60


def process_subject(subject_id: int) -> pd.DataFrame:
    signals = load_subject(WESAD_ROOT, subject_id)
    n_seconds = len(signals["label"]) // 700

    rr_ms, peak_times_s = extract_rr(signals["ecg"], sampling_rate=700)
    rmssd_1hz = rmssd_series(rr_ms, peak_times_s, n_seconds, window_s=WINDOW_S)
    hr_1hz = hr_series(rr_ms, peak_times_s, n_seconds)

    df = build_covariates(signals, rr_ms, peak_times_s, rmssd_1hz, hr_1hz)
    df.index.name = "second"
    return df


def verify(df: pd.DataFrame, subject_id: int) -> None:
    label_names = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
    parts = []
    for lbl, name in label_names.items():
        subset = df.loc[df["label"] == lbl, "rmssd"].dropna()
        parts.append(f"{name}={subset.mean():.1f}" if len(subset) else f"{name}=n/a")
    base = df.loc[df["label"] == 1, "rmssd"].dropna().mean()
    stress = df.loc[df["label"] == 2, "rmssd"].dropna().mean()
    ok = "✓" if (not np.isnan(base) and not np.isnan(stress) and base > stress) else "✗"
    print(f"  S{subject_id}: {ok}  " + "  ".join(parts))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(SUBJECT_IDS)} subjects → {OUT_DIR}\n")

    passed = 0
    for sid in tqdm(SUBJECT_IDS, desc="subjects"):
        df = process_subject(sid)
        df.to_parquet(OUT_DIR / f"S{sid}.parquet")
        verify(df, sid)

        base = df.loc[df["label"] == 1, "rmssd"].dropna().mean()
        stress = df.loc[df["label"] == 2, "rmssd"].dropna().mean()
        if not np.isnan(base) and not np.isnan(stress) and base > stress:
            passed += 1

    print(f"\nT1.7 sanity: baseline > stress RMSSD in {passed}/{len(SUBJECT_IDS)} subjects")
    if passed >= len(SUBJECT_IDS) * 0.6:
        print("PASS — consistent with WESAD literature (not all subjects show TSST response on HRV)")
    else:
        print("WARN — fewer than 60% show the pattern; consider inspecting R-peak quality per subject")


if __name__ == "__main__":
    main()
