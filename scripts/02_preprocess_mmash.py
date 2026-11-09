"""T2.2–T2.4 — MMASH users → data/mmash_processed/user_{id}.parquet (1 Hz).

Run with:
    uv run python scripts/02_preprocess_mmash.py

Prints per user: RR artifact rate, share of seconds with a valid RMSSD, hours
in bed, minutes of logged exercise.  Exclusion rules live in
thesis.eval.phase2 and are applied at evaluation time, so every user is saved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thesis.covariates.mmash import build
from thesis.io.mmash import SUBJECT_IDS, load_subject

MMASH_ROOT = Path("DataPaper")
OUT_DIR = Path("data/mmash_processed")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for uid in SUBJECT_IDS:
        df, artifact_frac = build(load_subject(MMASH_ROOT, uid))
        df.attrs["artifact_frac"] = artifact_frac
        df.to_parquet(OUT_DIR / f"user_{uid}.parquet")
        print(
            f"user_{uid:<2d} hours={len(df) / 3600:5.1f}  artifacts={artifact_frac:5.1%}"
            f"  rmssd_valid={np.isfinite(df['rmssd']).mean():5.1%}"
            f"  in_bed_h={df['sleeping'].sum() / 3600:4.1f}  exercise_min={df['exercise'].sum() / 60:5.0f}"
        )


if __name__ == "__main__":
    main()
