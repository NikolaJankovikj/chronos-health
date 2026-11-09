from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBJECT_IDS = list(range(1, 23))
# Users 8 and 9 encode the second recording day as -29 (a month-rollover bug).
DAY_FIXES = {-29: 2}
OPTIONAL_FILES = ("saliva",)  # user 21 has no saliva.csv


def load_subject(mmash_root: Path, subject_id: int) -> dict[str, pd.DataFrame]:
    """Load all CSVs for one MMASH subject; a missing optional file loads as an empty frame."""
    d = mmash_root / f"user_{subject_id}"
    names = ["RR", "Actigraph", "Activity", "sleep", "questionnaire", "saliva", "user_info"]
    out = {}
    for name in names:
        path = d / f"{name}.csv"
        key = name.lower()
        if not path.exists() and key in OPTIONAL_FILES:
            out[key] = pd.DataFrame()
            continue
        out[key] = pd.read_csv(path, index_col=0)
    return out


def seconds(day: pd.Series, time: pd.Series) -> np.ndarray:
    """Seconds since midnight of recording day 1; accepts HH:MM or HH:MM:SS."""
    clock = time.astype(str).where(time.astype(str).str.count(":") == 2, time.astype(str) + ":00")
    days = day.replace(DAY_FIXES).astype(int)
    return ((days - 1) * 86_400 + pd.to_timedelta(clock).dt.total_seconds()).to_numpy()
