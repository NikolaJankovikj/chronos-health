from __future__ import annotations

from pathlib import Path

import pandas as pd

SUBJECT_IDS = list(range(1, 23))


def load_subject(mmash_root: Path, subject_id: int) -> dict[str, pd.DataFrame]:
    """Load all CSVs for one MMASH subject; return as named dict."""
    d = mmash_root / f"user_{subject_id}"
    return {
        "rr": pd.read_csv(d / "RR.csv"),
        "actigraph": pd.read_csv(d / "Actigraph.csv"),
        "activity": pd.read_csv(d / "Activity.csv"),
        "sleep": pd.read_csv(d / "sleep.csv"),
        "questionnaire": pd.read_csv(d / "questionnaire.csv"),
        "saliva": pd.read_csv(d / "saliva.csv"),
        "user_info": pd.read_csv(d / "user_info.csv"),
    }
