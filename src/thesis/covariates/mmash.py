from __future__ import annotations

import pandas as pd

# Activity codes from the MMASH dataset paper — verify against Rossi et al. 2020
# TODO T2.1: confirm these by reading the MMASH paper
ACTIVITY_CODES: dict[int, str] = {}


def build(subject_data: dict[str, pd.DataFrame], start_time: pd.Timestamp) -> pd.DataFrame:
    """Align MMASH Actigraph + Activity + sleep covariates onto a 1 Hz grid.

    Columns: vector_mag, steps, hr, inclinometer, activity_code (categorical),
             is_sleeping (binary).
    """
    ...
