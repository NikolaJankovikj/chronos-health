from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
VALID_LABELS = {1, 2, 3, 4}  # baseline, stress, amusement, meditation
SAMPLING_RATE = 700  # Hz


def load_subject(wesad_root: Path, subject_id: int) -> dict:
    """Return the raw signal dict from SX.pkl plus the label array."""
    pkl_path = wesad_root / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data
