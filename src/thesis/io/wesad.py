from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
VALID_LABELS = frozenset({1, 2, 3, 4})
SAMPLING_RATE = 700


def load_subject(wesad_root: Path, subject_id: int) -> dict[str, np.ndarray]:
    """Load SX.pkl and return chest signals + label array, all at 700 Hz.

    Signals are already in physical units inside the pkl — no ADC conversion needed.
    Returns dict with keys: ecg, eda, resp, temp, acc, label.
    acc shape: (N, 3); all others: (N,).
    """
    pkl_path = wesad_root / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    chest = data["signal"]["chest"]
    labels = data["label"].astype(np.int32)
    # Remap ignore labels (5, 6, 7) to 0 so downstream code sees only {0,1,2,3,4}
    labels[~np.isin(labels, list(VALID_LABELS))] = 0

    return {
        "ecg":   chest["ECG"][:, 0].astype(np.float32),
        "eda":   chest["EDA"][:, 0].astype(np.float32),
        "resp":  chest["Resp"][:, 0].astype(np.float32),
        "temp":  chest["Temp"][:, 0].astype(np.float32),
        "acc":   chest["ACC"].astype(np.float32),
        "label": labels,
    }
