"""T2.2–T2.4 — Load raw MMASH subjects, compute RMSSD + covariates, save parquet.

Run with:
    uv run python scripts/02_preprocess_mmash.py
"""
from pathlib import Path

MMASH_ROOT = Path("DataPaper")
OUT_DIR = Path("data/mmash_processed")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Implementation: tasks T2.2 → T2.4


if __name__ == "__main__":
    main()
