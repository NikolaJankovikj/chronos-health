"""T1.4–T1.8 — Load raw WESAD subjects, compute RMSSD + covariates, save parquet.

Run with:
    uv run python scripts/01_preprocess_wesad.py
"""
from pathlib import Path

WESAD_ROOT = Path("WESAD")
OUT_DIR = Path("data/wesad_processed")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Implementation: tasks T1.4 → T1.8


if __name__ == "__main__":
    main()
