"""T1.3 — Verify Chronos-2-small loads and runs on CPU within an acceptable time.

Run with:
    uv run python scripts/00_smoke_test.py
"""
import time

import numpy as np
import pandas as pd


def main() -> None:
    print("Loading autogluon/chronos-2-small on CPU ...")
    t0 = time.perf_counter()
    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        "autogluon/chronos-2-small",
        device_map="cpu",
    )
    load_time = time.perf_counter() - t0
    print(f"  loaded in {load_time:.1f}s")

    # Synthetic 5-minute RMSSD-like series at 1 Hz with one past covariate
    rng = np.random.default_rng(42)
    n = 300
    context_df = pd.DataFrame({
        "item_id": "subject_0",
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "rmssd": 30 + 5 * rng.standard_normal(n),
        "acc_mag": np.abs(rng.standard_normal(n)) * 0.1,
    })

    # warm up JIT compilation (first call is always slow due to tracing)
    print("Warming up (3 calls) ...")
    for _ in range(3):
        pipeline.predict_df(
            context_df, prediction_length=30, quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id", timestamp_column="timestamp", target="rmssd",
        )

    print("Running one forecast (horizon = 30 s) ...")
    t1 = time.perf_counter()
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=30,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="rmssd",
    )
    infer_time = time.perf_counter() - t1
    print(f"  inference time: {infer_time:.2f}s")
    print("  forecast preview:")
    print(pred_df.head(5).to_string())

    print()
    if infer_time < 5.0:
        print("PASS — inference is fast enough for Phase-1 windowed evaluation")
        estimated_wesad = 2100 * infer_time / 60
        print(f"  Estimated WESAD total (2 100 windows): ~{estimated_wesad:.0f} min")
        estimated_mmash = 127_000 * infer_time / 3600
        print(f"  Estimated MMASH naive total (127 k windows): ~{estimated_mmash:.0f} h  ← will need batching + wide stride")
    else:
        print(f"WARN — {infer_time:.1f}s per single call is slow")
        print("  Consider: larger batch_size, wider stride, or running on colab GPU for Phase-2")


if __name__ == "__main__":
    main()
