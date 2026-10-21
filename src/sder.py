
"""Scheduling Decision Error Rate (SDER) for SJF and SRTF.

SDER compares pairwise scheduling priorities at decision points.

SJF:
    Compare actual and predicted burst-time priorities among
    ready processes at each non-preemptive dispatch.

SRTF:
    Compare actual and predicted remaining-time priorities
    among ready processes at each arrival/completion decision.

Tie policy:
    - Actual ties are excluded because no unique ideal order exists.
    - Correct predicted ordering contributes 0 errors.
    - Reversed predicted ordering contributes 1 error.
    - Predicted ties with unequal actual values contribute 0.5 errors.

The 0.5 tie penalty represents expected disagreement under
random tie-breaking. It does not represent the simulator's
specific deterministic tie-breaking policy.

Both functions return:
    (weighted_incorrect_pairs, eligible_pairs)
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

EPS = 1e-9
TIE_PENALTY = 0.5


def pairwise_error(actual_difference, predicted_difference):
    """Return 0, 0.5, or 1 for a pairwise comparison."""

    if abs(actual_difference) <= EPS:
        return None

    if abs(predicted_difference) <= EPS:
        return TIE_PENALTY

    if actual_difference * predicted_difference < 0:
        return 1.0

    return 0.0


def sder_sjf(df, prediction_column):
    """Calculate tie-aware SDER for non-preemptive SJF."""

    df = df.sort_values(
        ["arrival_time", "process_id"]
    ).reset_index(drop=True)

    if df.empty:
        return 0.0, 0

    arrivals = df["arrival_time"].to_numpy(float)
    actual = df["actual_burst"].to_numpy(float)
    predicted = df[prediction_column].to_numpy(float)
    ids = df["process_id"].to_numpy()

    remaining = set(range(len(df)))
    time = float(arrivals.min())

    errors = 0.0
    total = 0

    while remaining:

        ready = [
            i for i in remaining
            if arrivals[i] <= time + EPS
        ]

        if not ready:
            time = min(
                arrivals[i] for i in remaining
            )
            continue

        for ix, i in enumerate(ready):
            for j in ready[ix + 1:]:

                error = pairwise_error(
                    actual[i] - actual[j],
                    predicted[i] - predicted[j],
                )

                if error is None:
                    continue

                total += 1
                errors += error

        chosen = min(
            ready,
            key=lambda i: (
                predicted[i],
                arrivals[i],
                ids[i],
            ),
        )

        time += actual[chosen]
        remaining.remove(chosen)

    return errors, total


def sder_srtf(df, prediction_column):
    """Calculate tie-aware SDER for preemptive SRTF."""

    df = df.sort_values(
        ["arrival_time", "process_id"]
    ).reset_index(drop=True)

    n = len(df)

    if n == 0:
        return 0.0, 0

    arrivals = df["arrival_time"].to_numpy(float)
    actual = df["actual_burst"].to_numpy(float)
    predicted = df[prediction_column].to_numpy(float)
    ids = df["process_id"].to_numpy()

    remaining_actual = actual.copy()
    remaining_predicted = predicted.copy()

    time = float(arrivals[0])

    errors = 0.0
    total = 0

    while (remaining_actual > EPS).any():

        ready = [
            i for i in range(n)
            if arrivals[i] <= time + EPS
            and remaining_actual[i] > EPS
        ]

        if not ready:
            time = min(
                arrivals[i]
                for i in range(n)
                if remaining_actual[i] > EPS
                and arrivals[i] > time + EPS
            )
            continue

        for ix, i in enumerate(ready):
            for j in ready[ix + 1:]:

                error = pairwise_error(
                    remaining_actual[i] - remaining_actual[j],
                    remaining_predicted[i] - remaining_predicted[j],
                )

                if error is None:
                    continue

                total += 1
                errors += error

        chosen = min(
            ready,
            key=lambda i: (
                remaining_predicted[i],
                arrivals[i],
                ids[i],
            ),
        )

        next_arrival = min(
            (
                arrivals[i]
                for i in range(n)
                if remaining_actual[i] > EPS
                and arrivals[i] > time + EPS
            ),
            default=float("inf"),
        )

        duration = min(
            remaining_actual[chosen],
            next_arrival - time,
        )

        if duration <= EPS:
            raise RuntimeError(
                "SRTF simulation did not advance."
            )

        remaining_actual[chosen] = max(
            0.0,
            remaining_actual[chosen] - duration,
        )

        remaining_predicted[chosen] = max(
            0.0,
            remaining_predicted[chosen] - duration,
        )

        time += duration

    return errors, total


def main():
    df = pd.read_csv(
        RESULTS / "predictions.csv"
    )

    rows = []

    print("\nTIE-AWARE SCHEDULING DECISION ERROR RATE")
    print("=" * 70)

    methods = [
        ("actual_burst", "Oracle"),
        ("pred_exp", "Exponential-style baseline"),
        ("pred_lr", "Linear Regression"),
        ("pred_rf", "Random Forest"),
    ]

    for scheduler, function in [
        ("SJF", sder_sjf),
        ("SRTF", sder_srtf),
    ]:

        for column, label in methods:

            if column not in df.columns:
                continue

            errors, total = function(
                df,
                column,
            )

            rate = (
                100 * errors / total
                if total
                else float("nan")
            )

            rows.append({
                "scheduler": scheduler,
                "method": label,
                "weighted_incorrect_pairs": errors,
                "eligible_pairs": total,
                "sder_percent": rate,
                "tie_penalty": TIE_PENALTY,
            })

            print(
                f"{scheduler:4} | "
                f"{label:27} | "
                f"{errors:10.1f}/{total:10d} | "
                f"SDER={rate:7.3f}%"
            )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        RESULTS / "sder_results.csv",
        index=False,
    )

    print(
        "\nSaved results/sder_results.csv"
    )


if __name__ == "__main__":
    main()
