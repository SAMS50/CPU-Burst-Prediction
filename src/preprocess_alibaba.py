
from pathlib import Path
import heapq

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data" / "alibaba-trace-2017" / "batch_instance.csv"
)

OUTPUT = ROOT / "data" / "alibaba_workload.csv"


# Official Alibaba Cluster Trace 2017 batch-instance schema.
COLUMNS = [
    "start_timestamp",
    "end_timestamp",
    "job_id",
    "task_id",
    "machine_id",
    "status",
    "seq_no",
    "total_seq_no",
    "real_cpu_max",
    "real_cpu_avg",
    "real_mem_max",
    "real_mem_avg",
]


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Alibaba dataset not found: {SOURCE}"
        )

    selected = []
    total_rows = 0

    print("Reading Alibaba production workload trace...")

    # Process the large CSV in chunks to avoid loading
    # the entire dataset into memory at once.
    reader = pd.read_csv(
        SOURCE,
        header=None,
        names=COLUMNS,
        chunksize=100000,
        low_memory=False,
    )

    for chunk in reader:
        total_rows += len(chunk)

        # Keep successfully terminated instances.
        chunk = chunk.loc[
            chunk["status"].eq("Terminated")
        ].copy()

        numeric_columns = [
            "start_timestamp",
            "end_timestamp",
            "real_cpu_avg",
        ]

        for col in numeric_columns:
            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce",
            )

        chunk = chunk.dropna(
            subset=numeric_columns
        )

        # Remove invalid timestamps and CPU measurements.
        chunk = chunk.loc[
            (chunk["start_timestamp"] > 0)
            & (
                chunk["end_timestamp"]
                > chunk["start_timestamp"]
            )
            & (chunk["real_cpu_avg"] > 0)
        ].copy()

        # Instance elapsed duration in seconds.
        chunk["duration"] = (
            chunk["end_timestamp"]
            - chunk["start_timestamp"]
        )

        # Approximate CPU demand in CPU-core-seconds.
        #
        # This is a scheduling service-time proxy,
        # NOT a directly observed OS CPU burst.
        chunk["cpu_demand"] = (
            chunk["duration"]
            * chunk["real_cpu_avg"]
        )

        chunk = chunk.loc[
            np.isfinite(chunk["cpu_demand"])
            & (chunk["cpu_demand"] > 0)
        ].copy()

        selected.append(
            chunk[
                [
                    "start_timestamp",
                    "end_timestamp",
                    "job_id",
                    "task_id",
                    "cpu_demand",
                ]
            ]
        )

        print(
            f"Scanned {total_rows:,} source records",
            end="\r",
        )

    if not selected:
        raise RuntimeError(
            "No valid Alibaba workload records found."
        )

    data = pd.concat(
        selected,
        ignore_index=True,
    )

    # Chronological ordering ensures the train/test
    # split follows the observed timeline.
    data = data.sort_values(
        [
            "start_timestamp",
            "job_id",
            "task_id",
        ],
        kind="stable",
    ).reset_index(drop=True)
    
    sample_indices = np.linspace(
        0,
        len(data) - 1,
        num=min(3000, len(data)),
        dtype=int,
    )
    data = data.iloc[sample_indices].copy().reset_index(drop=True)

    if len(data) < 500:
        raise RuntimeError(
            "Too few valid Alibaba records."
        )

    # -------------------------------------------------
    # Leakage-safe historical feature construction
    # -------------------------------------------------
    #
    # For each instance, calculate the mean CPU demand
    # of instances that completed before its start time.
    #
    # The feature does not use the target instance's
    # future CPU demand.
    #
    # A min-heap stores (completion_time, CPU_demand).
    # -------------------------------------------------

    pending = []

    completed_count = 0
    completed_sum = 0.0

    historical = []
    completed_before = []

    for row in data.itertuples(index=False):

        start = float(row.start_timestamp)

        # Add only instances already completed.
        while pending and pending[0][0] <= start:
            _, demand = heapq.heappop(pending)

            completed_sum += demand
            completed_count += 1

        if completed_count > 0:
            historical.append(
                completed_sum / completed_count
            )
        else:
            historical.append(np.nan)

        completed_before.append(completed_count)

        # The current instance becomes available for
        # historical statistics only after completion.
        heapq.heappush(
            pending,
            (
                float(row.end_timestamp),
                float(row.cpu_demand),
            ),
        )

    data["historical_burst"] = historical
    data["completed_before"] = completed_before

    # Remove instances without completed historical data.
    data = data.dropna(
        subset=["historical_burst"]
    ).copy().reset_index(drop=True)

    if len(data) < 500:
        raise RuntimeError(
            "Too few instances have completed-history "
            "features. Increase the source sample size."
        )

    # Shift timestamps to start at zero.
    data["arrival_time"] = (
        data["start_timestamp"]
        - data["start_timestamp"].min()
    )

    # Assign sequential process IDs.
    data["process_id"] = np.arange(
        1,
        len(data) + 1,
    )

    # Estimated CPU-core-seconds as the scheduling
    # service-time target.
    data["actual_burst"] = data["cpu_demand"]

    output = data[
        [
            "process_id",
            "arrival_time",
            "historical_burst",
            "actual_burst",
            "completed_before",
        ]
    ].copy()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 60)
    print("ALIBABA PREPROCESSING COMPLETED")
    print("=" * 60)

    print(f"Scanned: {total_rows:,} source records")
    print(f"Saved: {len(output):,} valid workload instances")
    print(f"Output: {OUTPUT}")

    print(
        "Feature: mean CPU demand of previously "
        "completed instances"
    )

    print(
        "Service time: estimated CPU-core-seconds"
    )

    print(
        "Arrival time: observed instance start, "
        "shifted to zero"
    )

    print(
        "Instances without completed history "
        "were excluded"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
