
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def simulate_srtf(df, prediction_column):
    """
    Preemptive SRTF simulation.

    Scheduling priority is based on predicted remaining time.
    Actual burst time determines when a process completes.
    """
    required = {"process_id", "arrival_time", "actual_burst", prediction_column}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    processes = df.copy().sort_values(
        ["arrival_time", "process_id"]
    ).reset_index(drop=True)

    if processes.empty:
        raise ValueError("Workload is empty")

    if (processes["actual_burst"] <= 0).any():
        raise ValueError("Actual burst times must be positive")

    if (processes[prediction_column] <= 0).any():
        raise ValueError("Predicted burst times must be positive")

    if (processes["arrival_time"] < 0).any():
        raise ValueError("Arrival times cannot be negative")

    n = len(processes)

    arrival = processes["arrival_time"].to_numpy(dtype=float)
    actual = processes["actual_burst"].to_numpy(dtype=float)
    predicted = processes[prediction_column].to_numpy(dtype=float)
    ids = processes["process_id"].to_numpy()

    remaining_actual = actual.copy()
    remaining_predicted = predicted.copy()

    start = [None] * n
    completion = [None] * n

    time = float(arrival[0])
    completed = 0
    current = None
    events = []
    epsilon = 1e-9

    while completed < n:
        ready = [
            i for i in range(n)
            if arrival[i] <= time + epsilon
            and remaining_actual[i] > epsilon
        ]

        if not ready:
            future = [
                arrival[i] for i in range(n)
                if remaining_actual[i] > epsilon
                and arrival[i] > time + epsilon
            ]

            if not future:
                break

            time = min(future)
            current = None
            continue

        chosen = min(
            ready,
            key=lambda i: (
                max(0.0, remaining_predicted[i]),
                arrival[i],
                ids[i]
            )
        )

        if start[chosen] is None:
            start[chosen] = time

        next_arrival = min(
            (
                arrival[i] for i in range(n)
                if remaining_actual[i] > epsilon
                and arrival[i] > time + epsilon
            ),
            default=float("inf")
        )

        run_for = min(
            remaining_actual[chosen],
            next_arrival - time
        )

        if run_for <= epsilon:
            raise RuntimeError("Simulation failed to advance")

        events.append({
            "process_id": ids[chosen],
            "start_time": time,
            "end_time": time + run_for,
            "prediction_method": prediction_column
        })

        remaining_actual[chosen] -= run_for
        remaining_predicted[chosen] = max(
            0.0, remaining_predicted[chosen] - run_for
        )

        time += run_for

        if remaining_actual[chosen] <= epsilon:
            remaining_actual[chosen] = 0.0
            completion[chosen] = time
            completed += 1

        current = chosen

    records = []

    for i in range(n):
        turnaround = completion[i] - arrival[i]
        waiting = turnaround - actual[i]
        response = start[i] - arrival[i]

        records.append({
            "process_id": ids[i],
            "arrival_time": arrival[i],
            "actual_burst": actual[i],
            "predicted_burst": predicted[i],
            "start_time": start[i],
            "completion_time": completion[i],
            "waiting_time": waiting,
            "turnaround_time": turnaround,
            "response_time": response,
            "prediction_method": prediction_column
        })

    return pd.DataFrame(records), pd.DataFrame(events)


def main():
    RESULTS.mkdir(exist_ok=True)

    predictions_path = RESULTS / "predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            "Run python src/predict.py first"
        )

    df = pd.read_csv(predictions_path)

    all_results = []
    all_events = []

    methods = [
        ("actual_burst", "Oracle SRTF"),
        ("pred_exp", "Exponential Averaging SRTF"),
        ("pred_lr", "Linear Regression SRTF")
    ]

    print("\nSRTF SCHEDULING RESULTS")
    print("=" * 65)

    for column, name in methods:
        result, events = simulate_srtf(df, column)

        all_results.append(result)
        all_events.append(events)

        print(f"\n{name}")
        print(f"Average Waiting Time: {result['waiting_time'].mean():.4f}")
        print(f"Average Turnaround Time: {result['turnaround_time'].mean():.4f}")
        print(f"Average Response Time: {result['response_time'].mean():.4f}")
        print(f"Execution Segments: {len(events)}")

    pd.concat(all_results, ignore_index=True).to_csv(
        RESULTS / "srtf_results.csv", index=False
    )

    pd.concat(all_events, ignore_index=True).to_csv(
        RESULTS / "srtf_events.csv", index=False
    )

    print("\nSaved results/srtf_results.csv")
    print("Saved results/srtf_events.csv")


if __name__ == "__main__":
    main()
