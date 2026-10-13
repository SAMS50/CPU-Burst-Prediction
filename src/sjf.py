
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "results" / "predictions.csv"
OUTPUT_FILE = ROOT / "results" / "sjf_results.csv"


def simulate_sjf(df, prediction_column):
    processes = df.copy()
    completed = set()
    results = []
    current_time = 0

    while len(completed) < len(processes):

        ready = processes[
            (~processes["process_id"].isin(completed)) &
            (processes["arrival_time"] <= current_time)
        ]

        if ready.empty:
            remaining = processes[
                ~processes["process_id"].isin(completed)
            ]
            current_time = remaining["arrival_time"].min()
            continue

        selected = ready.sort_values(
            by=[prediction_column, "arrival_time", "process_id"]
        ).iloc[0]

        pid = int(selected["process_id"])
        arrival = selected["arrival_time"]
        actual_burst = selected["actual_burst"]

        start = current_time
        completion = start + actual_burst
        turnaround = completion - arrival
        waiting = start - arrival
        response = waiting

        results.append({
            "process_id": pid,
            "arrival_time": arrival,
            "actual_burst": actual_burst,
            "start_time": start,
            "completion_time": completion,
            "waiting_time": waiting,
            "turnaround_time": turnaround,
            "response_time": response,
            "method": prediction_column
        })

        completed.add(pid)
        current_time = completion

    return pd.DataFrame(results)


def main():
    df = pd.read_csv(INPUT_FILE)

    methods = {
        "actual_burst": "Oracle SJF",
        "pred_exp": "Exponential Averaging",
        "pred_lr": "Linear Regression"
    }

    all_results = []

    for column, name in methods.items():
        result = simulate_sjf(df, column)
        all_results.append(result)

        print(f"\n{name}")
        print("-" * 35)
        print(
            "Average waiting time:",
            round(result["waiting_time"].mean(), 4)
        )
        print(
            "Average turnaround time:",
            round(result["turnaround_time"].mean(), 4)
        )
        print(
            "Average response time:",
            round(result["response_time"].mean(), 4)
        )

        print("\nFirst 5 scheduled processes:")
        print(result.head().to_string(index=False))

    combined = pd.concat(all_results, ignore_index=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nResults saved to results/sjf_results.csv")


if __name__ == "__main__":
    main()
