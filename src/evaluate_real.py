
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sjf import simulate_sjf
from srtf import simulate_srtf

DATA = ROOT / "data" / "real_workload.csv"
OUT = ROOT / "results" / "real_workload"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA)
    df = df.sort_values(
        ["arrival_time", "process_id"]
    ).reset_index(drop=True)

    required = [
        "process_id", "arrival_time",
        "historical_burst", "actual_burst"
    ]
    if df[required].isna().any().any():
        raise ValueError("Dataset contains missing values")

    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    if len(test) == 0:
        raise ValueError("Not enough test records")

    # Same baseline methodology as synthetic experiment
    alpha = 0.5
    train_mean = train["actual_burst"].mean()

    test["pred_exp"] = (
        alpha * test["historical_burst"]
        + (1 - alpha) * train_mean
    )

    model = LinearRegression()
    model.fit(
        train[["historical_burst"]],
        train["actual_burst"]
    )

    test["pred_lr"] = model.predict(
        test[["historical_burst"]]
    )

    test["pred_exp"] = test["pred_exp"].clip(lower=0.001)
    test["pred_lr"] = test["pred_lr"].clip(lower=0.001)

    test.to_csv(OUT / "predictions.csv", index=False)

    print("\nREAL MEASURED CPU-TIME EXPERIMENT")
    print("=" * 55)
    print(f"Training records: {len(train)}")
    print(f"Testing records: {len(test)}")

    for column, label in [
        ("pred_exp", "Exponential Averaging"),
        ("pred_lr", "Linear Regression")
    ]:
        mae = mean_absolute_error(
            test["actual_burst"], test[column]
        )
        rmse = np.sqrt(mean_squared_error(
            test["actual_burst"], test[column]
        ))
        print(f"\n{label}")
        print(f"MAE: {mae:.4f} ms")
        print(f"RMSE: {rmse:.4f} ms")

    summary = []
    all_sjf = []
    all_srtf = []

    for scheduler, simulator, collection in [
        ("SJF", simulate_sjf, all_sjf),
        ("SRTF", simulate_srtf, all_srtf)
    ]:
        for column, label in [
            ("actual_burst", "Oracle"),
            ("pred_exp", "Exponential Averaging"),
            ("pred_lr", "Linear Regression")
        ]:
            result = simulator(test, column)

            # SRTF returns (metrics, events).
            if isinstance(result, tuple):
                result = result[0]

            collection.append(result)

            record = {
                "scheduler": scheduler,
                "method": label,
                "avg_waiting": result["waiting_time"].mean(),
                "avg_turnaround": result["turnaround_time"].mean(),
                "avg_response": result["response_time"].mean()
            }
            summary.append(record)

            print(f"\n{scheduler} | {label}")
            print(f"Waiting: {record['avg_waiting']:.4f} ms")
            print(f"Turnaround: {record['avg_turnaround']:.4f} ms")
            print(f"Response: {record['avg_response']:.4f} ms")

    pd.DataFrame(summary).to_csv(
        OUT / "summary.csv", index=False
    )

    pd.concat(all_sjf, ignore_index=True).to_csv(
        OUT / "sjf_results.csv", index=False
    )
    pd.concat(all_srtf, ignore_index=True).to_csv(
        OUT / "srtf_results.csv", index=False
    )

    print("\nSaved all real-workload experiment results.")


if __name__ == "__main__":
    main()
