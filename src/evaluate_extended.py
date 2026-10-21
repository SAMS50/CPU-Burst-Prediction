
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sjf import simulate_sjf
from srtf import simulate_srtf
from sder import sder_sjf, sder_srtf

OUT = ROOT / "results" / "extended"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "synthetic": ROOT / "data" / "processed_workload.csv",
    "measured": ROOT / "data" / "real_workload.csv",
}


def get_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Actual bursts are strictly positive in our datasets.
    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    return mae, rmse, mape


def get_sder(result):
    if isinstance(result, dict):
        incorrect = result["incorrect_pairs"]
        eligible = result["eligible_pairs"]
    else:
        incorrect, eligible = result[:2]

    return (
        100 * incorrect / eligible
        if eligible else float("nan")
    )


def get_throughput(df):
    # Number of completed processes per simulated time unit.
    start = df["arrival_time"].min()
    end = df["completion_time"].max()
    elapsed = end - start

    return len(df) / elapsed if elapsed > 0 else float("nan")


def evaluate_dataset(name, path):
    df = pd.read_csv(path)
    df = df.sort_values(
        ["arrival_time", "process_id"]
    ).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Insufficient records in {name}")

    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    feature_cols = ["historical_burst"]
    x_train = train[feature_cols]
    y_train = train["actual_burst"]
    x_test = test[feature_cols]

    # Existing exponential-style baseline.
    alpha = 0.5
    test["pred_exp"] = (
        alpha * test["historical_burst"]
        + (1 - alpha) * y_train.mean()
    )

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    test["pred_lr"] = lr.predict(x_test)

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(x_train, y_train)
    test["pred_rf"] = rf.predict(x_test)

    for col in ["pred_exp", "pred_lr", "pred_rf"]:
        test[col] = test[col].clip(lower=0.001)

    test.to_csv(
        OUT / f"{name}_predictions.csv",
        index=False
    )

    prediction_rows = []
    scheduling_rows = []

    methods = [
        ("actual_burst", "Oracle"),
        ("pred_exp", "Exponential-style baseline"),
        ("pred_lr", "Linear Regression"),
        ("pred_rf", "Random Forest")
    ]

    print(f"\n{'=' * 65}")
    print(f"DATASET: {name.upper()}")
    print(f"Training: {len(train)} | Testing: {len(test)}")
    print(f"{'=' * 65}")

    for column, label in methods:
        if column != "actual_burst":
            mae, rmse, mape = get_metrics(
                test["actual_burst"], test[column]
            )

            prediction_rows.append({
                "dataset": name,
                "method": label,
                "mae": mae,
                "rmse": rmse,
                "mape_percent": mape
            })

            print(
                f"{label:27} "
                f"MAE={mae:.4f} "
                f"RMSE={rmse:.4f} "
                f"MAPE={mape:.2f}%"
            )

        for scheduler, simulator, sder_func in [
            ("SJF", simulate_sjf, sder_sjf),
            ("SRTF", simulate_srtf, sder_srtf)
        ]:
            result = simulator(test, column)

            if isinstance(result, tuple):
                result = result[0]

            sder = get_sder(sder_func(test, column))
            throughput = get_throughput(result)

            scheduling_rows.append({
                "dataset": name,
                "scheduler": scheduler,
                "method": label,
                "avg_waiting": result["waiting_time"].mean(),
                "avg_turnaround": result["turnaround_time"].mean(),
                "avg_response": result["response_time"].mean(),
                "throughput": throughput,
                "sder_percent": sder
            })

            print(
                f"  {scheduler:4} | "
                f"Waiting={result['waiting_time'].mean():.3f} | "
                f"Throughput={throughput:.5f} | "
                f"SDER={sder:.3f}%"
            )

    return prediction_rows, scheduling_rows


def main():
    all_prediction_rows = []
    all_scheduling_rows = []

    for name, path in DATASETS.items():
        prediction_rows, scheduling_rows = evaluate_dataset(
            name, path
        )
        all_prediction_rows.extend(prediction_rows)
        all_scheduling_rows.extend(scheduling_rows)

    pd.DataFrame(all_prediction_rows).to_csv(
        OUT / "prediction_metrics.csv",
        index=False
    )

    pd.DataFrame(all_scheduling_rows).to_csv(
        OUT / "scheduling_metrics.csv",
        index=False
    )

    print("\nSaved:")
    print("results/extended/prediction_metrics.csv")
    print("results/extended/scheduling_metrics.csv")


if __name__ == "__main__":
    main()
