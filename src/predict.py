
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed_workload.csv"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df = df.sort_values(
    ["arrival_time", "process_id"]
).reset_index(drop=True)

# Chronological 80/20 split
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

# Method 1: One-step exponential averaging
alpha = 0.5
initial_prediction = train["actual_burst"].mean()

test["pred_exp"] = (
    alpha * test["historical_burst"]
    + (1 - alpha) * initial_prediction
)

# Method 2: Linear regression
features = ["historical_burst"]

X_train = train[features]
y_train = train["actual_burst"]

X_test = test[features]
y_test = test["actual_burst"]

model = LinearRegression()
model.fit(X_train, y_train)

test["pred_lr"] = model.predict(X_test)

# Ensure predicted bursts are positive
test["pred_exp"] = test["pred_exp"].clip(lower=1)
test["pred_lr"] = test["pred_lr"].clip(lower=1)

def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )
    return mae, rmse

exp_mae, exp_rmse = evaluate(
    y_test, test["pred_exp"]
)

lr_mae, lr_rmse = evaluate(
    y_test, test["pred_lr"]
)

print("\nCPU BURST-TIME PREDICTION RESULTS")
print("--------------------------------")

print("Training processes:", len(train))
print("Testing processes:", len(test))

print("\nExponential Averaging:")
print(f"MAE: {exp_mae:.4f}")
print(f"RMSE: {exp_rmse:.4f}")

print("\nLinear Regression:")
print(f"MAE: {lr_mae:.4f}")
print(f"RMSE: {lr_rmse:.4f}")

output_columns = [
    "process_id",
    "arrival_time",
    "historical_burst",
    "actual_burst",
    "pred_exp",
    "pred_lr"
]

test[output_columns].to_csv(
    RESULTS_DIR / "predictions.csv",
    index=False
)

print("\nPredictions saved to results/predictions.csv")
