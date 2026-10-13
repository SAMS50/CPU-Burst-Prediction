
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

NUM_PROCESSES = 1000

process_ids = np.arange(1, NUM_PROCESSES + 1)

arrival_times = np.sort(
    rng.integers(0, 500, NUM_PROCESSES)
)

historical_bursts = rng.integers(
    2, 25, NUM_PROCESSES
)

actual_bursts = np.maximum(
    1,
    np.round(
        historical_bursts +
        rng.normal(0, 4, NUM_PROCESSES)
    )
).astype(int)

dataset = pd.DataFrame({
    "process_id": process_ids,
    "arrival_time": arrival_times,
    "historical_burst": historical_bursts,
    "actual_burst": actual_bursts
})

dataset.to_csv(
    "data/synthetic_workload.csv",
    index=False
)

print("Dataset generated successfully!")
print(dataset.head())
print("Total processes:", len(dataset))
