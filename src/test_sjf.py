
import pandas as pd
from sjf import simulate_sjf

data = pd.DataFrame({
    "process_id": [1, 2, 3],
    "arrival_time": [0, 0, 0],
    "actual_burst": [6, 2, 4],
    "pred_lr": [6, 2, 4]
})

result = simulate_sjf(data, "pred_lr")

print(result[[
    "process_id",
    "start_time",
    "completion_time",
    "waiting_time",
    "turnaround_time"
]])

assert result["process_id"].tolist() == [2, 3, 1]
assert result["waiting_time"].tolist() == [0, 2, 6]
assert result["turnaround_time"].tolist() == [2, 6, 12]

print("\nSJF test passed!")
