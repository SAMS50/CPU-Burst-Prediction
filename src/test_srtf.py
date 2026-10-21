
import pandas as pd
from srtf import simulate_srtf

df = pd.DataFrame({
    "process_id": [1, 2, 3],
    "arrival_time": [0, 1, 2],
    "actual_burst": [8, 4, 2],
    "predicted_burst": [8, 4, 2]
})

result, events = simulate_srtf(df, "predicted_burst")
result = result.set_index("process_id")

assert result.loc[1, "completion_time"] == 14
assert result.loc[2, "completion_time"] == 7
assert result.loc[3, "completion_time"] == 4

assert result.loc[1, "waiting_time"] == 6
assert result.loc[2, "waiting_time"] == 2
assert result.loc[3, "waiting_time"] == 0

assert len(events) >= 4

print("SRTF preemption test passed!")
