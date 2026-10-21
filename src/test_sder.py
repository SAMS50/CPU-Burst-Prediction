"""Minimal hand-checkable tests for SDER."""
import pandas as pd
from sder import sder_sjf, sder_srtf


def data(arrivals, actual, predicted):
    return pd.DataFrame({
        'process_id': range(1, len(actual) + 1),
        'arrival_time': arrivals,
        'actual_burst': actual,
        'pred_test': predicted,
    })

# Three simultaneous jobs: oracle order 1,2,3; predicted order 2,1,3.
d = data([0, 0, 0], [3, 5, 9], [6, 4, 8])
assert sder_sjf(d, 'pred_test') == (1, 4), sder_sjf(d, 'pred_test')
# SJF counts pairs again at subsequent dispatches, hence 4 eligible comparisons.
assert sder_sjf(d, 'actual_burst')[0] == 0
assert sder_srtf(d, 'actual_burst')[0] == 0

# No pairwise comparison possible when jobs do not overlap in the ready queue.
d = data([0, 10], [2, 2], [2, 2])
assert sder_sjf(d, 'pred_test') == (0, 0)
assert sder_srtf(d, 'pred_test') == (0, 0)

# Two simultaneously ready jobs in reversed predicted order.
d = data([0, 0], [3, 7], [8, 2])
assert sder_sjf(d, 'pred_test') == (1, 1)
assert sder_srtf(d, 'pred_test')[0] >= 1

print('SDER tests passed!')
