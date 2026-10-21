"""Scheduling Decision Error Rate (SDER) for SJF and SRTF.

SJF: compares pairwise priorities among processes eligible at each dispatch.
SRTF: compares predicted and actual remaining-time priorities at each arrival
or completion decision, with actual service consumed up to that point.
Only pairs with strictly different actual and predicted priorities are counted;
prediction ties are counted as errors if actual priorities differ.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
EPS = 1e-9


def sder_sjf(df, prediction_column):
    """Ready-queue pairwise inversion rate at non-preemptive dispatches."""
    df = df.sort_values(['arrival_time', 'process_id']).reset_index(drop=True)
    arrivals = df.arrival_time.to_numpy(float)
    actual = df.actual_burst.to_numpy(float)
    pred = df[prediction_column].to_numpy(float)
    ids = df.process_id.to_numpy()
    remaining = set(range(len(df)))
    time = float(min(arrivals))
    errors = total = 0
    while remaining:
        ready = [i for i in remaining if arrivals[i] <= time + EPS]
        if not ready:
            time = min(arrivals[i] for i in remaining)
            continue
        for ai in range(len(ready)):
            i = ready[ai]
            for j in ready[ai+1:]:
                a = actual[i] - actual[j]
                if abs(a) <= EPS:
                    continue  # actual tie: no unique ideal ordering
                p = pred[i] - pred[j]
                total += 1
                if a * p < -EPS or abs(p) <= EPS:
                    errors += 1
        chosen = min(ready, key=lambda i: (pred[i], arrivals[i], ids[i]))
        time += actual[chosen]
        remaining.remove(chosen)
    return errors, total


def sder_srtf(df, prediction_column):
    """Decision-point pair inversions on the *predicted* SRTF execution path.

    Decisions occur at arrivals and completions. At each, compare the true
    remaining burst with estimated remaining burst for all ready pairs.
    An arrival that coincides with completion counts as one decision point.
    """
    df = df.sort_values(['arrival_time', 'process_id']).reset_index(drop=True)
    arrivals = df.arrival_time.to_numpy(float)
    actual = df.actual_burst.to_numpy(float)
    pred = df[prediction_column].to_numpy(float)
    ids = df.process_id.to_numpy()
    n = len(df)
    if n == 0:
        return 0, 0
    rem_actual = actual.copy()
    rem_pred = pred.copy()
    time = float(arrivals[0])
    errors = total = 0
    while (rem_actual > EPS).any():
        ready = [i for i in range(n) if arrivals[i] <= time + EPS and rem_actual[i] > EPS]
        if not ready:
            time = min(arrivals[i] for i in range(n) if rem_actual[i] > EPS and arrivals[i] > time + EPS)
            continue
        for ix, i in enumerate(ready):
            for j in ready[ix+1:]:
                a = rem_actual[i] - rem_actual[j]
                if abs(a) <= EPS:
                    continue
                p = rem_pred[i] - rem_pred[j]
                total += 1
                if a * p < -EPS or abs(p) <= EPS:
                    errors += 1
        chosen = min(ready, key=lambda i: (rem_pred[i], arrivals[i], ids[i]))
        next_arrival = min((arrivals[i] for i in range(n) if rem_actual[i] > EPS and arrivals[i] > time + EPS), default=float('inf'))
        duration = min(rem_actual[chosen], next_arrival - time)
        if duration <= EPS:
            raise RuntimeError('SRTF simulation did not advance')
        rem_actual[chosen] = max(0.0, rem_actual[chosen] - duration)
        rem_pred[chosen] = max(0.0, rem_pred[chosen] - duration)
        time += duration
    return errors, total


def main():
    df = pd.read_csv(RESULTS / 'predictions.csv')
    rows = []
    print('\nSCHEDULING DECISION ERROR RATE (SDER)')
    print('=' * 60)
    for algorithm, fn in [('SJF', sder_sjf), ('SRTF', sder_srtf)]:
        for col, label in [('actual_burst', 'Oracle'), ('pred_exp', 'Exponential Averaging'), ('pred_lr', 'Linear Regression')]:
            errors, total = fn(df, col)
            rate = 100 * errors / total if total else 0.0
            rows.append({'scheduler': algorithm, 'method': label, 'incorrect_pairs': errors, 'eligible_pairs': total, 'sder_percent': rate})
            print(f'{algorithm:4} | {label:22} | {errors:6}/{total:6} | SDER {rate:7.3f}%')
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / 'sder_results.csv', index=False)
    print('\nSaved results/sder_results.csv')


if __name__ == '__main__':
    main()
