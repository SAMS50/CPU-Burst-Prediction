
from pathlib import Path
import csv
import random
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "real_workload.csv"

random.seed(42)


def cpu_work(units):
    accumulator = 0
    for i in range(units):
        accumulator += (i * i) % 97
    return accumulator


def measure_burst(units):
    start = time.process_time()
    cpu_work(units)
    return (time.process_time() - start) * 1000


def main():
    OUTPUT.parent.mkdir(exist_ok=True)

    records = []
    arrival = 0.0

    for process_id in range(1, 201):
        units = random.randint(100000, 1200000)

        historical = measure_burst(units)
        actual = measure_burst(units)

        arrival += random.uniform(0, 2)

        records.append({
            "process_id": process_id,
            "arrival_time": round(arrival, 3),
            "historical_burst": max(0.001, historical),
            "actual_burst": max(0.001, actual)
        })

        if process_id % 50 == 0:
            print(f"Measured {process_id}/200 processes")

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} measured records")
    print(f"File: {OUTPUT}")
    print("CPU durations measured in milliseconds")
    print("Arrival times are simulated, not observed")


if __name__ == "__main__":
    main()
