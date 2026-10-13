
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "synthetic_workload.csv"
OUTPUT_FILE = ROOT / "data" / "processed_workload.csv"

REQUIRED_COLUMNS = [
    "process_id",
    "arrival_time",
    "historical_burst",
    "actual_burst"
]

def preprocess_data():
    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("\nMissing values:")
    print(df.isnull().sum())

    df = df.dropna(subset=REQUIRED_COLUMNS)

    duplicates = df["process_id"].duplicated().sum()
    print("\nDuplicate process IDs:", duplicates)

    df = df.drop_duplicates(subset="process_id")

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    df = df[
        (df["process_id"] > 0) &
        (df["arrival_time"] >= 0) &
        (df["historical_burst"] > 0) &
        (df["actual_burst"] > 0)
    ].copy()

    df = df.sort_values(
        by=["arrival_time", "process_id"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nProcessed dataset shape:", df.shape)
    print("\nDataset statistics:")
    print(df.describe())

    print("\nProcessed dataset saved successfully!")

if __name__ == "__main__":
    preprocess_data()
