"""
data_loader.py

Reusable functions for loading and profiling the raw Telco churn dataset.
Keeping this logic here (instead of inline in a notebook) means train.py,
predict.py, and any notebook can all import the same, tested loading logic
instead of copy-pasting pandas calls everywhere.
"""

import pandas as pd
from pathlib import Path


def load_raw_data(filepath: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
    """
    Load the raw Telco churn CSV into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file, relative to project root.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path — this is intentional:
        failing loudly here is better than pandas producing a confusing
        downstream error later.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at '{filepath}'. "
            "Did you download it into data/raw/? See README for instructions."
        )
    df = pd.read_csv(path)
    return df


def profile_dataset(df: pd.DataFrame) -> None:
    """
    Print a first-pass profile of the dataset: shape, dtypes, missing values,
    duplicates, and a preview of the first few rows.

    This is deliberately a print-based function (not returning a report
    object) because at this phase we want a human reading the terminal
    output, not a machine consuming it.
    """
    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    print("=" * 60)
    print("COLUMN DTYPES")
    print("=" * 60)
    print(df.dtypes, "\n")

    print("=" * 60)
    print("MISSING VALUES PER COLUMN")
    print("=" * 60)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values detected by pandas.\n")
    else:
        print(missing, "\n")

    print("=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(f"Duplicate rows: {df.duplicated().sum()}\n")

    print("=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)
    print(df.head(), "\n")

    print("=" * 60)
    print("TARGET VARIABLE DISTRIBUTION (Churn)")
    print("=" * 60)
    if "Churn" in df.columns:
        print(df["Churn"].value_counts())
        print(df["Churn"].value_counts(normalize=True).round(4) * 100, "\n")
    else:
        print("Column 'Churn' not found — check column names.\n")


if __name__ == "__main__":
    # Allows running this file directly for a quick check:
    # python src/data_loader.py
    data = load_raw_data()
    profile_dataset(data)