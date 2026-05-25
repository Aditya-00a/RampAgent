import pandas as pd
import numpy as np
from pathlib import Path


def load_transactions(file_path: str | None = None) -> pd.DataFrame:
    if file_path is None:
        file_path = Path(__file__).parent.parent / "data" / "sample_transactions.csv"

    df = pd.read_csv(file_path, parse_dates=["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["receipt_attached"] = df["receipt_attached"].astype(bool)
    df["pre_approved"] = df["pre_approved"].astype(bool)
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def load_policy(file_path: str | None = None) -> str:
    if file_path is None:
        file_path = Path(__file__).parent.parent / "data" / "sample_policy.md"

    return Path(file_path).read_text(encoding="utf-8")


def preprocess_for_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    cat_map = {cat: i for i, cat in enumerate(df["category"].unique())}
    df["category_encoded"] = df["category"].map(cat_map)

    dept_map = {dept: i for i, dept in enumerate(df["department"].unique())}
    df["department_encoded"] = df["department"].map(dept_map)

    df["log_amount"] = np.log1p(df["amount"])
    return df
