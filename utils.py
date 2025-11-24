"""Helper utilities."""
import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
