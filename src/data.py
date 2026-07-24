from pathlib import Path

import pandas as pd


def load_tabular_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a pandas DataFrame.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def save_dataframe(df: pd.DataFrame, file_path: str | Path) -> None:
    """
    Save a DataFrame as CSV or Excel and create the parent folder if needed.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
        return

    raise ValueError(f"Unsupported output file type: {path.suffix}")