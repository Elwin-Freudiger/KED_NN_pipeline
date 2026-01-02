"""
Merge interpolated variable CSVs into a single Parquet dataset.
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


VARLIST = [
    "precip",
    "East_wind",
    "North_wind",
    "humidity",
    "pressure",
    "temperature",
]

def load_variable_csv(input_dir: Path, var: str) -> pd.DataFrame:
    """
    Load a single interpolated variable CSV and clean flags.
    """
    path = input_dir / f"{var}_interpolated.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing interpolated file: {path}")

    df = pd.read_csv(path)

    flag_col = f"{var}_interpolated"
    if flag_col not in df.columns:
        raise ValueError(f"Missing column {flag_col} in {path}")

    df[flag_col] = df[flag_col].fillna(False).astype(bool)

    df = df.drop(
        columns=["east", "north", "altitude", "average_wind"],
        errors="ignore",
    )

    return df


def merge_interpolated_variables(
    input_dir: str,
    output_path: str,
    variables=VARLIST,
):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df = None

    for var in tqdm(variables, desc="Merging variables"):
        df_var = load_variable_csv(input_dir, var)

        if merged_df is None:
            merged_df = df_var
        else:
            merged_df = merged_df.merge(
                df_var,
                on=["station", "time"],
                how="outer",
                validate="one_to_one",
            )

    ordered_columns = (
        ["time", "station"]
        + variables
        + [f"{v}_interpolated" for v in variables]
    )

    merged_df = merged_df[ordered_columns]
    merged_df.sort_values(["station", "time"], inplace=True)

    merged_df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
    )