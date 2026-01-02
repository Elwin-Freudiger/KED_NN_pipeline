from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

DEFAULT_RENAME_DICT = {
    'rre150z0': 'precip',
    'fve010z0': 'wind_speed',
    'dkl010z0': 'wind_direction',
    'ure200s0': 'humidity',
    'pp0qnhs0': 'pressure',
    'tre200s0': 'temperature',
}

DEFAULT_VARLIST = [
    'precip',
    'East_wind',
    'North_wind',
    'humidity',
    'pressure',
    'temperature',
]

DEFAULT_START = pd.Timestamp("2019-01-01 00:00")
DEFAULT_END   = pd.Timestamp("2023-12-31 23:59")

CHUNK_SIZE = 300_000

def process_file(
    csv_path: Path,
    output_dir: Path,
    rename_dict: dict,
    varlist: list,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
):
    """
    Process a chunk of CSV file and append it.
    """

    station = csv_path.stem.split("_")[0].upper()

    for chunk in pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        sep=';',
    ):
        # Parse time
        chunk["time"] = pd.to_datetime(
            chunk["reference_timestamp"],
            format="%d.%m.%Y %H:%M",
            errors="coerce",
        )

        # Time filter
        mask = (chunk["time"] >= start_time) & (chunk["time"] <= end_time)
        chunk = chunk.loc[mask].copy()

        if chunk.empty:
            continue

        # Station and renaming
        chunk["station"] = station
        chunk = chunk.rename(columns=rename_dict)

        # Wind vector transformation
        if {'wind_speed', 'wind_direction'}.issubset(chunk.columns):
            direction_rad = np.deg2rad(chunk['wind_direction'])
            chunk['North_wind'] = -(chunk['wind_speed'] * np.cos(direction_rad)).round(1)
            chunk['East_wind']  = -(chunk['wind_speed'] * np.sin(direction_rad)).round(1)

        # Write per-variable outputs
        for var in varlist:
            if var not in chunk.columns:
                continue

            out_file = output_dir / f"{var}.csv"
            subset = chunk[["time", "station", var]]

            subset.to_csv(
                out_file,
                mode="a",
                header=not out_file.exists(),
                index=False,
            )


def clean_raw_data(
    raw_dir: Path,
    output_dir: Path,
    rename_dict: dict = DEFAULT_RENAME_DICT,
    varlist: list = DEFAULT_VARLIST,
    start_time: pd.Timestamp = DEFAULT_START,
    end_time: pd.Timestamp = DEFAULT_END,
) -> dict:
    """
    Clean all raw station CSV files into per-variable datasets.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw downloaded CSV files
    output_dir : Path
        Directory where cleaned CSVs are written

    Returns
    -------
    dict
        Summary statistics
    """

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.csv"))

    stats = {
        "files_processed": 0,
        "files_skipped": 0,
    }

    for csv_file in tqdm(files, desc="Cleaning raw files"):
        try:
            process_file(
                csv_path=csv_file,
                output_dir=output_dir,
                rename_dict=rename_dict,
                varlist=varlist,
                start_time=start_time,
                end_time=end_time,
            )
            stats["files_processed"] += 1
        except Exception as e:
            stats["files_skipped"] += 1

    return stats
