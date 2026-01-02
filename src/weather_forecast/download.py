import requests
from pathlib import Path
from requests.exceptions import RequestException, Timeout, HTTPError
import pandas as pd
from tqdm import tqdm

def download_file(
    url: str,
    output_path: Path,
    timeout: int = 60,
    chunk_size: int = 8192,
    overwrite: bool = False,
):
    """
    Download a file with error handling.
    """
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return "skipped"

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise ValueError("Server returned HTML instead of CSV")

            total_bytes = 0
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)

        if total_bytes == 0:
            raise ValueError("Downloaded file is empty")

        tmp_path.rename(output_path)
        return "downloaded"

    except (Timeout, HTTPError, RequestException, OSError, ValueError):
        return "failed"

    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def create_url(station_name: str, precip_only: bool) -> dict:
    """
    Create download URLs for a station.
    """

    station_name = station_name.lower()

    if precip_only:
        base = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn-precip"
        prefix = "ogd-smn-precip"
    else:
        base = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
        prefix = "ogd-smn"

    return {
        "2010-2019": f"{base}/{station_name}/{prefix}_{station_name}_t_historical_2010-2019.csv",
        "2020-2029": f"{base}/{station_name}/{prefix}_{station_name}_t_historical_2020-2029.csv",
    }


def download_data(
    stations_file: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    """
    Download all raw data for a list of stations.

    Parameters
    ----------
    stations_file : Path
        CSV containing station metadata
    output_dir : Path
        Directory where raw CSV files are saved
    overwrite : bool
        Force re-download of existing files

    Returns
    -------
    dict
        Summary counts of downloads
    """

    stations_file = Path(stations_file)
    output_dir = Path(output_dir)

    df_stations = pd.read_csv(stations_file)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for row in tqdm(
        df_stations.itertuples(index=False),
        total=len(df_stations),
        desc="Downloading stations",
    ):
        station = row.station.lower()
        precip_only = bool(row.precip_only)

        url_dict = create_url(station, precip_only)

        for year_range, url in url_dict.items():
            output_path = output_dir / f"{station}_{year_range}.csv"

            status = download_file(
                url=url,
                output_path=output_path,
                overwrite=overwrite,
            )

            stats[status] += 1

    return stats