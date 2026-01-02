"""
Interpolation tools using Kriging with External Drift.
Supports fitted or fixed variogram models.
"""

import os
import numpy as np
import pandas as pd
import gstools as gs
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

_STATION_CACHE = None

def get_station_metadata(stations_csv="data/valais_stations.csv"):
    global _STATION_CACHE
    if _STATION_CACHE is None:
        _STATION_CACHE = pd.read_csv(
            stations_csv,
            usecols=["station", "east", "north", "altitude", "average_wind"]
        )
    return _STATION_CACHE

def prepare_data(df_var, target_time, var, drift):
    time_data = df_var[df_var["time"] == target_time]
    known = time_data[~time_data[var].isna()]
    unknown = time_data[time_data[var].isna()]

    if len(known) == 0 or len(unknown) == 0:
        return None, None, None

    known_points = known[["east", "north", drift, var]].values
    unknown_points = unknown[["east", "north", drift]].values
    unknown_stations = unknown["station"].values

    return known_points, unknown_points, unknown_stations

def ked_interpolation_gstools(known_points, unknown_points):
    xk, yk, zk = known_points[:, 0], known_points[:, 1], known_points[:, 2]
    values = known_points[:, 3]
    xu, yu, zu = unknown_points[:, 0], unknown_points[:, 1], unknown_points[:, 2]

    if np.all(values == 0):
        return np.zeros(len(unknown_points))

    bin_center, gamma = gs.vario_estimate(
        (xk, yk),
        values,
        bin_edges=np.linspace(0, 100000, 15)
    )

    model = gs.Exponential(dim=2)
    model.fit_variogram(bin_center, gamma)

    ked = gs.krige.ExtDrift(
        model=model,
        cond_pos=(xk, yk),
        cond_val=values,
        ext_drift=zk,
    )

    preds, _ = ked((xu, yu), ext_drift=zu, return_var=True)
    return preds


def ked_interpolation_gstools_fixed(known_points, unknown_points):
    xk, yk, zk = known_points[:, 0], known_points[:, 1], known_points[:, 2]
    values = known_points[:, 3]
    xu, yu, zu = unknown_points[:, 0], unknown_points[:, 1], unknown_points[:, 2]

    if np.all(values == 0):
        return np.zeros(len(unknown_points))

    model = gs.Spherical(dim=2, var=1.0, len_scale=10000, nugget=0.1)

    ked = gs.krige.ExtDrift(
        model=model,
        cond_pos=(xk, yk),
        cond_val=values,
        ext_drift=zk,
    )

    preds, _ = ked((xu, yu), ext_drift=zu, return_var=True)
    return preds


def interpolate_points(known_points, unknown_points, mode):
    if mode == "fixed":
        return ked_interpolation_gstools_fixed(known_points, unknown_points)
    elif mode == "fitted":
        return ked_interpolation_gstools(known_points, unknown_points)
    else:
        raise ValueError(f"Unknown variogram mode: {mode}")


def interpolate_variable(
    var_csv,
    var,
    drift,
    output_dir,
    variogram_mode,
    position=0,
):
    df = pd.read_csv(var_csv, parse_dates=["time"])
    df = df.merge(get_station_metadata(), on="station", how="left")

    df_var = df[["time", "station", "east", "north", drift, var]].copy()
    missing_times = df_var.loc[df_var[var].isna(), "time"].drop_duplicates()
    results = []

    for t in tqdm(missing_times, desc=f"Interpolating {var}", position=position):
        known, unknown, stations = prepare_data(df_var, t, var, drift)

        if known is None or len(known) < 5:
            continue

        try:
            preds = interpolate_points(known, unknown, variogram_mode)

            for st, val in zip(stations, preds):
                if var == "precip":
                    val = max(val, 0)
                elif var in ["North_wind", "East_wind"]:
                    val = max(val, 0)
                elif var == "humidity":
                    val = min(max(val, 0), 100)

                results.append({
                    "time": t,
                    "station": st,
                    var: round(val, 1),
                    f"{var}_interpolated": True,
                })

        except Exception as e:
            print(f"[{var}] {t}: {e}")

    if not results:
        print(f"No interpolation needed for {var}")
        return

    df_known = df_var.dropna(subset=[var]).copy()
    df_known[f"{var}_interpolated"] = False

    df_interp = pd.DataFrame(results)
    df_final = pd.concat([df_known, df_interp], ignore_index=True)
    df_final.sort_values(["station", "time"], inplace=True)

    output_path = os.path.join(output_dir, f"{var}_interpolated.csv")
    df_final.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def run_interpolation_pipeline(
    input_dir,
    output_dir,
    variables,
    variogram_mode="fitted",
    max_workers=None,
):
    os.makedirs(output_dir, exist_ok=True)

    args = []
    for i, var in enumerate(variables):
        var_csv = os.path.join(input_dir, f"{var}.csv")
        if not os.path.exists(var_csv):
            continue

        drift = "average_wind" if var in ["East_wind", "North_wind"] else "altitude"
        args.append((var_csv, var, drift, output_dir, variogram_mode, i))

    process_map(
        lambda p: interpolate_variable(*p),
        args,
        max_workers=max_workers or os.cpu_count(),
        desc="Interpolating variables",
    )
