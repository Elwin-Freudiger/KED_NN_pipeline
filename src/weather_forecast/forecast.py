"""
Produce recursive multi-step forecasts using pretrained
classifier and regressor models.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from tqdm import tqdm

#default config
HIST_LEN = 36
N_SAMPLES = 20
RAIN_THRESHOLD = 0.8

#set seed
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

#load data
def load_data(weather_parquet, stations_csv):
    df_weather = pd.read_parquet(weather_parquet)
    df_stations = pd.read_csv(stations_csv)

    df = df_weather.merge(
        df_stations[["station", "east", "north", "altitude"]],
        on="station",
        how="left",
    )
    return df


def build_wide_dataframe(df):
    features = [
        "precip", "temperature",
        "East_wind", "North_wind",
        "pressure", "humidity",
        "east", "north", "altitude",
    ]

    df_pivot = df.pivot(
        index="time",
        columns="station",
        values=features,
    )

    df_pivot.columns = [
        f"{feat}_{station}" for feat, station in df_pivot.columns
    ]

    return df_pivot.sort_index().dropna()


def scale_test_data(df_pivot, scaler):
    split1 = int(0.6 * len(df_pivot))
    split2 = int(0.8 * len(df_pivot))

    df_test = df_pivot.iloc[split2:]

    df_test_scaled = pd.DataFrame(
        scaler.transform(df_test),
        columns=df_test.columns,
        index=df_test.index,
    )

    return df_test, df_test_scaled


def recursive_forecast(
    classifier,
    regressor,
    test_scaled,
    true_precip,
    precip_cols,
    horizon,
    threshold,
):
    indices = np.random.choice(
        len(test_scaled) - HIST_LEN - horizon,
        size=N_SAMPLES,
        replace=False,
    )

    results = {h: {"mae": [], "rmse": []} for h in range(1, horizon + 1)}

    for idx in tqdm(indices, desc="Forecasting"):
        x = test_scaled.iloc[idx:idx + HIST_LEN].values[np.newaxis, :, :]
        y_true_seq = true_precip[
            idx + HIST_LEN + 1 : idx + HIST_LEN + 1 + horizon
        ]

        forecasts = []

        for h in range(horizon):
            rain_probs = classifier.predict(x, verbose=0)[0]
            rain_mask = rain_probs > threshold

            step_forecast = []
            for j in range(len(precip_cols)):
                if rain_mask[j]:
                    log_y = regressor.predict(x, verbose=0)[0][0]
                    y_hat = np.expm1(log_y)
                else:
                    y_hat = 0.0
                step_forecast.append(y_hat)

            forecasts.append(step_forecast)

            if h < horizon - 1:
                x = update_recursive_input(
                    x, step_forecast, test_scaled.columns, precip_cols
                )

        forecasts = np.array(forecasts)

        for h in range(1, horizon + 1):
            results[h]["mae"].append(
                mean_absolute_error(y_true_seq[h - 1], forecasts[h - 1])
            )
            results[h]["rmse"].append(
                np.sqrt(
                    mean_squared_error(y_true_seq[h - 1], forecasts[h - 1])
                )
            )

    return results


def update_recursive_input(x, forecast_row, columns, precip_cols):
    next_input = x[0, 1:, :].copy()
    last_row = x[0, -1, :].copy()

    for j, col in enumerate(precip_cols):
        idx = columns.get_loc(col)
        last_row[idx] = forecast_row[j]

    return np.concatenate([next_input, [last_row]])[np.newaxis, :, :]

#main forecast function
def run_forecast(
    horizon: int,
    weather_parquet: str,
    stations_csv: str,
    classifier_path: str,
    regressor_path: str,
    scaler_path: str,
    seed: int = 42,
):
    set_seed(seed)

    df = load_data(weather_parquet, stations_csv)
    df_pivot = build_wide_dataframe(df)

    scaler = joblib.load(scaler_path)
    df_test, test_scaled = scale_test_data(df_pivot, scaler)

    precip_cols = [c for c in df_test.columns if c.startswith("precip_")]
    true_precip = df_test[precip_cols].values

    classifier = load_model(classifier_path)
    regressor = load_model(regressor_path)

    results = recursive_forecast(
        classifier,
        regressor,
        test_scaled,
        true_precip,
        precip_cols,
        horizon,
        RAIN_THRESHOLD,
    )

    return pd.DataFrame({
        "Horizon": range(1, horizon + 1),
        "MAE": [np.mean(results[h]["mae"]) for h in range(1, horizon + 1)],
        "RMSE": [np.mean(results[h]["rmse"]) for h in range(1, horizon + 1)],
    })