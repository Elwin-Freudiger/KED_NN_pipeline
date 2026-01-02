"""
Train and compare multiple Keras regression architectures
to predict precipitation intensity at stations.
Each architecture and the scaler are saved.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from keras_efficient_kan import KANLinear

#config
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-4
UNDERSAMPLING_RATE = 0.2

HIST_LEN = 36
HORIZON = 1


#set seed for reproducibility
def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


#loading and preparing the data
def load_and_prepare_data(
    weather_parquet: Path,
    stations_csv: Path,
):
    df_weather = pd.read_parquet(weather_parquet)
    df_stations = pd.read_csv(stations_csv)

    df = df_weather.merge(
        df_stations[["station", "east", "north", "altitude"]],
        on="station",
        how="left",
    )

    station_list = sorted(df["station"].unique())
    station_to_index = {s: i for i, s in enumerate(station_list)}

    features = [
        "precip", "temperature", "East_wind",
        "North_wind", "pressure", "humidity",
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

    df_pivot = df_pivot.sort_index().dropna()

    return df_pivot, station_list, station_to_index

def build_samples(
    df_scaled,
    df_raw,
    station_list,
    station_to_index,
    hist_len,
    horizon,
    undersample,
):
    x, y = [], []
    num_stations = len(station_list)

    for station in station_list:
        precip_col = f"precip_{station}"
        station_idx = station_to_index[station]
        one_hot = to_categorical(station_idx, num_classes=num_stations)

        for i in range(hist_len, len(df_scaled) - horizon):
            target = df_raw.iloc[i + horizon][precip_col]

            if target <= 0:
                continue

            if undersample and np.random.rand() > UNDERSAMPLING_RATE:
                continue

            x_window = df_scaled.iloc[i - hist_len:i].values
            one_hot_rep = np.tile(one_hot, (hist_len, 1))
            x_aug = np.concatenate([x_window, one_hot_rep], axis=1)

            x.append(x_aug)
            y.append(np.log1p(target))

    return np.array(x), np.array(y)

#---------MODELS------------

#LSTM KAN Hybrid
def build_lstm_kan(input_shape, lr):
    ts_input = Input(shape=input_shape)

    x = layers.LSTM(64, return_sequences=True)(ts_input)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Reshape((1, 32))(x)
    x = KANLinear(32)(x)
    x = KANLinear(16)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    model = Model(ts_input, output)
    model.compile(
        optimizer=Adam(lr),
        loss="mse",
        metrics=["mae"],
    )
    return model

#LSTM only
def build_lstm_only(input_shape, lr):
    ts_input = Input(shape=input_shape)

    x = layers.LSTM(64, return_sequences=True)(ts_input)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1)(x)

    model = Model(ts_input, output)
    model.compile(
        optimizer=Adam(lr),
        loss="mse",
        metrics=["mae"],
    )
    return model

DEFAULT_ARCHITECTURES = {
    "reg_lstm_kan": build_lstm_kan,
    "reg_lstm_only": build_lstm_only,
}


#training
def train_regressors(
    weather_parquet: str,
    stations_csv: str,
    model_dir: str,
    architectures: dict = DEFAULT_ARCHITECTURES,
    seed: int = 42,
):
    set_seed(seed)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df_pivot, station_list, station_to_index = load_and_prepare_data(
        weather_parquet,
        stations_csv,
    )

    split1 = int(0.6 * len(df_pivot))
    split2 = int(0.8 * len(df_pivot))

    df_train = df_pivot.iloc[:split1]
    df_val = df_pivot.iloc[split1:split2]

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train),
        columns=df_train.columns,
        index=df_train.index,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(df_val),
        columns=df_val.columns,
        index=df_val.index,
    )

    joblib.dump(scaler, model_dir / "scaler_regression.joblib")

    x_train, y_train = build_samples(
        train_scaled, df_train,
        station_list, station_to_index,
        HIST_LEN, HORIZON,
        undersample=True,
    )

    x_val, y_val = build_samples(
        val_scaled, df_val,
        station_list, station_to_index,
        HIST_LEN, HORIZON,
        undersample=False,
    )

    input_shape = x_train.shape[1:]

    for name, builder in architectures.items():
        model = builder(input_shape=input_shape, lr=LEARNING_RATE)

        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1,
        )

        model.save(model_dir / f"{name}.keras")