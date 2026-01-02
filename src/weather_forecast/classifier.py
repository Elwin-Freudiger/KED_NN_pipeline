"""
Train and compare multiple Keras architectures for precipitation classification.
Models and scalers are saved to disk.
"""

#import all libraries
from pathlib import Path
import random
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from keras_efficient_kan import KANLinear


#config
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-4
UNDERSAMPLING_RATE = 0.2

HIST_LEN = 36
HORIZON = 1

#setting a seed for reproducibility
def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

#loading and preparing data
def load_and_prepare_data(
    parquet_path: Path,
    stations_csv: Path,
):
    df_weather = pd.read_parquet(parquet_path)
    df_stations = pd.read_csv(stations_csv)

    df = df_weather.merge(
        df_stations[["station", "east", "north", "altitude"]],
        on="station",
        how="left",
    )

    selected_features = [
        "precip",
        "temperature",
        "East_wind",
        "North_wind",
        "pressure",
        "humidity",
    ]
    metadata_features = ["east", "north", "altitude"]
    all_features = selected_features + metadata_features

    df_features = df[["time", "station"] + all_features].copy()

    df_pivot = df_features.pivot(
        index="time",
        columns="station",
        values=all_features,
    )

    df_pivot.columns = [
        f"{feat}_{station}" for feat, station in df_pivot.columns
    ]

    df_pivot = df_pivot.sort_index().dropna()

    return df_pivot


def train_val_test_split(df, train=0.6, val=0.2):
    n = len(df)
    i1 = int(train * n)
    i2 = int((train + val) * n)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def scale_data(df_train, df_val, df_test, model_dir: Path):
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(df_train)
    val_scaled = scaler.transform(df_val)
    test_scaled = scaler.transform(df_test)

    joblib.dump(scaler, model_dir / "scaler.joblib")

    return (
        pd.DataFrame(train_scaled, columns=df_train.columns),
        pd.DataFrame(val_scaled, columns=df_val.columns),
        pd.DataFrame(test_scaled, columns=df_test.columns),
    )


#building windows
def build_windows(
    df_scaled,
    df_raw,
    precip_cols,
    hist_len,
    horizon,
    undersampling_rate,
):
    x, y = [], []

    for i in range(hist_len, len(df_scaled) - horizon):
        x_window = df_scaled.iloc[i - hist_len:i].values
        future_vals = df_raw.iloc[i + 1:i + 1 + horizon][precip_cols].values
        y_window = (np.any(future_vals > 0, axis=0)).astype(int)

        if np.sum(future_vals) == 0 and np.random.rand() > undersampling_rate:
            continue

        x.append(x_window)
        y.append(y_window)

    return np.array(x), np.array(y)

#---------MODELS------------

#LSTM KAN Hybrid
def build_lstm_kan(input_shape, num_stations, lr):
    ts_input = Input(shape=input_shape)

    x = layers.LSTM(64, return_sequences=True)(ts_input)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Reshape((1, 32))(x)
    x = KANLinear(32)(x)
    x = KANLinear(16)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(num_stations, activation="sigmoid")(x)

    model = Model(ts_input, output)
    model.compile(
        optimizer=Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model

#LSTM only model
def build_lstm_only(input_shape, num_stations, lr):
    ts_input = Input(shape=input_shape)

    x = layers.LSTM(64, return_sequences=True)(ts_input)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(num_stations, activation="sigmoid")(x)

    model = Model(ts_input, output)
    model.compile(
        optimizer=Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model


DEFAULT_ARCHITECTURES = {
    "lstm_kan": build_lstm_kan,
    "lstm_only": build_lstm_only,
}

#training
def train_classifier_models(
    parquet_path: str,
    stations_csv: str,
    model_dir: str,
    architectures: dict,
    seed: int = 42,
):
    set_seed(seed)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df_pivot = load_and_prepare_data(
        Path(parquet_path),
        Path(stations_csv),
    )

    df_train, df_val, df_test = train_val_test_split(df_pivot)

    train_scaled, val_scaled, test_scaled = scale_data(
        df_train, df_val, df_test, model_dir
    )

    precip_cols = [c for c in df_pivot.columns if c.startswith("precip_")]
    num_stations = len(precip_cols)

    x_train, y_train = build_windows(
        train_scaled,
        df_train,
        precip_cols,
        HIST_LEN,
        HORIZON,
        UNDERSAMPLING_RATE,
    )
    x_val, y_val = build_windows(
        val_scaled,
        df_val,
        precip_cols,
        HIST_LEN,
        HORIZON,
        UNDERSAMPLING_RATE,
    )

    input_shape = x_train.shape[1:]

    for name, builder in architectures.items():
        model = builder(
            input_shape=input_shape,
            num_stations=num_stations,
            lr=LEARNING_RATE,
        )

        model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=1,
        )

        model_path = model_dir / f"{name}.keras"
        model.save(model_path)