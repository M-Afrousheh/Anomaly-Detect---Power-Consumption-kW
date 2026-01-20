from pathlib import Path
from datetime import datetime
import sys
import traceback

import numpy as np
import pandas as pd

from kw_anomaly_utils import (
    load_timeseries,
    group_anomaly_events,
    plot_anomalies,
    infer_freq_minutes,
)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception as exc:
    tf = None
    tf_import_error = exc

try:
    from sklearn.preprocessing import StandardScaler
except Exception as exc:
    StandardScaler = None
    sklearn_import_error = exc


DATA_PATH = r"C:\Users\mafrousheh\Codes\kW_Anomaly Detection\kW data - S - MD US.xlsx"
SHEET_NAME = "Sysco Baltimore - Jessup, MD - "
TIME_COL = "time"
KW_COL = "Aggregated Demand (Peak)"

AE_EPOCHS = 30
AE_BATCH = 256
AE_PERCENTILE = 95

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")


def build_features(df, lags=(1, 2, 4, 8, 12, 24), roll_window=96):
    feat = pd.DataFrame(index=df.index)
    feat["kw"] = df["kw"].astype(float)
    for lag in lags:
        feat[f"lag_{lag}"] = df["kw"].shift(lag)
    feat[f"roll_mean_{roll_window}"] = df["kw"].rolling(roll_window, min_periods=roll_window).mean()
    feat[f"roll_std_{roll_window}"] = df["kw"].rolling(roll_window, min_periods=roll_window).std()

    hour = df.index.hour + df.index.minute / 60.0
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = df.index.dayofweek
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return feat.dropna()


def build_autoencoder(n_features):
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_features),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    if tf is None:
        print("TensorFlow is required for the autoencoder step.")
        print(f"Import error: {tf_import_error}")
        sys.exit(1)
    if StandardScaler is None:
        print("scikit-learn is required for scaling.")
        print(f"Import error: {sklearn_import_error}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_timeseries(
        DATA_PATH,
        sheet_name=SHEET_NAME,
        time_col=TIME_COL,
        value_col=KW_COL,
    )

    features = build_features(df)
    X = features.to_numpy()

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X)

    tf.random.set_seed(42)
    model = build_autoencoder(X_all.shape[1])
    model.fit(
        X_all,
        X_all,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH,
        validation_split=0.1,
        verbose=0,
    )

    recon = model.predict(X_all, verbose=0)
    errors = np.mean((X_all - recon) ** 2, axis=1)
    threshold = np.percentile(errors, AE_PERCENTILE)
    ae_anom = errors >= threshold

    out = pd.DataFrame(
        index=features.index,
        data={
            "value": features["kw"],
            "ae_error": errors,
            "is_anom": ae_anom,
        },
    )

    freq_minutes = infer_freq_minutes(out.index)
    events = group_anomaly_events(
        out,
        is_anom_col="is_anom",
        max_gap_minutes=freq_minutes * 1.5,
        freq_minutes=freq_minutes,
    )

    out_path_points = OUTPUT_DIR / "kw_anomaly_points_autoencoder_only.csv"
    out_path_events = OUTPUT_DIR / "kw_anomaly_events_autoencoder_only.csv"
    out.to_csv(out_path_points, index=True)
    events.to_csv(out_path_events, index=False)

    plot_path = OUTPUT_DIR / "kw_anomalies_autoencoder_only.png"
    plot_anomalies(out, value_col="value", is_anom_col="is_anom", out_path=str(plot_path))

    print(f"Saved points CSV to {out_path_points}")
    print(f"Saved events CSV to {out_path_events}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
