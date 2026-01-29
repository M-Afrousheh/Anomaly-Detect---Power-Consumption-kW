from pathlib import Path
from datetime import datetime
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception as exc:
    SimpleImputer = None
    sklearn_import_error = exc

try:
    from xgboost import XGBRegressor
except Exception as exc:
    XGBRegressor = None
    xgb_import_error = exc


DATA_PATH = r"C:\Users\mafrousheh\Codes\kW_Anomaly Detection\Kw data plus features v1\S MD - data V1.xlsx"
SHEET_NAME = "S MD - raw data"
TIME_COL = "time"
DEMAND_COL = "Aggregated Demand"
FREQ = "15min"
START_DATE = None

TRAIN_SPLIT = 0.8
LAGS = [1, 4, 12, 96]  # 15m, 1h, 3h, 24h
ERROR_PERCENTILE = 90
MIN_CONSECUTIVE = 4

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")


def load_all_features(path, sheet_name, time_col, freq, start_date=None):
    df = pd.read_excel(path, sheet_name=sheet_name)
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    if pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], unit="D", origin="1899-12-30", errors="coerce")
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    if start_date:
        df = df.loc[df.index >= pd.to_datetime(start_date)]

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if DEMAND_COL not in numeric_cols:
        raise ValueError(f"Missing demand column: {DEMAND_COL}")

    df[numeric_cols] = df[numeric_cols].ffill()
    df = df.dropna(subset=[numeric_cols[0]])
    return df, numeric_cols


def add_time_features(df):
    idx = df.index
    day_of_week = idx.dayofweek
    is_weekend = (day_of_week >= 5).astype(int)
    hour = idx.hour + idx.minute / 60.0
    is_night = ((hour >= 0.0) & (hour < 8.0)).astype(int)

    out = df.copy()
    out["is_weekend"] = is_weekend
    out["is_night"] = is_night
    out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["time_regime"] = np.select(
        [is_weekend == 1, is_night == 1],
        [3, 2],
        default=1,
    ).astype(int)
    return out


def add_demand_lags(df, demand_col, lags):
    out = df.copy()
    for lag in lags:
        out[f"{demand_col}_lag_{lag}"] = out[demand_col].shift(lag)
    return out


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def main():
    if SimpleImputer is None:
        print("scikit-learn is required.")
        print(f"Import error: {sklearn_import_error}")
        sys.exit(1)
    if XGBRegressor is None:
        print("xgboost is required.")
        print(f"Import error: {xgb_import_error}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, numeric_cols = load_all_features(
        DATA_PATH,
        sheet_name=SHEET_NAME,
        time_col=TIME_COL,
        freq=FREQ,
        start_date=START_DATE,
    )
    df = add_time_features(df)
    df = add_demand_lags(df, DEMAND_COL, LAGS)

    prod_cols = [
        "Cooler Selection - Daily",
        "Dry Selection - Daily",
        "Freezer Selection - Daily",
    ]
    shift_periods = 32
    for col in prod_cols:
        if col in df.columns:
            df[col] = df[col].shift(shift_periods)

    night_mask = df["time_regime"] == 2
    weekend_mask = df["time_regime"] == 3
    for col in prod_cols:
        if col in df.columns:
            df.loc[night_mask | weekend_mask, col] = 0.0

    df = df.dropna()

    features_path = OUTPUT_DIR / "features_engineered.csv"
    df.to_csv(features_path, index=True)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != DEMAND_COL]
    X = df[feature_cols].to_numpy()
    y = df[DEMAND_COL].to_numpy()

    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    errors = (y_test - y_pred) ** 2
    errors = np.asarray(errors, dtype=float)
    threshold = np.nanpercentile(errors, ERROR_PERCENTILE)
    is_point = errors >= threshold
    grp = (pd.Series(is_point) != pd.Series(is_point).shift(1)).cumsum()
    run_len = pd.Series(is_point).groupby(grp).transform("sum")
    is_anom = is_point & (run_len >= MIN_CONSECUTIVE)
    is_anom = np.asarray(is_anom, dtype=bool)

    out_idx = df.index[split_idx:]
    out = pd.DataFrame(
        index=out_idx,
        data={"demand": y_test, "y_pred": y_pred, "error": errors, "is_anom": is_anom},
    )
    out["is_anom"] = out["is_anom"].fillna(False).astype(bool)
    out.to_csv(OUTPUT_DIR / "anomaly_points_test.csv", index=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(out.index, out["demand"], color="tab:blue", label="demand (kW)")
    ax.scatter(out.index[out["is_anom"]], out["demand"][out["is_anom"]], color="red", s=10, label="anomaly")
    ax.set_xlabel("Time")
    ax.set_ylabel("kW")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "demand_anomalies_test.png", dpi=150)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(out.index, out["error"], color="tab:purple", label="demand error")
    ax.set_xlabel("Time")
    ax.set_ylabel("MSE")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "demand_error_test.png", dpi=150)

    nan_count = int(np.isnan(errors).sum())
    point_count = int(np.sum(is_point))
    max_run = int(pd.Series(is_point).groupby(grp).sum().max())
    print(f"Saved metrics to {metrics_path}")
    print(f"Error nan count: {nan_count}")
    print(f"Threshold (p{ERROR_PERCENTILE}): {float(threshold)}")
    print(f"Points >= threshold: {point_count}")
    print(f"Max run length: {max_run}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
