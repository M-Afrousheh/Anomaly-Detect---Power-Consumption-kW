import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_timeseries(path, sheet_name=None, time_col="time", value_col="Aggregated Demand (Peak)"):
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path)

    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    if value_col not in df.columns:
        raise ValueError(f"Missing value column: {value_col}")

    df = df[[time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.dropna().sort_values(time_col)
    df = df.set_index(time_col)
    df = df.rename(columns={value_col: "kw"})
    return df


def infer_freq_minutes(index):
    deltas = pd.Series(index).diff().dropna()
    if deltas.empty:
        return 15.0
    return deltas.median().total_seconds() / 60.0


def robust_rolling_zscore(s, window=672, z_thresh=5.0, min_consecutive=3):
    s = s.astype(float).copy()
    minp = max(20, window // 10)
    med = s.rolling(window=window, min_periods=minp).median()
    mad = (s - med).abs().rolling(window=window, min_periods=minp).median()
    eps = 1e-9
    robust_z = 0.6745 * (s - med) / (mad + eps)
    is_point_anom = robust_z.abs() >= z_thresh

    grp = (is_point_anom != is_point_anom.shift(1)).cumsum()
    run_len = is_point_anom.groupby(grp).transform("sum")
    is_anom = is_point_anom & (run_len >= min_consecutive)

    return pd.DataFrame({
        "value": s,
        "median": med,
        "mad": mad,
        "robust_z": robust_z,
        "is_point_anom": is_point_anom,
        "is_anom": is_anom
    })


def group_anomaly_events(df, is_anom_col="is_anom", max_gap_minutes=None, freq_minutes=None):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_id", "start_time", "end_time", "duration_minutes",
                "points", "max_abs_z", "max_value", "min_value", "mean_value"
            ]
        )

    if freq_minutes is None:
        freq_minutes = infer_freq_minutes(df.index)
    if max_gap_minutes is None:
        max_gap_minutes = freq_minutes * 1.5

    anoms = df[df[is_anom_col]].copy()
    if anoms.empty:
        return pd.DataFrame(
            columns=[
                "event_id", "start_time", "end_time", "duration_minutes",
                "points", "max_abs_z", "max_value", "min_value", "mean_value"
            ]
        )

    gap = anoms.index.to_series().diff()
    new_event = gap > pd.Timedelta(minutes=max_gap_minutes)
    event_id = new_event.cumsum()

    temp = anoms.copy()
    temp["event_id"] = event_id.values
    temp = temp.reset_index().rename(columns={temp.index.name or "index": "timestamp"})

    grouped = temp.groupby("event_id")
    agg_spec = {
        "start_time": ("timestamp", "min"),
        "end_time": ("timestamp", "max"),
        "points": ("timestamp", "size"),
        "max_value": ("value", "max"),
        "min_value": ("value", "min"),
        "mean_value": ("value", "mean"),
    }
    if "robust_z" in temp.columns:
        agg_spec["max_abs_z"] = ("robust_z", lambda x: float(np.nanmax(np.abs(x))))
    summary = grouped.agg(**agg_spec).reset_index()
    if "max_abs_z" not in summary.columns:
        summary["max_abs_z"] = np.nan

    duration = summary["end_time"] - summary["start_time"]
    summary["duration_minutes"] = duration.dt.total_seconds() / 60.0 + freq_minutes
    return summary


def plot_anomalies(df, value_col="value", is_anom_col="is_anom", out_path="outputs/kw_anomalies.png"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df[value_col], color="tab:blue", label="kW")
    ax.scatter(
        df.index[df[is_anom_col]],
        df[value_col][df[is_anom_col]],
        color="red",
        s=10,
        label="anomaly"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("kW")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path
