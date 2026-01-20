from pathlib import Path
from datetime import datetime
import traceback
import sys

from kw_anomaly_utils import (
    load_timeseries,
    robust_rolling_zscore,
    group_anomaly_events,
    plot_anomalies,
    infer_freq_minutes,
)


DATA_PATH = r"C:\Users\mafrousheh\Codes\kW_Anomaly Detection\kW data - S - MD US.xlsx"
SHEET_NAME = "Sysco Baltimore - Jessup, MD - "
TIME_COL = "time"
KW_COL = "Aggregated Demand (Peak)"

WINDOW = 96 * 7  # 7 days of 15-min data
Z_THRESH = 5.0
MIN_CONSECUTIVE = 3

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_timeseries(
        DATA_PATH,
        sheet_name=SHEET_NAME,
        time_col=TIME_COL,
        value_col=KW_COL,
    )

    res = robust_rolling_zscore(
        df["kw"],
        window=WINDOW,
        z_thresh=Z_THRESH,
        min_consecutive=MIN_CONSECUTIVE,
    )

    out = df.join(res)
    freq_minutes = infer_freq_minutes(out.index)
    events = group_anomaly_events(
        out,
        is_anom_col="is_anom",
        max_gap_minutes=freq_minutes * 1.5,
        freq_minutes=freq_minutes,
    )

    out_path_points = OUTPUT_DIR / "kw_anomaly_points.csv"
    out_path_events = OUTPUT_DIR / "kw_anomaly_events.csv"
    out.to_csv(out_path_points, index=True)
    events.to_csv(out_path_events, index=False)

    plot_path = OUTPUT_DIR / "kw_anomalies.png"
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
