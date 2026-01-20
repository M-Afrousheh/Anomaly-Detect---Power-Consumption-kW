# Anomaly Detection - Power Consumption (kW)

This repo contains anomaly detection pipelines for refrigeration facility kW data.

## Files
- `kW Anomaly detect, MD.py`: rolling robust z-score baseline + event grouping
- `kw_anomaly_autoencoder.py`: z-score + autoencoder (reconstruction error)
- `kw_anomaly_autoencoder_only.py`: autoencoder only
- `kw_anomaly_utils.py`: shared utilities
- `kW data - S - MD US.xlsx`: dataset

## Setup (Windows)
```
python -m venv C:\venv
C:\venv\Scripts\python.exe -m pip install --upgrade pip
C:\venv\Scripts\python.exe -m pip install pandas matplotlib numpy scikit-learn tensorflow openpyxl
```

## Run
Baseline:
```
C:\venv\Scripts\python.exe "kW Anomaly detect, MD.py"
```

Autoencoder + z-score:
```
C:\venv\Scripts\python.exe "kw_anomaly_autoencoder.py"
```

Autoencoder only:
```
C:\venv\Scripts\python.exe "kw_anomaly_autoencoder_only.py"
```

## Output
Each run writes results to a timestamped folder under `outputs/`, including:
- anomaly points CSV
- anomaly events CSV
- plot PNG
