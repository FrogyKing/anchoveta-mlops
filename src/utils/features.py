import pandas as pd
import numpy as np

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies shared feature engineering to the dataset.
    Ensures zero train-serve skew.
    """
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["cell_id", "fecha"]).reset_index(drop=True)
    
    # Cyclic features
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    
    # Anomalies and Rolling Windows
    df["sst_anom"] = df["sst_c"] - df.groupby("cell_id")["sst_c"].transform("mean")
    df["chl_roll4"] = df.groupby("cell_id")["clorofila_mg_m3"] \
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["ox_lag1"] = df.groupby("cell_id")["oxigeno_ml_l"].shift(1)
    
    # Fill NAs introduced by lags/rolling (simple backfill/forwardfill per group if needed, or 0)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

FEATURES = [
    "latitud", "longitud", "zona_pesca", "dist_costa_deg",
    "sst_c", "clorofila_mg_m3", "profundidad_m", "salinidad_psu",
    "oxigeno_ml_l", "enso_index",
    "corriente_vel_ms", "corriente_u_ms", "corriente_v_ms",
    "semana_anio", "mes_sin", "mes_cos",
    "sst_roll4", "chl_lag1", "sst_anom", "chl_roll4", "ox_lag1"
]
TARGET = "log_densidad"
