import pandas as pd

def filter_speed_anomalies(df: pd.DataFrame, max_speed_mps: float = 38.0) -> pd.DataFrame:
    """
    Flags and filters physical telemetry anomalies.
    Default threshold is 38.0 m/s (~85 mph).
    """
    if "speed" not in df.columns:
        return df
    
    clean_df = df[df["speed"] <= max_speed_mps].copy()
    anomalies_removed = len(df) - len(clean_df)
    print(f"[Physics Audit] Removed {anomalies_removed} anomaly records (> {max_speed_mps} m/s).")
    return clean_df
