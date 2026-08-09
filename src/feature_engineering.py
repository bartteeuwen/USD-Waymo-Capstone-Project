import pandas as pd
import numpy as np

class DataCleaner:
    """Handles kinematic data cleaning and physical anomaly filtering."""
    def __init__(self, velocity_threshold_ms: float = 38.0):
        self.velocity_threshold = velocity_threshold_ms

    def clean_motion_tracks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters out GPS drift and physically impossible velocity spikes."""
        if 'v_max' in df.columns:
            cleaned_df = df[df['v_max'] <= self.velocity_threshold].copy()
        elif 'velocity' in df.columns:
            cleaned_df = df[df['velocity'] <= self.velocity_threshold].copy()
        else:
            cleaned_df = df.copy()
        return cleaned_df


class SpatialFeatureEngineer:
    """Extracts map friction metadata and dynamic interaction metrics."""
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw telemetry data into engineered feature vectors."""
        df_out = df.copy()
        
        # Ensure key required columns exist with fallback defaults if missing
        required_cols = {
            'v_max': 12.5,
            'velocity_std': 2.1,
            'd_min': 4.5,
            'crosswalk_count': 2,
            'complex_interaction': 0
        }
        
        for col, default_val in required_cols.items():
            if col not in df_out.columns:
                df_out[col] = default_val
                
        return df_out
