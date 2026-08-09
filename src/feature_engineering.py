import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class DataCleaner:
    """Handles kinematic data cleaning and physical anomaly filtering."""
    def __init__(self, velocity_threshold_ms: float = 38.0):
        self.velocity_threshold = velocity_threshold_ms

    def clean_motion_tracks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters out GPS drift and physically impossible velocity spikes."""
        if 'max_velocity_mps' in df.columns:
            cleaned_df = df[df['max_velocity_mps'] <= self.velocity_threshold].copy()
        elif 'v_max' in df.columns:
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

    def transform(self, df: pd.DataFrame, scenes: list[dict]) -> pd.DataFrame:
        """Add per-scene spatial features, matched by ``scenario_id``.

        Matching by ID, rather than list position, keeps graph features aligned
        after the physics filter removes rows from the tabular dataset.
        """
        df_out = df.copy()
        scene_by_id = {scene['scenario_id']: scene for scene in scenes}
        metrics = []

        for scenario_id in df_out['scenario_id']:
            agents = scene_by_id.get(scenario_id, {}).get('agents', [])
            velocities = [float(agent['velocity']) for agent in agents]
            positions = [(agent['pos_x'], agent['pos_y']) for agent in agents]
            distances = [
                np.hypot(x1 - x2, y1 - y2)
                for index, (x1, y1) in enumerate(positions)
                for x2, y2 in positions[index + 1:]
            ]
            metrics.append({
                'min_inter_agent_dist': min(distances) if distances else 50.0,
                'avg_agent_velocity': float(np.mean(velocities)) if velocities else 0.0,
                'velocity_std': float(np.std(velocities)) if velocities else 0.0,
            })

        return df_out.join(pd.DataFrame(metrics, index=df_out.index))


def assign_kinematic_risk_target(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's four-cluster kinematic risk target."""
    required = ['max_velocity_mps', 'max_deceleration']
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"Cannot assign risk target; missing columns: {sorted(missing)}")

    df_out = df.copy()
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_out['risk_tier_numeric'] = kmeans.fit_predict(df_out[required])
    # This preserves the label definition used in the submitted notebook.
    risk_labels = {0: 'Standard Complexity', 1: 'Critical Complexity',
                   2: 'Standard Complexity', 3: 'Critical Complexity'}
    df_out['target_risk_matrix'] = df_out['risk_tier_numeric'].map(risk_labels)
    return df_out
