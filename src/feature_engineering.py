import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans

# Default path where artifacts are stored
KMEANS_ARTIFACT_PATH = "artifacts/kmeans_target.pkl"

def assign_kinematic_risk_target(
    df: pd.DataFrame, 
    kmeans_path: str = KMEANS_ARTIFACT_PATH
) -> pd.DataFrame:
    """
    Assigns kinematic risk target using a persistent KMeans model 
    to prevent target label swapping across dataset runs.
    """
    required = ['max_velocity_mps', 'max_deceleration']
    
    # Check for required columns
    if missing := set(required).difference(df.columns):
        raise ValueError(f"Missing required columns for risk assignment: {sorted(missing)}")

    df_out = df.copy()

    # --- STEP 1: LOAD OR FIT KMEANS ---
    if os.path.exists(kmeans_path):
        # Load pre-fitted model to ensure label indices (0, 1, 2, 3) stay deterministic
        kmeans = joblib.load(kmeans_path)
        df_out['risk_tier_numeric'] = kmeans.predict(df_out[required])
    else:
        # Fit once on baseline dataset, save artifact, then predict
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_out['risk_tier_numeric'] = kmeans.fit_predict(df_out[required])
        
        # Save clusterer artifact
        os.makedirs(os.path.dirname(kmeans_path), exist_ok=True)
        joblib.dump(kmeans, kmeans_path)

    # --- STEP 2: MAP CLUSTERS TO CATEGORIES ---
    # Note: Inspect cluster centers on first run to confirm cluster indices
    risk_labels = {
        0: 'Standard Complexity', 
        1: 'Critical Complexity',
        2: 'Standard Complexity', 
        3: 'Critical Complexity'
    }
    
    df_out['target_risk_matrix'] = df_out['risk_tier_numeric'].map(risk_labels)
    return df_out
