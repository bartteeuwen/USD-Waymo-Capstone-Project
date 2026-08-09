from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

def load_processed_data():
    """
    Loads pre-computed scenario indices and triaged data directly from disk.
    Avoids re-parsing GCS raw TFRecords.
    """
    index_path = DATA_DIR / "tfrecord_index.parquet"
    triaged_path = DATA_DIR / "triaged_scenarios.csv"
    
    if not index_path.exists() or not triaged_path.exists():
        raise FileNotFoundError("Processed artifacts missing in data/. Run Data_Preparation.ipynb first.")
        
    index_df = pd.read_parquet(index_path)
    triaged_df = pd.read_csv(triaged_path)
    
    print(f"Successfully loaded {len(index_df):,} indexed scenarios.")
    return index_df, triaged_df
