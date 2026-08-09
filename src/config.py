import os

# --- Paths & Storage ---
GCS_BUCKET_PATH = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s"
# The public training split contains 1,000 shards.  We intentionally process
# only the first 100 by default, but their filenames still end in `of-01000`.
NUM_SHARDS = 100
TOTAL_TRAINING_SHARDS = 1000
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

# --- Feature Definitions ---
STATIC_FEATURES = [
    'lane_count', 'stop_sign_count', 'crosswalk_count', 'speed_bump_count',
    'vehicle_count', 'pedestrian_count', 'cyclist_count'
]

EXPANDED_FEATURES = STATIC_FEATURES + [
    'min_inter_agent_dist', 'avg_agent_velocity', 'velocity_std'
]

# --- Model Hyperparameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_VELOCITY_THRESHOLD = 38.0  # m/s (~85 mph) physics filter limit
