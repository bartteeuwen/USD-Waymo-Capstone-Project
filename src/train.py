import os
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import (
    GCS_BUCKET_PATH, NUM_SHARDS, TOTAL_TRAINING_SHARDS, ARTIFACTS_DIR,
    STATIC_FEATURES, EXPANDED_FEATURES, RANDOM_STATE, TEST_SIZE
)
from src.data_loader import load_and_extract_waymo_scenarios
from src.modeling import SpatialGCN, GraphAttentionNet
from src.feature_engineering import SpatialFeatureEngineer, assign_kinematic_risk_target

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("🚀 Starting Waymo Risk Modeling Pipeline...")

    # 1. Load Data
    raw_df, gnn_extracted_scenes = load_and_extract_waymo_scenarios(
        GCS_BUCKET_PATH, NUM_SHARDS, TOTAL_TRAINING_SHARDS
    )
    
    # Filter valid physics
    df_final = raw_df[raw_df['is_valid_physics']].copy()

    # 2. Assign the same K-Means target used in the notebook.
    df_final = assign_kinematic_risk_target(df_final)
    y = df_final['target_risk_matrix'].map({'Standard Complexity': 0, 'Critical Complexity': 1})

    # 3. Engineer expanded features while retaining scene-to-row alignment.
    df_final = SpatialFeatureEngineer().transform(df_final, gnn_extracted_scenes)

    # 4. Tabular Data Splitting & Model Training
    X_expanded = df_final[EXPANDED_FEATURES]
    X_tr_exp, X_te_exp, y_train, y_test = train_test_split(
        X_expanded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Training Tabular Models (RF, XGBoost, LightGBM)...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)
    rf_model.fit(X_tr_exp, y_train)

    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE, eval_metric='logloss')
    xgb_model.fit(X_tr_exp, y_train)

    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)
    lgb_model.fit(X_tr_exp, y_train)

    # Save Tabular Artifacts
    joblib.dump(rf_model, os.path.join(ARTIFACTS_DIR, 'rf_model.pkl'))
    joblib.dump(xgb_model, os.path.join(ARTIFACTS_DIR, 'xgb_model.pkl'))
    joblib.dump(lgb_model, os.path.join(ARTIFACTS_DIR, 'lgb_model.pkl'))
    print("✅ Tabular models trained and saved to artifacts/")

    # 5. Graph Neural Network Dataset Construction & Training
    print("Constructing GNN Scene Graphs...")
    gnn_dataset = []
    scene_by_id = {scene['scenario_id']: scene for scene in gnn_extracted_scenes}
    clean_scenes = [scene_by_id[row.scenario_id] for _, row in df_final.iterrows()]

    for scene, target, (_, row) in zip(clean_scenes, y, df_final.iterrows()):
        agents = scene['agents']
        if len(agents) == 0: continue

        node_features = []
        positions = []
        for a in agents:
            node_features.append([
                float(a['agent_type']), a['velocity'], scene['roadgraph_features_count'],
                row['lane_count'], row['stop_sign_count'], row['crosswalk_count'], row['speed_bump_count']
            ])
            positions.append([a['pos_x'], a['pos_y']])

        x = torch.tensor(node_features, dtype=torch.float)
        sources, targets = [], []
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j and math.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]) <= 60.0:
                    sources.append(i)
                    targets.append(j)

        edge_index = torch.empty((2, 0), dtype=torch.long) if len(sources) == 0 else torch.tensor([sources, targets], dtype=torch.long)
        y_val = torch.tensor([target], dtype=torch.long)
        gnn_dataset.append(Data(x=x, edge_index=edge_index, y=y_val))

    gnn_targets = [g.y.item() for g in gnn_dataset]
    tr_g, te_g = train_test_split(gnn_dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=gnn_targets)

    train_loader = DataLoader(tr_g, batch_size=32, shuffle=True)
    test_loader = DataLoader(te_g, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn_model = SpatialGCN(in_channels=7).to(device)
    optimizer = Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = CrossEntropyLoss()

    print("Training Spatial GCN Model...")
    gcn_model.train()
    for epoch in range(1, 16):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = gcn_model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    # Save GNN Artifacts
    torch.save(gcn_model.state_dict(), os.path.join(ARTIFACTS_DIR, 'spatial_gcn.pt'))
    print("✅ Spatial GCN trained and saved to artifacts/")

if __name__ == "__main__":
    main()
