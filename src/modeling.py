import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
import math


class SpatialGCN(torch.nn.Module):
    """Two-layer graph convolutional baseline used by the training pipeline."""
    def __init__(self, in_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(global_mean_pool(x, batch))


def build_scene_graphs(df, scenes, targets, proximity_meters: float = 60.0):
    """Build graph samples while joining extracted scenes to rows by ID."""
    scene_by_id = {scene['scenario_id']: scene for scene in scenes}
    graph_dataset = []
    for row_index, row in df.iterrows():
        scene = scene_by_id.get(row['scenario_id'])
        if scene is None or not scene['agents']:
            continue
        agents = scene['agents']
        positions = [(agent['pos_x'], agent['pos_y']) for agent in agents]
        node_features = [[
            float(agent['agent_type']), agent['velocity'],
            scene['roadgraph_features_count'], row['lane_count'],
            row['stop_sign_count'], row['crosswalk_count'], row['speed_bump_count'],
        ] for agent in agents]
        sources, destinations = [], []
        for source, (x1, y1) in enumerate(positions):
            for destination, (x2, y2) in enumerate(positions):
                if source != destination and math.hypot(x1 - x2, y1 - y2) <= proximity_meters:
                    sources.append(source)
                    destinations.append(destination)
        edge_index = (
            torch.tensor([sources, destinations], dtype=torch.long)
            if sources else torch.empty((2, 0), dtype=torch.long)
        )
        graph_dataset.append(Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor([int(targets.loc[row_index])], dtype=torch.long),
        ))
    return graph_dataset

def initialize_tabular_models(X_tr_exp, y_train, X_te_exp, y_test):
    """Initializes and trains tabular XGBoost and LightGBM models."""
    
    # 1. XGBoost Expanded Classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.05, 
        random_state=42, 
        eval_metric='logloss'
    )
    xgb_model.fit(X_tr_exp, y_train)
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_te_exp))

    # 2. LightGBM Expanded Classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.05, 
        random_state=42, 
        verbose=-1
    )
    lgb_model.fit(X_tr_exp, y_train)
    lgb_acc = accuracy_score(y_test, lgb_model.predict(X_te_exp))

    print(f"Tabular Models Trained | XGB Acc: {xgb_acc:.4f} | LGB Acc: {lgb_acc:.4f}")
    return xgb_model, lgb_model

class GraphAttentionNet(torch.nn.Module):
    """
    Graph Attention Network (GAT) for spatial agent topology modeling 
    with dual mean and max pooling.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.gat1 = GATConv(in_channels, 16, heads=2, concat=True)
        self.gat2 = GATConv(32, 16, heads=1, concat=False)
        self.classifier = torch.nn.Linear(32, 2)  # Binary risk classification

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        
        # Combine mean pooling (overall scene context) + max pooling (extreme velocity/braking signal)
        scene_vec = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.classifier(scene_vec)
