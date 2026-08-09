import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score

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
