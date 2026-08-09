"""Reusable tuning and evaluation functions used by the white-paper notebook."""
import copy

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from torch_geometric.nn import GCNConv, global_mean_pool


def evaluate_classifier(model, X_test, y_test):
    """Return the white paper's classification metrics for a fitted model."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else predictions
    return {
        'Test Acc': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1-Score': f1_score(y_test, predictions, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, probabilities),
    }


def tune_tabular_models(X_train, y_train):
    """Use the same randomized-search grids reported in the original notebook."""
    searches = {
        'Random Forest (Tuned)': RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            {'n_estimators': [100, 200, 300], 'max_depth': [5, 8, 12, None],
             'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
            n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1),
        'XGBoost (Tuned)': RandomizedSearchCV(
            xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7, 9],
             'learning_rate': [0.01, 0.03, 0.05, 0.1], 'subsample': [0.7, 0.8, 1.0],
             'colsample_bytree': [0.7, 0.8, 1.0]},
            n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1),
        'LightGBM (Tuned)': RandomizedSearchCV(
            lgb.LGBMClassifier(random_state=42, verbose=-1),
            {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7, 9],
             'learning_rate': [0.01, 0.03, 0.05, 0.1], 'num_leaves': [15, 31, 63],
             'subsample': [0.7, 0.8, 1.0]},
            n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1),
    }
    return {name: search.fit(X_train, y_train).best_estimator_ for name, search in searches.items()}


class TunedRescuedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_rate):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(F.relu(self.fc1(global_mean_pool(x, batch))))


def tune_gcn(train_loader, test_loader, device):
    """Run the original four-setting ROC-AUC-guided GCN tuning loop."""
    labels = np.array([graph.y.item() for graph in train_loader.dataset])
    counts = np.bincount(labels, minlength=2)
    weights = torch.tensor(len(labels) / (2.0 * counts), dtype=torch.float, device=device)
    parameter_grid = [
        {'hidden_channels': 32, 'lr': 0.005, 'weight_decay': 1e-4, 'dropout': 0.1},
        {'hidden_channels': 32, 'lr': 0.003, 'weight_decay': 1e-4, 'dropout': 0.2},
        {'hidden_channels': 64, 'lr': 0.003, 'weight_decay': 1e-4, 'dropout': 0.2},
        {'hidden_channels': 64, 'lr': 0.001, 'weight_decay': 5e-4, 'dropout': 0.3},
    ]
    best = None
    for params in parameter_grid:
        model = TunedRescuedGCN(7, params['hidden_channels'], params['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss(weight=weights)
        model.train()
        for _ in range(35):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                criterion(model(batch.x, batch.edge_index, batch.batch), batch.y).backward()
                optimizer.step()
        model.eval()
        targets, probabilities, predictions = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index, batch.batch)
                probabilities.extend(F.softmax(output, dim=1)[:, 1].cpu().numpy())
                predictions.extend(output.argmax(dim=1).cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        metrics = {'Test Acc': accuracy_score(targets, predictions),
                   'Precision': precision_score(targets, predictions, zero_division=0),
                   'Recall': recall_score(targets, predictions, zero_division=0),
                   'F1-Score': f1_score(targets, predictions, zero_division=0),
                   'ROC-AUC': roc_auc_score(targets, probabilities)}
        if best is None or metrics['ROC-AUC'] > best['metrics']['ROC-AUC']:
            best = {'model': copy.deepcopy(model), 'params': params, 'metrics': metrics}
    return best
