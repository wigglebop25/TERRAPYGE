"""Run GNN dual-edge training and baseline comparison."""
import sys
sys.path.insert(0, r'D:\TERRAPYGE')

import torch
import numpy as np
from pathlib import Path

from src.terrapyge.data.graph import load_graph
from src.terrapyge.models.gnn import get_model
from src.terrapyge.models.training import (
    load_config, generate_synthetic_labels, create_masks,
    train_model, save_results
)
from src.terrapyge.models.baselines import run_baselines, save_comparison

# Paths
PROC_DIR = Path(r'D:\TERRAPYGE\data\processed\buhisan')
RESULTS_DIR = Path(r'D:\TERRAPYGE\results')
MODELS_DIR = Path(r'D:\TERRAPYGE\models')

# Load config
config = load_config(r'D:\TERRAPYGE\config.yaml')

# Load graph
print('Loading graph...')
data = load_graph(PROC_DIR / 'buhisan_hetero.pt')
print(f'  Nodes: {data["su"].num_nodes}')
print(f'  Features: {data["su"].x.shape[1]}')
print(f'  Spatial edges: {data["su", "spatial", "su"].edge_index.shape[1]}')
print(f'  Hydro edges: {data["su", "hydro", "su"].edge_index.shape[1]}')

# Generate synthetic labels if not present
if 'y' not in data['su']:
    print('\nGenerating synthetic labels...')
    data['su'].y = generate_synthetic_labels(
        data['su'].x,
        pos_rate=config['training']['pos_rate'],
        seed=config['training']['seed']
    )
print(f'  Labels: {data["su"].y.sum().item()} positive / {data["su"].num_nodes} total ({data["su"].y.float().mean():.1%})')

# Create train/val/test masks
print('\nCreating data splits...')
train_mask, val_mask, test_mask = create_masks(
    data['su'].num_nodes,
    train_ratio=config['training']['train_ratio'],
    val_ratio=config['training']['val_ratio'],
    seed=config['training']['seed']
)
data['su'].train_mask = train_mask
data['su'].val_mask = val_mask
data['su'].test_mask = test_mask
print(f'  Train: {train_mask.sum().item()} | Val: {val_mask.sum().item()} | Test: {test_mask.sum().item()}')

# === GNN Dual-Edge Training ===
print('\n' + '=' * 60)
print('GNN DUAL-EDGE TRAINING (Spatial + Hydro)')
print('=' * 60)

gnn_cfg = config['gnn']
model = get_model(
    model_type=gnn_cfg['model_type'],
    in_channels=data['su'].x.shape[1],
    hidden_channels=gnn_cfg['hidden_channels'],
    out_channels=2,
    num_layers=gnn_cfg['num_layers'],
    dropout=gnn_cfg['dropout'],
)

gnn_results = train_model(
    model, data, config,
    model_name='gnn_dual_edge',
    save_dir=MODELS_DIR,
)

# Save GNN results
save_results(gnn_results, RESULTS_DIR / 'model_comparison.json')

# === Baseline Training ===
print('\n' + '=' * 60)
print('BASELINE COMPARISON')
print('=' * 60)

# Extract features and labels for baselines
X = data['su'].x.numpy()
y = data['su'].y.numpy()

X_train = X[train_mask.numpy()]
y_train = y[train_mask.numpy()]
X_test = X[test_mask.numpy()]
y_test = y[test_mask.numpy()]

baseline_results = run_baselines(X_train, y_train, X_test, y_test, config)

# Add GNN results to comparison
baseline_results['GNN (dual-edge)'] = {
    'model': 'GNN (dual-edge)',
    'auc': gnn_results['test_auc'],
    'ap': gnn_results['test_ap'],
    'f1': gnn_results['test_f1'],
}

# Add spatial-only reference (from existing metrics.json)
import json
existing_metrics_path = PROC_DIR / 'metrics.json'
if existing_metrics_path.exists():
    with open(existing_metrics_path) as f:
        existing = json.load(f)
    baseline_results['GNN (spatial-only)'] = {
        'model': 'GNN (spatial-only)',
        'auc': existing.get('test_auc', 0),
        'ap': existing.get('test_ap', 0),
        'f1': existing.get('test_f1', 0),
    }

# Save comparison
save_comparison(baseline_results, RESULTS_DIR / 'baseline_comparison.json')

# Print final comparison
print('\n' + '=' * 60)
print('FINAL COMPARISON')
print('=' * 60)
print(f'{"Model":<25} {"AUC":>8} {"AP":>8} {"F1":>8}')
print('-' * 50)
for name, metrics in baseline_results.items():
    print(f'{name:<25} {metrics["auc"]:>8.4f} {metrics["ap"]:>8.4f} {metrics["f1"]:>8.4f}')

print('\nDone.')
