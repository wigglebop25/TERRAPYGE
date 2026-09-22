"""Shared GNN experiment helpers for TERRAPYGE.

Training / evaluation utilities reused by the physics ablation, the spatial
masking experiment, and the hazard-map inference script. Kept separate from
``training.py`` (which holds the general training loop) so the experiment
drivers share one implementation without importing each other's scripts.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.terrapyge.models.gnn import get_model
from src.terrapyge.utils.paths import ABLATION_MODELS

# Node-feature schema for the physics graph (21 features; index 20 is the
# label source ``newmark_dn_cm`` and is therefore excluded from model inputs).
N_FEATURES = 20
FEATURE_SLICE = slice(0, N_FEATURES)

# Topographic + slope-derived features that encode the label's main driver.
# Masking slope alone is insufficient: TWI, SPI, plan/profile curvature and the
# physics descriptors (static FS, critical acceleration) are all functions of
# slope and leak the label, so the whole topographic block is masked together.
MASK_COLS = [0, 1, 2, 3, 4, 5, 6, 17, 18]
KEPT_COLS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19]

# GNN backbone classes exposed by the ablation experiment.
GNN_MODELS = ['HeteroGCN', 'HeteroSAGE', 'HeteroGAT']
EDGE_CONFIGS = ['spatial_only', 'hydro_only', 'dual_edge']


def get_edge_configs(data):
    """Return the 3 edge-config variants (spatial / hydro / dual) from a graph."""
    spatial_edge_index = data['su', 'spatial', 'su'].edge_index
    hydro_edge_index = data['su', 'hydro', 'su'].edge_index
    empty = torch.zeros((2, 0), dtype=torch.long)

    return {
        'spatial_only': {
            ('su', 'spatial', 'su'): spatial_edge_index,
            ('su', 'hydro', 'su'): empty,
        },
        'hydro_only': {
            ('su', 'spatial', 'su'): empty,
            ('su', 'hydro', 'su'): hydro_edge_index,
        },
        'dual_edge': {
            ('su', 'spatial', 'su'): spatial_edge_index,
            ('su', 'hydro', 'su'): hydro_edge_index,
        },
    }


def train_gnn_experiment(data, edge_index_dict, model_type, config, model_name,
                         models_dir=None):
    """Train a single GNN experiment and return a results dict.

    Uses the split masks already present on ``data['su']`` and early-stops on
    validation AUC. Saves the best state dict to ``models_dir/{model_name}.pt``.
    """
    if models_dir is None:
        models_dir = ABLATION_MODELS

    gnn_cfg = config.get('gnn', {})
    lr = gnn_cfg.get('learning_rate', 0.001)
    epochs = gnn_cfg.get('epochs', 100)
    patience = gnn_cfg.get('early_stopping_patience', 15)
    seed = config.get('training', {}).get('seed', 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build x_dict for HeteroConv
    x_dict = {'su': data['su'].x}

    # Create model
    model = get_model(
        model_type=model_type,
        in_channels=data['su'].x.shape[1],
        hidden_channels=gnn_cfg.get('hidden_channels', 64),
        out_channels=2,
        num_layers=gnn_cfg.get('num_layers', 3),
        dropout=gnn_cfg.get('dropout', 0.2),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_mask = data['su'].train_mask
    val_mask = data['su'].val_mask
    test_mask = data['su'].test_mask

    best_val_auc = 0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_ap': [], 'val_f1': []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, edge_index_dict)
        loss = criterion(out[train_mask], data['su'].y[train_mask])
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(x_dict, edge_index_dict)
            val_loss = criterion(out[val_mask], data['su'].y[val_mask]).item()

            probs = torch.softmax(out[val_mask], dim=1)[:, 1].cpu().numpy()
            labels = data['su'].y[val_mask].cpu().numpy()

            try:
                val_auc = roc_auc_score(labels, probs)
                val_ap = average_precision_score(labels, probs)
            except ValueError:
                val_auc = 0.0
                val_ap = 0.0

            preds = (probs > 0.5).astype(int)
            val_f1 = f1_score(labels, preds, zero_division=0)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_ap'].append(val_ap)
        history['val_f1'].append(val_f1)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    training_time = time.time() - start_time

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        test_loss = criterion(out[test_mask], data['su'].y[test_mask]).item()
        probs = torch.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
        labels = data['su'].y[test_mask].cpu().numpy()

        try:
            test_auc = roc_auc_score(labels, probs)
            test_ap = average_precision_score(labels, probs)
        except ValueError:
            test_auc = 0.0
            test_ap = 0.0

        preds = (probs > 0.5).astype(int)
        test_f1 = f1_score(labels, preds, zero_division=0)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Save model
    save_path = models_dir / f'{model_name}.pt'
    torch.save(model.state_dict(), save_path)

    # Extract attention weights for GAT
    attention_weights = None
    if model_type == 'HeteroGAT' and best_state is not None:
        try:
            model.eval()
            with torch.no_grad():
                # Get attention from last GAT layer
                for conv in model.convs:
                    if hasattr(conv, 'convs'):
                        for edge_type, edge_conv in conv.convs.items():
                            if hasattr(edge_conv, 'attention'):
                                attention_weights = edge_conv.attention.detach().cpu().numpy()
                                break
        except Exception:
            pass

    return {
        'model': model_type,
        'edge_config': model_name.split('_', 1)[1] if '_' in model_name else model_name,
        'test_auc': float(test_auc),
        'test_ap': float(test_ap),
        'test_f1': float(test_f1),
        'num_params': int(num_params),
        'convergence_epoch': int(best_epoch),
        'training_time_sec': float(training_time),
        'history': {k: [float(x) for x in v] for k, v in history.items()},
        'attention_weights': attention_weights,
    }
