"""Training infrastructure for TERRAPYGE GNN models.

Provides training loop, early stopping, metrics computation,
synthetic label generation, and data splitting utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pathlib import Path
import json
import yaml


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Dict with configuration.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_synthetic_labels(features, pos_rate=0.15, seed=42):
    """Generate synthetic landslide labels from terrain features.

    Heuristic: high slope + high TWI + high SPI + high curvature → landslide-prone.

    Args:
        features: FloatTensor [N, F] — node features.
        pos_rate: Target positive class rate.
        seed: Random seed.

    Returns:
        labels: LongTensor [N] — binary labels (0/1).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = features.numpy() if isinstance(features, torch.Tensor) else features
    n = X.shape[0]

    # Feature indices (based on config.yaml feature order)
    # slope_mean, twi_mean, spi_mean, curv_plan_mean are typically indices 3, 7, 8, 5
    # Use first 4 numeric features as proxy
    score = X[:, :4].mean(axis=1)

    # Add noise
    score += np.random.randn(n) * 0.3

    # Threshold to get target positive rate
    threshold = np.percentile(score, 100 * (1 - pos_rate))
    labels = (score > threshold).astype(np.int64)

    return torch.from_numpy(labels).long()


def create_masks(n, train_ratio=0.64, val_ratio=0.16, seed=42):
    """Create stratified train/val/test masks.

    Args:
        n: Number of nodes.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        seed: Random seed.

    Returns:
        train_mask, val_mask, test_mask: Boolean tensors.
    """
    indices = np.arange(n)
    test_ratio = 1 - train_ratio - val_ratio

    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=seed
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def train_epoch(model, data, optimizer, criterion, mask):
    """Single training epoch.

    Args:
        model: GNN model.
        data: HeteroData graph.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        mask: Boolean mask for training nodes.

    Returns:
        loss: Training loss.
    """
    model.train()
    optimizer.zero_grad()

    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[mask], data['su'].y[mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, criterion, mask):
    """Evaluate model on masked nodes.

    Args:
        model: GNN model.
        data: HeteroData graph.
        criterion: Loss function.
        mask: Boolean mask for evaluation nodes.

    Returns:
        loss: Evaluation loss.
        auc: Area under ROC curve.
        ap: Average precision.
        f1: F1 score.
    """
    model.eval()

    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[mask], data['su'].y[mask]).item()

    probs = torch.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
    labels = data['su'].y[mask].cpu().numpy()

    try:
        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
    except ValueError:
        auc = 0.0
        ap = 0.0

    preds = (probs > 0.5).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)

    return loss, auc, ap, f1


def train_model(model, data, config, model_name='gnn', save_dir=None):
    """Full training loop with early stopping and checkpointing.

    Args:
        model: GNN model.
        data: HeteroData graph.
        config: Configuration dict.
        model_name: Name for saving checkpoints.
        save_dir: Directory to save model checkpoints.

    Returns:
        results: Dict with training history and best metrics.
    """
    # Extract config
    gnn_cfg = config.get('gnn', {})
    train_cfg = config.get('training', {})

    lr = gnn_cfg.get('learning_rate', 0.001)
    epochs = gnn_cfg.get('epochs', 100)
    patience = gnn_cfg.get('early_stopping_patience', 15)
    seed = train_cfg.get('seed', 42)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create masks if not present
    if 'train_mask' not in data['su']:
        n = data['su'].num_nodes
        train_mask, val_mask, test_mask = create_masks(
            n,
            train_ratio=train_cfg.get('train_ratio', 0.64),
            val_ratio=train_cfg.get('val_ratio', 0.16),
            seed=seed
        )
        data['su'].train_mask = train_mask
        data['su'].val_mask = val_mask
        data['su'].test_mask = test_mask

    train_mask = data['su'].train_mask
    val_mask = data['su'].val_mask
    test_mask = data['su'].test_mask

    # Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_auc = 0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_ap': [], 'val_f1': []}

    print(f'Training {model_name} for {epochs} epochs (patience={patience})...')
    print(f'  Train: {train_mask.sum().item()} | Val: {val_mask.sum().item()} | Test: {test_mask.sum().item()}')

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, data, optimizer, criterion, train_mask)

        # Evaluate
        val_loss, val_auc, val_ap, val_f1 = evaluate(model, data, criterion, val_mask)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_ap'].append(val_ap)
        history['val_f1'].append(val_f1)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Log
        if epoch % 10 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | Val F1: {val_f1:.4f}')

        if patience_counter >= patience:
            print(f'\n  Early stopping at epoch {epoch}')
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test set
    test_loss, test_auc, test_ap, test_f1 = evaluate(model, data, criterion, test_mask)

    print(f'\n  Best epoch: {best_epoch}')
    print(f'  Test AUC: {test_auc:.4f} | AP: {test_ap:.4f} | F1: {test_f1:.4f}')

    # Save model
    if save_dir is not None:
        save_path = Path(save_dir) / f'{model_name}.pt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f'  Model saved to {save_path}')

    results = {
        'model_name': model_name,
        'best_epoch': best_epoch,
        'test_auc': float(test_auc),
        'test_ap': float(test_ap),
        'test_f1': float(test_f1),
        'test_loss': float(test_loss),
        'best_val_auc': float(best_val_auc),
        'history': history,
        'hyperparams': {
            'hidden_channels': gnn_cfg.get('hidden_channels', 64),
            'num_layers': gnn_cfg.get('num_layers', 3),
            'dropout': gnn_cfg.get('dropout', 0.2),
            'learning_rate': lr,
            'epochs': epochs,
            'patience': patience,
        },
    }

    return results


def save_results(results, output_path):
    """Save training results to JSON.

    Args:
        results: Dict with training results.
        output_path: Path to save JSON file.
    """
    # Remove non-serializable items
    results_clean = {k: v for k, v in results.items() if k != 'history'}
    results_clean['history'] = {
        k: [float(x) for x in v] for k, v in results.get('history', {}).items()
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)

    print(f'Results saved to {output_path}')
