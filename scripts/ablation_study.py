"""TERRAPYGE Ablation Study: 12 experiments (9 GNN + 3 baselines).

Runs all experiments with identical conditions (same labels, splits, seed)
and generates comparison plots + summary table.
"""

import sys
sys.path.insert(0, r'D:\TERRAPYGE')

import torch
import numpy as np
import json
import time
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.terrapyge.data.graph import load_graph
from src.terrapyge.models.gnn import get_model
from src.terrapyge.models.training import (
    load_config, generate_synthetic_labels, create_masks,
    train_epoch, evaluate, save_results
)
from src.terrapyge.models.baselines import (
    LogisticRegressionBaseline, RandomForestBaseline, XGBoostBaseline
)

# Paths
PROC_DIR = Path(r'D:\TERRAPYGE\data\processed\buhisan')
RESULTS_DIR = Path(r'D:\TERRAPYGE\results')
MODELS_DIR = Path(r'D:\TERRAPYGE\models\ablation')
FIGURES_DIR = RESULTS_DIR / 'figures'

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load config
config = load_config(r'D:\TERRAPYGE\config.yaml')
gnn_cfg = config.get('gnn', {})
train_cfg = config.get('training', {})
baseline_cfg = config.get('baselines', {})


def get_edge_configs(data):
    """Return 3 edge config variants from the graph."""
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


def train_gnn_experiment(data, edge_index_dict, model_type, config, model_name):
    """Train a single GNN experiment and return results."""
    import torch.nn as nn

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

            from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
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
    save_path = MODELS_DIR / f'{model_name}.pt'
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
        except:
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


def run_ablation():
    """Run full ablation study."""
    print('=' * 70)
    print('TERRAPYGE ABLATION STUDY')
    print('=' * 70)

    # Load graph
    print('\nLoading graph...')
    data = load_graph(PROC_DIR / 'buhisan_hetero.pt')
    print(f'  Nodes: {data["su"].num_nodes}')
    print(f'  Features: {data["su"].x.shape[1]}')
    print(f'  Spatial edges: {data["su", "spatial", "su"].edge_index.shape[1]}')
    print(f'  Hydro edges: {data["su", "hydro", "su"].edge_index.shape[1]}')

    # Generate labels
    print('\nGenerating synthetic labels (seed=42)...')
    data['su'].y = generate_synthetic_labels(
        data['su'].x,
        pos_rate=train_cfg.get('pos_rate', 0.15),
        seed=train_cfg.get('seed', 42)
    )
    print(f'  Labels: {data["su"].y.sum().item()} positive / {data["su"].num_nodes} total')

    # Create splits
    print('\nCreating data splits (seed=42)...')
    train_mask, val_mask, test_mask = create_masks(
        data['su'].num_nodes,
        train_ratio=train_cfg.get('train_ratio', 0.64),
        val_ratio=train_cfg.get('val_ratio', 0.16),
        seed=train_cfg.get('seed', 42)
    )
    data['su'].train_mask = train_mask
    data['su'].val_mask = val_mask
    data['su'].test_mask = test_mask
    print(f'  Train: {train_mask.sum().item()} | Val: {val_mask.sum().item()} | Test: {test_mask.sum().item()}')

    # Get edge configs
    edge_configs = get_edge_configs(data)

    # Part 1: GNN Experiments (9 runs)
    print('\n' + '=' * 70)
    print('PART 1: GNN EXPERIMENTS (9 runs)')
    print('=' * 70)

    gnn_models = ['HeteroGCN', 'HeteroSAGE', 'HeteroGAT']
    gnn_experiments = {}

    for model_type in gnn_models:
        for edge_name, edge_index_dict in edge_configs.items():
            exp_name = f'{model_type.split("Hetero")[1].lower()}_{edge_name}'
            print(f'\n--- {exp_name} ---')

            result = train_gnn_experiment(
                data, edge_index_dict, model_type, config, exp_name
            )
            gnn_experiments[exp_name] = result

            print(f'  AUC: {result["test_auc"]:.4f} | AP: {result["test_ap"]:.4f} | F1: {result["test_f1"]:.4f}')
            print(f'  Params: {result["num_params"]} | Conv epoch: {result["convergence_epoch"]} | Time: {result["training_time_sec"]:.1f}s')

    # Part 2: Baseline Experiments (3 runs)
    print('\n' + '=' * 70)
    print('PART 2: BASELINE EXPERIMENTS (3 runs)')
    print('=' * 70)

    X = data['su'].x.numpy()
    y = data['su'].y.numpy()
    X_train = X[train_mask.numpy()]
    y_train = y[train_mask.numpy()]
    X_test = X[test_mask.numpy()]
    y_test = y[test_mask.numpy()]

    baseline_experiments = {}

    # Logistic Regression
    print('\n--- logistic_regression ---')
    lr_model = LogisticRegressionBaseline(baseline_cfg.get('logistic_regression', {}))
    start = time.time()
    lr_model.train(X_train, y_train)
    lr_time = time.time() - start
    lr_metrics = lr_model.evaluate(X_test, y_test)
    lr_metrics['training_time_sec'] = float(lr_time)
    lr_model.save(MODELS_DIR / 'lr.pkl')
    baseline_experiments['logistic_regression'] = lr_metrics
    print(f'  AUC: {lr_metrics["auc"]:.4f} | AP: {lr_metrics["ap"]:.4f} | F1: {lr_metrics["f1"]:.4f} | Time: {lr_time:.1f}s')

    # Random Forest
    print('\n--- random_forest ---')
    rf_model = RandomForestBaseline(baseline_cfg.get('random_forest', {}))
    start = time.time()
    rf_model.train(X_train, y_train)
    rf_time = time.time() - start
    rf_metrics = rf_model.evaluate(X_test, y_test)
    rf_metrics['training_time_sec'] = float(rf_time)
    rf_model.save(MODELS_DIR / 'rf.pkl')
    # Get feature importance
    rf_metrics['feature_importance'] = {f'f{i}': float(v) for i, v in enumerate(rf_model.model.feature_importances_)}
    baseline_experiments['random_forest'] = rf_metrics
    print(f'  AUC: {rf_metrics["auc"]:.4f} | AP: {rf_metrics["ap"]:.4f} | F1: {rf_metrics["f1"]:.4f} | Time: {rf_time:.1f}s')

    # XGBoost
    print('\n--- xgboost ---')
    xgb_model = XGBoostBaseline(baseline_cfg.get('xgboost', {}))
    start = time.time()
    xgb_model.train(X_train, y_train)
    xgb_time = time.time() - start
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_metrics['training_time_sec'] = float(xgb_time)
    xgb_model.save(MODELS_DIR / 'xgb.pkl')
    # Get feature importance
    xgb_metrics['feature_importance'] = {f'f{i}': float(v) for i, v in enumerate(xgb_model.model.feature_importances_)}
    baseline_experiments['xgboost'] = xgb_metrics
    print(f'  AUC: {xgb_metrics["auc"]:.4f} | AP: {xgb_metrics["ap"]:.4f} | F1: {xgb_metrics["f1"]:.4f} | Time: {xgb_time:.1f}s')

    # Part 3: Derived Metrics
    print('\n' + '=' * 70)
    print('PART 3: DERIVED METRICS')
    print('=' * 70)

    # Hydro improvement
    hydro_improvement = {}
    for model_short in ['gcn', 'sage', 'gat']:
        spatial_key = f'{model_short}_spatial_only'
        dual_key = f'{model_short}_dual_edge'
        if spatial_key in gnn_experiments and dual_key in gnn_experiments:
            spatial_auc = gnn_experiments[spatial_key]['test_auc']
            dual_auc = gnn_experiments[dual_key]['test_auc']
            diff = dual_auc - spatial_auc
            pct = (diff / spatial_auc * 100) if spatial_auc > 0 else 0
            hydro_improvement[model_short] = {
                'spatial_auc': float(spatial_auc),
                'dual_auc': float(dual_auc),
                'diff': float(diff),
                'pct': float(pct),
            }

    avg_diff = np.mean([v['diff'] for v in hydro_improvement.values()])
    avg_pct = np.mean([v['pct'] for v in hydro_improvement.values()])

    # Best model
    best_gnn = max(gnn_experiments.items(), key=lambda x: x[1]['test_auc'])
    best_baseline = max(baseline_experiments.items(), key=lambda x: x[1]['auc'])

    # Best edge config
    edge_aucs = {}
    for exp_name, exp in gnn_experiments.items():
        edge = exp['edge_config']
        if edge not in edge_aucs:
            edge_aucs[edge] = []
        edge_aucs[edge].append(exp['test_auc'])
    best_edge = max(edge_aucs.items(), key=lambda x: np.mean(x[1]))

    summary = {
        'hydro_improvement': {
            **hydro_improvement,
            'average_diff': float(avg_diff),
            'average_pct': float(avg_pct),
        },
        'best_model': {'name': best_gnn[0], 'auc': best_gnn[1]['test_auc']},
        'best_edge_config': {'name': best_edge[0], 'avg_auc': float(np.mean(best_edge[1]))},
        'best_baseline': {'name': best_baseline[0], 'auc': best_baseline[1]['auc']},
        'gnn_vs_baseline': {
            'best_gnn_auc': best_gnn[1]['test_auc'],
            'best_baseline_auc': best_baseline[1]['auc'],
            'diff': float(best_gnn[1]['test_auc'] - best_baseline[1]['auc']),
        },
    }

    # Part 4: Summary Table
    print('\n' + '=' * 70)
    print('ABLATION STUDY RESULTS (12 Experiments)')
    print('=' * 70)
    print(f'{"Model":<15} {"Edge Config":<15} {"AUC":>8} {"AP":>8} {"F1":>8} {"Params":>8} {"Conv":>6} {"Time":>8}')
    print('-' * 70)

    for exp_name, exp in gnn_experiments.items():
        model_short = exp['model'].replace('Hetero', '')
        print(f'{model_short:<15} {exp["edge_config"]:<15} {exp["test_auc"]:>8.4f} {exp["test_ap"]:>8.4f} {exp["test_f1"]:>8.4f} {exp["num_params"]:>8} {exp["convergence_epoch"]:>6} {exp["training_time_sec"]:>7.1f}s')

    print('-' * 70)
    for name, exp in baseline_experiments.items():
        print(f'{name:<15} {"-":<15} {exp["auc"]:>8.4f} {exp["ap"]:>8.4f} {exp["f1"]:>8.4f} {"-":>8} {"-":>6} {exp["training_time_sec"]:>7.1f}s')

    print('=' * 70)
    print('\nHYDRO IMPROVEMENT (dual vs spatial):')
    for model_short, imp in hydro_improvement.items():
        print(f'  {model_short.upper()}:   +{imp["diff"]:.4f} AUC (+{imp["pct"]:.1f}%)')
    print(f'  Avg:   +{avg_diff:.4f} AUC (+{avg_pct:.1f}%)')

    print(f'\nBEST MODEL: {best_gnn[0]} (AUC: {best_gnn[1]["test_auc"]:.4f})')
    print(f'BEST EDGE CONFIG: {best_edge[0]} (Avg AUC: {np.mean(best_edge[1]):.4f})')
    print(f'BEST GNN vs BEST BASELINE: {best_gnn[1]["test_auc"]:.4f} vs {best_baseline[1]["auc"]:.4f} (diff: {best_gnn[1]["test_auc"] - best_baseline[1]["auc"]:.4f})')
    print('=' * 70)

    # Part 5: Visualizations
    print('\n' + '=' * 70)
    print('PART 5: GENERATING VISUALIZATIONS')
    print('=' * 70)

    # Get feature names
    feat_names = data['su'].feat_names if hasattr(data['su'], 'feat_names') else [f'f{i}' for i in range(data['su'].x.shape[1])]

    # Plot 1: Ablation Comparison Bar Chart
    print('\nGenerating ablation_comparison.png...')
    plot_ablation_comparison(gnn_experiments, baseline_experiments, feat_names)

    # Plot 2: Training Curves
    print('Generating training_curves.png...')
    plot_training_curves(gnn_experiments)

    # Plot 3: Attention Heatmap
    print('Generating attention_heatmap.png...')
    plot_attention_heatmap(gnn_experiments, data)

    # Plot 4: Feature Importance
    print('Generating feature_importance.png...')
    plot_feature_importance(baseline_experiments, feat_names)

    # Save results
    results = {
        'gnn_experiments': {k: {kk: vv for kk, vv in v.items() if kk != 'attention_weights'} for k, v in gnn_experiments.items()},
        'baseline_experiments': baseline_experiments,
        'summary': summary,
    }

    output_path = RESULTS_DIR / 'ablation_study.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {output_path}')

    print('\nABLATION STUDY COMPLETE')
    return results


def plot_ablation_comparison(gnn_experiments, baseline_experiments, feat_names):
    """Plot 1: Grouped bar chart of AUC by model × edge config."""
    fig, ax = plt.subplots(figsize=(14, 7))

    models = ['GCN', 'SAGE', 'GAT']
    edge_configs = ['spatial_only', 'hydro_only', 'dual_edge']
    edge_labels = ['Spatial-only', 'Hydro-only', 'Dual-edge']
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    x = np.arange(len(models) + len(baseline_experiments))
    width = 0.25

    # GNN bars
    for i, edge in enumerate(edge_configs):
        aucs = []
        for model in models:
            key = f'{model.lower()}_{edge}'
            aucs.append(gnn_experiments[key]['test_auc'] if key in gnn_experiments else 0)
        bars = ax.bar(x[:3] + i * width, aucs, width, label=edge_labels[i], color=colors[i])
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Baseline bars
    baseline_names = ['LR', 'RF', 'XGBoost']
    baseline_keys = ['logistic_regression', 'random_forest', 'xgboost']
    baseline_aucs = [baseline_experiments[k]['auc'] for k in baseline_keys]
    bars = ax.bar(x[3:] + width, baseline_aucs, width, label='Baseline', color='#9E9E9E')
    for bar, auc in zip(bars, baseline_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Test AUC', fontsize=12)
    ax.set_title('Ablation Study: AUC by Model x Edge Config', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models + baseline_names)
    ax.legend(loc='lower right')
    ax.set_ylim(0.85, 0.98)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(gnn_experiments):
    """Plot 2: Training curves (loss vs epoch) for all 9 GNN experiments."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    models = ['GCN', 'SAGE', 'GAT']
    edge_configs = ['spatial_only', 'hydro_only', 'dual_edge']
    edge_labels = ['Spatial-only', 'Hydro-only', 'Dual-edge']

    for row, model in enumerate(models):
        for col, edge in enumerate(edge_configs):
            ax = axes[row, col]
            key = f'{model.lower()}_{edge}'

            if key in gnn_experiments:
                exp = gnn_experiments[key]
                epochs = range(1, len(exp['history']['train_loss']) + 1)

                ax.plot(epochs, exp['history']['train_loss'], 'b-', label='Train', linewidth=1.5)
                ax.plot(epochs, exp['history']['val_loss'], 'r--', label='Val', linewidth=1.5)

                # Mark convergence epoch
                conv_epoch = exp['convergence_epoch']
                if conv_epoch > 0:
                    ax.axvline(x=conv_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best ({conv_epoch})')

            ax.set_title(f'{model} - {edge_labels[col]}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle('Training Curves: Loss vs Epoch', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_heatmap(gnn_experiments, data, n_samples=50):
    """Plot 3: GAT attention weight heatmap (sampled 50 slope units)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    edge_configs = ['spatial_only', 'hydro_only', 'dual_edge']
    edge_labels = ['Spatial-only', 'Hydro-only', 'Dual-edge']

    for col, edge in enumerate(edge_configs):
        ax = axes[col]
        key = f'gat_{edge}'

        if key in gnn_experiments and gnn_experiments[key].get('attention_weights') is not None:
            attn = gnn_experiments[key]['attention_weights']
            # Sample n_samples units
            n = min(n_samples, attn.shape[0])
            idx = np.random.choice(attn.shape[0], n, replace=False)
            attn_sampled = attn[np.ix_(idx, idx)]

            im = ax.imshow(attn_sampled, cmap='YlOrRd', aspect='auto')
            ax.set_title(f'GAT - {edge_labels[col]}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Target SU', fontsize=8)
            ax.set_ylabel('Source SU', fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'No attention\nweights', ha='center', va='center', fontsize=12)
            ax.set_title(f'GAT - {edge_labels[col]}', fontsize=10, fontweight='bold')

    fig.suptitle('GAT Attention Weight Analysis (50 sampled SUs)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'attention_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(baseline_experiments, feat_names):
    """Plot 4: Feature importance from RF and XGBoost."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (model_name, model_label) in enumerate([('random_forest', 'Random Forest'), ('xgboost', 'XGBoost')]):
        ax = axes[idx]
        if model_name in baseline_experiments and 'feature_importance' in baseline_experiments[model_name]:
            fi = baseline_experiments[model_name]['feature_importance']
            # Map feature indices to names
            fi_named = {}
            for k, v in fi.items():
                idx_num = int(k.replace('f', ''))
                if idx_num < len(feat_names):
                    fi_named[feat_names[idx_num]] = v
                else:
                    fi_named[k] = v

            # Sort by importance
            sorted_fi = sorted(fi_named.items(), key=lambda x: x[1], reverse=True)
            names = [x[0] for x in sorted_fi]
            values = [x[1] for x in sorted_fi]

            # Horizontal bar chart
            y_pos = np.arange(len(names))
            ax.barh(y_pos, values, color='#2196F3' if idx == 0 else '#FF9800')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_title(f'{model_label} Feature Importance', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No feature\nimportance', ha='center', va='center', fontsize=12)
            ax.set_title(f'{model_label} Feature Importance', fontsize=12, fontweight='bold')

    fig.suptitle('Feature Importance: Baseline Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run_ablation()