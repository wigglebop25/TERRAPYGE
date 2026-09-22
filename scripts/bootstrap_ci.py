"""Bootstrap 95% confidence intervals for model performance (B1).

Loads the physics graph and the trained GNN checkpoints, reconstructs the
deterministic seed-42 test split, and bootstraps the test set to report 95%
confidence intervals for AUC / AP / F1. Baselines are retrained on the same
training split (their saved checkpoints predate the physics pipeline).

Outputs:
  results/bootstrap_ci.json
  results/figures/bootstrap_ci.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.terrapyge.data.graph import load_graph
from src.terrapyge.models.training import load_config, create_masks
from src.terrapyge.models.baselines import (
    LogisticRegressionBaseline, RandomForestBaseline, XGBoostBaseline,
)
from src.terrapyge.models.experiments import (
    get_edge_configs, predict_node_probs, load_gnn_model, PHYSICS_FEATURE_IDX,
)
from src.terrapyge.utils.paths import (
    ABLATION_MODELS, FIGURES, PROCESSED_BUHISAN as PROC_DIR, RESULTS, ROOT,
)

GRAPH = PROC_DIR / 'buhisan_hetero_physics.pt'
OUT_JSON = RESULTS / 'bootstrap_ci.json'
OUT_FIG = FIGURES / 'bootstrap_ci.png'

N_BOOT = 2000
SEED = 42

GNN_VARIANTS = [
    ('heterogcn', 'dual_edge'),
    ('heterosage', 'dual_edge'),
    ('heterogat', 'dual_edge'),
    ('heterogcn', 'spatial_only'),
]


def bootstrap_metrics(y, p, n_boot=N_BOOT, seed=SEED):
    """Return point estimates plus 2.5/97.5 percentiles for AUC/AP/F1."""
    def _point(yy, pp):
        if len(np.unique(yy)) < 2:
            return 0.0, 0.0, 0.0
        return (roc_auc_score(yy, pp),
                average_precision_score(yy, pp),
                f1_score(yy, (pp > 0.5).astype(int), zero_division=0))

    auc, ap, f1 = _point(y, p)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {'auc': [], 'ap': [], 'f1': []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot['auc'].append(roc_auc_score(yb, pb))
        boot['ap'].append(average_precision_score(yb, pb))
        boot['f1'].append(f1_score(yb, (pb > 0.5).astype(int), zero_division=0))

    def _ci(v):
        a = np.asarray(v, dtype=float)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        'auc': {'point': float(auc), 'ci95': _ci(boot['auc'])},
        'ap': {'point': float(ap), 'ci95': _ci(boot['ap'])},
        'f1': {'point': float(f1), 'ci95': _ci(boot['f1'])},
    }


def main():
    print('=' * 70)
    print('TERRAPYGE: BOOTSTRAP CONFIDENCE INTERVALS')
    print('=' * 70)

    config = load_config(str(ROOT / 'config.yaml'))
    train_cfg = config.get('training', {})
    data = load_graph(GRAPH)
    n = data['su'].num_nodes
    x = data['su'].x[:, PHYSICS_FEATURE_IDX].contiguous()
    y = data['su'].y.cpu().numpy()

    train_mask, _, test_mask = create_masks(
        n, train_ratio=train_cfg.get('train_ratio', 0.64),
        val_ratio=train_cfg.get('val_ratio', 0.16), seed=train_cfg.get('seed', 42),
    )
    test_idx = np.where(test_mask.numpy())[0]
    tr_idx = np.where(train_mask.numpy())[0]
    y_test = y[test_idx]
    print(f'Nodes: {n}  train: {len(tr_idx)}  test: {len(test_idx)}')

    edge_configs = get_edge_configs(data)
    results = {}

    for backbone, edge in GNN_VARIANTS:
        name = f'with_physics__{backbone}__{edge}'
        path = ABLATION_MODELS / f'{name}.pt'
        if not path.exists():
            print(f'[SKIP] {name}: checkpoint not found')
            continue
        model = load_gnn_model(path, f'Hetero{backbone.replace("hetero", "").upper()}',
                               x.shape[1], config)
        probs = predict_node_probs(model, x, edge_configs[edge])
        results[name] = bootstrap_metrics(y_test, probs[test_idx])
        print(f'  {name:42s} AUC={results[name]["auc"]["point"]:.4f} '
              f'CI={results[name]["auc"]["ci95"]}')

    # Baselines (retrained on the same train split)
    X = x.numpy()
    baseline_cfg = config.get('baselines', {})
    base_models = [
        ('Logistic Regression', LogisticRegressionBaseline(baseline_cfg.get('logistic_regression', {}))),
        ('Random Forest', RandomForestBaseline(baseline_cfg.get('random_forest', {}))),
        ('XGBoost', XGBoostBaseline(baseline_cfg.get('xgboost', {}))),
    ]
    for label, model in base_models:
        model.train(X[tr_idx], y[tr_idx])
        probs = model.predict_proba(X[test_idx])
        results[label] = bootstrap_metrics(y_test, probs)
        print(f'  {label:42s} AUC={results[label]["auc"]["point"]:.4f} '
              f'CI={results[label]["auc"]["ci95"]}')

    with open(OUT_JSON, 'w') as f:
        json.dump({'n_boot': N_BOOT, 'seed': SEED, 'metrics': results}, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')

    # Figure
    names = list(results.keys())
    aucs = [results[m]['auc']['point'] for m in names]
    lo = [results[m]['auc']['point'] - results[m]['auc']['ci95'][0] for m in names]
    hi = [results[m]['auc']['ci95'][1] - results[m]['auc']['point'] for m in names]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(np.arange(len(names)), aucs, xerr=[lo, hi], color='#2196F3', alpha=0.8,
            capsize=4)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([m.replace('with_physics__', '') for m in names], fontsize=9)
    ax.set_xlabel('Test AUC (95% bootstrap CI)')
    ax.set_title('Bootstrap 95% Confidence Intervals')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_FIG}')
    print('\nDONE')


if __name__ == '__main__':
    main()
