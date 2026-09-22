"""Paired bootstrap significance tests for model differences (B2).

Compares AUC between model pairs on the same seed-42 test split using a paired
bootstrap: for each resample the AUC difference is computed, yielding a 95%
confidence interval and a two-sided p-value (proportion of resamples on the
opposite side of zero).

Pairs:
  - dual-edge vs spatial-only   (each backbone)
  - with-physics vs without-physics (GCN dual-edge)

Note: physics-derived labels are a near-deterministic function of the input
features, so full-feature AUC approaches 1.0 for every model (including
non-graph baselines). The spatial-masking experiment is the honest test of
graph value; these tests quantify differences under the full-feature regime.

Outputs:
  results/significance.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
from sklearn.metrics import roc_auc_score

from src.terrapyge.data.graph import load_graph
from src.terrapyge.models.training import load_config, create_masks
from src.terrapyge.models.experiments import (
    get_edge_configs, predict_node_probs, load_gnn_model,
    BASE_FEATURE_IDX, PHYSICS_FEATURE_IDX,
)
from src.terrapyge.utils.paths import (
    ABLATION_MODELS, PROCESSED_BUHISAN as PROC_DIR, RESULTS, ROOT,
)

GRAPH = PROC_DIR / 'buhisan_hetero_physics.pt'
OUT_JSON = RESULTS / 'significance.json'

N_BOOT = 2000
SEED = 42

# (label, model_a, edge_a, features_a, model_b, edge_b, features_b)
PAIRS = [
    ('dual_vs_spatial_gcn',
     'with_physics__heterogcn__dual_edge', 'dual_edge', 'physics',
     'with_physics__heterogcn__spatial_only', 'spatial_only', 'physics'),
    ('dual_vs_spatial_sage',
     'with_physics__heterosage__dual_edge', 'dual_edge', 'physics',
     'with_physics__heterosage__spatial_only', 'spatial_only', 'physics'),
    ('dual_vs_spatial_gat',
     'with_physics__heterogat__dual_edge', 'dual_edge', 'physics',
     'with_physics__heterogat__spatial_only', 'spatial_only', 'physics'),
    ('physics_vs_without_gcn_dual',
     'with_physics__heterogcn__dual_edge', 'dual_edge', 'physics',
     'without_physics__heterogcn__dual_edge', 'dual_edge', 'base'),
]


def _model_type(name):
    backbone = name.split('__')[1]
    return 'Hetero' + backbone.replace('hetero', '').upper()


def _probs(name, edge, feat_kind, x_phys, x_base, edge_configs, config):
    path = ABLATION_MODELS / f'{name}.pt'
    if not path.exists():
        return None
    x = x_phys if feat_kind == 'physics' else x_base
    model = load_gnn_model(path, _model_type(name), x.shape[1], config)
    return predict_node_probs(model, x, edge_configs[edge])


def paired_bootstrap_auc(y, pa, pb, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(roc_auc_score(yb, pa[idx]) - roc_auc_score(yb, pb[idx]))
    d = np.asarray(diffs, dtype=float)
    frac_le0 = float((d <= 0).mean())
    frac_ge0 = float((d >= 0).mean())
    return {
        'auc_a': float(roc_auc_score(y, pa)),
        'auc_b': float(roc_auc_score(y, pb)),
        'auc_diff': float(roc_auc_score(y, pa) - roc_auc_score(y, pb)),
        'diff_ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        'p_value': float(2 * min(frac_le0, frac_ge0)),
        'n_boot': int(len(d)),
    }


def main():
    print('=' * 70)
    print('TERRAPYGE: SIGNIFICANCE TESTS (paired bootstrap)')
    print('=' * 70)

    config = load_config(str(ROOT / 'config.yaml'))
    train_cfg = config.get('training', {})
    data = load_graph(GRAPH)
    n = data['su'].num_nodes
    x_phys = data['su'].x[:, PHYSICS_FEATURE_IDX].contiguous()
    x_base = data['su'].x[:, BASE_FEATURE_IDX].contiguous()
    y = data['su'].y.cpu().numpy()

    _, _, test_mask = create_masks(
        n, train_ratio=train_cfg.get('train_ratio', 0.64),
        val_ratio=train_cfg.get('val_ratio', 0.16), seed=train_cfg.get('seed', 42),
    )
    test_idx = np.where(test_mask.numpy())[0]
    y_test = y[test_idx]
    edge_configs = get_edge_configs(data)

    results = {}
    for label, a, ea, fa, b, eb, fb in PAIRS:
        pa = _probs(a, ea, fa, x_phys, x_base, edge_configs, config)
        pb = _probs(b, eb, fb, x_phys, x_base, edge_configs, config)
        if pa is None or pb is None:
            print(f'[SKIP] {label}: missing checkpoint')
            continue
        res = paired_bootstrap_auc(y_test, pa[test_idx], pb[test_idx])
        res['model_a'], res['model_b'] = a, b
        results[label] = res
        print(f'  {label:32s} dAUC={res["auc_diff"]:+.4f} '
              f'CI={[round(x, 4) for x in res["diff_ci95"]]} p={res["p_value"]:.4f}')

    with open(OUT_JSON, 'w') as f:
        json.dump({'n_boot': N_BOOT, 'seed': SEED, 'pairs': results}, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')
    print('\nDONE')


if __name__ == '__main__':
    main()
