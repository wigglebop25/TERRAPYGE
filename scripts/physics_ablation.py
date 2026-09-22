"""TERRAPYGE physics ablation: with vs. without physics-informed features.

Uses the physics-informed graph (buhisan_hetero_physics.pt) with
physics-derived labels. Trains all GNN architectures across all edge
configurations on two feature variants:

  - without_physics: first 14 base features (terrain + environment)
  - with_physics:    all 21 features (base + 7 physics descriptors)

Baselines (LR, RF, XGBoost) are trained on both variants as well.
Results are saved to results/physics_ablation.json and a comparison
figure is written to results/figures/physics_ablation.png.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.terrapyge.data.graph import load_graph
from src.terrapyge.models.training import load_config, create_masks
from src.terrapyge.models.baselines import run_baselines
from src.terrapyge.models.experiments import train_gnn_experiment, get_edge_configs
from src.terrapyge.utils.paths import (
    ABLATION_MODELS as MODELS_DIR,
    FIGURES as FIGURES_DIR,
    PROCESSED_BUHISAN as PROC_DIR,
    RESULTS as RESULTS_DIR,
    ROOT,
)

GRAPH = PROC_DIR / 'buhisan_hetero_physics.pt'
OUT_JSON = RESULTS_DIR / 'physics_ablation.json'
OUT_FIG = FIGURES_DIR / 'physics_ablation.png'

N_BASE = 14   # base features (indices 0-13)

# Feature index sets.
# Note: index 20 (newmark_dn_cm) is the direct source of the physics label
# and MUST be excluded from features to avoid trivial leakage.
# Physics descriptors retained: gamma, c', phi, static FS, critical acc., Arias.
IDX_WITHOUT = list(range(14))
IDX_WITH = list(range(14)) + [14, 15, 16, 17, 18, 19]


def main():
    print('=' * 70)
    print('TERRAPYGE PHYSICS ABLATION')
    print('=' * 70)

    config = load_config(str(ROOT / 'config.yaml'))
    data = load_graph(GRAPH)
    n = data['su'].num_nodes
    print(f'Nodes: {n}')
    print(f'Features: {data["su"].x.shape[1]}')
    print(f'Physics labels: {data["su"].y.sum().item()} positive / {n} ({data["su"].y.float().mean():.1%})')

    # Shared splits (seed=42)
    train_mask, val_mask, test_mask = create_masks(
        n,
        train_ratio=config['training']['train_ratio'],
        val_ratio=config['training']['val_ratio'],
        seed=config['training']['seed'],
    )
    data['su'].train_mask = train_mask
    data['su'].val_mask = val_mask
    data['su'].test_mask = test_mask

    variants = {
        'without_physics': IDX_WITHOUT,
        'with_physics': IDX_WITH,
    }

    model_types = ['HeteroGCN', 'HeteroSAGE', 'HeteroGAT']
    results = {'gnn': {}, 'baselines': {}}
    all_gnn = {}

    for variant, feat_idx in variants.items():
        print('\n' + '=' * 70)
        print(f'VARIANT: {variant} ({len(feat_idx)} features)')
        print('=' * 70)

        d = copy.deepcopy(data)
        d['su'].x = d['su'].x[:, feat_idx].contiguous()
        edge_configs = get_edge_configs(d)

        for model_type in model_types:
            for edge_name, eidx in edge_configs.items():
                exp = f'{variant}__{model_type.lower()}__{edge_name}'
                res = train_gnn_experiment(d, eidx, model_type, config, exp)
                results['gnn'][exp] = {k: v for k, v in res.items()
                                       if k not in ('attention_weights',)}
                all_gnn[exp] = res

        # Baselines on the same feature variant and splits
        X = d['su'].x.numpy()
        y = d['su'].y.numpy()
        X_tr, y_tr = X[train_mask.numpy()], y[train_mask.numpy()]
        X_te, y_te = X[test_mask.numpy()], y[test_mask.numpy()]
        base = run_baselines(X_tr, y_tr, X_te, y_te, config)
        results['baselines'][variant] = base

    # Summary
    print('\n' + '=' * 70)
    print('PHYSICS ABLATION SUMMARY')
    print('=' * 70)

    for variant in variants:
        print(f'\n--- {variant} ---')
        print(f'{"Model":<18} {"Edge":<14} {"AUC":>8} {"AP":>8} {"F1":>8}')
        print('-' * 60)
        for exp, r in results['gnn'].items():
            if not exp.startswith(variant + '__'):
                continue
            parts = exp.split('__')
            print(f'{parts[1]:<18} {parts[2]:<14} '
                  f'{r["test_auc"]:>8.4f} {r["test_ap"]:>8.4f} {r["test_f1"]:>8.4f}')
        for name, m in results['baselines'][variant].items():
            print(f'{name:<18} {"-":<14} {m["auc"]:>8.4f} {m["ap"]:>8.4f} {m["f1"]:>8.4f}')

    # Physics improvement (best GNN per variant)
    def best_gnn(variant):
        subset = {k: v for k, v in results['gnn'].items() if k.startswith(variant + '__')}
        k = max(subset, key=lambda x: subset[x]['test_auc'])
        return k, subset[k]['test_auc']

    k_wo, auc_wo = best_gnn('without_physics')
    k_w, auc_w = best_gnn('with_physics')
    print('\nBEST GNN (without physics): %.4f  (%s)' % (auc_wo, k_wo))
    print('BEST GNN (with physics):    %.4f  (%s)' % (auc_w, k_w))
    print('Physics contribution: %+.4f AUC' % (auc_w - auc_wo))

    results['summary'] = {
        'best_without_physics': {'experiment': k_wo, 'auc': auc_wo},
        'best_with_physics': {'experiment': k_w, 'auc': auc_w},
        'physics_auc_delta': auc_w - auc_wo,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')

    # Figure: grouped bars by model x edge for both variants (best config per model)
    plot_ablation(results)


def plot_ablation(results):
    models = ['heterogcn', 'heterosage', 'heterogat']
    edges = ['spatial_only', 'hydro_only', 'dual_edge']
    variants = ['without_physics', 'with_physics']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    width = 0.25
    x = np.arange(len(models))

    for ax, variant in zip(axes, variants):
        for i, edge in enumerate(edges):
            aucs = []
            for m in models:
                key = f'{variant}__{m}__{edge}'
                aucs.append(results['gnn'][key]['test_auc'] if key in results['gnn'] else 0)
            bars = ax.bar(x + i * width, aucs, width, label=edge)
            for b, a in zip(bars, aucs):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f'{a:.3f}', ha='center', va='bottom', fontsize=8)

        # Baseline reference lines
        for name, m in results['baselines'][variant].items():
            ax.axhline(m['auc'], linestyle=':', alpha=0.6, label=f'{name} {m["auc"]:.3f}')

        ax.set_title(variant.replace('_', ' ').title(), fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['GCN', 'SAGE', 'GAT'])
        ax.set_ylim(0.8, 1.0)
        ax.set_ylabel('Test AUC')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Physics Ablation: With vs Without Physics-Informed Features',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_FIG}')


if __name__ == '__main__':
    main()
