"""TERRAPYGE spatial masking experiment.

Tests whether the graph-based GNN recovers susceptibility for slope units
whose direct slope observation is missing, better than non-graph baselines.

Design:
  - Load the physics-informed graph (buhisan_hetero_physics.pt).
  - Use feature indices 0-19 (exclude index 20 = newmark_dn_cm, the direct
    label source, to avoid trivial leakage).
  - For each mask rate in {0.10, 0.20, 0.30}:
      * nested random node subsets (10% subset of 20% subset of 30%)
      * set slope_mean (index 1) = 0 for masked nodes
      * train_mask = 80% of unmasked; val_mask = 20% of unmasked
      * test_mask = masked nodes (held out entirely)
  - Train GNNs (GCN/SAGE/GAT x spatial/hydro/dual) and non-graph baselines
    (LR/RF/XGBoost) on the same split.
  - Evaluate AUC / AP / F1 on the masked test nodes.

Outputs:
  - results/spatial_masking.json
  - results/figures/spatial_masking.png
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
from src.terrapyge.models.training import load_config
from src.terrapyge.models.baselines import run_baselines
from src.terrapyge.models.experiments import (
    train_gnn_experiment, get_edge_configs,
    FEATURE_SLICE, MASK_COLS, KEPT_COLS,
)
from src.terrapyge.utils.paths import (
    ABLATION_MODELS as MODELS_DIR,
    FIGURES as FIGURES_DIR,
    PROCESSED_BUHISAN as PROC_DIR,
    RESULTS as RESULTS_DIR,
    ROOT,
)

GRAPH = PROC_DIR / 'buhisan_hetero_physics.pt'
OUT_JSON = RESULTS_DIR / 'spatial_masking.json'
OUT_FIG = FIGURES_DIR / 'spatial_masking.png'

# FEATURE_SLICE, MASK_COLS and KEPT_COLS are defined in
# src/terrapyge/models/experiments.py so the masking scheme stays consistent
# across experiments.
MASK_RATES = [0.10, 0.20, 0.30]
VAL_FRACTION = 0.20
SEED = 42
REFERENCE_AUC = 0.999          # approx unmasked full-feature ceiling

MODEL_TYPES = ['HeteroGCN', 'HeteroSAGE', 'HeteroGAT']


def build_masks(n, mask_rate, perm):
    """Return train/val/test boolean masks for a given rate.

    Masked test indices are the first int(rate*n) of a fixed permutation,
    so mask subsets are nested across increasing rates.
    """
    n_masked = int(mask_rate * n)
    masked_idx = perm[:n_masked]
    unmasked_idx = perm[n_masked:]

    n_val = int(VAL_FRACTION * len(unmasked_idx))
    val_idx = unmasked_idx[:n_val]
    train_idx = unmasked_idx[n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[masked_idx] = True
    return train_mask, val_mask, test_mask, masked_idx, train_idx


def main():
    print('=' * 70)
    print('TERRAPYGE SPATIAL MASKING EXPERIMENT')
    print('=' * 70)

    config = load_config(str(ROOT / 'config.yaml'))
    data = load_graph(GRAPH)
    n = data['su'].num_nodes

    # Restrict to feature indices 0-19 (drop direct label source)
    data['su'].x = data['su'].x[:, FEATURE_SLICE].contiguous()
    print(f'Nodes: {n}')
    print(f'Features used: {data["su"].x.shape[1]} (indices 0-19)')
    print(f'Masked (topographic) columns: {MASK_COLS}')
    print(f'Kept (environmental) columns: {KEPT_COLS}')
    print(f'Labels positive: {data["su"].y.sum().item()} / {n} '
          f'({data["su"].y.float().mean():.1%})')

    perm = np.random.default_rng(SEED).permutation(n)
    results = {'mask_rates': MASK_RATES, 'reference_auc': REFERENCE_AUC,
               'mask_cols': MASK_COLS, 'kept_cols': KEPT_COLS, 'runs': {}}

    for rate in MASK_RATES:
        tag = f'mask{int(rate * 100)}'
        print('\n' + '=' * 70)
        print(f'MASK RATE {rate:.0%}  ({tag})')
        print('=' * 70)

        d = copy.deepcopy(data)
        train_mask, val_mask, test_mask, masked_idx, train_idx = build_masks(n, rate, perm)

        # Mask the topographic block for test nodes (set to standardized mean = 0)
        x = d['su'].x.clone()
        x[masked_idx.reshape(-1, 1), MASK_COLS] = 0.0
        d['su'].x = x
        d['su'].train_mask = train_mask
        d['su'].val_mask = val_mask
        d['su'].test_mask = test_mask

        # Sanity checks
        assert torch.all(d['su'].x[masked_idx.reshape(-1, 1), MASK_COLS] == 0), 'masking failed'
        pos_rate = d['su'].y[test_mask].float().mean().item()
        print(f'  Masked nodes: {len(masked_idx)} ({rate:.0%})')
        print(f'  Train: {int(train_mask.sum())} | Val: {int(val_mask.sum())}')
        print(f'  Test (masked): {len(masked_idx)} | positive rate: {pos_rate:.1%}')

        run = {'n_masked': int(len(masked_idx)), 'test_pos_rate': pos_rate,
               'gnn': {}, 'baselines': {}}

        edge_configs = get_edge_configs(d)
        for model_type in MODEL_TYPES:
            for edge_name, eidx in edge_configs.items():
                exp = f'{tag}__{model_type.lower()}__{edge_name}'
                res = train_gnn_experiment(d, eidx, model_type, config, exp)
                run['gnn'][f'{model_type.lower()}__{edge_name}'] = {
                    'test_auc': res['test_auc'],
                    'test_ap': res['test_ap'],
                    'test_f1': res['test_f1'],
                    'convergence_epoch': res['convergence_epoch'],
                }
                print(f'  GNN {model_type:<10} {edge_name:<13} '
                      f'AUC={res["test_auc"]:.4f} AP={res["test_ap"]:.4f} F1={res["test_f1"]:.4f}')

        # Baselines: identical split, no graph, slope masked on test
        X = d['su'].x.numpy()
        y = d['su'].y.numpy()
        train_idx_np = train_mask.numpy()
        test_idx_np = test_mask.numpy()
        base = run_baselines(X[train_idx_np], y[train_idx_np],
                             X[test_idx_np], y[test_idx_np], config)
        run['baselines'] = base
        for name, m in base.items():
            print(f'  BASE {name:<10} {"-":<13} '
                  f'AUC={m["auc"]:.4f} AP={m["ap"]:.4f} F1={m["f1"]:.4f}')

        results['runs'][tag] = run

    # Summary table
    print('\n' + '=' * 70)
    print('MASKING SUMMARY (AUC on masked nodes)')
    print('=' * 70)
    header = f'{"Model":<20}' + ''.join(f'{int(r*100):>8}%' for r in MASK_RATES)
    print(header)
    print('-' * len(header))

    def gnn_mean(tag, model):
        vals = [v['test_auc'] for k, v in results['runs'][tag]['gnn'].items()
                if k.startswith(model + '__')]
        return float(np.mean(vals)) if vals else 0.0

    for model in ['heterogcn', 'heterosage', 'heterogat']:
        row = f'{model:<20}' + ''.join(f'{gnn_mean(f"mask{int(r*100)}", model):>9.4f}' for r in MASK_RATES)
        print(row)
    for name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
        row = f'{name:<20}'
        for r in MASK_RATES:
            m = results['runs'][f'mask{int(r*100)}']['baselines'][name]
            row += f'{m["auc"]:>9.4f}'
        print(row)
    print(f'\nUnmasked full-feature ceiling (approx): {REFERENCE_AUC:.4f}')

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')

    plot_degradation(results)


def plot_degradation(results):
    rates = [int(r * 100) for r in results['mask_rates']]

    def gnn_curve(model):
        out = []
        for r in rates:
            vals = [v['test_auc'] for k, v in results['runs'][f'mask{r}']['gnn'].items()
                    if k.startswith(model + '__')]
            out.append(float(np.mean(vals)))
        return out

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, label in [('heterogcn', 'GCN'), ('heterosage', 'SAGE'), ('heterogat', 'GAT')]:
        ax.plot(rates, gnn_curve(model), marker='o', linewidth=2, label=f'GNN {label} (mean)')

    for name, style in [('Logistic Regression', '--'), ('Random Forest', '--'), ('XGBoost', '--')]:
        curve = [results['runs'][f'mask{r}']['baselines'][name]['auc'] for r in rates]
        ax.plot(rates, curve, marker='s', linestyle=style, alpha=0.8, label=name)

    ax.axhline(results['reference_auc'], color='gray', linestyle=':', linewidth=1.5,
               label=f'Unmasked ceiling (~{results["reference_auc"]:.3f})')

    ax.set_xlabel('Mask rate (%) — topographic covariates withheld from test nodes')
    ax.set_ylabel('Test AUC (masked nodes)')
    ax.set_title('Spatial Masking Experiment: Graph vs Non-Graph Recovery')
    ax.set_xticks(rates)
    ax.set_xticklabels([f'{r}%' for r in rates])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_FIG}')


if __name__ == '__main__':
    main()
