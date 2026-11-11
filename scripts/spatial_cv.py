"""Spatial / block cross-validation for TERRAPYGE (Stage B4).

Motivation
----------
Labels are a deterministic function of slope (Newmark displacement), so random
k-fold CV over slope units is optimistic: neighbouring slope units share
terrain, climate and soil, and a transductive GNN can additionally propagate
held-out node features through the graph. This script measures how much of the
reported performance is attributable to that spatial autocorrelation by
re-running the same models under spatially blocked folds.

Protocols
---------
random_transductive
    Status quo. Random stratified k-fold over nodes; the GNN is trained on the
    whole graph, with held-out nodes present but unlabelled.
block_transductive
    Spatially blocked k-fold. Whole grid blocks are held out, but the GNN still
    sees the whole graph, so feature propagation to held-out nodes remains.
block_inductive
    Spatially blocked k-fold with inductive training. The GNN is trained on the
    induced subgraph of train + val nodes only, so held-out blocks are unseen
    during training and cannot leak through message passing.

Feature modes
-------------
full
    Features 0-19 (index 20, ``newmark_dn_cm``, is the label source and is
    excluded from every model input).
masked
    As ``full``, but the topographic block (``MASK_COLS``) of held-out nodes is
    zeroed, matching the spatial masking experiment.

Outputs
-------
results/spatial_cv.json
results/figures/spatial_cv.png

Usage
-----
python scripts/spatial_cv.py
python scripts/spatial_cv.py --folds 5 --n-blocks 25 --model-types all
python scripts/spatial_cv.py --feature-modes full --protocols block_inductive
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import json

import numpy as np
import torch
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.terrapyge.data.graph import load_graph
from src.terrapyge.models.training import load_config
from src.terrapyge.models.baselines import run_baselines
from src.terrapyge.models.experiments import (
    train_gnn_experiment, get_edge_configs, build_induced_subgraph,
    FEATURE_SLICE, MASK_COLS, GNN_MODELS,
)
from src.terrapyge.utils.paths import (
    FIGURES, PROCESSED_BUHISAN as PROC_DIR, RESULTS, ROOT,
)

GRAPH = PROC_DIR / 'buhisan_hetero_physics.pt'
SU_GPKG = PROC_DIR / 'slope_units.gpkg'
OUT_JSON = RESULTS / 'spatial_cv.json'
OUT_FIG = FIGURES / 'spatial_cv.png'

PROTOCOLS = ['random_transductive', 'block_transductive', 'block_inductive']
FEATURE_MODES = ['full', 'masked']

BASELINE_NAMES = ['Logistic Regression', 'Random Forest', 'XGBoost']


# --------------------------------------------------------------------------
# Spatial blocking
# --------------------------------------------------------------------------

def load_centroids(su_gpkg):
    """Return [N, 2] representative-point coordinates per slope unit.

    ``representative_point`` is guaranteed to fall inside the polygon, unlike a
    centroid for concave slope units. Node order matches ``slope_units.gpkg``
    row order, which is the order used when the graph was built.
    """
    gdf = gpd.read_file(su_gpkg)
    points = gdf.geometry.representative_point()
    return np.column_stack([points.x.to_numpy(), points.y.to_numpy()])


def make_blocks(centroids, n_blocks):
    """Assign each node to a square grid block over the study-area extent.

    Args:
        centroids: [N, 2] projected coordinates in metres.
        n_blocks: target number of blocks; the grid is ``ceil(sqrt(n_blocks))``
            cells per side, so the actual count is close to but not exactly
            ``n_blocks``.

    Returns:
        block: int array [N] with the block id of each node.
        n_side: number of cells per side.
        cell_m: block side length in metres.
    """
    x, y = centroids[:, 0], centroids[:, 1]
    n_side = max(1, int(np.ceil(np.sqrt(n_blocks))))
    span = max(x.max() - x.min(), y.max() - y.min())
    cell_m = span / n_side
    if not np.isfinite(cell_m) or cell_m <= 0:
        raise ValueError('degenerate study-area extent: cannot build blocks')

    ix = np.clip(((x - x.min()) / cell_m).astype(int), 0, n_side - 1)
    iy = np.clip(((y - y.min()) / cell_m).astype(int), 0, n_side - 1)
    return iy * n_side + ix, n_side, float(cell_m)


def assign_blocks_to_folds(block, y, n_folds):
    """Assign whole blocks to folds, balancing node counts.

    Greedy: blocks are placed largest-first into the fold with the fewest nodes
    so far (ties broken by positive count). Whole blocks are never split, which
    is what makes the folds spatially contiguous.

    Returns:
        fold_of_node: int array [N] with the fold id of each node.
    """
    unique_blocks = np.unique(block)
    sizes = {b: int(np.sum(block == b)) for b in unique_blocks}
    positives = {b: int(np.sum(y[block == b])) for b in unique_blocks}
    order = sorted(unique_blocks, key=lambda b: (-sizes[b], b))

    fold_nodes = [0] * n_folds
    fold_positives = [0] * n_folds
    block_fold = {}
    for b in order:
        target = min(range(n_folds), key=lambda f: (fold_nodes[f], fold_positives[f]))
        block_fold[b] = target
        fold_nodes[target] += sizes[b]
        fold_positives[target] += positives[b]

    fold_of_node = np.empty(len(block), dtype=int)
    for b, fold in block_fold.items():
        fold_of_node[block == b] = fold
    return fold_of_node


def random_fold_ids(y, n_folds, seed):
    """Random stratified k-fold assignment over nodes."""
    fold_of_node = np.empty(len(y), dtype=int)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold_of_node[test_idx] = fold
    return fold_of_node


def apply_buffer(train_idx, test_idx, centroids, buffer_m):
    """Drop training nodes within ``buffer_m`` of any held-out node.

    Standard spatial-CV practice: a buffer keeps training points close enough to
    the test block from reproducing its signal through spatial autocorrelation.
    """
    if buffer_m <= 0 or len(train_idx) == 0:
        return train_idx
    tree = cKDTree(centroids[test_idx])
    distance, _ = tree.query(centroids[train_idx], k=1)
    return train_idx[distance > buffer_m]


def fold_masks(fold_of_node, fold, val_fraction, seed):
    """Return (train_idx, val_idx, test_idx) for one fold."""
    test_idx = np.flatnonzero(fold_of_node == fold)
    remaining = np.flatnonzero(fold_of_node != fold)
    rng = np.random.default_rng(seed + fold)
    rng.shuffle(remaining)

    n_val = max(1, int(val_fraction * len(remaining)))
    return remaining[n_val:], remaining[:n_val], test_idx


def masks_from_idx(n, train_idx, val_idx, test_idx):
    """Build boolean train/val/test masks over ``n`` nodes."""
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.as_tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.as_tensor(val_idx, dtype=torch.long)] = True
    test_mask[torch.as_tensor(test_idx, dtype=torch.long)] = True
    return train_mask, val_mask, test_mask


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

def aggregate(values):
    """Mean/std over defined fold values (``None`` folds are skipped)."""
    defined = [v for v in values if v is not None]
    if not defined:
        return {'mean': None, 'std': None, 'n': 0}
    std = float(np.std(defined, ddof=1)) if len(defined) > 1 else 0.0
    return {'mean': float(np.mean(defined)), 'std': std, 'n': len(defined)}


def summarize(folds, section):
    """Aggregate AUC/AP/F1 across folds for every model in ``section``."""
    keys = sorted({key for fold in folds for key in fold[section]})
    summary = {}
    for key in keys:
        entries = [fold[section][key] for fold in folds if key in fold[section]]
        summary[key] = {
            metric: aggregate([entry[metric] for entry in entries])
            for metric in ('auc', 'ap', 'f1')
        }
    return summary


def gnn_record(result, degenerate):
    """Convert a training result into a fold record, nulling degenerate folds."""
    if degenerate:
        return {'auc': None, 'ap': None, 'f1': None,
                'convergence_epoch': result['convergence_epoch']}
    return {'auc': result['test_auc'], 'ap': result['test_ap'],
            'f1': result['test_f1'],
            'convergence_epoch': result['convergence_epoch']}


def baseline_record(metrics, degenerate):
    """Convert baseline metrics into a fold record, nulling degenerate folds."""
    if degenerate:
        return {'auc': None, 'ap': None, 'f1': None}
    return {'auc': metrics['auc'], 'ap': metrics['ap'], 'f1': metrics['f1']}


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--folds', type=int, default=5,
                        help='number of CV folds (default: 5)')
    parser.add_argument('--n-blocks', type=int, default=25,
                        help='target number of spatial blocks (default: 25)')
    parser.add_argument('--buffer-m', type=float, default=0.0,
                        help='exclude training nodes within this distance (m) of '
                             'the held-out block; 0 disables the buffer')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='fraction of the non-test nodes used for validation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model-types', default='HeteroGCN',
                        help="comma-separated GNN backbones, or 'all' "
                             f"(available: {', '.join(GNN_MODELS)})")
    parser.add_argument('--edge-configs', default='dual_edge',
                        help="comma-separated edge configs, or 'all' "
                             "(spatial_only, hydro_only, dual_edge)")
    parser.add_argument('--feature-modes', default=','.join(FEATURE_MODES),
                        help='comma-separated feature modes (full, masked)')
    parser.add_argument('--protocols', default=','.join(PROTOCOLS),
                        help='comma-separated protocols')
    return parser.parse_args(argv)


def resolve_list(value, allowed, label):
    """Expand 'all' or a comma-separated list, validating against ``allowed``."""
    if value.strip().lower() == 'all':
        return list(allowed)
    items = [item.strip() for item in value.split(',') if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f'unknown {label}: {unknown} (allowed: {list(allowed)})')
    if not items:
        raise ValueError(f'no {label} selected')
    return items


def main(argv=None):
    args = parse_args(argv)

    model_types = resolve_list(args.model_types, GNN_MODELS, 'model types')
    edge_configs_requested = resolve_list(
        args.edge_configs, ['spatial_only', 'hydro_only', 'dual_edge'], 'edge configs')
    feature_modes = resolve_list(args.feature_modes, FEATURE_MODES, 'feature modes')
    protocols = resolve_list(args.protocols, PROTOCOLS, 'protocols')

    print('=' * 70)
    print('TERRAPYGE SPATIAL / BLOCK CROSS-VALIDATION (Stage B4)')
    print('=' * 70)

    config = load_config(str(ROOT / 'config.yaml'))
    data = load_graph(GRAPH)
    data['su'].x = data['su'].x[:, FEATURE_SLICE].contiguous()

    n = data['su'].num_nodes
    y = data['su'].y.numpy()

    centroids = load_centroids(SU_GPKG)
    if len(centroids) != n:
        raise ValueError(f'slope-unit/centroid mismatch: {len(centroids)} vs {n} nodes')

    block, n_side, cell_m = make_blocks(centroids, args.n_blocks)
    print(f'Nodes: {n}')
    print(f'Features used: {data["su"].x.shape[1]} (indices 0-19)')
    print(f'Positive rate: {y.mean():.1%}')
    print(f'Blocks: {n_side} x {n_side} grid, {cell_m:,.0f} m per side, '
          f'{len(np.unique(block))} populated')
    print(f'Folds: {args.folds} | val fraction: {args.val_fraction:.0%} | '
          f'buffer: {args.buffer_m:,.0f} m')
    print(f'Protocols: {", ".join(protocols)}')
    print(f'Feature modes: {", ".join(feature_modes)}')
    print(f'Backbones: {", ".join(model_types)} | edge configs: '
          f'{", ".join(edge_configs_requested)}')

    results = {
        'n_nodes': int(n),
        'n_folds': args.folds,
        'n_blocks_requested': args.n_blocks,
        'n_blocks_populated': int(len(np.unique(block))),
        'block_side': int(n_side),
        'block_size_m': cell_m,
        'buffer_m': args.buffer_m,
        'val_fraction': args.val_fraction,
        'seed': args.seed,
        'feature_modes': feature_modes,
        'protocols': protocols,
        'model_types': model_types,
        'edge_configs': edge_configs_requested,
        'mask_cols': MASK_COLS,
        'runs': {},
    }

    # Baselines are feature-only, so the same split and feature mode always
    # produce the same numbers. Cache them across protocols that share a split.
    baseline_cache = {}

    for protocol in protocols:
        blocked = protocol.startswith('block')
        inductive = protocol.endswith('inductive')

        if blocked:
            fold_of_node = assign_blocks_to_folds(block, y, args.folds)
        else:
            fold_of_node = random_fold_ids(y, args.folds, args.seed)

        print('\n' + '=' * 70)
        print(f'PROTOCOL {protocol}')
        print('=' * 70)

        for mode in feature_modes:
            mode_runs = results['runs'].setdefault(mode, {})
            folds = []

            for fold in range(args.folds):
                train_idx, val_idx, test_idx = fold_masks(
                    fold_of_node, fold, args.val_fraction, args.seed)
                train_idx = apply_buffer(train_idx, test_idx, centroids, args.buffer_m)

                degenerate = len(np.unique(y[test_idx])) < 2
                pos_rate = float(y[test_idx].mean()) if len(test_idx) else float('nan')

                print(f'\n  fold {fold} [{mode}] '
                      f'train={len(train_idx)} val={len(val_idx)} '
                      f'test={len(test_idx)} pos={pos_rate:.1%}')

                # Per-fold graph copy; masking is applied to held-out nodes only.
                d = copy.deepcopy(data)
                if mode == 'masked':
                    x = d['su'].x.clone()
                    x[torch.as_tensor(test_idx, dtype=torch.long).reshape(-1, 1),
                      MASK_COLS] = 0.0
                    d['su'].x = x

                train_mask, val_mask, test_mask = masks_from_idx(
                    n, train_idx, val_idx, test_idx)
                d['su'].train_mask = train_mask
                d['su'].val_mask = val_mask
                d['su'].test_mask = test_mask

                fold_record = {
                    'fold': fold,
                    'n_train': int(len(train_idx)),
                    'n_val': int(len(val_idx)),
                    'n_test': int(len(test_idx)),
                    'test_pos_rate': pos_rate,
                    'degenerate': bool(degenerate),
                    'gnn': {},
                    'baselines': {},
                }

                # ---- GNN ----
                full_edges = get_edge_configs(d)
                if inductive:
                    keep_idx = np.concatenate([train_idx, val_idx])
                    sub, remap = build_induced_subgraph(d, keep_idx)
                    sub_train = torch.zeros(sub['su'].num_nodes, dtype=torch.bool)
                    sub_val = torch.zeros(sub['su'].num_nodes, dtype=torch.bool)
                    sub_train[remap[torch.as_tensor(train_idx, dtype=torch.long)]] = True
                    sub_val[remap[torch.as_tensor(val_idx, dtype=torch.long)]] = True
                    sub['su'].train_mask = sub_train
                    sub['su'].val_mask = sub_val
                    train_graph, train_edges = sub, get_edge_configs(sub)
                else:
                    train_graph, train_edges = d, full_edges

                for model_type in model_types:
                    for edge_name in edge_configs_requested:
                        key = f'{model_type.lower()}__{edge_name}'
                        name = f'spatialcv__{protocol}__{mode}__f{fold}__{key}'
                        fold_config = copy.deepcopy(config)
                        fold_config.setdefault('training', {})['seed'] = args.seed + fold

                        res = train_gnn_experiment(
                            train_graph, train_edges[edge_name], model_type,
                            fold_config, name, save_model=False,
                            eval_data=d, eval_edge_index_dict=full_edges[edge_name],
                            eval_mask=test_mask,
                        )
                        fold_record['gnn'][key] = gnn_record(res, degenerate)
                        auc = fold_record['gnn'][key]['auc']
                        print(f'    GNN {key:<26} '
                              f'AUC={"n/a" if auc is None else f"{auc:.4f}"}')

                # ---- Baselines (cached across protocols sharing a split) ----
                cache_key = (protocol.replace('_inductive', '_transductive'), fold, mode)
                if cache_key not in baseline_cache:
                    X = d['su'].x.numpy()
                    base = run_baselines(X[train_idx], y[train_idx],
                                         X[test_idx], y[test_idx], config)
                    baseline_cache[cache_key] = {
                        name: baseline_record(metrics, degenerate)
                        for name, metrics in base.items()
                    }
                fold_record['baselines'] = copy.deepcopy(baseline_cache[cache_key])

                for name, metrics in fold_record['baselines'].items():
                    auc = metrics['auc']
                    print(f'    BASE {name:<25} '
                          f'AUC={"n/a" if auc is None else f"{auc:.4f}"}')

                folds.append(fold_record)

            mode_runs[protocol] = {
                'folds': folds,
                'summary': {
                    'gnn': summarize(folds, 'gnn'),
                    'baselines': summarize(folds, 'baselines'),
                },
            }

    results['leakage_gap'] = compute_leakage_gap(results)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as handle:
        json.dump(results, handle, indent=2)
    print(f'\nSaved: {OUT_JSON}')

    print_summary(results)
    plot_results(results)


def compute_leakage_gap(results):
    """AUC lost when moving from random to spatially blocked folds."""
    gap = {}
    for mode, protocol_runs in results['runs'].items():
        if 'random_transductive' not in protocol_runs:
            continue
        random_summary = protocol_runs['random_transductive']['summary']
        mode_gap = {}
        for protocol in ('block_transductive', 'block_inductive'):
            if protocol not in protocol_runs:
                continue
            block_summary = protocol_runs[protocol]['summary']
            for section in ('gnn', 'baselines'):
                for key, metrics in block_summary[section].items():
                    random_auc = random_summary[section].get(key, {}).get('auc', {}).get('mean')
                    block_auc = metrics['auc']['mean']
                    if random_auc is None or block_auc is None:
                        continue
                    mode_gap.setdefault(f'{section}:{key}', {})[protocol] = {
                        'auc_random': random_auc,
                        'auc_block': block_auc,
                        'gap': random_auc - block_auc,
                    }
        gap[mode] = mode_gap
    return gap


def print_summary(results):
    """Print a per-mode AUC table: model x protocol."""
    for mode, protocol_runs in results['runs'].items():
        protocols = list(protocol_runs)
        print('\n' + '=' * 78)
        print(f'AUC BY PROTOCOL [{mode}]')
        print('=' * 78)

        header = f'{"Model":<34}' + ''.join(f'{p[:18]:>20}' for p in protocols)
        print(header)
        print('-' * len(header))

        rows = []
        for section in ('gnn', 'baselines'):
            for key in sorted(protocol_runs[protocols[0]]['summary'][section]):
                rows.append((section, key))

        for section, key in rows:
            cells = ''
            for protocol in protocols:
                stats = protocol_runs[protocol]['summary'][section].get(key, {})
                auc = stats.get('auc', {})
                if auc.get('mean') is None:
                    cells += f'{"n/a":>20}'
                else:
                    cells += f'{auc["mean"]:>13.4f} +/-{auc["std"]:.3f}'
            print(f'{key:<34}{cells}')

        print('\nLeakage gap (random AUC - blocked AUC):')
        for key, values in results['leakage_gap'].get(mode, {}).items():
            for protocol, stats in values.items():
                print(f'  {key:<34} {protocol:<20} '
                      f'{stats["auc_random"]:.4f} -> {stats["auc_block"]:.4f} '
                      f'({stats["gap"]:+.4f})')


def plot_results(results):
    """Grouped bar chart of mean AUC per model, protocol and feature mode."""
    modes = [mode for mode in results['runs'] if results['runs'][mode]]
    if not modes:
        return

    figure, axes = plt.subplots(1, len(modes), figsize=(7.5 * len(modes), 5.5),
                                squeeze=False)

    for axis, mode in zip(axes[0], modes):
        protocol_runs = results['runs'][mode]
        protocols = list(protocol_runs)

        rows = []
        for section in ('gnn', 'baselines'):
            for key in sorted(protocol_runs[protocols[0]]['summary'][section]):
                rows.append((section, key))
        labels = [key for _, key in rows]

        width = 0.8 / max(1, len(protocols))
        positions = np.arange(len(rows))

        for offset, protocol in enumerate(protocols):
            means, errors = [], []
            for section, key in rows:
                auc = protocol_runs[protocol]['summary'][section].get(key, {}).get('auc', {})
                means.append(np.nan if auc.get('mean') is None else auc['mean'])
                errors.append(0.0 if auc.get('std') is None else auc['std'])
            axis.bar(positions + offset * width, means, width,
                     yerr=errors, capsize=3, label=protocol)

        axis.set_xticks(positions + width * (len(protocols) - 1) / 2)
        axis.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        axis.set_ylabel('Test AUC (mean over folds)')
        axis.set_title(f'Feature mode: {mode}')
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis='y', alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle('Spatial / Block Cross-Validation: Random vs Spatially Blocked Folds')
    figure.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close(figure)
    print(f'Saved: {OUT_FIG}')


if __name__ == '__main__':
    main()
