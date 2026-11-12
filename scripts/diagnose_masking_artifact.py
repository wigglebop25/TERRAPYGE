"""Reproduce the evidence behind Story 2.3: the XGBoost masking anomaly.

The anomaly is that baseline AUC appears to rise as more topographic evidence is
withheld from the test nodes. This script tests four candidate explanations and
writes the numbers to ``results/masking_artifact_diagnosis.json``:

1. Test-set composition. Masked test sets are nested, so a rise could simply
   reflect which nodes happen to be in the larger subset. Tested by scoring every
   trained model against every nested test subset.
2. Training-set size. Mask rate is confounded with training-set size, because
   more masked nodes means fewer training nodes.
3. Seed instability. The trend is re-estimated under independent permutation
   seeds. A real effect should survive; a tie-ordering artifact will not.
4. Score collapse. XGBoost concentrates almost all of its gain on the masked
   block, so zeroing those columns at test time collapses the score to a
   near-constant and reduces AUC to the ordering of a few leaf values.

A fifth section records why the masked block is not the only route to the label:
four kept covariates are strong spatial proxies for slope.

Run: python scripts/diagnose_masking_artifact.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.terrapyge.data.graph import load_graph
from src.terrapyge.models.experiments import FEATURE_SLICE, KEPT_COLS, MASK_COLS
from src.terrapyge.utils.paths import PROCESSED_BUHISAN as PROC_DIR, RESULTS as RESULTS_DIR

GRAPH = PROC_DIR / 'buhisan_hetero_physics.pt'
OUT_JSON = RESULTS_DIR / 'masking_artifact_diagnosis.json'

FEATURE_NAMES = [
    'dem', 'slope', 'aspect', 'curv_plan', 'curv_profile', 'twi', 'spi',
    'soil', 'worldcover', 'dynamicworld', 'chirps', 'worldclim', 'era5',
    'jrc_water', 'gamma', 'c_prime', 'phi_prime', 'static_fs',
    'critical_acc', 'arias',
]

MASK_RATES = [0.10, 0.20, 0.30]
VAL_FRACTION = 0.20
PERM_SEEDS = [42, 0, 1, 2, 3]


def build_masks(n, mask_rate, perm):
    """Nested train/val/test split for a mask rate (mirrors spatial_masking_test)."""
    n_masked = int(mask_rate * n)
    masked_idx = perm[:n_masked]
    unmasked_idx = perm[n_masked:]
    n_val = int(VAL_FRACTION * len(unmasked_idx))
    return masked_idx, unmasked_idx[:n_val], unmasked_idx[n_val:]


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42,
                         eval_metric='logloss')


def main():
    data = load_graph(GRAPH)
    n = data['su'].num_nodes
    X = data['su'].x[:, FEATURE_SLICE].numpy()
    y = data['su'].y.numpy()

    # Every node with the topographic block zeroed, so a model trained at one
    # rate can be scored on any test subset consistently.
    X_masked = X.copy()
    X_masked[:, MASK_COLS] = 0.0

    out = {'n_nodes': int(n), 'positive_rate': float(y.mean()),
           'mask_rates': MASK_RATES, 'mask_cols': MASK_COLS,
           'kept_cols': KEPT_COLS}

    # ------------------------------------------------------------------
    # 1 and 2. Model x test-subset matrix, and the training-size effect
    # ------------------------------------------------------------------
    print('=' * 78)
    print('1/2. Model trained at rate r, scored on nested test subset s')
    print('=' * 78)
    perm = np.random.default_rng(42).permutation(n)
    splits = {r: build_masks(n, r, perm) for r in MASK_RATES}

    models = {}
    for rate in MASK_RATES:
        _, _, train_idx = splits[rate]
        m = make_xgb()
        m.fit(X[train_idx], y[train_idx])
        models[rate] = m

    matrix = {}
    header = f'{"train@":<10}' + ''.join(f'{"test@" + str(int(s * 100)) + "%":>14}' for s in MASK_RATES)
    print(header + f'{"all masked":>14}')
    print('-' * len(header))
    for rate in MASK_RATES:
        row = {}
        line = f'{int(rate * 100):>3}%{"":<6}'
        for s in MASK_RATES:
            masked_idx, _, _ = splits[s]
            probs = models[rate].predict_proba(X_masked[masked_idx])[:, 1]
            auc = float(roc_auc_score(y[masked_idx], probs))
            row[f'test{int(s * 100)}'] = auc
            line += f'{auc:>14.4f}'
        probs_all = models[rate].predict_proba(X_masked)[:, 1]
        auc_all = float(roc_auc_score(y, probs_all))
        row['all_masked'] = auc_all
        line += f'{auc_all:>14.4f}'
        matrix[f'train{int(rate * 100)}'] = row
        print(line)
    out['model_by_test_matrix'] = matrix
    print()

    # ------------------------------------------------------------------
    # 3. Seed instability
    # ------------------------------------------------------------------
    print('=' * 78)
    print('3. XGBoost AUC across independent permutation seeds')
    print('=' * 78)
    print(f'{"perm_seed":<12}' + ''.join(f'{int(r * 100):>9}%' for r in MASK_RATES) + '   trend')
    print('-' * 78)

    seed_aucs = {}
    for pseed in PERM_SEEDS:
        p = np.random.default_rng(pseed).permutation(n)
        aucs = []
        for rate in MASK_RATES:
            masked_idx, _, train_idx = build_masks(n, rate, p)
            m = make_xgb()
            m.fit(X[train_idx], y[train_idx])
            probs = m.predict_proba(X_masked[masked_idx])[:, 1]
            aucs.append(float(roc_auc_score(y[masked_idx], probs)))
        seed_aucs[str(pseed)] = aucs
        rising = 'RISES' if aucs[0] < aucs[1] < aucs[2] else 'not monotonic'
        print(f'{pseed:<12}' + ''.join(f'{a:>10.4f}' for a in aucs) + f'   {rising}')

    arr = np.array(list(seed_aucs.values()))
    n_rising = sum(1 for a in arr if a[0] < a[1] < a[2])
    print('-' * 78)
    print(f'{"mean":<12}' + ''.join(f'{v:>10.4f}' for v in arr.mean(axis=0)))
    print(f'{"std":<12}' + ''.join(f'{v:>10.4f}' for v in arr.std(axis=0)))
    print(f'\nMonotonic rise in {n_rising}/{len(arr)} seeds')
    out['seed_aucs'] = seed_aucs
    out['seed_mean'] = [float(v) for v in arr.mean(axis=0)]
    out['seed_std'] = [float(v) for v in arr.std(axis=0)]
    out['n_seeds_with_monotonic_rise'] = int(n_rising)
    out['n_seeds'] = int(len(arr))
    print()

    # ------------------------------------------------------------------
    # 4. Score collapse and gain concentration
    # ------------------------------------------------------------------
    print('=' * 78)
    print('4. Score collapse and gain concentration')
    print('=' * 78)
    masked_idx, _, _ = splits[0.30]
    collapse = {}
    for rate in MASK_RATES:
        probs = models[rate].predict_proba(X_masked[masked_idx])[:, 1]
        collapse[f'train{int(rate * 100)}'] = {
            'std': float(probs.std()),
            'n_unique': int(len(np.unique(probs))),
            'n_scored': int(len(probs)),
        }
        print(f'  train@{int(rate * 100)}%: std={probs.std():.6f} '
              f'n_unique={len(np.unique(probs))} of {len(probs)}')
    out['score_collapse_test30'] = collapse

    gain = {}
    for rate in MASK_RATES:
        g = models[rate].feature_importances_
        gain[f'train{int(rate * 100)}'] = {
            'masked_block': float(g[MASK_COLS].sum()),
            'kept_block': float(g[KEPT_COLS].sum()),
        }
        print(f'  train@{int(rate * 100)}%: masked_block={g[MASK_COLS].sum():.3f} '
              f'kept_block={g[KEPT_COLS].sum():.3f}')
    out['gain_importance'] = gain
    print()

    # ------------------------------------------------------------------
    # 5. Single-feature AUC and correlation with slope
    # ------------------------------------------------------------------
    print('=' * 78)
    print('5. Single-feature AUC, and Spearman rho with slope')
    print('=' * 78)
    print(f'{"feature":<16}{"block":<10}{"n_unique":>9}{"AUC":>9}{"rho_vs_slope":>14}')
    print('-' * 60)
    single = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        block = 'MASKED' if i in MASK_COLS else 'kept'
        try:
            auc = float(roc_auc_score(y, col))
            auc = max(auc, 1.0 - auc)
        except ValueError:
            auc = float('nan')
        if name == 'slope':
            rho = float('nan')
        else:
            rho = float(spearmanr(col, X[:, 1]).statistic)
        single[name] = {'block': block, 'n_unique': int(len(np.unique(col))),
                        'auc': auc, 'rho_vs_slope': rho}
        print(f'{name:<16}{block:<10}{len(np.unique(col)):>9}{auc:>9.4f}{rho:>14.4f}')
    out['single_feature'] = single
    print()

    # ------------------------------------------------------------------
    # 6. Kept-block-only model: the honest baseline
    # ------------------------------------------------------------------
    print('=' * 78)
    print('6. XGBoost trained on the kept block only (no train/test mismatch)')
    print('=' * 78)
    print(f'{"perm_seed":<12}' + ''.join(f'{int(r * 100):>9}%' for r in MASK_RATES))
    print('-' * 74)
    kept_aucs = {}
    for pseed in PERM_SEEDS:
        p = np.random.default_rng(pseed).permutation(n)
        aucs = []
        for rate in MASK_RATES:
            masked_idx, _, train_idx = build_masks(n, rate, p)
            m = make_xgb()
            m.fit(X[np.ix_(train_idx, KEPT_COLS)], y[train_idx])
            probs = m.predict_proba(X[np.ix_(masked_idx, KEPT_COLS)])[:, 1]
            aucs.append(float(roc_auc_score(y[masked_idx], probs)))
        kept_aucs[str(pseed)] = aucs
        print(f'{pseed:<12}' + ''.join(f'{a:>10.4f}' for a in aucs))

    karr = np.array(list(kept_aucs.values()))
    print('-' * 74)
    print(f'{"mean":<12}' + ''.join(f'{v:>10.4f}' for v in karr.mean(axis=0)))
    print(f'{"std":<12}' + ''.join(f'{v:>10.4f}' for v in karr.std(axis=0)))
    out['kept_only_aucs'] = kept_aucs
    out['kept_only_mean'] = [float(v) for v in karr.mean(axis=0)]
    out['kept_only_std'] = [float(v) for v in karr.std(axis=0)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')


if __name__ == '__main__':
    main()
