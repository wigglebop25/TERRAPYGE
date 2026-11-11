"""Unit tests for the spatial / block cross-validation utilities.

Covers the split construction (grid blocks, block-to-fold assignment, fold
partitions, distance buffers) and the induced-subgraph helper used for
inductive evaluation. These are pure-function tests and do not need the graph
or any geospatial data on disk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from scripts.spatial_cv import (
    aggregate,
    apply_buffer,
    assign_blocks_to_folds,
    fold_masks,
    make_blocks,
    random_fold_ids,
    resolve_list,
)
from src.terrapyge.models.experiments import build_induced_subgraph


def test_make_blocks_grid_assignment():
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    block, n_side, cell_m = make_blocks(centroids, 4)

    assert n_side == 2
    assert cell_m == pytest.approx(5.0)
    # iy * n_side + ix for the four corners of a 2x2 grid
    assert list(block) == [0, 1, 2, 3]


def test_make_blocks_respects_grid_bounds():
    rng = np.random.default_rng(0)
    centroids = rng.uniform(0.0, 1000.0, size=(200, 2))
    block, n_side, _ = make_blocks(centroids, 9)

    assert n_side == 3
    assert block.min() >= 0
    assert block.max() <= n_side * n_side - 1


def test_make_blocks_rejects_degenerate_extent():
    centroids = np.full((5, 2), 42.0)
    with pytest.raises(ValueError):
        make_blocks(centroids, 4)


def test_assign_blocks_to_folds_keeps_blocks_intact():
    block = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    y = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0])

    folds = assign_blocks_to_folds(block, y, 3)

    for b in np.unique(block):
        assert len(np.unique(folds[block == b])) == 1, 'a block was split across folds'
    assert len(np.unique(folds)) == 3, 'not every fold received a block'


def test_assign_blocks_to_folds_balances_node_counts():
    block = np.repeat(np.arange(10), [20, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    y = np.zeros(len(block), dtype=int)

    folds = assign_blocks_to_folds(block, y, 5)
    sizes = np.bincount(folds, minlength=5)

    assert (sizes > 0).all()
    # imbalance is bounded by the largest single block
    assert sizes.max() - sizes.min() <= 20


def test_random_fold_ids_is_stratified():
    y = np.array([0] * 90 + [1] * 10)

    folds = random_fold_ids(y, 5, seed=42)

    assert len(np.unique(folds)) == 5
    for fold in range(5):
        assert 1 <= y[folds == fold].sum() <= 3


def test_fold_masks_partitions_remaining_nodes():
    fold_of_node = np.array([0, 0, 1, 1, 2, 2])

    train_idx, val_idx, test_idx = fold_masks(fold_of_node, 0, 0.5, seed=0)

    assert set(test_idx.tolist()) == {0, 1}
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx) | set(val_idx) == {2, 3, 4, 5}
    assert len(val_idx) == 2


def test_apply_buffer_drops_nearby_training_nodes():
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]])
    train_idx = np.array([1, 2])
    test_idx = np.array([0])

    kept = apply_buffer(train_idx, test_idx, centroids, buffer_m=50.0)

    assert kept.tolist() == [2], 'node within the buffer should be excluded'


def test_apply_buffer_disabled_keeps_everything():
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]])
    train_idx = np.array([1, 2])
    test_idx = np.array([0])

    kept = apply_buffer(train_idx, test_idx, centroids, buffer_m=0.0)

    assert kept.tolist() == [1, 2]


def _toy_graph():
    data = HeteroData()
    data['su'].x = torch.arange(6, dtype=torch.float).reshape(3, 2)
    data['su'].y = torch.tensor([0, 1, 0])
    data['su'].num_nodes = 3
    data['su', 'spatial', 'su'].edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data['su', 'hydro', 'su'].edge_index = torch.tensor([[0, 2], [2, 0]])
    return data


def test_build_induced_subgraph_drops_cross_boundary_edges():
    data = _toy_graph()

    sub, remap = build_induced_subgraph(data, [0, 2])

    assert sub['su'].num_nodes == 2
    assert torch.equal(sub['su'].x, torch.tensor([[0.0, 1.0], [4.0, 5.0]]))
    assert torch.equal(sub['su'].y, torch.tensor([0, 0]))
    # every spatial edge in the toy graph touches node 1, which was dropped
    assert sub['su', 'spatial', 'su'].edge_index.shape == (2, 0)
    assert remap.tolist() == [0, -1, 1]


def test_build_induced_subgraph_reindexes_edges():
    data = _toy_graph()

    sub, _ = build_induced_subgraph(data, [0, 2])

    # hydro edges (0 -> 2) and (2 -> 0) survive and become 0 -> 1 and 1 -> 0
    assert sub['su', 'hydro', 'su'].edge_index.tolist() == [[0, 1], [1, 0]]


def test_aggregate_skips_undefined_folds():
    stats = aggregate([0.5, None, 0.7])

    assert stats['mean'] == pytest.approx(0.6)
    assert stats['n'] == 2


def test_aggregate_all_undefined():
    assert aggregate([None, None])['mean'] is None


def test_resolve_list_all_and_validation():
    assert resolve_list('all', ['a', 'b'], 'thing') == ['a', 'b']
    assert resolve_list('a, b', ['a', 'b'], 'thing') == ['a', 'b']
    with pytest.raises(ValueError):
        resolve_list('z', ['a'], 'thing')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
