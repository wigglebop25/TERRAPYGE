"""Build the base slope-unit graph with undirected spatial adjacency edges.

Spatial edges are Queen contiguity over the slope-unit polygons (`libpysal`),
stored as a symmetric edge_index. Hydrological (directed D8) edges are added
afterwards by `scripts/compute_hydro_edges.py`.

Node features come from `su_features.csv`; the output `buhisan_hetero.pt` is the
pre-physics graph consumed by `build_physics_features.py`.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from libpysal.weights import Queen

from src.terrapyge.utils.paths import PROCESSED_BUHISAN as PROC_DIR

SU_GPKG = PROC_DIR / 'slope_units.gpkg'
FEATURES_CSV = PROC_DIR / 'su_features.csv'
OUT_GRAPH = PROC_DIR / 'buhisan_hetero.pt'


def main():
    print('=' * 70)
    print('TERRAPYGE: BUILD SPATIAL EDGES')
    print('=' * 70)

    gdf = gpd.read_file(SU_GPKG).reset_index(drop=True)
    feats = pd.read_csv(FEATURES_CSV)
    if len(gdf) != len(feats):
        raise ValueError(f'SU/feature mismatch: {len(gdf)} vs {len(feats)}')

    print(f'Slope units: {len(gdf)}')
    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)

    src, dst = [], []
    for i, neighbours in w.neighbors.items():
        for j in neighbours:
            src.append(i)
            dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    print(f'Spatial edges (symmetric): {edge_index.shape[1]}')

    feature_cols = [c for c in feats.columns if c != 'su_id']
    X = torch.tensor(feats[feature_cols].astype(float).values, dtype=torch.float32)

    from torch_geometric.data import HeteroData
    data = HeteroData()
    data['su'].x = X
    data['su'].num_nodes = len(gdf)
    data['su'].feat_names = feature_cols
    data['su', 'spatial', 'su'].edge_index = edge_index
    data['su', 'hydro', 'su'].edge_index = torch.zeros((2, 0), dtype=torch.long)

    torch.serialization.add_safe_globals([HeteroData])
    torch.save(data, OUT_GRAPH)
    print(f'Saved: {OUT_GRAPH}')
    print(f'  Features: {len(feature_cols)}')
    print('\nDONE')


if __name__ == '__main__':
    main()
