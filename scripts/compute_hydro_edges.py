"""Compute hydrological edges and update graph."""
import sys
sys.path.insert(0, r'D:\TERRAPYGE')

from src.terrapyge.data.graph import (
    load_pysheds_grid, compute_hydro_edges, validate_graph, load_graph, save_graph
)
from pathlib import Path

PROC_DIR = Path(r'D:\TERRAPYGE\data\processed\buhisan')

# 1. Load pysheds grid
print('Loading DEM into pysheds...')
grid, fdir, acc = load_pysheds_grid(PROC_DIR / 'dem_conditioned.tif')
print(f'  Flow direction shape: {fdir.shape}')
print(f'  Flow accumulation max: {acc.max()}')

# 2. Compute hydro edges
print('\nComputing hydrological edges...')
hydro_edge_index, edge_set = compute_hydro_edges(
    PROC_DIR / 'su.tif', fdir, nodata_su=0
)
print(f'  Hydro edges: {hydro_edge_index.shape[1]} directed edges')
print(f'  Unique source SUs: {len(set(hydro_edge_index[0].tolist()))}')
print(f'  Unique target SUs: {len(set(hydro_edge_index[1].tolist()))}')

# 3. Load existing graph
print('\nLoading existing graph...')
data = load_graph(PROC_DIR / 'buhisan_hetero.pt')
n = data['su'].num_nodes
spatial_e = data['su', 'spatial', 'su'].edge_index.shape[1]
hydro_e_before = data['su', 'hydro', 'su'].edge_index.shape[1]
print(f'  Nodes: {n}')
print(f'  Spatial edges: {spatial_e}')
print(f'  Hydro edges (before): {hydro_e_before}')

# 4. Update hydro edges
data['su', 'hydro', 'su'].edge_index = hydro_edge_index
hydro_e_after = data['su', 'hydro', 'su'].edge_index.shape[1]
print(f'  Hydro edges (after): {hydro_e_after}')

# 5. Validate
print('\nValidating updated graph...')
stats = validate_graph(data)

# 6. Save
print('\nSaving updated graph...')
save_graph(
    data,
    PROC_DIR / 'buhisan_hetero.pt',
    PROC_DIR / 'graph_metadata.json'
)
print('Done.')
