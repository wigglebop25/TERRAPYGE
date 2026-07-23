"""Graph construction for TERRAPYGE.

Builds HeteroData graphs with spatial (Queen contiguity) and
hydrological (D8 flow) edge types for slope unit-based landslide
susceptibility mapping.
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import json


def load_pysheds_grid(dem_path):
    """Load conditioned DEM into pysheds and compute D8 flow direction.

    The DEM should already be conditioned (depressions filled) by WhiteboxTools.
    This function computes D8 flow direction and flow accumulation directly.

    Args:
        dem_path: Path to conditioned DEM (GeoTIFF, EPSG:32651).

    Returns:
        grid: pysheds Grid object.
        fdir: D8 flow direction array (int, 1/2/4/8/16/32/64/128).
        acc: Flow accumulation array (cells).
    """
    import rasterio
    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder

    dem_path = str(dem_path)

    # Load DEM data with rasterio
    with rasterio.open(dem_path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    # Create pysheds grid
    grid = Grid.from_raster(dem_path, data_name='dem')

    # Create Raster object (required by pysheds 0.5 API)
    vf = ViewFinder(shape=data.shape, nodata=nodata, affine=transform, crs=crs)
    dem_raster = Raster(data, viewfinder=vf)

    # D8 direction offsets: (N, NE, E, SE, S, SW, W, NW)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Compute D8 flow direction (DEM already conditioned by WhiteboxTools)
    fdir = grid.flowdir(dem_raster, routing='d8', dirmap=dirmap)

    # Compute flow accumulation
    acc = grid.accumulation(fdir, routing='d8')

    return grid, np.array(fdir), np.array(acc)


def compute_hydro_edges(su_raster_path, fdir, nodata_su=0):
    """Compute directed hydrological edges between slope units using D8 flow.

    For each cell in slope unit i, if D8 flow points to a cell in slope unit j
    (where j != i and j != nodata), add directed edge i -> j.

    Args:
        su_raster_path: Path to slope unit raster (GeoTIFF, aligned to DEM grid).
        fdir: D8 flow direction array from pysheds.
        nodata_su: NoData value in slope unit raster (default 0).

    Returns:
        edge_index: LongTensor [2, E] — directed edges [source, target].
        edge_set: Set of (source, target) tuples.
    """
    import rasterio

    # Load slope unit raster
    with rasterio.open(su_raster_path) as src:
        su = src.read(1)
        transform = src.transform

    # D8 direction offsets: (row_offset, col_offset)
    # dirmap: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
    d8_offsets = {
        1:   (0, 1),    # E
        2:   (1, 1),    # SE
        4:   (1, 0),    # S
        8:   (1, -1),   # SW
        16:  (0, -1),   # W
        32:  (-1, -1),  # NW
        64:  (-1, 0),   # N
        128: (-1, 1),   # NE
    }

    rows, cols = su.shape
    edge_set = set()

    # For each cell, check if D8 flow exits the current SU
    for row in range(rows):
        for col in range(cols):
            su_id = su[row, col]
            if su_id == nodata_su or su_id <= 0:
                continue

            direction = fdir[row, col]
            if direction == 0 or direction not in d8_offsets:
                continue

            dr, dc = d8_offsets[direction]
            nr, nc = row + dr, col + dc

            # Check bounds
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            target_su = su[nr, nc]
            if target_su == nodata_su or target_su <= 0:
                continue

            # Only add edge if flow exits to a different SU
            if target_su != su_id:
                edge_set.add((int(su_id) - 1, int(target_su) - 1))  # 0-based indexing

    # Convert to edge_index tensor
    if len(edge_set) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edges = list(edge_set)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    return edge_index, edge_set


def build_spatial_edges(gdf):
    """Build spatial edges via Queen contiguity (boundary intersection).

    Args:
        gdf: GeoDataFrame with slope unit geometries.

    Returns:
        edge_index: LongTensor [2, E] — undirected edges (both directions).
    """
    from shapely.strtree import STRtree
    from tqdm import tqdm

    tree = STRtree(gdf.geometry)
    edges = []

    for i, geom in enumerate(tqdm(gdf.geometry, desc='Spatial edges')):
        candidates = tree.query(geom, predicate='intersects')
        for j in candidates:
            if i != j:
                edges.append([i, j])
                edges.append([j, i])  # undirected

    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_hetero_data(features, labels, spatial_edge_index, hydro_edge_index,
                      feat_names=None, scaler=None):
    """Construct HeteroData graph.

    Args:
        features: FloatTensor [N, F] — node features.
        labels: LongTensor [N] — node labels.
        spatial_edge_index: LongTensor [2, E_spatial] — spatial edges.
        hydro_edge_index: LongTensor [2, E_hydro] — hydrological edges.
        feat_names: List of feature names (optional).
        scaler: StandardScaler used for normalization (optional).

    Returns:
        data: HeteroData object.
    """
    data = HeteroData()

    # Node features
    data['su'].x = features.float()
    data['su'].y = labels.long()
    data['su'].num_nodes = features.shape[0]

    if feat_names is not None:
        data['su'].feat_names = feat_names

    if scaler is not None:
        data['su'].scaler_mean = scaler.mean_.tolist()
        data['su'].scaler_scale = scaler.scale_.tolist()

    # Edge types
    data['su', 'spatial', 'su'].edge_index = spatial_edge_index
    data['su', 'hydro', 'su'].edge_index = hydro_edge_index

    return data


def validate_graph(data):
    """Validate graph connectivity and print statistics.

    Args:
        data: HeteroData object.

    Returns:
        stats: Dict with graph statistics.
    """
    n = data['su'].num_nodes
    spatial_edges = data['su', 'spatial', 'su'].edge_index.shape[1]
    hydro_edges = data['su', 'hydro', 'su'].edge_index.shape[1]

    # Check for isolated nodes
    all_nodes = set(range(n))
    connected_nodes = set()

    if spatial_edges > 0:
        spatial_src = set(data['su', 'spatial', 'su'].edge_index[0].tolist())
        spatial_dst = set(data['su', 'spatial', 'su'].edge_index[1].tolist())
        connected_nodes |= spatial_src | spatial_dst

    if hydro_edges > 0:
        hydro_src = set(data['su', 'hydro', 'su'].edge_index[0].tolist())
        hydro_dst = set(data['su', 'hydro', 'su'].edge_index[1].tolist())
        connected_nodes |= hydro_src | hydro_dst

    isolated = all_nodes - connected_nodes

    # Degree statistics
    spatial_degrees = torch.zeros(n)
    hydro_degrees = torch.zeros(n)

    if spatial_edges > 0:
        for node in data['su', 'spatial', 'su'].edge_index[0]:
            spatial_degrees[node] += 1

    if hydro_edges > 0:
        for node in data['su', 'hydro', 'su'].edge_index[0]:
            hydro_degrees[node] += 1

    total_degrees = spatial_degrees + hydro_degrees

    stats = {
        'n_nodes': n,
        'n_features': data['su'].x.shape[1],
        'spatial_edges': spatial_edges,
        'hydro_edges': hydro_edges,
        'total_edges': spatial_edges + hydro_edges,
        'isolated_nodes': len(isolated),
        'spatial_degree_mean': float(spatial_degrees.mean()),
        'spatial_degree_max': int(spatial_degrees.max()),
        'hydro_degree_mean': float(hydro_degrees.mean()),
        'hydro_degree_max': int(hydro_degrees.max()),
        'total_degree_mean': float(total_degrees.mean()),
        'total_degree_max': int(total_degrees.max()),
        'label_pos_rate': float(data['su'].y.float().mean()) if 'y' in data['su'] and data['su'].y is not None else None,
    }

    print('\n=== Graph Validation ===')
    for k, v in stats.items():
        print(f'  {k}: {v}')

    if len(isolated) > 0:
        print(f'\n  WARNING: {len(isolated)} isolated nodes: {sorted(list(isolated))[:20]}...')
    else:
        print('\n  No isolated nodes — graph is connected.')

    return stats


def load_graph(graph_path):
    """Load HeteroData graph from file.

    Args:
        graph_path: Path to .pt file.

    Returns:
        data: HeteroData object.
    """
    from torch_geometric.data import HeteroData
    torch.serialization.add_safe_globals([HeteroData])
    return torch.load(graph_path, weights_only=False)


def save_graph(data, graph_path, metadata_path=None):
    """Save HeteroData graph to file.

    Args:
        data: HeteroData object.
        graph_path: Path to save .pt file.
        metadata_path: Optional path to save metadata JSON.
    """
    torch.save(data, graph_path)

    if metadata_path is not None:
        meta = {
            'n_nodes': data['su'].num_nodes,
            'n_features': data['su'].x.shape[1],
            'spatial_edges': data['su', 'spatial', 'su'].edge_index.shape[1],
            'hydro_edges': data['su', 'hydro', 'su'].edge_index.shape[1],
            'feat_names': getattr(data['su'], 'feat_names', None),
            'crs': 'EPSG:32651',
        }
        with open(metadata_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'Metadata saved to {metadata_path}')
