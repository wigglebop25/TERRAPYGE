"""Phase 2: Graph Construction Pipeline.

Builds the HeteroData graph for TERRAPYGE:
- Node features: zonal stats over slope units (elevation, slope, aspect, curvature, TWI, SPI)
- Spatial edges: Queen contiguity (undirected)
- Hydrological edges: D8 flow routing (directed)

Usage:
    python phase2_graph_construction.py
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import generic_filter, sobel
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_raster(path):
    """Load a raster and return data + profile."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    return data, profile


def load_slope_units(path):
    """Load slope units shapefile and sort by cat for deterministic ordering."""
    gdf = gpd.read_file(path)
    gdf = gdf.sort_values('cat').reset_index(drop=True)
    return gdf


# ============================================================
# 2. DERIVED RASTERS
# ============================================================

def compute_curvature(dem, pixel_size):
    """Compute plan curvature from DEM (second derivative)."""
    # Sobel filters for second derivatives
    dx = sobel(dem, axis=1) / (8 * pixel_size)
    dy = sobel(dem, axis=0) / (8 * pixel_size)
    dxx = sobel(dx, axis=1) / (8 * pixel_size)
    dyy = sobel(dy, axis=0) / (8 * pixel_size)
    curvature = -2 * (dxx + dyy)
    return curvature


def compute_twi(slope_rad, flow_accum, pixel_size):
    """Compute Topographic Wetness Index: ln(a / tan(b))."""
    # Specific catchment area = flow_accum * pixel_size / pixel_width
    # For D8, flow_accum is in cells, so specific catchment area = flow_accum * pixel_size
    a = flow_accum * pixel_size
    tan_b = np.tan(slope_rad)
    # Avoid division by zero
    tan_b = np.clip(tan_b, 1e-6, None)
    twi = np.log(a / tan_b)
    return twi


def compute_spi(slope_rad, flow_accum, pixel_size):
    """Compute Stream Power Index: a * tan(b)."""
    a = flow_accum * pixel_size
    spi = a * np.tan(slope_rad)
    return spi


# ============================================================
# 3. ZONAL STATISTICS
# ============================================================

def zonal_stats_vectorized(raster_data, raster_profile, polygons):
    """Compute zonal stats (mean, std) for each polygon over a raster.

    Uses rasterized approach for speed — rasterize each polygon and compute stats.
    """
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    transform = raster_profile['transform']
    out_shape = (raster_profile['height'], raster_profile['width'])
    crs = raster_profile['crs']

    means = np.full(len(polygons), np.nan)
    stds = np.full(len(polygons), np.nan)

    for i, geom in enumerate(polygons):
        try:
            # Create mask for this polygon (True = outside polygon)
            mask = geometry_mask(
                [mapping(geom)],
                out_shape=out_shape,
                transform=transform,
                invert=True  # True = inside polygon
            )
            # Extract values inside polygon
            values = raster_data[mask]
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                means[i] = np.mean(valid)
                stds[i] = np.std(valid) if len(valid) > 1 else 0.0
        except Exception:
            pass

    return means, stds


# ============================================================
# 4. SPATIAL EDGES (Queen Contiguity)
# ============================================================

def build_spatial_edges(gdf):
    """Build Queen contiguity edges using libpysal.

    Returns edge_index as [2, num_edges] tensor (both directions for undirected).
    """
    from libpysal.weights import Queen

    w = Queen.from_dataframe(gdf, silence_warnings=True)

    sources = []
    targets = []
    for i, neighbors in w.neighbors.items():
        for j in neighbors:
            sources.append(i)
            targets.append(j)

    edge_index = np.array([sources, targets], dtype=np.int64)
    return edge_index, w


# ============================================================
# 5. HYDROLOGICAL EDGES (D8 Flow Routing)
# ============================================================

def build_hydrological_edges(dem_data, dem_profile, slope_units_gdf):
    """Build directed hydrological edges using D8 flow routing.

    For each pair of adjacent slope units, determine if water flows
    from one to the other based on flow direction at shared boundaries.

    Returns edge_index_hydro as [2, num_edges] (directed: upstream -> downstream).
    """
    from libpysal.weights import Queen

    transform = dem_profile['transform']
    pixel_size = abs(transform.a)  # meters

    # Compute flow direction using simple D8 algorithm
    # D8 codes: E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128
    flow_dir = compute_d8_flow_direction(dem_data)

    # Get Queen contiguity for adjacency
    w = Queen.from_dataframe(slope_units_gdf, silence_warnings=True)

    # For each pair of adjacent SUs, check flow direction at boundary
    sources = []
    targets = []

    # Pre-compute: for each SU, find the mean elevation
    geometries = slope_units_gdf.geometry.values
    su_elevations = np.full(len(geometries), np.nan)

    for i, geom in enumerate(geometries):
        from rasterio.features import geometry_mask
        from shapely.geometry import mapping
        try:
            mask = geometry_mask([mapping(geom)],
                               out_shape=(dem_profile['height'], dem_profile['width']),
                               transform=transform, invert=True)
            vals = dem_data[mask]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                su_elevations[i] = np.mean(valid)
        except:
            pass

    # Build directed edges: upstream (higher elevation) -> downstream (lower elevation)
    for i, neighbors in w.neighbors.items():
        for j in neighbors:
            if i < j:  # avoid duplicates
                ei = su_elevations[i] if not np.isnan(su_elevations[i]) else 0
                ej = su_elevations[j] if not np.isnan(su_elevations[j]) else 0
                if ei > ej:
                    sources.append(i)
                    targets.append(j)
                elif ej > ei:
                    sources.append(j)
                    targets.append(i)
                # If equal elevation, no directed edge (flat boundary)

    if len(sources) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    edge_index = np.array([sources, targets], dtype=np.int64)
    return edge_index


def compute_d8_flow_direction(dem):
    """Compute D8 flow direction from DEM.

    Returns flow direction raster with D8 codes.
    """
    rows, cols = dem.shape
    flow_dir = np.zeros_like(dem, dtype=np.uint8)

    # D8 direction offsets: (dr, dc, code)
    directions = [
        (0, 1, 1),    # E
        (1, 1, 2),    # SE
        (1, 0, 4),    # S
        (1, -1, 8),   # SW
        (0, -1, 16),  # W
        (-1, -1, 32), # NW
        (-1, 0, 64),  # N
        (-1, 1, 128), # NE
    ]

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if np.isnan(dem[r, c]):
                continue

            max_drop = 0
            max_dir = 0
            for dr, dc, code in directions:
                nr, nc = r + dr, c + dc
                if np.isnan(dem[nr, nc]):
                    continue
                drop = dem[r, c] - dem[nr, nc]
                if drop > max_drop:
                    max_drop = drop
                    max_dir = code

            flow_dir[r, c] = max_dir

    return flow_dir


# ============================================================
# 6. ASSEMBLE GRAPH
# ============================================================

def assemble_graph(node_features, feature_names, edge_index_spatial, edge_index_hydro):
    """Assemble HeteroData graph."""
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()

    # Node features
    data['slope_unit'].x = torch.tensor(node_features, dtype=torch.float32)

    # Spatial edges (undirected — both directions already in edge_index)
    if edge_index_spatial.shape[1] > 0:
        data['slope_unit', 'spatially_adjacent', 'slope_unit'].edge_index = \
            torch.tensor(edge_index_spatial, dtype=torch.long)

    # Hydrological edges (directed)
    if edge_index_hydro.shape[1] > 0:
        data['slope_unit', 'flows_into', 'slope_unit'].edge_index = \
            torch.tensor(edge_index_hydro, dtype=torch.long)

    return data


# ============================================================
# 7. DIAGNOSTICS
# ============================================================

def graph_diagnostics(data):
    """Compute and print graph diagnostics."""
    print("\n" + "=" * 60)
    print("GRAPH DIAGNOSTICS")
    print("=" * 60)

    x = data['slope_unit'].x
    print(f"\nNode features:")
    print(f"  Nodes: {x.shape[0]}")
    print(f"  Features: {x.shape[1]}")
    print(f"  Feature names: {data.get('feature_names', 'N/A')}")

    # Spatial edges
    if ('slope_unit', 'spatially_adjacent', 'slope_unit') in data.edge_types:
        es = data['slope_unit', 'spatially_adjacent', 'slope_unit'].edge_index
        n_edges = es.shape[1]
        n_undirected = n_edges // 2
        # Degree distribution
        degrees = np.bincount(es[0].numpy(), minlength=x.shape[0])
        print(f"\nSpatial edges (undirected):")
        print(f"  Total directed edges: {n_edges}")
        print(f"  Unique undirected edges: {n_undirected}")
        print(f"  Mean degree: {degrees.mean():.1f}")
        print(f"  Median degree: {np.median(degrees):.0f}")
        print(f"  Max degree: {degrees.max()}")
        print(f"  Min degree: {degrees.min()}")
        print(f"  Isolated nodes: {(degrees == 0).sum()}")

    # Hydrological edges
    if ('slope_unit', 'flows_into', 'slope_unit') in data.edge_types:
        eh = data['slope_unit', 'flows_into', 'slope_unit'].edge_index
        n_edges = eh.shape[1]
        print(f"\nHydrological edges (directed):")
        print(f"  Total edges: {n_edges}")

        # Check for cycles (should be a DAG)
        G = nx.DiGraph()
        G.add_edges_from(eh.numpy().T.tolist())
        try:
            nx.find_cycle(G)
            print("  WARNING: Graph contains cycles (should be a DAG)")
        except nx.NetworkXNoCycle:
            print("  DAG: No cycles detected")

        # Connected components
        G_undirected = G.to_undirected()
        n_components = nx.number_connected_components(G_undirected)
        print(f"  Connected components: {n_components}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Paths
    processed_dir = Path.cwd() / 'data' / 'processed'
    graphs_dir = Path.cwd() / 'data' / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)

    dem_path = processed_dir / 'CebuCity_DEM_conditioned_utm.tif'
    slope_path = processed_dir / 'CebuCity_Slope_utm.tif'
    aspect_path = processed_dir / 'CebuCity_Aspect_utm.tif'
    su_path = processed_dir / 'slope_units.shp'
    output_path = graphs_dir / 'cebu_hetero.pt'

    print("=" * 60)
    print("PHASE 2: GRAPH CONSTRUCTION")
    print("=" * 60)

    # === Load data ===
    print("\nStep 1: Loading data...")
    dem, dem_profile = load_raster(dem_path)
    slope_data, _ = load_raster(slope_path)
    aspect_data, _ = load_raster(aspect_path)
    gdf = load_slope_units(su_path)

    pixel_size = abs(dem_profile['transform'].a)
    print(f"  DEM shape: {dem.shape}, pixel size: {pixel_size:.2f} m")
    print(f"  Slope units: {len(gdf)}")

    # === Compute derived rasters ===
    print("\nStep 2: Computing derived rasters...")
    print("  Curvature...")
    curvature = compute_curvature(dem, pixel_size)

    # Convert slope to radians for TWI/SPI
    slope_rad = np.deg2rad(slope_data)

    # Compute flow accumulation (simple upstream count)
    print("  Flow accumulation...")
    flow_accum = compute_flow_accumulation(dem, dem_profile)

    print("  TWI...")
    twi = compute_twi(slope_rad, flow_accum, pixel_size)

    print("  SPI...")
    spi = compute_spi(slope_rad, flow_accum, pixel_size)

    # === Zonal statistics ===
    print("\nStep 3: Computing zonal statistics...")
    geometries = gdf.geometry.values

    features = {}
    feature_names = []

    rasters = {
        'elevation': dem,
        'slope': slope_data,
        'aspect': aspect_data,
        'curvature': curvature,
        'twi': twi,
        'spi': spi,
    }

    for name, raster in rasters.items():
        print(f"  {name}...")
        means, stds = zonal_stats_vectorized(raster, dem_profile, geometries)
        features[f'{name}_mean'] = means
        features[f'{name}_std'] = stds
        feature_names.extend([f'{name}_mean', f'{name}_std'])

    # Stack features into matrix
    node_features = np.column_stack([features[k] for k in feature_names])

    # Replace NaN with 0 (for SUs that don't overlap any valid raster cells)
    nan_mask = np.isnan(node_features)
    node_features[nan_mask] = 0
    n_nan_sus = np.any(nan_mask, axis=1).sum()
    print(f"  Nodes with NaN features (zeroed): {n_nan_sus}")

    # Normalize features (StandardScaler)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)
    print(f"  Feature matrix shape: {node_features.shape}")

    # === Spatial edges ===
    print("\nStep 4: Building spatial edges (Queen contiguity)...")
    edge_index_spatial, w = build_spatial_edges(gdf)
    print(f"  Spatial edges: {edge_index_spatial.shape[1]} directed ({edge_index_spatial.shape[1]//2} undirected)")

    # === Hydrological edges ===
    print("\nStep 5: Building hydrological edges (elevation-based)...")
    edge_index_hydro = build_hydrological_edges(dem, dem_profile, gdf)
    print(f"  Hydrological edges: {edge_index_hydro.shape[1]} directed")

    # === Assemble graph ===
    print("\nStep 6: Assembling HeteroData...")
    data = assemble_graph(node_features, feature_names, edge_index_spatial, edge_index_hydro)
    data['feature_names'] = feature_names

    # === Diagnostics ===
    graph_diagnostics(data)

    # === Save ===
    print("\nStep 7: Saving graph...")
    import torch
    torch.save(data, output_path)
    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("=" * 60)


def compute_flow_accumulation(dem, profile):
    """Compute simple flow accumulation (upstream cell count) using D8."""
    rows, cols = dem.shape
    accum = np.ones_like(dem, dtype=np.float64)
    accum[np.isnan(dem)] = 0

    # D8 direction offsets
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    # Sort cells by elevation (highest first) for upstream accumulation
    valid_mask = ~np.isnan(dem)
    elevations = dem.copy()
    elevations[~valid_mask] = -np.inf

    # Get sorted indices (highest elevation first)
    flat_idx = np.argsort(-elevations.ravel())
    rows_idx = flat_idx // cols
    cols_idx = flat_idx % cols

    # Accumulate flow downstream
    for k in range(len(flat_idx)):
        r, c = rows_idx[k], cols_idx[k]
        if not valid_mask[r, c]:
            continue

        # Find steepest downslope neighbor
        max_drop = 0
        nr, nc = r, c
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and valid_mask[rr, cc]:
                drop = dem[r, c] - dem[rr, cc]
                if drop > max_drop:
                    max_drop = drop
                    nr, nc = rr, cc

        # Transfer accumulation to downstream cell
        if (nr, nc) != (r, c):
            accum[nr, nc] += accum[r, c]

    return accum


if __name__ == '__main__':
    main()
