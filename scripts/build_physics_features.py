"""Build physics-informed features for TERRAPYGE slope units.

Steps:
1. Aggregate the 6-band soil raster (clay, sand, silt, ph, bd, soc) to slope units.
2. Load existing terrain/climate features (su_features.csv).
3. Compute physics features: unit weight, cohesion, friction angle,
   static Factor of Safety, critical acceleration, Arias intensity,
   Newmark displacement.
4. Generate physics-derived susceptibility labels.
5. Save physics_features.csv and a physics-informed graph
   (buhisan_hetero_physics.pt) without touching the original graph.
"""

import sys
sys.path.insert(0, r'D:\TERRAPYGE')

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import torch
from pathlib import Path
from rasterstats import zonal_stats

from src.terrapyge.features.physics import (
    derive_geotechnical_params,
    static_fs,
    critical_acceleration_g,
    arias_intensity,
    newmark_displacement_cm,
    physics_labels,
)

PROC_DIR = Path(r'D:\TERRAPYGE\data\processed\buhisan')
SOIL_6BAND = PROC_DIR / 'soil_6band_utm.tif'
SU_GPKG = PROC_DIR / 'slope_units.gpkg'
FEATURES_CSV = PROC_DIR / 'su_features.csv'
GRAPH = PROC_DIR / 'buhisan_hetero.pt'
OUT_CSV = PROC_DIR / 'physics_features.csv'
OUT_GRAPH = PROC_DIR / 'buhisan_hetero_physics.pt'

# Physics parameters (from config.yaml)
import yaml
with open(r'D:\TERRAPYGE\config.yaml') as _f:
    _cfg = yaml.safe_load(_f)
_PHYS = _cfg.get('physics', {})
PGA_G = float(_PHYS.get('pga_g', 0.4))
DURATION_S = float(_PHYS.get('duration_s', 10.0))
SOIL_DEPTH_M = float(_PHYS.get('soil_depth_m', 2.0))
PORE_PRESSURE_RATIO = float(_PHYS.get('pore_pressure_ratio', 0.15))
LABEL_THRESHOLDS = tuple(_PHYS.get('label_thresholds_cm', [2.0, 5.0, 15.0]))
BINARY_THRESHOLD = float(_PHYS.get('binary_threshold_cm', 2.0))

BAND_NAMES = ['clay', 'sand', 'silt', 'ph', 'bulk_density', 'soc']


def aggregate_soil(gdf):
    """Zonal mean of each soil band per slope unit."""
    print('Aggregating 6-band soil to slope units...')
    with rasterio.open(SOIL_6BAND) as src:
        transform = src.transform
        nodata = src.nodata
        bands = {BAND_NAMES[i]: src.read(i + 1) for i in range(src.count)}

    out = {}
    for name, arr in bands.items():
        stats = zonal_stats(gdf.geometry, arr, affine=transform,
                            stats=['mean'], nodata=nodata, all_touched=False)
        vals = np.array([s['mean'] if s['mean'] is not None else np.nan
                         for s in stats], dtype=float)
        out[f'{name}_mean'] = vals
        print(f'  {name}: valid={np.isfinite(vals).sum()}/{len(vals)}')

    soil_df = pd.DataFrame(out, index=gdf.index)
    # Fill gaps at raster edges with per-band mean
    for col in soil_df.columns:
        soil_df[col] = soil_df[col].fillna(soil_df[col].mean())
    return soil_df


def main():
    print('=' * 70)
    print('TERRAPYGE: BUILD PHYSICS FEATURES')
    print('=' * 70)

    # 1. Slope units
    gdf = gpd.read_file(SU_GPKG)
    if 'cat' in gdf.columns:
        gdf = gdf.rename(columns={'cat': 'su_id'})
    elif 'value' in gdf.columns:
        gdf = gdf.rename(columns={'value': 'su_id'})
    gdf = gdf.set_index('su_id').sort_index()
    print(f'Slope units: {len(gdf)}')

    # 2. Soil aggregation
    soil_df = aggregate_soil(gdf)

    # 3. Existing features
    feats = pd.read_csv(FEATURES_CSV)
    feats = feats.set_index('su_id').sort_index()
    if len(feats) != len(gdf):
        raise ValueError(f'Feature/SU mismatch: {len(feats)} vs {len(gdf)}')
    print(f'Existing features: {feats.shape[1]} columns')

    # 4. Geotechnical parameters
    print('\nDeriving geotechnical parameters...')
    gamma, c_prime, phi = derive_geotechnical_params(
        soil_df['clay_mean'].values,
        soil_df['sand_mean'].values,
        soil_df['bulk_density_mean'].values,
    )

    slope = feats['slope_mean'].values.astype(float)
    pga_ms2 = PGA_G * 9.81

    # 5. Physics features
    print('Computing physics features...')
    fs = static_fs(slope, c_prime, phi, gamma, SOIL_DEPTH_M,
                   ru=PORE_PRESSURE_RATIO)
    ac_g = critical_acceleration_g(fs, slope)
    ia = arias_intensity(pga_ms2, DURATION_S)
    dn = newmark_displacement_cm(ia, ac_g)
    labels = physics_labels(dn, thresholds=LABEL_THRESHOLDS)

    print(f'  FS:      min={np.nanmin(fs):.3f} max={np.nanmax(fs):.3f} mean={np.nanmean(fs):.3f}')
    print(f'  Ac (g):  min={np.nanmin(ac_g):.4f} max={np.nanmax(ac_g):.4f} mean={np.nanmean(ac_g):.4f}')
    print(f'  Ia:      {ia:.3f} m/s (uniform)')
    print(f'  Dn (cm): min={np.nanmin(dn):.3f} max={np.nanmax(dn):.3f} mean={np.nanmean(dn):.3f}')
    print(f'  Dn >= {BINARY_THRESHOLD:g} cm (Moderate+ / positive class): '
          f'{(dn >= BINARY_THRESHOLD).sum()} / {len(dn)} ({(dn >= BINARY_THRESHOLD).mean():.1%})')

    # 6. Assemble physics table
    physics_df = pd.DataFrame({
        'su_id': gdf.index,
        'clay_mean': soil_df['clay_mean'].values.round(4),
        'sand_mean': soil_df['sand_mean'].values.round(4),
        'silt_mean': soil_df['silt_mean'].values.round(4),
        'bulk_density_mean': soil_df['bulk_density_mean'].values.round(4),
        'gamma_kNm3': gamma.round(4),
        'c_prime_kpa': c_prime.round(4),
        'phi_prime_deg': phi.round(4),
        'static_fs': fs.round(4),
        'critical_acc_g': ac_g.round(6),
        'arias_intensity': np.full(len(dn), ia).round(6),
        'newmark_dn_cm': dn.round(4),
        'physics_class': labels,
        'physics_label_binary': (dn >= BINARY_THRESHOLD).astype(int),
    }).set_index('su_id')

    physics_df.to_csv(OUT_CSV)
    print(f'\nSaved: {OUT_CSV}')

    # 7. Build physics-informed graph (original graph untouched)
    print('\nBuilding physics-informed graph...')
    from torch_geometric.data import HeteroData
    torch.serialization.add_safe_globals([HeteroData])
    data = torch.load(GRAPH, weights_only=False)

    # Feature matrix: existing numeric features + physics features
    base_cols = [c for c in feats.columns
                 if c not in ('value', 'label') and pd.api.types.is_numeric_dtype(feats[c])]
    base_X = feats[base_cols].astype(float).values

    phys_cols = ['gamma_kNm3', 'c_prime_kpa', 'phi_prime_deg',
                 'static_fs', 'critical_acc_g', 'arias_intensity',
                 'newmark_dn_cm']
    phys_X = physics_df[phys_cols].astype(float).values

    X = np.hstack([base_X, phys_X]).astype(np.float32)

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Physics-derived labels (binary)
    y = torch.tensor(physics_df['physics_label_binary'].values, dtype=torch.long)

    data_phys = HeteroData()
    data_phys['su'].x = torch.from_numpy(X)
    data_phys['su'].y = y
    data_phys['su'].num_nodes = data['su'].num_nodes
    data_phys['su', 'spatial', 'su'].edge_index = data['su', 'spatial', 'su'].edge_index
    data_phys['su', 'hydro', 'su'].edge_index = data['su', 'hydro', 'su'].edge_index

    feat_names = base_cols + phys_cols
    data_phys['su'].feat_names = feat_names
    data_phys['su'].scaler_mean = scaler.mean_.tolist()
    data_phys['su'].scaler_scale = scaler.scale_.tolist()
    data_phys['su'].crs = 'EPSG:32651'

    torch.save(data_phys, OUT_GRAPH)
    print(f'Saved: {OUT_GRAPH}')
    print(f'  Nodes: {data_phys["su"].num_nodes}')
    print(f'  Features: {len(feat_names)} (base={len(base_cols)}, physics={len(phys_cols)})')
    print(f'  Positive label rate: {y.float().mean():.1%}')
    print(f'  Spatial edges: {data_phys["su", "spatial", "su"].edge_index.shape[1]}')
    print(f'  Hydro edges: {data_phys["su", "hydro", "su"].edge_index.shape[1]}')

    print('\nPHYSICS FEATURE BUILD COMPLETE')


if __name__ == '__main__':
    main()
