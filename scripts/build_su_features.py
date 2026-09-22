"""Aggregate processed rasters to slope units (zonal mean).

Produces `su_features.csv` with one row per slope unit and a mean value of each
input raster, matching the schema consumed by `build_physics_features.py`
(14 feature columns).

Raster band -> column mapping is explicit in RASTERS so provenance is clear.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

from src.terrapyge.utils.paths import PROCESSED_BUHISAN as PROC_DIR

SU_GPKG = PROC_DIR / 'slope_units.gpkg'
OUT_CSV = PROC_DIR / 'su_features.csv'

# output column -> raster file
RASTERS = {
    'dem_mean': 'dem_utm.tif',
    'slope_mean': 'slope.tif',
    'aspect_mean': 'aspect.tif',
    'curv_plan_mean': 'curv_plan.tif',
    'curv_profile_mean': 'curv_profile.tif',
    'twi_mean': 'twi.tif',
    'spi_mean': 'spi.tif',
    'soil_mean': 'soil_utm.tif',
    'worldcover_mean': 'worldcover_utm.tif',
    'dynamicworld_mean': 'dynamicworld_utm.tif',
    'chirps_mean': 'chirps_utm.tif',
    'worldclim_mean': 'worldclim_utm.tif',
    'era5_mean': 'era5_utm.tif',
    'jrc_water_mean': 'jrc_water_utm.tif',
}


def _zonal_mean(gdf, raster_name):
    path = PROC_DIR / raster_name
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        transform = src.transform
        nodata = src.nodata
    stats = zonal_stats(gdf.geometry, arr, affine=transform, stats=['mean'],
                        nodata=nodata, all_touched=False)
    vals = np.array([s['mean'] if s['mean'] is not None else np.nan
                     for s in stats], dtype=float)
    return vals


def main():
    print('=' * 70)
    print('TERRAPYGE: BUILD SU FEATURES')
    print('=' * 70)

    gdf = gpd.read_file(SU_GPKG)
    if 'cat' in gdf.columns:
        su_id = gdf['cat'].astype(int)
    elif 'value' in gdf.columns:
        su_id = gdf['value'].astype(int)
    else:
        su_id = pd.Series(range(len(gdf)))
    print(f'Slope units: {len(gdf)}')

    out = {'su_id': su_id.values}
    for col, raster in RASTERS.items():
        vals = _zonal_mean(gdf, raster)
        out[col] = vals
        print(f'  {col:20s} valid={np.isfinite(vals).sum()}/{len(vals)} '
              f'mean={np.nanmean(vals):.4f}')

    df = pd.DataFrame(out)
    df['value'] = df['su_id']
    df['label'] = np.nan
    cols = ['su_id', 'value', 'label'] + list(RASTERS.keys())
    df[cols].to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}  shape={df.shape}')
    print('\nDONE')


if __name__ == '__main__':
    main()
