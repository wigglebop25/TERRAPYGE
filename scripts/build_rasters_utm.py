"""Reproject raw Buhisan rasters to the processed UTM 51N grid.

The DEM (SRTM 30 m, EPSG:4326) defines the target grid: it is reprojected to
EPSG:32651 and every other raw raster is resampled band-1 onto that grid.

Band selection: multi-band sources use band 1 by default (see BAND in
BAND_MAP). Adjust per source if a different band is intended.

Outputs (data/processed/buhisan/):
  dem_utm.tif, chirps_utm.tif, worldclim_utm.tif, era5_utm.tif,
  worldcover_utm.tif, dynamicworld_utm.tif, jrc_water_utm.tif
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from src.terrapyge.utils.paths import PROCESSED_BUHISAN as PROC_DIR, RAW

RAW_DIR = RAW / 'buhisan'
DEM_RAW = RAW_DIR / 'dem' / 'Buhisan_DEM_SRTM_30m.tif'
TARGET_CRS = 'EPSG:32651'

# output name -> (raw relative path, band index, resampling)
BAND_MAP = {
    'chirps_utm.tif': ('climate/Buhisan_CHIRPS_AnnualRainfall.tif', 1, Resampling.bilinear),
    'worldclim_utm.tif': ('climate/Buhisan_WorldClim_Annual.tif', 1, Resampling.bilinear),
    'era5_utm.tif': ('climate/Buhisan_ERA5Land.tif', 1, Resampling.bilinear),
    'worldcover_utm.tif': ('landcover/Buhisan_WorldCover_10m.tif', 1, Resampling.nearest),
    'dynamicworld_utm.tif': ('landcover/Buhisan_DynamicWorld.tif', 1, Resampling.bilinear),
    'jrc_water_utm.tif': ('water/Buhisan_JRC_Water.tif', 1, Resampling.bilinear),
}


def _target_grid():
    """Compute the UTM 51N target grid from the raw DEM."""
    with rasterio.open(DEM_RAW) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
        return transform, width, height


def reproject_dem(transform, width, height):
    out = PROC_DIR / 'dem_utm.tif'
    with rasterio.open(DEM_RAW) as src:
        dst = np.full((height, width), np.nan, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
        profile = src.profile.copy()
        profile.update(crs=TARGET_CRS, transform=transform, width=width,
                       height=height, count=1, dtype='float64', nodata=np.nan,
                       compress='lzw')
    with rasterio.open(out, 'w', **profile) as dst_ds:
        dst_ds.write(dst.astype(np.float64), 1)
    print(f'dem_utm.tif  shape=({height},{width})  res={transform.a:.4f}')
    return out


def reproject_band(out_name, rel_path, band, resampling, transform, width, height):
    src_path = RAW_DIR / rel_path
    out = PROC_DIR / out_name
    with rasterio.open(src_path) as src:
        dst = np.full((height, width), np.nan, dtype=np.float64)
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
        dtype = 'uint8' if resampling == Resampling.nearest else 'float32'
        nodata = 255 if dtype == 'uint8' else np.nan
        profile = {'driver': 'GTiff', 'dtype': dtype, 'width': width,
                   'height': height, 'count': 1, 'crs': TARGET_CRS,
                   'transform': transform, 'nodata': nodata, 'compress': 'lzw'}
        arr = dst if dtype == 'float32' else np.nan_to_num(dst, nan=255).astype('uint8')
    with rasterio.open(out, 'w', **profile) as dst_ds:
        dst_ds.write(arr.astype(dtype), 1)
    print(f'{out_name}  from {rel_path} band {band}')


def main():
    print('=' * 70)
    print('TERRAPYGE: BUILD UTM RASTERS')
    print('=' * 70)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    transform, width, height = _target_grid()
    reproject_dem(transform, width, height)
    for out_name, (rel_path, band, resampling) in BAND_MAP.items():
        reproject_band(out_name, rel_path, band, resampling, transform, width, height)
    print('\nDONE')


if __name__ == '__main__':
    main()
