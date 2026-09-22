"""Reproject the raw 6-band SoilGrids raster to the processed UTM grid.

Fixes the earlier band-collapse bug where only band 1 was retained.
Preserves: clay, sand, silt, ph, bulk_density, soc.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from src.terrapyge.utils.paths import PROCESSED_BUHISAN, RAW

RAW_SOIL = RAW / 'buhisan' / 'soil' / 'Buhisan_SoilGrids_250m.tif'
DEM_REF = PROCESSED_BUHISAN / 'dem_utm.tif'
OUT = PROCESSED_BUHISAN / 'soil_6band_utm.tif'

BAND_NAMES = ['clay', 'sand', 'silt', 'ph', 'bulk_density', 'soc']


def main():
    # Reference grid from DEM (already in EPSG:32651)
    with rasterio.open(DEM_REF) as ref:
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_crs = ref.crs

    print(f'Target grid: {dst_height}x{dst_width}, CRS={dst_crs}')

    with rasterio.open(RAW_SOIL) as src:
        print(f'Source: {src.count} bands, {src.shape}, {src.crs}')
        bands = []
        for i in range(1, src.count + 1):
            dst = np.full((dst_height, dst_width), np.nan, dtype=np.float64)
            reproject(
                source=rasterio.band(src, i),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=None,
                dst_nodata=np.nan,
            )
            bands.append(dst)
            valid = ~np.isnan(dst)
            print(f'  band {i} ({BAND_NAMES[i-1]}): valid={valid.sum()}/{dst.size}, '
                  f'min={np.nanmin(dst):.2f}, max={np.nanmax(dst):.2f}')

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': dst_width,
        'height': dst_height,
        'count': len(bands),
        'crs': dst_crs,
        'transform': dst_transform,
        'nodata': np.nan,
        'compress': 'lzw',
    }

    with rasterio.open(OUT, 'w', **profile) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr.astype(np.float32), i)
        dst.descriptions = tuple(BAND_NAMES)

    print(f'\nSaved: {OUT} ({len(bands)} bands)')


if __name__ == '__main__':
    main()
