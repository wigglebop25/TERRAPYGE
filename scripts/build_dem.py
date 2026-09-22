"""Condition the UTM DEM for hydrological routing.

Fills depressions and pits with WhiteboxTools so D8 flow routing is defined
everywhere. Reads `dem_utm.tif`, writes `dem_conditioned.tif` (EPSG:32651).

The processed DEM raster stores nodata as -32768 (matching the rest of the
terrain stack) rather than NaN, since WhiteboxTools requires a numeric nodata.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from whitebox import WhiteboxTools

from src.terrapyge.utils.paths import PROCESSED_BUHISAN as PROC_DIR

DEM = PROC_DIR / 'dem_utm.tif'
OUT = PROC_DIR / 'dem_conditioned.tif'
NODATA = -32768.0


def main():
    print('=' * 70)
    print('TERRAPYGE: CONDITION DEM')
    print('=' * 70)

    with rasterio.open(DEM) as src:
        arr = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    arr = np.where(np.isnan(arr), NODATA, arr)
    profile.update(dtype='float64', nodata=NODATA, compress='lzw')

    tmp_in = PROC_DIR / 'dem_utm_nodata.tif'
    with rasterio.open(tmp_in, 'w', **profile) as dst:
        dst.write(arr, 1)

    wbt = WhiteboxTools()
    wbt.verbose = False
    wbt.fill_depressions(str(tmp_in), str(OUT))
    tmp_in.unlink(missing_ok=True)

    with rasterio.open(OUT) as src:
        cond = src.read(1)
        profile = src.profile.copy()
    profile.update(dtype='float64', nodata=NODATA, compress='lzw')
    valid = cond != NODATA
    print(f'dem_conditioned.tif  shape={cond.shape} valid={valid.sum()} '
          f'min={cond[valid].min():.1f} max={cond[valid].max():.1f}')
    print('\nDONE')


if __name__ == '__main__':
    main()
