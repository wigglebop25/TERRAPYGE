"""Derive terrain attributes from the conditioned DEM.

Uses WhiteboxTools on `dem_conditioned.tif` to produce the topographic stack
consumed by the feature aggregation step:

  slope.tif          slope (degrees)
  aspect.tif         aspect (degrees)
  curv_plan.tif      plan curvature
  curv_profile.tif   profile curvature
  twi.tif            topographic wetness index  (from SCA + slope)
  spi.tif            stream power index         (from SCA + slope)

TWI/SPI require specific catchment area (D8 flow accumulation in SCA units),
so an intermediate `sca.tif` is produced and removed afterwards.

All outputs are EPSG:32651, float32, nodata -32768.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rasterio
import numpy as np
from whitebox import WhiteboxTools

from src.terrapyge.utils.paths import PROCESSED_BUHISAN as PROC_DIR

DEM = PROC_DIR / 'dem_conditioned.tif'
SLOPE = PROC_DIR / 'slope.tif'
SCA = PROC_DIR / 'sca.tif'
NODATA = -32768.0


def _normalize(path):
    """Coerce a WhiteboxTools float64 output to float32 with nodata -32768."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    profile.update(dtype='float32', nodata=NODATA, compress='lzw')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)


def main():
    print('=' * 70)
    print('TERRAPYGE: TERRAIN DERIVATIVES')
    print('=' * 70)

    wbt = WhiteboxTools()
    wbt.verbose = False

    # Direct DEM derivatives
    wbt.slope(str(DEM), str(SLOPE), units='degrees')
    wbt.aspect(str(DEM), str(PROC_DIR / 'aspect.tif'))
    wbt.plan_curvature(str(DEM), str(PROC_DIR / 'curv_plan.tif'))
    wbt.profile_curvature(str(DEM), str(PROC_DIR / 'curv_profile.tif'))

    # Specific catchment area (SCA) for TWI/SPI
    wbt.d8_flow_accumulation(str(DEM), str(SCA), out_type='sca')
    wbt.wetness_index(str(SCA), str(SLOPE), str(PROC_DIR / 'twi.tif'))
    wbt.stream_power_index(str(SCA), str(SLOPE), str(PROC_DIR / 'spi.tif'))
    SCA.unlink(missing_ok=True)

    for name in ['slope.tif', 'aspect.tif', 'curv_plan.tif', 'curv_profile.tif',
                 'twi.tif', 'spi.tif']:
        _normalize(PROC_DIR / name)
        print(f'{name} written')

    print('\nDONE')


if __name__ == '__main__':
    main()
