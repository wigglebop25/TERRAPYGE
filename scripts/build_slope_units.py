"""Extract slope units with GRASS GIS `r.slopeunits`.

This script is designed to run INSIDE a GRASS session, e.g.:

    grass84.bat --text --exec python scripts/build_slope_units.py

It imports the conditioned DEM, runs `r.slopeunits` using the parameters under
`slope_units:` in config.yaml, vectorizes the result, and exports:

    data/processed/buhisan/su.tif            (uint16)
    data/processed/buhisan/slope_units.gpkg

If GRASS is not available the step is skipped; the other geospatial steps run
independently.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.terrapyge.utils.paths import PROCESSED_BUHISAN as PROC_DIR, ROOT

DEM = PROC_DIR / 'dem_conditioned.tif'
SU_TIF = PROC_DIR / 'su.tif'
SU_GPKG = PROC_DIR / 'slope_units.gpkg'


def main():
    print('=' * 70)
    print('TERRAPYGE: SLOPE UNITS (GRASS r.slopeunits)')
    print('=' * 70)

    try:
        import grass.script as gs
    except ImportError:
        print('ERROR: grass.script not found. Run this script inside GRASS:')
        print('  grass84.bat --text --exec python scripts/build_slope_units.py')
        sys.exit(1)

    with open(ROOT / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    su_cfg = cfg.get('slope_units', {})

    gs.run_command('g.region', raster='dem_conditioned', flags='a')
    gs.run_command('r.in.gdal', input=str(DEM), output='dem_conditioned', overwrite=True)
    gs.run_command('g.region', raster='dem_conditioned')

    # r.slopeunits: threshold = initial polygon size, minsize = minimum SU area.
    gs.run_command(
        'r.slopeunits',
        raster='dem_conditioned',
        output='su',
        threshold=int(su_cfg.get('threshold', 5000)),
        minsize=int(su_cfg.get('min_area', 10000)),
        reduction=float(su_cfg.get('reduction_factor', 1.0)),
        maxiteration=int(su_cfg.get('max_iterations', 100)),
        overwrite=True,
    )

    gs.run_command('r.to.vect', input='su', output='su_vect', type='area', overwrite=True)
    gs.run_command('v.out.ogr', input='su_vect', output=str(SU_GPKG),
                   format='GPKG', overwrite=True)
    gs.run_command('r.out.gdal', input='su', output=str(SU_TIF),
                   format='GTiff', type='UInt16', nodata=65535, overwrite=True)

    print(f'Exported: {SU_TIF}')
    print(f'Exported: {SU_GPKG}')
    print('\nDONE')


if __name__ == '__main__':
    main()
