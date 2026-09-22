"""TERRAPYGE geospatial pipeline orchestrator (raw -> processed).

Produces the processed artifacts consumed by `run_physics_pipeline.py`:

  1. Reproject raw rasters to UTM 51N        (build_rasters_utm.py)
  2. Condition the DEM                        (build_dem.py)
  3. Terrain derivatives                      (build_terrain_derivatives.py)
  4. Slope units via GRASS r.slopeunits       (build_slope_units.py)  [GRASS]
  5. Aggregate features to slope units        (build_su_features.py)
  6. Spatial adjacency edges                  (build_spatial_edges.py)
  7. Hydrological (D8) edges                  (compute_hydro_edges.py)
  8. Reproject 6-band soil                    (reproject_soil_6band.py)

Step 4 requires GRASS GIS and is invoked through `grass84.bat --exec`. Set
`GRASS_BAT` to override the default path.

Usage:
    python scripts/run_geospatial_pipeline.py            # all steps
    python scripts/run_geospatial_pipeline.py --from 3   # start at step 3
    python scripts/run_geospatial_pipeline.py --only 2   # a single step
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
GRASS_BAT = os.environ.get('GRASS_BAT', r'C:\OSGeo4W\bin\grass84.bat')

# (label, script, requires_grass)
STEPS = [
    ('1. Reproject rasters to UTM', 'build_rasters_utm.py', False),
    ('2. Condition DEM', 'build_dem.py', False),
    ('3. Terrain derivatives', 'build_terrain_derivatives.py', False),
    ('4. Slope units (GRASS r.slopeunits)', 'build_slope_units.py', True),
    ('5. Aggregate SU features', 'build_su_features.py', False),
    ('6. Spatial adjacency edges', 'build_spatial_edges.py', False),
    ('7. Hydrological (D8) edges', 'compute_hydro_edges.py', False),
    ('8. Reproject 6-band soil', 'reproject_soil_6band.py', False),
]


def run_step(label, script, requires_grass):
    path = SCRIPTS / script
    print('\n' + '#' * 70)
    print(f'# {label}')
    print(f'# {path.name}')
    print('#' * 70, flush=True)

    if requires_grass:
        if not Path(GRASS_BAT).exists():
            print(f'[SKIP] GRASS not found at {GRASS_BAT}; set GRASS_BAT to run this step.')
            return
        cmd = [GRASS_BAT, '--text', '--exec', sys.executable, str(path)]
    else:
        cmd = [sys.executable, str(path)]

    t0 = time.time()
    result = subprocess.run(cmd)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f'\n[FAILED] {label} (exit {result.returncode}) after {dt:.1f}s')
        sys.exit(result.returncode)
    print(f'\n[OK] {label} ({dt:.1f}s)')


def main():
    parser = argparse.ArgumentParser(description='TERRAPYGE geospatial pipeline')
    parser.add_argument('--from', dest='start', type=int, default=1,
                        help='start at this step number (1-8)')
    parser.add_argument('--only', dest='only', type=int, default=None,
                        help='run only this step number (1-8)')
    args = parser.parse_args()

    if args.only is not None:
        selected = [STEPS[args.only - 1]]
    else:
        selected = STEPS[args.start - 1:]

    print('TERRAPYGE GEOSPATIAL PIPELINE')
    print(f'Steps to run: {[s[0] for s in selected]}')

    t0 = time.time()
    for label, script, requires_grass in selected:
        run_step(label, script, requires_grass)

    print('\n' + '=' * 70)
    print(f'PIPELINE COMPLETE ({time.time() - t0:.1f}s total)')
    print('=' * 70)


if __name__ == '__main__':
    main()
