"""TERRAPYGE physics-informed pipeline orchestrator.

Runs the full reproducible physics pipeline in order:

  1. Reproject 6-band soil to the UTM grid           (reproject_soil_6band.py)
  2. Build physics features + physics labels          (build_physics_features.py)
  3. Physics ablation: with vs without physics        (physics_ablation.py)
  4. Spatial masking experiment (graph value)          (spatial_masking_test.py)
  5. Generate physics-informed hazard map              (generate_physics_hazard_map.py)

Usage:
    python scripts/run_physics_pipeline.py            # run all steps
    python scripts/run_physics_pipeline.py --from 3   # start at step 3
    python scripts/run_physics_pipeline.py --only 2   # run a single step
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

STEPS = [
    ('1. Reproject 6-band soil', 'reproject_soil_6band.py'),
    ('2. Build physics features + labels', 'build_physics_features.py'),
    ('3. Physics ablation (with vs without)', 'physics_ablation.py'),
    ('4. Spatial masking experiment', 'spatial_masking_test.py'),
    ('5. Physics-informed hazard map', 'generate_physics_hazard_map.py'),
]


def run_step(label, script):
    path = SCRIPTS / script
    print('\n' + '#' * 70)
    print(f'# {label}')
    print(f'# {path.name}')
    print('#' * 70, flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(path)])
    dt = time.time() - t0
    if result.returncode != 0:
        print(f'\n[FAILED] {label} (exit {result.returncode}) after {dt:.1f}s')
        sys.exit(result.returncode)
    print(f'\n[OK] {label} ({dt:.1f}s)')


def main():
    parser = argparse.ArgumentParser(description='TERRAPYGE physics pipeline')
    parser.add_argument('--from', dest='start', type=int, default=1,
                        help='start at this step number (1-5)')
    parser.add_argument('--only', dest='only', type=int, default=None,
                        help='run only this step number (1-5)')
    args = parser.parse_args()

    if args.only is not None:
        selected = [STEPS[args.only - 1]]
    else:
        selected = STEPS[args.start - 1:]

    print('TERRAPYGE PHYSICS PIPELINE')
    print(f'Steps to run: {[s[0] for s in selected]}')

    t0 = time.time()
    for label, script in selected:
        run_step(label, script)

    print('\n' + '=' * 70)
    print(f'PIPELINE COMPLETE ({time.time() - t0:.1f}s total)')
    print('=' * 70)


if __name__ == '__main__':
    main()
