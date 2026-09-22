"""Smoke tests for the TERRAPYGE installation and repository layout.

Core packages and the source layout are asserted strictly. Data artifacts
(``data/`` is gitignored) are treated as optional and skipped when absent.
"""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = [
    'rasterio',
    'geopandas',
    'matplotlib',
    'numpy',
    'pandas',
    'scipy',
    'shapely',
    'pyproj',
    'torch',
    'torch_geometric',
    'torch_scatter',
    'torch_sparse',
    'whitebox',
    'pysheds',
    'libpysal',
    'sklearn',
    'xgboost',
    'optuna',
    'networkx',
    'folium',
    'yaml',
]

REQUIRED_SRC_DIRS = [
    'src/terrapyge/data',
    'src/terrapyge/features',
    'src/terrapyge/models',
    'src/terrapyge/utils',
    'src/terrapyge/visualization',
]

REQUIRED_FILES = [
    'config.yaml',
    'pyproject.toml',
    'scripts/run_physics_pipeline.py',
]

# Optional, gitignored artifacts - skipped when not present.
OPTIONAL_DATA = [
    'data/processed/buhisan/buhisan_hetero_physics.pt',
]


def test_imports():
    """All required packages must be importable."""
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    assert not missing, f"missing packages: {missing}"


def test_project_structure():
    """The source layout and key files must exist."""
    missing = [d for d in REQUIRED_SRC_DIRS if not (ROOT / d).is_dir()]
    missing += [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
    assert not missing, f"missing paths: {missing}"


@pytest.mark.parametrize('rel_path', OPTIONAL_DATA)
def test_optional_data_present(rel_path):
    """Optional data artifacts are validated only when present."""
    path = ROOT / rel_path
    if not path.exists():
        pytest.skip(f"optional data not present: {rel_path}")
    assert path.stat().st_size > 0, f"empty data file: {rel_path}"


def main():
    """Manual runner: print a PASS/FAIL summary and return an exit code."""
    print("=" * 60)
    print("TERRAPYGE Installation Test")
    print("=" * 60)

    failures = []
    for name, fn in [
        ('imports', test_imports),
        ('project structure', test_project_structure),
    ]:
        try:
            fn()
            print(f"[OK] {name}")
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}")
            failures.append(name)

    if failures:
        print(f"\n[WARNING] some checks failed: {failures}")
        return 1
    print("\n[SUCCESS] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
