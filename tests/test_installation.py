#!/usr/bin/env python3
"""
Test script to verify TERRAPYGE installation
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed."""
    packages = [
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
    ]
    
    print("Testing package imports...")
    for package in packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError as e:
            print(f"[FAIL] {package}: {e}")
            return False
    
    print("\nAll packages imported successfully!")
    return True

def test_dem_file():
    """Test if DEM file exists."""
    dem_path = Path.cwd() / 'data' / 'raw' / 'dem' / 'CebuCity_DEM.tif'
    
    print(f"\nTesting DEM file: {dem_path}")
    if dem_path.exists():
        print("[OK] DEM file exists")
        
        # Try to open it
        try:
            import rasterio
            with rasterio.open(dem_path) as src:
                print(f"[OK] DEM can be opened")
                print(f"  Shape: {src.shape}")
                print(f"  CRS: {src.crs}")
                print(f"  Resolution: {src.res}")
        except Exception as e:
            print(f"[FAIL] Error opening DEM: {e}")
            return False
        
        return True
    else:
        print(f"[FAIL] DEM file not found: {dem_path}")
        return False

def test_project_structure():
    """Test if project structure is correct."""
    base_path = Path.cwd()
    
    required_dirs = [
        'data/raw/dem',
        'notebooks',
        'src/terrapyge/data',
        'src/terrapyge/models',
        'src/terrapyge/utils',
        'src/terrapyge/visualization',
        'models/trained',
        'models/checkpoints',
        'results/figures',
        'results/maps',
        'results/reports'
    ]
    
    print(f"\nTesting project structure...")
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"[OK] {dir_path}")
        else:
            print(f"[FAIL] {dir_path} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Main test function."""
    print("=" * 60)
    print("TERRAPYGE Installation Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test DEM file
    dem_ok = test_dem_file()
    
    # Test project structure
    structure_ok = test_project_structure()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Package imports: {'[OK] PASS' if imports_ok else '[FAIL] FAIL'}")
    print(f"DEM file: {'[OK] PASS' if dem_ok else '[FAIL] FAIL'}")
    print(f"Project structure: {'[OK] PASS' if structure_ok else '[FAIL] FAIL'}")
    
    if imports_ok and dem_ok and structure_ok:
        print("\n[SUCCESS] ALL TESTS PASSED! Environment is ready.")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
