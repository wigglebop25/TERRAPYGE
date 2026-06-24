#!/usr/bin/env python3
"""
TERRAPYGE Project Setup Script
Creates the complete project directory structure and initial files.
"""

import os
import sys
from pathlib import Path

def create_directory_structure(base_path):
    """Create the complete project directory structure."""
    
    directories = [
        # Data directories
        'data/raw',
        'data/processed',
        'data/graphs',
        
        # Notebooks
        'notebooks',
        
        # Source code
        'src/data',
        'src/models',
        'src/utils',
        'src/visualization',
        
        # Models and results
        'models/trained',
        'models/checkpoints',
        'results/figures',
        'results/maps',
        'results/reports',
        
        # Documentation
        'docs',
        
        # Tests
        'tests',
    ]
    
    print("Creating project directory structure...")
    for directory in directories:
        path = base_path / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created: {directory}")
    
    print("\nDirectory structure created successfully!")

def create_initial_files(base_path):
    """Create initial project files."""
    
    files_to_create = {
        # Python files
        'src/__init__.py': '',
        'src/data/__init__.py': '',
        'src/models/__init__.py': '',
        'src/utils/__init__.py': '',
        'src/visualization/__init__.py': '',
        
        # Configuration
        'config.yaml': """# TERRAPYGE Configuration File

# Project paths
paths:
  data_raw: data/raw
  data_processed: data/processed
  data_graphs: data/graphs
  models: models
  results: results

# Coordinate Reference System
crs:
  source: EPSG:4326  # WGS84
  target: EPSG:32651  # UTM Zone 51N

# Study area
study_area:
  name: Cebu City
  watershed: Buhisan  # or Malubog
  bbox: [123.8, 10.2, 124.0, 10.4]  # [min_lon, min_lat, max_lon, max_lat]

# Slope unit parameters
slope_units:
  flow_accumulation_threshold: 10000  # square meters
  reduction_factor: 2
  min_area: 300000  # square meters
  min_circular_variance: 0.1

# GNN parameters
gnn:
  hidden_channels: 64
  num_layers: 3
  dropout: 0.1
  learning_rate: 0.001
  epochs: 100
  batch_size: 32

# Training parameters
training:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  early_stopping_patience: 10
""",
        
        # Requirements
        'requirements.txt': """# Geospatial Processing
geopandas>=0.12.0
pysheds>=0.3.0
libpysal>=4.6.0
whitebox>=2.3.0
rasterio>=1.3.0
shapely>=2.0.0
pyproj>=3.4.0

# Deep Learning
torch>=1.13.0
torch-geometric>=2.2.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0
torch-cluster>=1.6.0

# Data Processing
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
folium>=0.13.0
plotly>=5.11.0

# Development
jupyter>=1.0.0
black>=22.0.0
isort>=5.10.0
pytest>=7.0.0
""",
        
        # Environment YAML
        'environment.yml': """name: terrapyge
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - geopandas
  - pysheds
  - libpysal
  - whitebox
  - rasterio
  - shapely
  - pyproj
  - numpy
  - pandas
  - scipy
  - matplotlib
  - seaborn
  - jupyter
  - pip
  - pip:
    - torch
    - torch-geometric
    - torch-scatter
    - torch-sparse
    - torch-cluster
    - folium
    - plotly
""",
        
        # Git ignore
        '.gitignore': """# TERRAPYGE .gitignore

# Data files (too large for Git)
data/raw/
data/processed/
data/graphs/

# Model files
models/trained/
models/checkpoints/

# Results
results/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
""",
        
        # README for notebooks
        'notebooks/README.md': """# Jupyter Notebooks

This directory contains all Jupyter notebooks for the TERRAPYGE project.

## Notebook Order

1. **01_data_acquisition.ipynb** - Data acquisition and initial setup
2. **02_data_preprocessing.ipynb** - Data cleaning and preprocessing
3. **03_slope_units_extraction.ipynb** - Slope unit extraction from DEM
4. **04_graph_construction.ipynb** - Graph construction and feature engineering
5. **05_gnn_modeling.ipynb** - GNN model training and evaluation
6. **06_evaluation.ipynb** - Model evaluation and comparison
7. **07_visualization.ipynb** - Results visualization and mapping

## Running Notebooks

Start with `01_data_acquisition.ipynb` and proceed in numerical order.

## Data Dependencies

- Government data from MGB and BSWM (eFOI requests)
- DEM from Geoportal Philippines
- Philippine Soil Series (already available)
""",
        
        # Source code templates
        'src/data/acquisition.py': '''"""Data acquisition utilities for TERRAPYGE."""

import geopandas as gpd
import rasterio
from pathlib import Path

def load_dem(dem_path):
    """Load Digital Elevation Model."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
    return dem, profile

def load_shapefile(shp_path):
    """Load shapefile with proper CRS handling."""
    gdf = gpd.read_file(shp_path)
    return gdf

def validate_crs(gdf, target_crs='EPSG:32651'):
    """Validate and reproject if needed."""
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf
''',
        
        'src/utils/geo_utils.py': '''"""Geospatial utilities for TERRAPYGE."""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def calculate_centroid(gdf):
    """Calculate centroid for each geometry."""
    return gdf.geometry.centroid

def spatial_join(target_gdf, source_gdf, how='inner'):
    """Perform spatial join between GeoDataFrames."""
    return gpd.sjoin(target_gdf, source_gdf, how=how, op='intersects')

def normalize_features(features):
    """Normalize features to [0, 1] range."""
    return (features - features.min()) / (features.max() - features.min())
''',
        
        'src/visualization/maps.py': '''"""Map visualization utilities."""

import folium
import geopandas as gpd
import matplotlib.pyplot as plt

def create_folium_map(gdf, center=None):
    """Create interactive Folium map."""
    if center is None:
        center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=12)
    return m

def plot_slope_units(gdf, ax=None):
    """Plot slope units."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor='black', linewidth=0.5)
    return ax
''',
    }
    
    print("\nCreating initial files...")
    for file_path, content in files_to_create.items():
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if full_path.exists():
            print(f"[SKIP] Already exists: {file_path}")
            continue

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"[OK] Created: {file_path}")
    
    print("\nInitial files created successfully!")

def main():
    """Main setup function."""
    
    # Get base path
    base_path = Path(__file__).parent
    
    print("=" * 60)
    print("TERRAPYGE Project Setup")
    print("=" * 60)
    print(f"Project location: {base_path}")
    print()
    
    # Create directory structure
    create_directory_structure(base_path)
    
    # Create initial files
    create_initial_files(base_path)
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Activate environment: conda activate terrapyge")
    print("3. Start Jupyter: jupyter notebook")
    print("4. Begin with notebooks/01_data_acquisition.ipynb")
    print("\nGood luck with your thesis! [OK]")

if __name__ == "__main__":
    main()