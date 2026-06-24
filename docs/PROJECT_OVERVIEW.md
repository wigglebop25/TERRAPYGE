# TERRAPYGE Project Overview

## About

**TERRAPYGE** is a specialized deep learning pipeline and Graph Neural Network (GNN) framework designed for advanced landslide vulnerability mapping in Cebu City, Philippines. This project moves beyond traditional Euclidean, grid-based mapping approaches by modeling geographical points of interest as discrete nodes connected via spatial adjacency and hydrological flow pathways.

## Key Innovation

- **Non-Euclidean Approach**: Uses Slope Units (SUs) instead of arbitrary grid cells
- **Dual Edge Modalities**: Combines spatial adjacency (undirected) and hydrological flow (directed)
- **Geometric Deep Learning**: Leverages PyTorch Geometric for message-passing convolutions
- **Real-world Application**: Addresses SDG 11 (Sustainable Cities) and SDG 13 (Climate Action)

## Technical Stack

- **Geospatial**: geopandas, pysheds, libpysal, whitebox, r.slopeunits (GRASS GIS)
- **Deep Learning**: PyTorch 2.12.0, PyTorch Geometric (PyG) 2.8.0
- **Data Processing**: NumPy, Pandas, SciPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn, Folium
- **Experiment Tracking**: Weights & Biases (W&B)
- **Hyperparameter Optimization**: Optuna

## Project Structure

```
TERRAPYGE/
├── data/
│   ├── raw/           # Downloaded DEM, shapefiles
│   ├── processed/     # Conditioned DEM, slope units
│   └── graphs/        # PyG HeteroData graph
├── notebooks/         # Jupyter notebooks (run in order)
├── src/
│   ├── data/          # Data acquisition utilities
│   ├── models/        # GNN models and baselines
│   ├── utils/         # Geospatial and path utilities
│   └── visualization/ # Map plotting
├── models/
│   ├── trained/       # Final trained models
│   └── checkpoints/   # Training checkpoints
├── results/           # QC reports, figures, maps
├── docs/              # Documentation
├── phase1_dem_preprocessing.py    # DEM conditioning pipeline
├── phase1_slope_units.py          # Slope unit extraction (GRASS GIS)
├── phase2_graph_construction.py   # Graph construction pipeline
├── config.yaml                    # Project configuration
└── requirements.txt               # Python dependencies
```

## Study Area

- **Location**: Cebu City, Region VII (Central Visayas), Philippines
- **Watersheds**: Buhisan and Malubog
- **Bounding Box**: 123.8-124.0 E, 10.2-10.4 N (WGS84)
- **CRS**: EPSG:32651 (UTM Zone 51N)
- **DEM**: SRTM 30m from Google Earth Engine

## Current Status

| Phase | Status | Key Output |
|---|---|---|
| Phase 0 - Environment | Complete | PyG 2.8.0, torch_scatter, libpysal, scikit-learn, xgboost, optuna, wandb |
| Phase 1 - DEM Repair | Complete | UTM DEM (mean slope 6.71 degrees), 812 slope units, all QC checks pass |
| Phase 2 - Graph Construction | Complete | HeteroData: 812 nodes, 12 features, dual-edge modalities |
| Phase 3 - Labels | Complete | Synthetic labels (132 positive, 680 negative) |
| Phase 4 - GNN Models | In Progress | Baselines + GNN models + training pipeline |
| Phase 5 - Evaluation | Not Started | Metrics, ablation, hazard maps |

## Timeline

- **Total Duration**: 28 weeks (7 months)
- **Thesis 1 (Proof of Concept)**: Weeks 1-14
- **Thesis 2 (Final Defense)**: Weeks 15-28

## Data Sources

- **Google Earth Engine**: Digital Elevation Model (DEM) - SRTM 30m
- **PHIVOLCS**: Earthquake-induced landslide hazard maps (request pending)
- **Philippine Soil Series**: Soil classification data

## License

This project is for academic research purposes. Data usage must comply with government agency regulations.

## Acknowledgments

- Mines and Geosciences Bureau (MGB) Region VII
- Bureau of Soils and Water Management (BSWM)
- Philippine Institute of Volcanology and Seismology (PHIVOLCS)
- National Mapping and Resource Information Authority (NAMRIA)
- PyTorch Geometric Development Team
