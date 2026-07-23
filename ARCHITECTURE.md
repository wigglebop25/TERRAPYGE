# TERRAPYGE System Architecture

## Overview
TERRAPYGE is a geospatial deep learning pipeline for landslide susceptibility mapping in Cebu City, Philippines. It replaces grid-based approaches with **Slope Units (SUs)** as graph nodes and encodes **dual-edge modalities**: spatial adjacency (undirected) + hydrological flow (directed D8).

---

## Pipeline Stages

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  DEM (SRTM) │────▶│  Conditioned │────▶│  Slope Units     │────▶│  Feature Matrix │
│  30m, EPSG: │     │  DEM (UTM)   │     │  (r.slopeunits)  │     │  + Labels       │
│  4326       │     │  30m         │     │  ~13k polygons   │     │  13297 × 15     │
└─────────────┘     └──────────────┘     └──────────────────┘     └────────┬────────┘
                                                                            │
                                                                            ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Hazard Map │◀────│  GNN Inference│◀────│  Trained Model   │◀────│  HeteroGraph    │
│  (5 classes)│     │  (GCN/SAGE)  │     │  (AUC: 0.888)   │     │  (Dual-Edge)    │
└─────────────┘     └──────────────┘     └──────────────────┘     └─────────────────┘
```

---

## Module Structure

```
src/terrapyge/
├── data/
│   ├── acquisition.py      # GEE downloads, GRASS wrappers
│   ├── preprocessing.py    # DEM conditioning (WhiteboxTools)
│   ├── slope_units.py      # GRASS r.slopeunits wrapper
│   ├── features.py         # Zonal stats → feature matrix
│   ├── graph.py            # HeteroData construction
│   └── labels.py           # Synthetic / real label generation
├── models/
│   ├── gnn.py              # HeteroGCN, HeteroSAGE, HeteroGAT
│   ├── baselines.py        # LR, RF, XGBoost
│   └── training.py         # Training loop, early stopping, Optuna
├── utils/
│   ├── geo.py              # CRS, raster/vector I/O
│   ├── viz.py              # Folium maps, matplotlib figures
│   └── metrics.py          # AUC, AP, F1, calibration
└── visualization/
    └── maps.py             # Hazard map generation
```

---

## Data Flow

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| 1. DEM | SRTM 30m (GEE) | Reproject → Fill depressions → Mask offshore | `dem_conditioned.tif` (UTM 51N) |
| 2. Slope Units | Conditioned DEM | GRASS `r.slopeunits` (thresh=5000) | `su.gpkg` (13,297 polygons) |
| 3. Features | 14 rasters + SU | Zonal stats (mean) per SU | Feature matrix (13297 × 15) |
| 4. Graph | SUs + Features | Queen contiguity + D8 flow | `HeteroData` (dual-edge) |
| 5. Labels | Synthetic / PHIVOLCS | Rule-based / point-in-polygon | Binary / 5-class labels |
| 6. GNN | Graph + Labels | GCN/SAGE/GAT + spatial CV | Probabilities per SU |
| 7. Map | Probabilities | Quantile bins → GeoJSON/HTML | 5-class hazard map |

---

## Graph Specification

```
HeteroData(
  node_type: 'su' (Slope Unit)
  num_nodes: 13,297
  x: [13297, 15]           # Standardized features
  y: [13297]               # Labels (0/1 or 0-4)
  
  edge_types:
    ('su', 'spatial', 'su'):  # Undirected Queen contiguity
      edge_index: [2, 88772]
      edge_attr: None
      
    ('su', 'hydro', 'su'):    # Directed D8 flow
      edge_index: [2, E_hydro]  # TODO: implement
      edge_attr: None
)
```

---

## Configuration (config.yaml)

```yaml
crs:
  source: "EPSG:4326"
  target: "EPSG:32651"  # UTM Zone 51N

study_area:
  name: "Buhisan Watershed"
  bbox: [123.86, 10.28, 123.95, 10.36]

slope_units:
  algorithm: "r.slopeunits"
  threshold: 5000
  reduction_factor: 1.0
  min_area: 10000

features:
  topographic: [dem, slope, aspect, curv_plan, curv_profile, twi, spi]
  soil: [clay, sand, silt, ph, bd, soc]
  landcover: [worldcover, dynamicworld]
  climate: [chirps, worldclim, era5]
  hydrology: [jrc_water]

gnn:
  hidden_channels: 64
  num_layers: 3
  dropout: 0.2
  lr: 0.001
  epochs: 100
  patience: 15
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Slope Units vs Grid | Geomorphologically meaningful; reduces MAUP; better hydrology |
| Dual-edge graph | Captures both spatial autocorrelation AND hydrological connectivity |
| HeteroData (PyG) | Native support for multi-edge-type message passing |
| Spatial CV | Prevents leakage from spatial autocorrelation |
| Synthetic labels | Enables pipeline development before PHIVOLCS data arrives |
| UTM 51N projection | Metric units for accurate area/distance; standard for Philippines |

---

## Extensibility Points

1. **Hydrological edges** — Add D8 downstream edges (`r.stream.extract` → vector → graph)
2. **Real labels** — Swap synthetic `y` with PHIVOLCS point-in-polygon labels
3. **Multi-class** — Extend to 5-class susceptibility (Very Low → Very High)
4. **Attention** — Replace GCN with HeteroGAT for interpretable edge weights
5. **Temporal** — Add time-series rainfall as dynamic node features
