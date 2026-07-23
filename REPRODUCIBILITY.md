# TERRAPYGE Reproducibility Guide

## Quick Start (One-Command)

```bash
# 1. Clone & environment
git clone <repo> TERRAPYGE
cd TERRAPYGE
conda env create -f environment.yml
conda activate terrapyge

# 2. System deps (Windows via OSGeo4W)
# - GRASS GIS 8.4+
# - Add to PATH: C:\OSGeo4W\bin

# 3. Authenticate GEE (once)
earthengine authenticate

# 4. Run full pipeline
bash scripts/run_pipeline.sh
```

---

## Prerequisites

### System Dependencies

| Tool | Version | Windows Install |
|------|---------|-----------------|
| **GRASS GIS** | 8.4+ | OSGeo4W → `grass84` |
| **GDAL** | 3.7+ | Included in conda |
| **Python** | 3.10 | Conda |
| **Git** | — | Git for Windows |

### Python Environment

```bash
conda env create -f environment.yml
conda activate terrapyge

# Verify
python -c "import torch, torch_geometric, rasterio, geopandas, whitebox, pysheds; print('OK')"
```

### GRASS Addon (Required for Slope Units)

```bash
# Run inside GRASS session or via grass84.bat
g.extension extension=r.slopeunits
```

---

## GEE Authentication

```bash
# One-time setup
earthengine authenticate
# Follow browser flow → token saved to ~/.config/earthengine/credentials

# Verify
earthengine ls
```

---

## Directory Structure (Before Running)

```
TERRAPYGE/
├── pyproject.toml             # ← Single source of truth for deps
├── config.yaml                # ← Runtime config
├── environment.yml            # ← Conda env
├── scripts/                   # Pipeline entry points
├── src/terrapyge/             # Package
│   ├── data/                  # Phases 1-4
│   ├── models/                # GNN + baselines
│   ├── utils/                 # Geo I/O, metrics
│   └── visualization/         # Hazard map
├── tests/                     # Test suite
├── data/
│   ├── raw/                   # GEE downloads (gitignored)
│   └── processed/             # Pipeline outputs (gitignored)
├── models/                    # Trained weights (gitignored)
│   ├── gnn_dual_edge.pt       # Dual-edge GNN model
│   └── baselines/             # Baseline models (.pkl)
└── results/                   # Metrics, maps (gitignored)
    ├── baseline_comparison.json  # LR, RF, XGBoost vs GNN metrics
    ├── model_comparison.json     # spatial-only vs dual-edge GNN
    ├── figures/                  # Plots, charts
    └── maps/                     # Hazard maps
```

---

## Pipeline Execution

### Phase 1: DEM Preprocessing (~3 min)

```bash
python -m scripts.phase1_dem_preprocessing
```

**Input**: `data/raw/buhisan/dem/Buhisan_DEM_SRTM_30m.tif`  
**Output**: `data/processed/buhisan/`
- `dem_utm.tif` — Reprojected to UTM 51N
- `dem_conditioned.tif` — Whitebox fill depressions + pits
- `slope.tif`, `aspect.tif` — Derivatives
- `curv_plan.tif`, `curv_profile.tif`
- `twi.tif`, `spi.tif` — Hydrological indices

### Phase 2: Slope Units (~5 min)

```bash
# Option A: Automated (requires GRASS in PATH)
python -m scripts.phase1_slope_units

# Option B: Manual GRASS
bash scripts/run_grass_slopeunits.sh
```

**Output**: `data/processed/buhisan/`
- `slope_units.gpkg` — 13,297 slope unit polygons
- `su.tif` — Raster (13,297 unique IDs)

### Phase 3: Feature Aggregation (~8 min)

```bash
python -m src.terrapyge.data.features
```

**Output**: `data/processed/buhisan/`
- `su_features.csv` — Feature matrix (13297 × 15)
- Updated `slope_units.gpkg` with feature columns

### Phase 4: Graph Construction (~2 min)

```bash
python -m src.terrapyge.data.graph
```

**Output**: `data/processed/buhisan/buhisan_hetero.pt` (HeteroData)

### Phase 5: GNN Training (~5 min)

```bash
python -m src.terrapyge.models.training
```

**Output**: `data/processed/buhisan/`
- `best_model.pt` — State dict
- `metrics.json` — AUC, AP, F1, loss curves

### Phase 6: Hazard Map (~1 min)

```bash
python -m src.terrapyge.visualization.maps
```

**Output**: `data/processed/buhisan/`
- `buhisan_hazard_map.gpkg` — Vector with 5-class hazard
- `buhisan_hazard_map.geojson` — Web-ready
- `buhisan_hazard_map.html` — Interactive Folium map

---

## Expected Outputs (Verification)

| File | Size | Validation Check |
|------|------|------------------|
| `data/processed/buhisan/dem_conditioned.tif` | ~800 KB | `np.nanmin() >= 0` |
| `data/processed/buhisan/su.tif` | ~600 KB | `len(np.unique()) == 13297` |
| `data/processed/buhisan/slope_units.gpkg` | ~4.5 MB | `len(gdf) == 13297` |
| `data/processed/buhisan/buhisan_hetero.pt` | ~2 MB | `data['su'].num_nodes == 13297` |
| `data/processed/buhisan/best_model.pt` | ~500 KB | Loads without error |
| `data/processed/buhisan/buhisan_hazard_map.gpkg` | ~4.5 MB | 13,297 features, 5 classes |

---

## Random Seeds (Fixed)

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

All splits (train/val/test), synthetic labels, and model initialization use this seed.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| GPU | 4 GB VRAM | 8+ GB VRAM (CUDA) |
| Disk | 10 GB free | 20 GB free |
| CPU | 4 cores | 8+ cores |

---

## Common Issues & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `OSError: GRASS not found` | GRASS not in PATH | Add `C:\OSGeo4W\bin` to PATH (Windows) |
| `ModuleNotFoundError: grass.script` | Python can't find GRASS | Run via `grass84.bat --text --exec python script.py` |
| `torch.load` UnpicklingError | PyTorch 2.6+ weights_only | Use `torch.load(..., weights_only=False)` |
| `rasterstats` MemoryError | Too many polygons | Process in chunks or use `all_touched=False` |
| GEE export fails | Quota exceeded | Wait 24h or request quota increase |
| `r.slopeunits` not found | Addon not installed | `g.extension extension=r.slopeunits` |

---

## Exact Version Pinning

| Package | Version |
|---------|---------|
| torch | 2.1.0 |
| torch-geometric | 2.4.0 |
| torch-scatter | 2.1.2 |
| torch-sparse | 0.6.18 |
| rasterio | 1.4.3 |
| geopandas | 0.14.4 |
| whitebox | 2.3.0 |
| pysheds | 0.5.0 |
| rasterstats | 0.21.0 |
| scikit-learn | 1.3.2 |
| xgboost | 2.0.3 |
| optuna | 3.4.0 |
| folium | 0.20.0 |
| earthengine-api | 0.1.374 |

---

## Contact & Maintenance

- **Last Verified**: 2026-07-20
- **Environment**: Windows 11 + OSGeo4W GRASS 8.4.2 + Conda env (Python 3.10)
- **All Phases**: Execute successfully end-to-end

---

## License

MIT License — See `LICENSE` file for details.
