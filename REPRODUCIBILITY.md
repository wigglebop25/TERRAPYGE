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

# 4. Run full physics pipeline
python scripts/run_physics_pipeline.py
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
├── config.yaml                # ← Runtime config (physics, gnn, hazard, inference)
├── environment.yml            # ← Conda env
├── scripts/                   # Pipeline entry points + legacy/ (archived)
├── src/terrapyge/             # Package
│   ├── data/                  # Acquisition, graph construction
│   ├── features/              # Physics-informed features + labels
│   ├── models/                # GNN, baselines, experiments
│   ├── utils/                 # Geo I/O, paths, metrics
│   └── visualization/         # Maps and plots
├── tests/                     # Test suite
├── data/
│   ├── raw/                   # GEE downloads (gitignored)
│   └── processed/             # Pipeline outputs (gitignored)
├── models/                    # Trained weights (gitignored; README tracked)
│   └── ablation/              # GNN + baseline checkpoints
├── docs/figures/              # Curated final figures (tracked)
└── results/                   # Metrics, figures (gitignored)
    ├── physics_ablation.json
    ├── spatial_masking.json
    └── figures/               # Plots, charts
```

---

## Pipeline Execution

The maintained pipeline is orchestrated by `scripts/run_physics_pipeline.py`,
which runs five steps in order:

```bash
# Run all steps
python scripts/run_physics_pipeline.py

# Resume from a step (1-5), or run a single step
python scripts/run_physics_pipeline.py --from 3
python scripts/run_physics_pipeline.py --only 2
```

| Step | Script | Output |
|------|--------|--------|
| 1. Reproject 6-band soil | `scripts/reproject_soil_6band.py` | `data/processed/buhisan/soil_6band_utm.tif` |
| 2. Physics features + labels | `scripts/build_physics_features.py` | `physics_features.csv`, `buhisan_hetero_physics.pt` |
| 3. Physics ablation | `scripts/physics_ablation.py` | `results/physics_ablation.json`, `results/figures/physics_ablation.png` |
| 4. Spatial masking | `scripts/spatial_masking_test.py` | `results/spatial_masking.json`, `results/figures/spatial_masking.png` |
| 5. Hazard map | `scripts/generate_physics_hazard_map.py` | `physics_hazard_map.{gpkg,geojson,csv,html}` |

### Geospatial pipeline (raw -> processed)

The physics steps above consume processed artifacts in
`data/processed/buhisan/`. Those are produced by the geospatial pipeline
(`scripts/run_geospatial_pipeline.py`), kept separate because it needs GRASS GIS
and heavier raster tooling:

| Step | Script | Output |
|------|--------|--------|
| 1. DEM export | Manual (Google Earth Engine; see GEE Authentication) | `data/raw/buhisan/dem/Buhisan_DEM_SRTM_30m.tif` |
| 2. Reproject rasters to UTM 51N | `scripts/build_rasters_utm.py` | `data/processed/buhisan/*_utm.tif` |
| 3. DEM conditioning | `scripts/build_dem.py` | `dem_conditioned.tif` |
| 4. Terrain derivatives | `scripts/build_terrain_derivatives.py` | `slope.tif`, `aspect.tif`, `curv_plan.tif`, `curv_profile.tif`, `twi.tif`, `spi.tif` |
| 5. Slope units | `scripts/build_slope_units.py` (GRASS `r.slopeunits`) | `slope_units.gpkg`, `su.tif` |
| 6. Feature aggregation | `scripts/build_su_features.py` | `su_features.csv` |
| 7. Base graph | `scripts/build_spatial_edges.py` + `scripts/compute_hydro_edges.py` | `buhisan_hetero.pt` |

```bash
# Full geospatial pipeline (requires GRASS GIS in PATH)
python scripts/run_geospatial_pipeline.py

# Resume from a step, or run a single step
python scripts/run_geospatial_pipeline.py --from 3
python scripts/run_geospatial_pipeline.py --only 2
```

Notes:
- Step 1 (DEM export) is manual; the GEE script is in `docs/GOOGLE_EARTH_ENGINE.md`.
- Step 5 runs GRASS `r.slopeunits` via `grass84.bat --exec`; parameters come from
  `slope_units:` in `config.yaml`.
- Spatial edges are undirected Queen contiguity (`libpysal`); hydrological edges
  are directed D8 flow (`pysheds`).
- Reconciliation targets: ~13,297 slope units, the `su_features.csv` column set,
  and the spatial + hydro edge modalities. Exact numeric equality is not
  guaranteed across library versions.

**Validation (vs the reference artifacts):** UTM rasters and SU spatial edges
reproduce exactly (0 difference; 88,772 spatial edges; 13,297 units). Soil,
landcover, climate and water feature means are exact. DEM conditioning and
slope/curvature differ only within small tolerances; TWI and SPI differ more
because they depend on the specific-catchment-area method. Slope-unit extraction
with `r.slopeunits` is stochastic and may yield a slightly different SU count.

The physics pipeline (step 2 onwards in the table above) then consumes these
artifacts. Feature indices follow the locked schema in `DATA_SCHEMA.md`
(21 features; index 20 is the label source).

---

## Expected Outputs (Verification)

| File | Validation Check |
|------|------------------|
| `data/processed/buhisan/soil_6band_utm.tif` | 6 bands (clay, sand, silt, pH, bulk density, SOC) |
| `data/processed/buhisan/slope_units.gpkg` | 13,297 slope units |
| `data/processed/buhisan/buhisan_hetero_physics.pt` | 13,297 nodes; 21 features; spatial + hydro edges |
| `data/processed/buhisan/physics_hazard_map.gpkg` | 13,297 features; 4 classes |
| `results/spatial_masking.json` | dual-edge GNN retains AUC under topographic masking |
| `models/ablation/with_physics__heterogcn__dual_edge.pt` | loads without error |

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

All splits (train/val/test), physics-derived labels, and model initialization use this seed.

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
