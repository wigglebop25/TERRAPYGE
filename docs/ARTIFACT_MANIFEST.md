# Artifact Manifest

Key inputs, outputs, and regeneration commands for TERRAPYGE. All large
artifacts are gitignored and regenerated from scripts; this manifest records
the expected shape so results can be checked after a run.

## Pipeline

Two orchestrators, run in order:

```bash
# 1. Geospatial pipeline (raw -> processed; step 4 needs GRASS GIS)
python scripts/run_geospatial_pipeline.py

# 2. Physics-informed pipeline (features -> graph -> ablation -> hazard map)
python scripts/run_physics_pipeline.py
```

`config.yaml` is the single source of truth (paths, CRS, physics parameters,
GNN/training hyperparameters, hazard classes, `inference:`).

## Key artifacts

| Artifact | Location | Expected |
|----------|----------|----------|
| Slope units | `data/processed/buhisan/slope_units.gpkg` | 13,297 polygons, EPSG:32651 |
| SU features | `data/processed/buhisan/su_features.csv` | 13,297 rows, 14 feature columns |
| Soil 6-band | `data/processed/buhisan/soil_6band_utm.tif` | 6 bands |
| Physics features | `data/processed/buhisan/physics_features.csv` | 13,297 rows |
| Physics graph | `data/processed/buhisan/buhisan_hetero_physics.pt` | 13,297 nodes; 21 features; 88,772 spatial + 24,666 hydro edges |
| Hazard map | `data/processed/buhisan/physics_hazard_map.{gpkg,geojson,csv,html}` | 4 classes (Low/Moderate/High/Very High) |
| Spatial masking metrics | `results/spatial_masking.json` | dual-edge GNN retains AUC under masking |
| Physics ablation metrics | `results/physics_ablation.json` | with vs without physics features |
| Curated figures | `docs/figures/*.png` | 7 tracked figures |
| Preferred model | `models/ablation/with_physics__heterogcn__dual_edge.pt` | loads without error |

## Reference results

- Hazard class distribution (4-class, quantile): Low 11,362 / Moderate 1,652 /
  High 239 / Very High 44.
- Spatial masking: dual-edge GNN retains ~0.965 AUC while non-graph baselines
  drop to ~0.32-0.87 (topographic covariates withheld from test nodes).

## Verification

Geospatial artifacts were validated against the reference:
- UTM rasters and SU spatial edges reproduce **exactly** (88,772 spatial edges).
- Soil / landcover / climate / water feature means are **exact**.
- DEM conditioning and slope/curvature are within small tolerances; TWI/SPI
  depend on the specific-catchment-area method.
- Slope-unit extraction (`r.slopeunits`) is stochastic.

Physics pipeline smoke test: `python scripts/run_physics_pipeline.py --only 5`
completes successfully and reproduces the reference class distribution.

Tests: `python -m pytest` (physics unit tests + installation smoke tests).
