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
| Spatial/block CV metrics | `results/spatial_cv.json` | random vs blocked folds, transductive vs inductive |
| Physics ablation metrics | `results/physics_ablation.json` | with vs without physics features |
| Curated figures | `docs/figures/*.png` | 7 tracked figures |
| Preferred model | `models/ablation/with_physics__heterogcn__dual_edge.pt` | loads without error |

## Reference results

- Hazard class distribution (4-class, quantile): Low 11,362 / Moderate 1,652 /
  High 239 / Very High 44.
- Spatial masking: dual-edge GNN retains ~0.965 AUC while non-graph baselines
  drop to ~0.32-0.87 (topographic covariates withheld from test nodes).
- Spatial/block CV (5 folds, 25 blocks of ~1,974 m, dual-edge HeteroGCN), test AUC:

  | Feature mode | Protocol | GNN | Best baseline |
  |--------------|----------|-----|---------------|
  | full | random transductive | 0.978 | 1.000 |
  | full | block inductive | 0.973 | 1.000 |
  | masked | random transductive | 0.960 | 0.651 |
  | masked | block transductive | 0.937 | 0.725 |
  | masked | block inductive | 0.873 | 0.725 |

  In `full` mode every baseline reaches AUC 1.000 under every protocol: the
  label is a deterministic function of slope, so spatial blocking finds no
  leakage to remove. Under topographic masking the graph advantage is real and
  survives the most conservative protocol (inductive, spatially blocked).

- Threshold and F1. F1 is reported at a validation-selected, prevalence-matched
  threshold rather than a fixed 0.5, because the positive rate is 14.55% and a
  0.5 cut predicts almost no positives. Under the masked inductive protocol the
  dual-edge GNN reaches F1 0.535 (precision 0.686, recall 0.442) against 0.394
  for the best baseline (XGBoost, precision 0.333, recall 0.582). Random forest
  is the degenerate case: its probabilities collapse toward zero, so
  prevalence-matching flags every unit and reports recall 1.000 at precision
  0.145.

  | Masked protocol | GNN F1 | GNN P | GNN R | Best baseline F1 |
  |-----------------|--------|-------|-------|------------------|
  | random transductive | 0.747 | 0.704 | 0.797 | 0.310 |
  | block transductive | 0.658 | 0.739 | 0.606 | 0.394 |
  | block inductive | 0.535 | 0.686 | 0.442 | 0.394 |

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
