# TERRAPYGE Roadmap

Physics-informed Graph Neural Network for earthquake-induced landslide
susceptibility mapping using slope units with spatial and hydrological edge
modalities (Buhisan Watershed, Cebu City).

## Status
Core research pipeline: **Complete**. Documentation and repository hygiene:
**Complete**. Remaining work is the thesis write-up and defense.

## Completed
- Scripted raw -> processed geospatial pipeline (`run_geospatial_pipeline.py`)
- License, citation metadata, and artifact manifest (`docs/ARTIFACT_MANIFEST.md`)
- Bootstrap 95% CIs and paired significance tests (`scripts/bootstrap_ci.py`, `scripts/significance_tests.py`)
- Soil 6-band reprojection to the UTM grid
- Slope units (13,297) and node-feature aggregation
- Physics-informed features and labels (infinite-slope FS, Arias intensity,
  Newmark displacement)
- Heterogeneous graph: spatial adjacency + directed hydrological edges
- GNN backbones (GCN, SAGE, GAT) across spatial-only / hydro-only / dual-edge
- Non-graph baselines (Logistic Regression, Random Forest, XGBoost)
- Physics ablation (with vs without physics features)
- Spatial masking experiment (graph value under topographic masking)
- Config-driven 4-class hazard map + interactive HTML map
- Physics unit tests and installation smoke tests
- Spatial/block cross-validation (Stage B4): random vs blocked folds, transductive
  vs inductive training (`scripts/spatial_cv.py`)
- Reproducibility and data-schema documentation sync (`REPRODUCIBILITY.md`,
  `DATA_SCHEMA.md`, `docs/ARTIFACT_MANIFEST.md`)
- Documentation aligned to the approved thesis title; repository cleaned

## In Progress
- Local task tracking and roadmap maintenance

## Next
- Optional: full end-to-end pipeline verification run
- Optional: extend Stage B4 to all three GNN backbones and a buffer-distance sweep

## Known Issues
- Full-feature AUC is ~1.0 for every model because the physics label is a
  deterministic function of slope, which is itself an input feature. Spatial
  blocking does not change this, so only the topographic-masked protocols in
  `results/spatial_cv.json` are informative about graph value.
- Raw -> processed geospatial steps are now scripted (`run_geospatial_pipeline.py`);
  slope-unit extraction still depends on GRASS and is stochastic
- TWI/SPI and DEM conditioning reproduce the documented method within tolerance,
  not bit-exact
- Legacy pre-physics scripts are retained under `scripts/legacy/` for provenance
  only and are superseded

## Milestones
- Physics-informed modeling pipeline: Complete
- Reproducibility and documentation: Complete
- Thesis write-up and defense: Pending
