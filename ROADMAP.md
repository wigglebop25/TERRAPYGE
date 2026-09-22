# TERRAPYGE Roadmap

Physics-informed Graph Neural Network for earthquake-induced landslide
susceptibility mapping using slope units with spatial and hydrological edge
modalities (Buhisan Watershed, Cebu City).

## Status
Core research pipeline: **Complete**. Documentation and repository hygiene:
**In Progress**.

## Completed
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
- Documentation aligned to the approved thesis title; repository cleaned

## In Progress
- Reproducibility and data-schema documentation sync
- Local task tracking and roadmap maintenance

## Next
- Document the raw -> processed path (DEM download and slope-unit extraction)
- Optional: full end-to-end pipeline verification run

## Known Issues
- The raw -> processed geospatial steps rely on external tooling (DEM source and
  GRASS GIS) and are not fully scripted in this repository
- Legacy pre-physics scripts are retained under `scripts/legacy/` for provenance
  only and are superseded

## Milestones
- Physics-informed modeling pipeline: Complete
- Reproducibility and documentation: In Progress
- Thesis write-up and defense: Pending
