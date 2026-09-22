# Model Artifacts

This directory holds trained model binaries. They are **gitignored** (large,
regenerable) and are intentionally not tracked. This README is the only tracked
file under `models/`.

## Regenerating

All models are deterministic given the seed in `config.yaml` (`gnn.seed: 42`,
`training.seed: 42`).

```bash
# Full physics pipeline (features -> graph -> ablation -> hazard map)
python scripts/run_physics_pipeline.py

# Physics ablation (with/without physics features)
python scripts/physics_ablation.py

# Spatial masking experiment (mask10 / mask20 / mask30)
python scripts/spatial_masking_test.py
```

## Layout

| Path pattern | Family | Notes |
|--------------|--------|-------|
| `gnn_dual_edge.pt` | legacy single model | superseded by `ablation/` |
| `ablation/lr.pkl`, `rf.pkl`, `xgb.pkl` | non-graph baselines | sklearn / xgboost |
| `ablation/without_physics__<backbone>__<edges>.pt` | physics ablation | no physics features |
| `ablation/with_physics__<backbone>__<edges>.pt` | physics ablation | with physics features |
| `ablation/mask10__<backbone>__<edges>.pt` | spatial masking | 10% topographic feature masking |
| `ablation/mask20__<backbone>__<edges>.pt` | spatial masking | 20% masking |
| `ablation/mask30__<backbone>__<edges>.pt` | spatial masking | 30% masking |

- `<backbone>`: `heterogcn` | `heterosage` | `heterogat`
- `<edges>`: `spatial_only` | `hydro_only` | `dual_edge`

Metrics for these runs are written to `results/` (also gitignored). Curated
figures live in `docs/figures/`.
