# Legacy scripts (archived)

**SUPERSEDED - not part of the physics-informed pipeline. Do not use.**

These scripts predate the current workflow and are retained only for
provenance. Some use synthetic labels and the non-physics graph
(`buhisan_hetero.pt`), and several hardcode absolute paths.

| Script | Superseded by |
|--------|---------------|
| `ablation_study.py` | `scripts/physics_ablation.py` (helpers now in `src/terrapyge/models/experiments.py`) |
| `run_training.py` | `scripts/run_physics_pipeline.py` |
| `generate_hazard_maps.py` | `scripts/generate_physics_hazard_map.py` |
| `setup_project.py` | obsolete - the directory tree already exists |

The authoritative pipeline is `scripts/run_physics_pipeline.py`; see
`docs/local/AGENTS.md`.
