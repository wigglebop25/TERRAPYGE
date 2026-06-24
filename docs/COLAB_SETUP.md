# Hybrid Compute Guide: Local (Ryzen 5600G) + Google Colab (T4 GPU)

## Overview

TERRAPYGE training and hyperparameter sweeps run on a hybrid compute setup.
The local Ryzen 5 5600G (CPU) handles development, single training runs, and
all geospatial processing. Google Colab Free (T4 GPU) handles hyperparameter
sweeps and ensemble training where the GPU provides a meaningful speedup.

## Strategy at a Glance

| Task | Where | Why |
|------|-------|-----|
| DEM conditioning (WhiteboxTools) | Local | seconds, small raster |
| Slope unit extraction | Local | 1-5 min, iterative dev |
| Graph construction (D8, Queen edges) | Local | seconds, small graph |
| Single GNN training run (100-300 ep) | Local | 15-30 min on CPU, fast enough |
| Debugging, interpretability, plotting | Local | needs iteration speed |
| Hyperparameter sweeps (Optuna, 50+ trials) | Colab Free (T4) | 2-4 hrs vs 12-25 hrs |
| Ensemble training, final ablations | Colab Free/Pro | longer runs |
| Inference + hazard map generation | Either | works on both |

## 1. Local Setup (5600G, one-time)

### 1.1 Activate existing venv

```powershell
cd D:\TERRAPYGE
.\venv\Scripts\Activate.ps1
```

### 1.2 Install the missing GNN stack

CPU wheels matching the installed torch 2.12.0+cpu. No compilation needed.

```powershell
pip install torch_geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
pip install city2graph libpysal scikit-learn rioxarray tqdm wandb optuna xgboost
```

### 1.3 W&B login (experiment tracking, free tier)

```powershell
pip install wandb
wandb login
# Paste API key from https://wandb.ai/authorize
```

### 1.4 Verify installation

```powershell
python -c "import torch, torch_geometric as pyg; print('torch', torch.__version__, 'pyg', pyg.__version__); print('cuda', torch.cuda.is_available())"
```

Expect: `cuda False`. This is correct for local — the 5600G has no discrete GPU.
All GNN training runs on CPU, which is fast enough for ~1k-node graphs.

### 1.5 Verify paths module

```powershell
python -c "from src.utils.paths import ROOT, GRAPHS, CHECKPOINTS; print('ROOT:', ROOT); print('GRAPHS:', GRAPHS); print('CHECKPOINTS:', CHECKPOINTS)"
```

Expect: `D:\TERRAPYGE` paths on local.

## 2. Colab Setup (per-session, ~5 min)

### 2.1 Runtime selection

1. Open https://colab.research.google.com
2. Runtime > Change runtime type > **T4 GPU**
3. Free tier: 12-hr session limit, ~12 GB GPU memory, ~12 GB RAM

### 2.2 Colab setup cell (paste as first cell in every Colab notebook)

```python
# === TERRAPYGE Colab Setup ===
import torch, sys
print(f"Python {sys.version}, Torch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Mount Google Drive (persistent storage across sessions)
from google.colab import drive
drive.mount('/content/drive')

# Install PyG with compiled extensions matching Colab's torch+CUDA
TORCH = torch.__version__.split('+')[0]
CUDA = 'cu' + torch.version.cuda.replace('.', '')
!pip install torch_geometric
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html

# Workflow utilities
!pip install wandb optuna xgboost tqdm -q

# Clone repo if not already on Drive
import os
REPO_DIR = '/content/TERRAPYGE'
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/wigglebop25/TERRAPYGE.git {REPO_DIR}
os.chdir(REPO_DIR)

# W&B login
import wandb
wandb.login()

print("Setup complete.")
```

Replace `<YOUR_GITHUB_USER>` with your GitHub username.

### 2.3 Folder layout on Google Drive (set up once, manually)

```
My Drive/
  TERRAPYGE/
    data/
      raw/dem/          <- upload CebuCity_DEM.tif here (one-time)
      graphs/            <- torch.save(data, ...) lands here
      labels/            <- synthetic or real labels
    checkpoints/         <- model .pt files (best per sweep run)
```

Upload the DEM file once from local to Drive via the Google Drive web UI or `rclone`.

### 2.4 Verify paths on Colab

```python
from src.utils.paths import ROOT, GRAPHS, CHECKPOINTS, _is_colab
print(f"Is Colab: {_is_colab()}")
print(f"ROOT: {ROOT}")
print(f"GRAPHS: {GRAPHS}")
print(f"CHECKPOINTS: {CHECKPOINTS}")
```

Expect: `/content/TERRAPYGE` for ROOT, `/content/drive/MyDrive/TERRAPYGE/checkpoints` for CHECKPOINTS.

## 3. Workflow: Develop Local, Sweep on Colab

### 3.1 Day-to-day dev loop (local)

1. Edit code on the local machine.
2. Quick smoke test: `python -m src.models.trainer --epochs 5` (finishes in ~1 min).
3. Commit and push to GitHub: `git add -A && git commit -m "msg" && git push`.
4. Pull on Colab in a notebook cell: `!cd /content/TERRAPYGE && git pull`.

### 3.2 Colab hyperparameter sweep loop

```python
# After the setup cell (section 2.2)...
!cd /content/TERRAPYGE && git pull

# Load the precomputed graph (built locally, uploaded to Drive)
import torch
data = torch.load('/content/drive/MyDrive/TERRAPYGE/data/graphs/cebu_hetero.pt')

# Run Optuna sweep with W&B logging
!python -m src.models.hpo \
    --trials 50 \
    --graph /content/drive/MyDrive/TERRAPYGE/data/graphs/cebu_hetero.pt \
    --sweep hetero_dual_edge \
    --epochs 300

# After sweep completes, best checkpoint is saved to Drive automatically
```

T4 GPU: ~1-2 s/epoch on a 1k-node graph. A 50-config sweep finishes in
2-4 hours, well within the 12-hr free-tier limit.

### 3.3 Mixed precision (T4-specific, auto-disables on CPU)

Add this to the training loop in `src/models/trainer.py`:

```python
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# Inside training loop:
with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

The `enabled=torch.cuda.is_available()` flag makes this a no-op on CPU (local),
so the same code runs on both environments without modification.

## 4. Avoiding 12-hr Colab Timeouts

| Mitigation | How |
|------------|-----|
| Checkpoint to Drive | Save model `.pt` to `/content/drive/MyDrive/TERRAPYGE/checkpoints/` every epoch if val improves. Survives session death. |
| Save Optuna state | `joblib.dump(study, '/content/drive/MyDrive/TERRAPYGE/checkpoints/optuna_study.pkl')` after every N trials. Resume with `joblib.load()`. |
| Prefit on Colab, fine-tune local | Load the best checkpoint, continue 10-20 epochs on CPU if a run is borderline. |
| Load graph once | Do `data = torch.load(...)` at session start. Colab RAM (~12 GB) is plenty for a 1k-node graph. |
| Avoid idle disconnect | Keep one Colab tab focused. Colab disconnects after ~90 min of tab inactivity on free tier. |

## 5. Cost Estimate

| Item | Cost | Notes |
|------|------|-------|
| Local (5600G) | $0 | Already owned |
| Colab Free | $0 | T4 GPU, 12-hr sessions. Sufficient for entire project. |
| Colab Pro | $10/mo | Only if hitting timeout walls or need V100/A100. Likely unnecessary. |
| W&B Free | $0 | 100 GB storage, unlimited tracked runs. More than enough. |

## 6. Troubleshooting

### torch-scatter/sparse wheel mismatch

If `pip install torch-scatter ...` fails with version errors:
- Check Colab's torch version: `print(torch.__version__)`
- Match the wheel URL: `https://data.pyg.org/whl/torch-{VERSION}+{CUDA}.html`
- Example for torch 2.5.1+cu124: `https://data.pyg.org/whl/torch-2.5.1+cu124.html`

### Google Drive mount fails

```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

If it keeps failing, Runtime > Restart runtime, then re-run the setup cell.

### CUDA out of memory on T4

The 1k-node graph should never cause OOM. If it does (unlikely):
- Reduce `hidden_channels` (64 -> 32)
- Reduce `num_heads` in GAT (4 -> 2)
- Fall back to CPU: `model.cpu(); data = data.cpu()`

### Colab session disconnects mid-sweep

- Save Optuna state to Drive every 5-10 trials (see section 4).
- On reconnect, re-mount Drive, re-clone repo, load the study, resume.

### paths.py returns wrong path

- On Colab: verify `COLAB_GPU` env var exists: `import os; print('COLAB_GPU' in os.environ)`
- On local: verify you're running from `D:\TERRAPYGE` as the working directory.

## 7. File Reference

| File | Purpose |
|------|---------|
| `src/utils/paths.py` | Path normalization (local vs Colab) |
| `docs/COLAB_SETUP.md` | This guide |
| `config.yaml` | GNN hyperparameters, study area, CRS |
| `data/graphs/cebu_hetero.pt` | Precomputed HeteroData graph (built locally) |

---

Last updated: 2026-06-20
