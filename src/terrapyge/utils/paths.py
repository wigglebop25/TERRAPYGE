"""Path resolution for TERRAPYGE.

Works identically on local Windows and Google Colab
(/content/TERRAPYGE + Google Drive mount). Detects the Colab environment
via the COLAB_GPU env var that Colab injects at session start.

Local resolution order (first match wins):
  1. ``TERRAPYGE_ROOT`` environment variable
  2. Nearest ancestor of this file that contains ``config.yaml`` + ``pyproject.toml``
  3. Nearest such ancestor of the current working directory
  4. Current working directory (last resort)

Usage:
    from src.terrapyge.utils.paths import ROOT, DATA, GRAPHS, MODELS
"""

from pathlib import Path
import os


def _is_colab() -> bool:
    """Return True if running inside Google Colab."""
    return 'COLAB_GPU' in os.environ


def _find_repo_root(start: Path) -> Path | None:
    """Return the nearest ancestor of ``start`` that looks like the repo root."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / 'config.yaml').exists() and (candidate / 'pyproject.toml').exists():
            return candidate
    return None


def get_project_root() -> Path:
    """Return the repo root path for the current environment."""
    if _is_colab():
        return Path('/content/TERRAPYGE')

    env_root = os.environ.get('TERRAPYGE_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()

    found = _find_repo_root(Path(__file__).parent)
    if found is not None:
        return found

    found = _find_repo_root(Path.cwd())
    if found is not None:
        return found

    return Path.cwd()


def get_drive_root() -> Path:
    """Return the persistent Drive root (Colab) or repo root (local).

    On Colab, points to the mounted Google Drive folder so checkpoints
    and graphs survive session timeouts. On local, returns the repo root
    since persistence is already guaranteed.
    """
    if _is_colab():
        return Path('/content/drive/MyDrive/TERRAPYGE')
    return get_project_root()


ROOT = get_project_root()
DATA = ROOT / 'data'
RAW = DATA / 'raw'
PROCESSED = DATA / 'processed'
GRAPHS = DATA / 'graphs'
RESULTS = ROOT / 'results'
LABELS = DATA / 'labels'

# Convenience constants for the physics pipeline.
PROCESSED_BUHISAN = PROCESSED / 'buhisan'
FIGURES = RESULTS / 'figures'

# Write checkpoints and graphs to Drive on Colab (persists across sessions),
# to repo on local.
CHECKPOINTS = get_drive_root() / 'checkpoints'
PERSISTENT_GRAPHS = get_drive_root() / 'data' / 'graphs'
MODELS = ROOT / 'models'
ABLATION_MODELS = MODELS / 'ablation'


__all__ = [
    '_is_colab', 'get_project_root', 'get_drive_root',
    'ROOT', 'DATA', 'RAW', 'PROCESSED', 'GRAPHS', 'RESULTS', 'LABELS',
    'PROCESSED_BUHISAN', 'FIGURES',
    'CHECKPOINTS', 'PERSISTENT_GRAPHS', 'MODELS', 'ABLATION_MODELS',
]
