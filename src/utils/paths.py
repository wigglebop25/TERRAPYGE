"""Path resolution for TERRAPYGE.

Works identically on local Windows (D:/TERRAPYGE) and Google Colab
(/content/TERRAPYGE + Google Drive mount). Detects environment via
the COLAB_GPU env var that Colab injects at session start.

Usage:
    from src.utils.paths import ROOT, DATA, GRAPHS, CHECKPOINTS
"""

from pathlib import Path
import os


def _is_colab() -> bool:
    """Return True if running inside Google Colab."""
    return 'COLAB_GPU' in os.environ


def get_project_root() -> Path:
    """Return the repo root path for the current environment."""
    if _is_colab():
        return Path('/content/TERRAPYGE')
    return Path('D:/TERRAPYGE')


def get_drive_root() -> Path:
    """Return the persistent Drive root (Colab) or repo root (local).

    On Colab, points to the mounted Google Drive folder so checkpoints
    and graphs survive session timeouts. On local, returns the repo root
    since persistence is already guaranteed.
    """
    if _is_colab():
        return Path('/content/drive/MyDrive/TERRAPYGE')
    return Path('D:/TERRAPYGE')


ROOT = get_project_root()
DATA = ROOT / 'data'
RAW = DATA / 'raw'
PROCESSED = DATA / 'processed'
GRAPHS = DATA / 'graphs'
RESULTS = ROOT / 'results'
LABELS = DATA / 'labels'

# Write checkpoints and graphs to Drive on Colab (persists across sessions),
# to repo on local.
CHECKPOINTS = get_drive_root() / 'checkpoints'
PERSISTENT_GRAPHS = get_drive_root() / 'data' / 'graphs'
MODELS = ROOT / 'models'


__all__ = [
    '_is_colab', 'get_project_root', 'get_drive_root',
    'ROOT', 'DATA', 'RAW', 'PROCESSED', 'GRAPHS', 'RESULTS', 'LABELS',
    'CHECKPOINTS', 'PERSISTENT_GRAPHS', 'MODELS',
]
