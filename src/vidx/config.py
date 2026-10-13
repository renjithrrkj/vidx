"""Paths and tunables. Everything can be overridden with VIDX_* env vars."""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("VIDX_HOME", Path.home() / ".vidx")).expanduser()
DB_PATH = HOME / "vidx.db"

WHISPER_MODEL = os.environ.get("VIDX_WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.environ.get("VIDX_WHISPER_LANGUAGE") or None  # None = auto-detect
CLIP_MODEL = os.environ.get("VIDX_CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.environ.get("VIDX_CLIP_PRETRAINED", "laion2b_s34b_b79k")

# Analyzer thresholds
SHOT_THRESHOLD = float(os.environ.get("VIDX_SHOT_THRESHOLD", "27"))
SILENCE_DB = float(os.environ.get("VIDX_SILENCE_DB", "-35"))
SILENCE_MIN = 0.5
LOUD_OVER_INTEGRATED = 10.0  # LU above integrated loudness counts as "loud"
BLACK_MIN = 0.3
FREEZE_MIN = 1.0

# Output budgets
OVERVIEW_MAX_CHARS = 12000
DEFAULT_LIMIT = 20
SNIPPET_CHARS = 60


def video_dir(video_id: str) -> Path:
    d = HOME / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def device() -> str:
    """Best torch device: cuda > mps > cpu."""
    forced = os.environ.get("VIDX_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
