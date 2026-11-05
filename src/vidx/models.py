"""Model weights. Shot cuts, OCR and speech detection ship inside their Python packages. Whisper and
CLIP are too large for a git repo, so `vidx setup` fetches them once; after that nothing uses the
network. Set VIDX_MODEL_DIR to keep them in a folder you can copy to another machine."""
from __future__ import annotations

from . import config


def prefetch(log=print) -> None:
    from huggingface_hub import snapshot_download

    log(f"whisper {config.WHISPER_MODEL} ...")
    snapshot_download(f"Systran/faster-whisper-{config.WHISPER_MODEL}")
    log("CLIP ...")
    from .analyzers.visual import load_clip

    load_clip()
    log("ready: vidx can now run offline")
