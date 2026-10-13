"""ffmpeg helpers and token-aware formatting."""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from .store import VidxError


def need_ffmpeg() -> None:
    for b in ("ffmpeg", "ffprobe"):
        if shutil.which(b) is None:
            raise VidxError(f"{b} not found on PATH (install with: brew install ffmpeg)")


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-3:]
        raise VidxError(f"{cmd[0]} failed: {' | '.join(tail)}")
    return p


def ffprobe(path: str) -> dict:
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(p.stdout)


def extract_audio16k(src: str, dst: Path) -> Path:
    """Mono 16 kHz PCM wav, shared by transcript + checks."""
    if not dst.exists():
        tmp = dst.with_suffix(".tmp.wav")
        run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(tmp)])
        tmp.rename(dst)
    return dst


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, sr


def rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return 20 * np.log10(max(r, 1e-6))


# ---------- formatting (rule 1: one decimal) ----------

def t1(x: float) -> str:
    return f"{x:.1f}"


def mmss(x: float) -> str:
    m, s = divmod(max(x, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def rng(a: float, b: float) -> str:
    return f"{a:.1f}-{b:.1f}"


def snip(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def more(total: int, shown: int) -> list[str]:
    return [f"… {total - shown} more"] if total > shown else []
