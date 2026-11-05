"""Generate the bundled sound library. Every sound is synthesized here, so the files carry no
third-party rights and ship under the repo's MIT license.

    uv run python scripts/make_sfx.py
"""
from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 48000
OUT = Path(__file__).parent.parent / "src/vidx/assets/sfx"


def t(sec: float) -> np.ndarray:
    return np.arange(int(sec * SR)) / SR


def lowpass(x: np.ndarray, cutoff) -> np.ndarray:
    """One-pole lowpass; cutoff may vary per sample (Hz)."""
    a = 1 - np.exp(-2 * np.pi * np.broadcast_to(cutoff, x.shape) / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def verb(x: np.ndarray, decay: float, rng) -> np.ndarray:
    """Convolve with a decaying noise burst: a cheap room."""
    n = int(decay * SR)
    ir = rng.standard_normal(n) * np.exp(-6 * np.arange(n) / n)
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return x + 0.25 * y / (np.abs(y).max() + 1e-9)


def drone(root: float, seed: int, sec: float = 12.0, fifth: bool = True, shimmer: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = t(sec)
    loop = 2 * np.pi / sec  # every modulation completes whole cycles, so the file loops cleanly
    out = np.zeros_like(x)
    for f, a in [(root, 1.0), (root * 2, 0.45), (root * 1.5 if fifth else root * 1.2, 0.35), (root * 3, 0.12)]:
        f = round(f * sec) / sec  # whole cycles per loop
        for det in (-1, 0, 1):
            fd = f + det / sec
            out += a * np.sin(2 * np.pi * fd * x + rng.uniform(0, 6.28))
    out *= 0.75 + 0.25 * np.sin(loop * 2 * x)
    if shimmer:
        out += shimmer * np.sin(2 * np.pi * round(root * 8 * sec) / sec * x) * (0.5 + 0.5 * np.sin(loop * 3 * x))
    return out


def impact(seed: int, low: float = 45, bright: float = 1.0, sec: float = 2.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = t(sec)
    sweep = low + 180 * np.exp(-x * 30)
    body = np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.exp(-x * 3.2)
    crack = lowpass(rng.standard_normal(len(x)), 900 + 3000 * bright) * np.exp(-x * 45) * bright
    return verb(body + 0.8 * crack, 1.4, rng)


def riser(seed: int, sec: float = 4.0, tone: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = t(sec)
    p = x / sec
    noise = lowpass(rng.standard_normal(len(x)), 200 + 7000 * p ** 2.5)
    f = 110 * 2 ** (p * 3.2)
    pitch = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return (noise * 1.6 + tone * pitch) * p ** 2.2 * (1 - np.exp(-(1 - p) * 60))


def whoosh(seed: int, sec: float = 1.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = t(sec)
    env = np.sin(np.pi * x / sec) ** 3
    return lowpass(rng.standard_normal(len(x)), 300 + 3500 * env) * env * 2


SOUNDS = {
    "drone_01": lambda: drone(55, 1), "drone_02": lambda: drone(41.2, 2), "drone_03": lambda: drone(49, 3, fifth=False),
    "drone_04": lambda: drone(36.7, 4), "drone_05": lambda: drone(65.4, 5, shimmer=0.12),
    "drone_06": lambda: drone(46.2, 6, fifth=False, shimmer=0.08),
    "impact_01": lambda: impact(11), "impact_02": lambda: impact(12, low=38, bright=0.5),
    "impact_03": lambda: impact(13, low=60, bright=1.4), "impact_04": lambda: impact(14, low=30, bright=0.25, sec=3.5),
    "impact_05": lambda: impact(15, low=52, bright=0.8, sec=1.8), "impact_06": lambda: impact(16, low=70, bright=1.8, sec=1.5),
    "riser_01": lambda: riser(21), "riser_02": lambda: riser(22, sec=6.0), "riser_03": lambda: riser(23, sec=2.5, tone=0.2),
    "riser_04": lambda: riser(24, tone=0.9), "riser_05": lambda: riser(25, sec=8.0, tone=0.35),
    "whoosh_01": lambda: whoosh(31), "whoosh_02": lambda: whoosh(32, sec=0.7), "whoosh_03": lambda: whoosh(33, sec=1.8),
}


def write(name: str, x: np.ndarray) -> None:
    x = x / (np.abs(x).max() + 1e-9) * 0.89
    edge = int(0.004 * SR)  # no clicks at the ends
    x[:edge] *= np.linspace(0, 1, edge)
    x[-edge:] *= np.linspace(1, 0, edge)
    pcm = (np.stack([x, x], 1) * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        with wave.open(f.name, "wb") as w:
            w.setnchannels(2), w.setsampwidth(2), w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", f.name, "-c:a", "aac", "-b:a", "128k",
                        str(OUT / f"{name}.m4a")], check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in SOUNDS.items():
        write(name, make())
        print(name)
