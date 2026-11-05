"""Automatic colour matching: shift a clip's colours toward a reference, as a 3D LUT.

Reinhard colour transfer in Lab: move each channel's mean and spread to the reference's. Unlike a
brightness offset this also fixes colour casts and contrast. The transform is baked into a small
.cube file applied with ffmpeg's lut3d, so it costs nothing extra at render time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import thumbs

LUT_SIZE = 17
SPREAD_LIMIT = (0.6, 1.6)  # beyond this a stretched channel starts to posterize


def lab_stats(images) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of L, a, b over PIL images (float Lab: L 0-100, a/b about 0)."""
    import cv2

    px = []
    for im in images:
        if im is not None:
            rgb = np.asarray(im.convert("RGB"), np.float32) / 255.0
            px.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3))
    if not px:
        raise ValueError("no frames to measure")
    allpx = np.concatenate(px)
    return allpx.mean(0), allpx.std(0)


def visible(im, active: list[int] | None, src_size: tuple[int, int], aspect: float):
    """Crop a frame to what the viewer will see: inside any letterbox, then the output's aspect.
    Colours outside that (black bars, the sides a 9:16 crop drops) must not steer the match."""
    if im is None:
        return None
    W, H = im.size
    x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
    if active:
        aw, ah, ax, ay = active
        sw, sh = src_size
        x0, y0, x1, y1 = ax / sw, ay / sh, (ax + aw) / sw, (ay + ah) / sh
    w, h = (x1 - x0) * W, (y1 - y0) * H
    if w / h > aspect:
        cut = (w - h * aspect) / 2 / W
        x0, x1 = x0 + cut, x1 - cut
    else:
        cut = (h - w / aspect) / 2 / H
        y0, y1 = y0 + cut, y1 - cut
    return im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))


def frames_at(path: str, a: float, b: float, n: int = 5, active=None, src_size=(1, 1), aspect: float | None = None):
    ts = [a + (b - a) * (i + 0.5) / n for i in range(n)]
    frames = thumbs.grab_frames(path, ts, long_side=384, workers=2)
    if aspect is None:
        return frames
    return [visible(f, active, src_size, aspect) for f in frames]


def transfer(lab: np.ndarray, src: tuple, ref: tuple, strength: float = 0.8) -> np.ndarray:
    (sm, ss), (rm, rs) = src, ref
    scale = np.clip(rs / np.maximum(ss, 1e-3), *SPREAD_LIMIT)
    moved = (lab - sm) * scale + rm
    return lab + (moved - lab) * strength


def cube_text(src: tuple, ref: tuple, strength: float = 0.8, size: int = LUT_SIZE) -> str:
    import cv2

    g = np.linspace(0, 1, size, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")  # .cube order: red varies fastest
    rgb = np.stack([r, gg, b], -1).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    out = cv2.cvtColor(transfer(lab, src, ref, strength).astype(np.float32), cv2.COLOR_LAB2RGB)
    out = np.clip(out.reshape(-1, 3), 0, 1)
    lines = [f"LUT_3D_SIZE {size}"] + [f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in out]
    return "\n".join(lines) + "\n"


def match_lut(cache: Path, src: dict, ref: dict, aspect: float, strength: float = 0.8) -> Path:
    """Build (or reuse) the LUT that moves a clip toward a reference.
    src / ref: {path, a, b, active, size} describing a time range of a video."""
    sig = json.dumps([src, ref, round(aspect, 4), strength], sort_keys=True, default=str)
    p = cache / f"{hashlib.sha1(sig.encode()).hexdigest()[:16]}.cube"
    if not p.exists():
        def measure(c: dict):
            return lab_stats(frames_at(c["path"], c["a"], c["b"], active=c["active"], src_size=c["size"], aspect=aspect))

        p.write_text(cube_text(measure(src), measure(ref), strength))
    return p
