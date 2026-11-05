"""Train your own detector from labelled examples: "knockdown", "crowd reaction", "AI morph".

CLIP stays frozen. What gets trained is a small new layer on top of its image embeddings, plus motion:
one weight per feature, about 500 numbers in all. That is enough to learn a visual concept from a
few dozen seconds of labelled video, runs in seconds on a laptop, and can't damage the models under it.

    vidx teach knockdown --pos ca833113:12-14,ca833113:30-31 --neg ca833113:50-60,8089b5dc:0-20
    vidx detect knockdown ca833113
    query(video_id, "knockdown")          # runs the detector if it hasn't been run yet

Honest scores come from cross-validation grouped by labelled range: seconds next to each other look
alike, so a random split would let the model peek at near-copies of its test data.
"""
from __future__ import annotations

import json
import re

import numpy as np

from . import config, store
from .store import Event, VidxError

SCAN_FPS = 1.0
MIN_RANGES = 3


def det_dir():
    d = config.HOME / "detectors"
    d.mkdir(parents=True, exist_ok=True)
    return d


def names() -> list[str]:
    return sorted(p.stem for p in det_dir().glob("*.npz"))


def check_name(name: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", name):
        raise VidxError(f"bad detector name {name!r} (lowercase letters, digits, _)")
    if name in ("silence", "loud", "black", "freeze", "laugh", "beat", "glitch", "speech", "music", "sfx"):
        raise VidxError(f"{name!r} is a built-in event, pick another name")
    return name


def scan(video_id: str) -> tuple[np.ndarray, np.ndarray]:
    """One CLIP embedding per second, plus motion around it. Cached: the slow part happens once."""
    v = store.get_video(video_id)
    p = v.dir / "scan.npz"
    if p.exists():
        d = np.load(p)
        return d["t"], d["x"].astype(np.float32)
    from PIL import Image

    from .analyzers.shots import frame_size, iter_frames
    from .analyzers.visual import encode_images

    w, h = frame_size(v.width, v.height, 256)
    emb, batch = [], []
    for f in iter_frames(v.path, SCAN_FPS, w, h):
        batch.append(Image.fromarray(np.ascontiguousarray(f[..., ::-1])))
        if len(batch) == 32:  # stream: never hold more than one batch of frames
            emb.append(encode_images(batch))
            batch = []
    if batch:
        emb.append(encode_images(batch))
    if not emb:
        raise VidxError(f"no frames decoded from {v.path}")
    e = np.concatenate(emb)
    t = np.arange(len(e)) / SCAN_FPS
    x = np.hstack([e, motion_features(v, t)]).astype(np.float32)
    np.savez_compressed(p, t=t, x=x.astype(np.float16))
    return t, x


def motion_features(v, t: np.ndarray) -> np.ndarray:
    """CLIP sees single frames; action lives in movement. Mean and peak change in each second."""
    out = np.zeros((len(t), 2), np.float32)
    mp = v.dir / "motion.npy"
    if mp.exists():
        m = np.load(mp).astype(np.float32)
        fps = v.fps
        for i, s in enumerate(t):
            seg = m[int(s * fps): int((s + 1) * fps)]
            if len(seg):
                out[i] = (seg.mean() * 4, seg.max() * 2)
    return out


def parse_ranges(text: str, default_video: str | None = None) -> list[tuple[str, float, float]]:
    """'id:12-14,id:30-31' -> [(id, 12, 14), ...]"""
    out = []
    for part in filter(None, (x.strip() for x in text.split(","))):
        m = re.fullmatch(r"(?:([0-9a-f]{8}):)?([\d.]+)-([\d.]+)", part)
        if not m or not (m.group(1) or default_video):
            raise VidxError(f"bad range {part!r} (use video_id:12-14)")
        a, b = float(m.group(2)), float(m.group(3))
        if b <= a:
            raise VidxError(f"empty range {part!r}")
        out.append((m.group(1) or default_video, a, b))
    return out


def samples(ranges: list[tuple[str, float, float]], label: int, first_group: int):
    xs, ys, gs = [], [], []
    for g, (vid, a, b) in enumerate(ranges, first_group):
        t, x = scan(vid)
        sel = np.nonzero((t >= a - 0.5) & (t <= b))[0]
        if not len(sel):
            raise VidxError(f"{vid}:{a}-{b} has no scanned frames (is the range inside the video?)")
        xs.append(x[sel])
        ys.append(np.full(len(sel), label))
        gs.append(np.full(len(sel), g))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)


def fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, steps: int = 600, lr: float = 2.0) -> tuple[np.ndarray, float]:
    """Logistic regression by gradient descent. Each class carries equal total weight, because
    interesting moments are rare and a detector that says 'no' to everything must not look good."""
    n, d = x.shape
    w, b = np.zeros(d), 0.0
    cw = np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1))
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(x @ w + b, -30, 30)))
        err = (p - y) * cw
        w -= lr * (x.T @ err + l2 * w / n)
        b -= lr * err.sum()
    return w, float(b)


def prob(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x @ w + b, -30, 30)))


def cross_validate(x, y, groups, folds: int = 5) -> dict:
    """Whole labelled ranges are held out together, so adjacent seconds never straddle train and test."""
    ids = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    scores = np.zeros(len(y))
    seen = np.zeros(len(y), bool)
    for f in range(min(folds, len(ids))):
        test = np.isin(groups, ids[f::min(folds, len(ids))])
        if len(np.unique(y[~test])) < 2:
            continue
        w, b = fit(x[~test], y[~test])
        scores[test] = prob(x[test], w, b)
        seen |= test
    p = scores[seen] >= 0.5
    t = y[seen] == 1
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    return {"precision": round(tp / max(tp + fp, 1), 2), "recall": round(tp / max(tp + fn, 1), 2),
            "seconds_tested": int(seen.sum())}


def teach(name: str, pos: str, neg: str) -> str:
    check_name(name)
    P, N = parse_ranges(pos), parse_ranges(neg)
    if len(P) < MIN_RANGES or len(N) < MIN_RANGES:
        raise VidxError(f"give at least {MIN_RANGES} positive and {MIN_RANGES} negative ranges "
                        f"(got {len(P)} and {len(N)}); more, from different videos, is better")
    xp, yp, gp = samples(P, 1, 0)
    xn, yn, gn = samples(N, 0, len(P))
    x, y, g = np.vstack([xp, xn]), np.concatenate([yp, yn]), np.concatenate([gp, gn])
    cv = cross_validate(x, y, g)
    w, b = fit(x, y)
    np.savez(det_dir() / f"{name}.npz", w=w, b=b, meta=json.dumps({"pos": P, "neg": N, "cv": cv}))
    videos = len({r[0] for r in P + N})
    lines = [f"detector {name}: {len(w)} weights from {int(y.sum())}s positive, {int((1 - y).sum())}s negative, "
             f"{videos} video(s)",
             f"held-out: precision {cv['precision']:.2f} recall {cv['recall']:.2f} on {cv['seconds_tested']}s"]
    if videos == 1:
        lines.append("warn: all from one video, so this may have learned that video rather than the idea; add others")
    lines.append(f"next: query(video_id, '{name}')")
    return "\n".join(lines)


def detect(name: str, video_id: str, threshold: float = 0.5, min_len: float = 1.0) -> list[Event]:
    p = det_dir() / f"{name}.npz"
    if not p.exists():
        raise VidxError(f"no detector {name!r} (have: {', '.join(names()) or 'none'}; train with `vidx teach`)")
    d = np.load(p)
    t, x = scan(video_id)
    s = prob(x, d["w"], float(d["b"]))
    s = np.convolve(s, np.ones(3) / 3, mode="same")  # one odd second shouldn't trigger or break a run
    ev: list[Event] = []
    i = 0
    while i < len(s):
        if s[i] < threshold:
            i += 1
            continue
        j = i
        while j + 1 < len(s) and s[j + 1] >= threshold:
            j += 1
        if t[j] + 1 - t[i] >= min_len:
            ev.append(Event(name, float(t[i]), float(t[j] + 1), round(float(s[i:j + 1].max()), 2)))
        i = j + 1
    store.put_events(video_id, [name], ev)
    return ev
