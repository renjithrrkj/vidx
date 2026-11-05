"""Shot boundaries + per-frame motion/color curves, from ONE low-res ffmpeg decode.

Cuts come from TransNetV2, a small neural shot detector (weights ship inside its wheel, so it
works offline). It catches dissolves and survives fast motion, where a frame-difference threshold
either misses transitions or fires on every punch. scenedetect's ContentDetector is the fallback.

Per-frame curves are saved next to the index (motion.npy, color.npy) so per-shot stats, peak
motion and colour matching never need to decode the video again.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from .. import config, store
from ..store import Shot, VideoInfo
from . import AnalyzerResult, Context

LONG_SIDE = 256
TN_W, TN_H = 48, 27      # TransNetV2's fixed input size
TN_WIN, TN_STEP, TN_PAD = 100, 50, 25
CUT_THRESHOLD = 0.5
CURVE_W = 64             # width of the copy used for motion/colour curves


def frame_size(w: int, h: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    s = long_side / max(w, h)
    return max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)


def iter_frames(path: str, fps: float, w: int, h: int, hwaccel: bool = True):
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if hwaccel and sys.platform == "darwin":
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", path, "-an", "-sn", "-dn", "-vf", f"scale={w}:{h}:flags=fast_bilinear",
            "-fps_mode", "cfr", "-r", f"{fps}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3 * 8)
    n = w * h * 3
    finished = False
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                finished = True
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        err = p.stderr.read().decode(errors="replace")
        p.wait()
        if finished and p.returncode not in (0, None) and hwaccel:  # not when the caller stopped early
            raise RuntimeError(err)


class NeuralCuts:
    """Streaming TransNetV2: same 100-frame windows / 50-frame step as the reference code, fed
    as frames arrive, so memory stays flat however long the video is."""

    def __init__(self):
        import torch
        from transnetv2_pytorch import TransNetV2

        self.torch = torch
        self.model = TransNetV2(device="cpu").eval()  # tiny inputs: CPU beats MPS transfer overhead
        self.buf: list[np.ndarray] = []
        self.preds: list[np.ndarray] = []
        self.n = 0

    def _run(self) -> None:
        batch = self.torch.from_numpy(np.stack(self.buf[:TN_WIN]))[None]
        with self.torch.no_grad():
            single, _ = self.model.predict_raw(batch)
        self.preds.append(single[0, TN_PAD:TN_PAD + TN_STEP, 0].numpy())
        del self.buf[:TN_STEP]

    def add(self, small_rgb: np.ndarray) -> None:
        if not self.buf:
            self.buf = [small_rgb] * TN_PAD
        self.buf.append(small_rgb)
        self.n += 1
        if len(self.buf) >= TN_WIN:
            self._run()

    def finish(self) -> np.ndarray:
        if self.n == 0:
            return np.zeros(0)
        last = self.buf[-1]
        while sum(len(p) for p in self.preds) < self.n:
            self.buf += [last] * (TN_WIN - len(self.buf))
            self._run()
        return np.concatenate(self.preds)[: self.n]


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    sa, sb = a.std(), b.std()
    if sa < 3 or sb < 3:  # near-flat frame (black, white): correlation means nothing
        return None
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def is_flash(thumbs: np.ndarray, k: int, gap: int = 4, same: float = 0.7, back: float = 0.75) -> bool:
    """True when frame k is a flash, not a cut. A flash keeps the picture (high correlation after
    normalizing brightness), or changes it for a moment and then it comes back within ~1.5s.
    Only the central rows count: letterbox bars match across every cut and fake a high score."""
    h = thumbs.shape[1]
    rows = slice(h // 4, h - h // 4)
    before = thumbs[max(0, k - gap), rows]
    c = _corr(before, thumbs[min(len(thumbs) - 1, k + gap - 1), rows])
    if c is not None and c > same:
        return True
    for j in range(k + 10, min(len(thumbs), k + 45), 5):
        c = _corr(before, thumbs[j, rows])
        if c is not None and c > back:
            return True
    return False


def jump_candidates(motion: np.ndarray, floor: float = 0.05, ratio: float = 4.0, win: int = 10) -> list[int]:
    """Frames whose change towers over their neighbourhood. Sustained motion (a fight, a whip pan)
    raises the local median, so only isolated jumps qualify."""
    out = []
    for k in np.nonzero(motion >= floor)[0]:
        lo, hi = max(1, k - win), min(len(motion), k + win + 1)
        around = np.concatenate([motion[lo:k], motion[k + 1:hi]])
        if len(around) and motion[k] >= ratio * max(float(np.median(around)), 1e-3) and motion[k] == motion[lo:hi].max():
            out.append(int(k))
    return out


def cuts_from_scores(scores: np.ndarray, motion: np.ndarray | None = None, thumbs: np.ndarray | None = None,
                     threshold: float = CUT_THRESHOLD, min_gap: int = 3) -> list[int]:
    """Frame indices where a new shot starts, from three signals tested on real footage:

    - the network finds cuts a pixel test can't (dark scenes, dissolves), but can sit frames off
      and, on long uncertain runs, a good half second late;
    - isolated pixel jumps give the exact frame and catch hard cuts the network skipped;
    - both fire on flashes, which is_flash() rejects.
    """
    n = len(scores)
    above = scores > threshold
    cand: list[int] = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        c = (i + j + 1) // 2
        if motion is not None and len(motion) >= n:
            lo, hi = max(1, i - 2), min(n, j + 4)
            k = lo + int(np.argmax(motion[lo:hi]))  # motion[k] compares frame k with k-1
            typical = float(np.median(motion[lo:hi]))
            if j - i < 4 or motion[k] >= max(0.04, 3 * typical):
                c = k  # a clear jump inside the run: that's the cut, not the run's middle
        cand.append(c)
        i = j + 1
    if motion is not None:
        cand += [k for k in jump_candidates(motion[:n]) if k < n]
    cuts: list[int] = []
    for c in sorted(set(cand)):
        if not 0 < c < n:
            continue
        if thumbs is not None and is_flash(thumbs, c):
            continue
        if cuts and c - cuts[-1] < min_gap:
            continue
        cuts.append(c)
    return cuts


def curves(frame_bgr: np.ndarray, prev_gray, cv2) -> tuple[float, np.ndarray, list[float]]:
    """-> (motion 0-1, gray copy, [L, a, b, luma_std, saturation]) for one frame."""
    h = max(2, int(frame_bgr.shape[0] * CURVE_W / frame_bgr.shape[1]))
    small = cv2.resize(frame_bgr, (CURVE_W, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    motion = 0.0 if prev_gray is None else float(np.abs(gray.astype(np.int16) - prev_gray).mean()) / 255.0
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].mean()
    return motion, gray, [*lab.mean(0).tolist(), float(lab[:, 0].std()), float(sat)]


def shot_stats(a: float, b: float, fps: float, motion: np.ndarray, color: np.ndarray) -> dict:
    """Per-shot motion curve summary and colour stats (0-100 style units, 1 decimal)."""
    i, j = int(a * fps), max(int(a * fps) + 1, int(b * fps))
    m = motion[i + 1: j] if j - i > 2 else motion[i: j]  # skip the cut frame itself
    c = color[i: j]
    if not len(m) or not len(c):
        return {}
    k = int(np.argmax(m))
    L, A, B = (float(c[:, n].mean()) for n in range(3))
    return {
        "motion": {"mean": round(float(m.mean()) * 100, 2), "peak": round(float(m.max()) * 100, 2),
                   "peak_t": round(a + (k + 1) / fps, 2),
                   "curve": [round(float(p.mean()) * 100, 1) for p in np.array_split(m, min(12, len(m)))]},
        "color": {"luma": round(L / 2.55, 1), "contrast": round(float(c[:, 3].mean()) / 2.55, 1),
                  "sat": round(float(c[:, 4].mean()) / 2.55, 1),
                  "lab": [round(L, 1), round(A - 128, 1), round(B - 128, 1)]},
    }


class Shots:
    name = "shots"
    optional = False
    version = 2  # 2: TransNetV2 + flash filter + motion/colour curves

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        import cv2

        fps = video.fps
        w, h = frame_size(video.width, video.height)

        def detect(hw: bool):
            try:
                neural = NeuralCuts()
            except ImportError:
                neural = None
            fallback = None
            if neural is None:
                from scenedetect.common import FrameTimecode
                from scenedetect.detectors import ContentDetector

                fallback = ContentDetector(threshold=config.SHOT_THRESHOLD, min_scene_len=max(2, round(fps * 0.5)))
            cuts: list[int] = []
            motion: list[float] = []
            color: list[list[float]] = []
            thumbs: list[np.ndarray] = []  # 32x18 gray, ~60 MB per hour of 30fps video
            prev = None
            i = -1
            for i, frame in enumerate(iter_frames(video.path, fps, w, h, hwaccel=hw)):
                m, prev, c = curves(frame, prev, cv2)
                motion.append(m)
                color.append(c)
                thumbs.append(cv2.resize(prev, (32, 18), interpolation=cv2.INTER_AREA))
                if neural is not None:
                    small = cv2.resize(frame, (TN_W, TN_H), interpolation=cv2.INTER_AREA)
                    neural.add(np.ascontiguousarray(small[..., ::-1]))  # model expects RGB
                else:
                    cuts += [tc.frame_num for tc in fallback.process_frame(FrameTimecode(i, fps), frame)]
            if neural is not None:
                cuts = cuts_from_scores(neural.finish(), np.asarray(motion, np.float32), np.stack(thumbs) if thumbs else None)
            elif i >= 0:
                cuts += [tc.frame_num for tc in fallback.post_process(FrameTimecode(i, fps))]
            return cuts, motion, color, i + 1, "transnetv2" if neural is not None else "content"

        try:
            cuts, motion, color, nframes, method = detect(True)
        except RuntimeError:
            cuts, motion, color, nframes, method = detect(False)  # hw decode unsupported for this codec

        mo = np.asarray(motion, np.float32) if motion else np.zeros(1, np.float32)
        co = np.asarray(color, np.float32) if color else np.zeros((1, 5), np.float32)
        np.save(video.dir / "motion.npy", mo.astype(np.float16))
        np.save(video.dir / "color.npy", co.astype(np.float16))

        dur = video.duration
        bounds = [0.0] + [c / fps for c in sorted(set(cuts)) if 0 < c / fps < dur] + [dur]
        shots = []
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            d = shot_stats(a, b, fps, mo, co)
            shots.append(Shot(idx=k + 1, start=round(a, 3), end=round(b, 3),
                              motion=d.get("motion", {}).get("mean", 0.0), detail=d))
        store.put_shots(video.id, shots)
        return AnalyzerResult(info={"shots": len(shots), "frames": nframes, "method": method})
