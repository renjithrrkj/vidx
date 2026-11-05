"""AI-glitch suspects: morphing hands, warping faces, things that melt.

Real motion can be explained by moving pixels from one frame to the next; a morph can't. For each
frame pair we estimate optical flow, warp the previous frame forward, and look at what is left over,
block by block. A block whose leftover suddenly jumps above that shot's own normal level, and stays
there for a few frames, is reported as a suspect, together with the last frame before it started.

This is a heuristic and only ever says "suspect": it can't see that a hand has six fingers.
Measured: planted melts in real footage were found in 4 of 5 clips, within 2 frames of the onset. It has
not been tested on real AI output. Handheld phone video gets about one false alarm per clip (a fast pan
over a blank wall looks the same to optical flow), so it is opt-in: index_video(glitch=True) for
AI-generated clips, then confirm each suspect with peek.
"""
from __future__ import annotations

import numpy as np

from .. import store
from ..store import Event, VideoInfo
from . import AnalyzerResult, Context

LONG_SIDE = 192
GRID = 8
Z_THRESH = 6.0
MIN_RESIDUAL = 12.0   # gray levels; below this nobody can see it
MIN_FRAMES = 3
EDGE_GUARD = 3        # frames either side of a cut are unreliable
MAX_PAN_PX = 2.0      # typical movement per frame at LONG_SIDE; above this the camera is whipping


def residual(prev, cur, cv2) -> tuple[float, float]:
    """-> (worst block of unexplained change, typical movement in pixels) for two gray frames."""
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 15, 2, 5, 1.1, 0)
    h, w = prev.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(prev, gx - flow[..., 0], gy - flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.abs(warped.astype(np.float32) - cur.astype(np.float32))
    err = err - np.median(err)  # global changes (exposure, fades) move everything equally
    bh, bw = h // GRID, w // GRID
    blocks = err[: bh * GRID, : bw * GRID].reshape(GRID, bh, GRID, bw).mean((1, 3))
    return float(blocks.max()), float(np.median(np.hypot(flow[..., 0], flow[..., 1])))


def find_suspect(res: np.ndarray, pan: np.ndarray | None = None) -> tuple[int, float] | None:
    """res: per-frame residual inside one shot -> (first bad index, score), or None.

    pan: per-frame typical movement. When the whole frame is racing (a whip pan), motion blur defeats
    the flow estimate and the leftover means nothing. A morph is local: most of the frame holds still.
    """
    n = len(res)
    if n < 12:
        return None
    body = res[EDGE_GUARD: n - EDGE_GUARD]
    med = float(np.median(body))
    mad = float(np.median(np.abs(body - med))) * 1.4826 + 0.5
    z = (res - med) / mad
    bad = (z >= Z_THRESH) & (res >= MIN_RESIDUAL)
    if pan is not None:
        racing = np.convolve(pan > MAX_PAN_PX, np.ones(5), mode="same") > 0  # blur outlasts the peak
        bad &= ~racing
    bad[:EDGE_GUARD] = bad[n - EDGE_GUARD:] = False
    run = 0
    for i, b in enumerate(bad):
        run = run + 1 if b else 0
        if run >= MIN_FRAMES:
            start = i - run + 1
            score = round(float(z[start: i + 1].mean()), 1)
            while start > EDGE_GUARD and z[start - 1] >= 2.0:  # the visible onset starts earlier than the alarm
                start -= 1
            return start, score
    return None


class Glitch:
    name = "glitch"
    optional = True  # runs with index_video(glitch=True)

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        import cv2

        from .shots import frame_size, iter_frames

        fps = video.fps
        w, h = frame_size(video.width, video.height, LONG_SIDE)
        res = []
        prev = None
        for frame in iter_frames(video.path, fps, w, h):
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res.append((0.0, 0.0) if prev is None else residual(prev, g, cv2))
            prev = g
        r, pan = np.asarray(res, np.float32).reshape(-1, 2).T
        events, updates = [], {}
        for s in store.get_shots(video.id):
            i, j = int(s.start * fps), int(s.end * fps)
            hit = find_suspect(r[i:j], pan[i:j])
            if hit is None:
                updates[s.idx] = {"glitch": None}
                continue
            k, score = hit
            t = (i + k) / fps
            updates[s.idx] = {"glitch": {"score": score, "start": round(t, 2),
                                         "last_clean": round(max(s.start, t - 1 / fps), 2)}}
            events.append(Event("glitch", round(t, 2), round(min(s.end, t + 0.5), 2), score))
        store.merge_shot_detail(video.id, updates)
        store.put_events(video.id, ["glitch"], events)
        return AnalyzerResult(info={"suspects": len(events)})
