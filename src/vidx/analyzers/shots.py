"""Shot boundaries via scenedetect ContentDetector, fed by a low-res ffmpeg pipe.

Decoding through ffmpeg (hardware-accelerated where available, scaled to ~256px) is several
times faster than scenedetect's own OpenCV backend on 1080p sources.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from .. import config, store
from ..store import Shot, VideoInfo
from . import AnalyzerResult, Context

LONG_SIDE = 256


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
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        err = p.stderr.read().decode(errors="replace")
        p.wait()
        if p.returncode not in (0, None) and hwaccel:
            raise RuntimeError(err)


class Shots:
    name = "shots"
    optional = False

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        from scenedetect.common import FrameTimecode
        from scenedetect.detectors import ContentDetector

        fps = video.fps
        w, h = frame_size(video.width, video.height)

        def detect(hw: bool):
            det = ContentDetector(threshold=config.SHOT_THRESHOLD, min_scene_len=max(2, round(fps * 0.5)))
            cuts: list[int] = []
            scores: list[float] = []
            i = -1
            for i, frame in enumerate(iter_frames(video.path, fps, w, h, hwaccel=hw)):
                for tc in det.process_frame(FrameTimecode(i, fps), frame):
                    cuts.append(tc.frame_num)
                scores.append(float(det._frame_score or 0.0))
            if i >= 0:
                cuts += [tc.frame_num for tc in det.post_process(FrameTimecode(i, fps))]
            return cuts, scores, i + 1

        try:
            cuts, scores, nframes = detect(True)
        except RuntimeError:
            cuts, scores, nframes = detect(False)  # hw decode unsupported for this codec

        dur = video.duration
        bounds = [0.0] + [c / fps for c in sorted(set(cuts)) if 0 < c / fps < dur] + [dur]
        shots = []
        sc = np.asarray(scores) if scores else np.zeros(1)
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            seg = sc[int(a * fps) + 1: max(int(a * fps) + 2, int(b * fps))]  # skip the cut frame itself
            shots.append(Shot(idx=k + 1, start=round(a, 3), end=round(b, 3),
                              motion=round(float(seg.mean()) if seg.size else 0.0, 2)))
        store.put_shots(video.id, shots)
        return AnalyzerResult(info={"shots": len(shots), "frames": nframes})
