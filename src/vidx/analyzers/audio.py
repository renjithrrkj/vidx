"""silence / loud / black / freeze events from ONE ffmpeg decode pass."""
from __future__ import annotations

import re
import subprocess
import sys

from .. import config, store
from ..store import Event, VideoInfo
from . import AnalyzerResult, Context

KINDS = ["silence", "loud", "black", "freeze"]
_num = r"(-?[\d.]+(?:e-?\d+)?)"


def detect_cmd(path: str, has_audio: bool, hwaccel: bool = True, video: bool = True,
               black_min: float = config.BLACK_MIN, freeze_min: float = config.FREEZE_MIN) -> list[str]:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats"]
    if hwaccel and sys.platform == "darwin":
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", path]
    if video:
        cmd += ["-vf", f"scale=320:-2,cropdetect=limit=24:round=2:reset=1,blackdetect=d={black_min}:pix_th=0.10,"
                       f"freezedetect=n=-60dB:d={freeze_min}"]
    else:
        cmd += ["-vn"]
    if has_audio:
        cmd += ["-af", f"silencedetect=n={config.SILENCE_DB}dB:d={config.SILENCE_MIN},ebur128=framelog=info"]
    else:
        cmd += ["-an"]
    return cmd + ["-f", "null", "-"]


def parse_detect_log(log: str, duration: float) -> tuple[list[Event], dict]:
    ev: list[Event] = []
    # black
    for m in re.finditer(rf"black_start:{_num} black_end:{_num} black_duration:{_num}", log):
        a, b, d = map(float, m.groups())
        ev.append(Event("black", a, b, round(d, 2)))
    # freeze (start / end on separate lines)
    starts = [float(x) for x in re.findall(rf"freeze_start: {_num}", log)]
    ends = [float(x) for x in re.findall(rf"freeze_end: {_num}", log)]
    for i, a in enumerate(starts):
        b = ends[i] if i < len(ends) else duration
        ev.append(Event("freeze", a, b, round(b - a, 2)))
    # silence
    s_starts = [float(x) for x in re.findall(rf"silence_start: {_num}", log)]
    s_ends = [float(x) for x in re.findall(rf"silence_end: {_num}", log)]
    for i, a in enumerate(s_starts):
        a = max(a, 0.0)
        b = s_ends[i] if i < len(s_ends) else duration
        ev.append(Event("silence", a, b, round(b - a, 2)))
    # loudness: momentary every 100ms + integrated summary
    mom = [(float(t), float(m)) for t, m in re.findall(rf"t: {_num}\s+TARGET:\S+ LUFS\s+M:\s*{_num}", log)]
    integ = re.findall(rf"Integrated loudness:\s*I:\s*{_num} LUFS", log)
    lufs = float(integ[-1]) if integ else None
    if mom and lufs is not None and lufs > -70:
        thr = lufs + config.LOUD_OVER_INTEGRATED
        run_a, peak, last = None, -200.0, None
        for t, m in mom + [(duration + 1, -200.0)]:
            t0 = t - 0.4  # momentary window is 400ms ending at t
            if m >= thr:
                if run_a is None:
                    run_a, peak = max(t0, 0.0), m
                peak = max(peak, m)
                last = t
            elif run_a is not None and (last is None or t - last > 0.3):
                if last - run_a >= 0.3:
                    ev.append(Event("loud", round(run_a, 2), round(last, 2), round(peak, 1)))
                run_a = None
    ev.sort(key=lambda e: (e.start, e.kind))
    return ev, {"lufs": lufs, "crop320": dominant_crop(log)}


def dominant_crop(log: str, min_share: float = 0.6) -> list[int] | None:
    """Per-frame cropdetect -> the letterbox most frames agree on. Title cards / credits that fill
    the frame shouldn't cancel a film's 2.35:1 bars, so height and width are voted separately."""
    from collections import Counter

    crops = [tuple(map(int, m)) for m in re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", log)]
    if len(crops) < 10:
        return None
    n = len(crops)
    (h, hc), = Counter(c[1] for c in crops).most_common(1)
    (w, wc), = Counter(c[0] for c in crops).most_common(1)
    if hc / n < min_share or wc / n < min_share:
        return None
    (y, _), = Counter(c[3] for c in crops if c[1] == h).most_common(1)
    (x, _), = Counter(c[2] for c in crops if c[0] == w).most_common(1)
    return [w, h, x, y]


def active_area(crop320: list[int] | None, width: int, height: int) -> list[int] | None:
    """Baked-in letterbox / pillarbox in display pixels [w, h, x, y], or None if the frame is full.
    cropdetect ran on a 320px-wide copy, so scale back up."""
    if not crop320:
        return None
    f = width / 320
    w, h, x, y = (int(round(v * f / 2)) * 2 for v in crop320)
    w, h = min(w, width), min(h, height)
    if w < width * 0.3 or h < height * 0.3:  # mostly black video: don't trust it
        return None
    if w >= width * 0.97 and h >= height * 0.97:
        return None
    return [w, h, x, y]


class AudioEvents:
    name = "audio"
    optional = False

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        def go(hw: bool) -> subprocess.CompletedProcess:
            return subprocess.run(detect_cmd(video.path, video.has_audio, hw), capture_output=True, text=True)

        p = go(True)
        if p.returncode != 0:
            p = go(False)
        events, info = parse_detect_log(p.stderr, video.duration)
        store.put_events(video.id, KINDS, events)
        store.update_meta(video.id, lufs=info["lufs"],
                          active=active_area(info.get("crop320"), video.width, video.height))
        counts = {k: sum(e.kind == k for e in events) for k in KINDS}
        info.pop("crop320", None)
        return AnalyzerResult(info={**info, **counts})
