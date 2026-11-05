"""Per-shot keyframes (start / middle / end) and on-screen text + watermark positions.

Text comes from RapidOCR, whose models ship inside its wheel (offline). Text seen in the same place
across most of the video is a watermark; a logo with no readable text is found as edges that stay
put while the shots around them change. Positions are 0-1 fractions of the frame.
"""
from __future__ import annotations

import difflib
from collections import Counter

import numpy as np

from .. import store, thumbs
from ..store import VideoInfo
from . import AnalyzerResult, Context

KEYFRAME_SIDE = 384
OCR_SIDE = 960
MAX_OCR_SAMPLES = 120
WATERMARK_SHARE = 0.3   # OCR misses faint marks on dark frames, so a third of samples is already a lot
MIN_SCORE = 0.8


def keyframe_times(start: float, end: float, fps: float) -> list[float]:
    inset = min(2.0 / fps, (end - start) / 4)
    return [start + inset, (start + end) / 2, end - inset]


def ocr_times(shots, duration: float, cap: int = MAX_OCR_SAMPLES) -> list[float]:
    """Each shot's middle, plus one sample every 10s inside long takes."""
    ts: list[float] = []
    for s in shots:
        n = max(1, int(s.dur // 10))
        ts += [s.start + s.dur * (i + 0.5) / n for i in range(n)]
    if len(ts) > cap:
        ts = [ts[i] for i in np.linspace(0, len(ts) - 1, cap).round().astype(int)]
    return ts


def _similar(a: str, b: str) -> bool:
    """OCR spells a small watermark differently on every frame, and often reads only part of it."""
    return a in b or b in a or difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.6


def cluster_text(hits: list[tuple], n_samples: int, step: float, multi_shot: bool = True) -> list[tuple]:
    """hits: (t, x, y, w, h, text) -> overlay rows, grouping the same text in the same place."""
    groups: list[dict] = []
    for t, x, y, w, h, text in sorted(hits):
        key = "".join(text.lower().split())
        for g in groups:
            if abs(g["x"] - x) < 0.05 and abs(g["y"] - y) < 0.05 and _similar(g["key"], key):
                g["ts"].append(t)
                g["names"][text] += 1
                g["w"], g["h"] = max(g["w"], w), max(g["h"], h)
                break
        else:
            groups.append({"key": key, "x": x, "y": y, "w": w, "h": h, "ts": [t], "names": Counter({text: 1})})
    rows = []
    for g in groups:
        share = len(set(g["ts"])) / max(n_samples, 1)
        kind = "watermark" if share >= WATERMARK_SHARE and n_samples >= 4 and multi_shot else "text"
        rows.append((kind, round(max(min(g["ts"]) - step / 2, 0.0), 2), round(max(g["ts"]) + step / 2, 2),
                     round(g["x"], 3), round(g["y"], 3), round(g["w"], 3), round(g["h"], 3),
                     g["names"].most_common(1)[0][0]))
    return rows


def static_logo_boxes(edge_maps: list[np.ndarray], keep: float = 0.8, max_area: float = 0.06) -> list[tuple]:
    """Boxes (x, y, w, h fractions) of edges that persist across samples from DIFFERENT shots."""
    import cv2

    if len(edge_maps) < 6:
        return []
    persist = (np.mean(np.stack(edge_maps) > 0, axis=0) >= keep).astype(np.uint8)
    persist = cv2.dilate(persist, np.ones((3, 3), np.uint8))
    h, w = persist.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(persist)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_edge = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        if touches_edge or bw * bh > max_area * w * h or area < 30:
            continue  # letterbox lines reach the frame edge; huge regions are static backgrounds
        boxes.append((float(x / w), float(y / h), float(bw / w), float(bh / h)))
    return boxes


class Detail:
    name = "detail"
    optional = False

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        import cv2
        from rapidocr_onnxruntime import RapidOCR

        shots = store.get_shots(video.id)
        kdir = video.dir / "keyframes"
        kdir.mkdir(exist_ok=True)
        updates: dict[int, dict] = {}
        edge_maps = []
        for s in shots:
            ts = keyframe_times(s.start, s.end, video.fps)
            frames = thumbs.grab_frames(video.path, ts, long_side=KEYFRAME_SIDE, workers=3)
            saved = []
            for tag, t, f in zip("ame", ts, frames):
                if f is None:
                    continue
                f.save(kdir / f"s{s.idx}_{tag}.jpg", quality=82)
                saved.append(round(t, 2))
            updates[s.idx] = {"keyframes": saved}
            if frames[1] is not None and s.dur >= 0.5:
                g = cv2.cvtColor(np.asarray(frames[1]), cv2.COLOR_RGB2GRAY)
                edge_maps.append(cv2.Canny(g, 80, 160))
        store.merge_shot_detail(video.id, updates)

        engine = RapidOCR()
        ts = ocr_times(shots, video.duration)
        step = video.duration / max(len(ts), 1)
        hits: list[tuple] = []
        for i in range(0, len(ts), 8):  # small batches: never hold many full frames at once
            batch = ts[i:i + 8]
            for t, f in zip(batch, thumbs.grab_frames(video.path, batch, long_side=OCR_SIDE, workers=2)):
                if f is None:
                    continue
                arr = np.asarray(f)[..., ::-1]
                res, _ = engine(arr)
                H, W = arr.shape[:2]
                for box, text, score in res or []:
                    if float(score) < MIN_SCORE or sum(ch.isalnum() for ch in text) < 3:
                        continue
                    b = np.asarray(box, np.float32)
                    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()
                    hits.append((t, float(x0 / W), float(y0 / H), float((x1 - x0) / W), float((y1 - y0) / H), text.strip()))
        rows = cluster_text(hits, len(ts), step, multi_shot=len(shots) >= 4)  # in one take, room text never moves either
        if len(shots) >= 6:  # a static background in a one-take video would look like a logo
            placed = [(r[3], r[4], r[5], r[6]) for r in rows if r[0] == "watermark"]
            for x, y, w, h in static_logo_boxes(edge_maps):
                if not any(abs(px - x) < 0.08 and abs(py - y) < 0.08 for px, py, _, _ in placed):
                    rows.append(("watermark", 0.0, round(video.duration, 2), round(x, 3), round(y, 3),
                                 round(w, 3), round(h, 3), ""))
        store.put_overlays(video.id, rows)
        marks = [r for r in rows if r[0] == "watermark"]
        store.update_meta(video.id, watermarks=[{"box": list(r[3:7]), "text": r[7]} for r in marks])
        return AnalyzerResult(info={"keyframes": sum(len(u["keyframes"]) for u in updates.values()),
                                    "text": len(rows) - len(marks), "watermarks": len(marks)})
