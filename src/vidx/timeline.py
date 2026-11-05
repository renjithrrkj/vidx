"""The edit as a file: one JSON document that lists clips, transitions, grade, captions and audio.

    {
      "video": "ca833113",                      # default source for clips
      "clips": [
        {"in": 12.0, "out": 18.5,
         "video": "other_id",                   # optional, another indexed video
         "speed": 1.0, "crop": "9:16",
         "grade": {"brightness": 0.02, "contrast": 1.05, "saturation": 1.1,
                   "match": {"t": 40.0, "video": "ca833113", "strength": 0.8}},
         "captions": [{"at": 0.5, "for": 2.0, "text": "Hello", "pos": "bottom"}],
         "zoom": 1.1,                           # optional punch-in
         "transition": {"type": "whip_left"}}   # into the NEXT clip: any name from vidx.cuts; omit = hard cut
      ],
      "audio": [
        {"sfx": "impact_01", "at": 3.2, "gain": -6},    # at = output time
        {"music": "drone_02", "gain": -18},             # loops under the whole edit
        {"file": "/path/voice.wav", "at": 0, "gain": 0}
      ]
    }

Every edit, whether written as DSL or JSON, compiles to this. `segments`, `speed`, `joins` are kept
beside `clips` so checks, diff and the OTIO export read one shape.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import cuts, store
from .store import VidxError

ASSETS = Path(__file__).parent / "assets" / "sfx"
CLIP_KEYS = {"in", "out", "video", "speed", "crop", "grade", "captions", "transition", "zoom"}
GRADE_KEYS = {"brightness", "contrast", "saturation", "match"}


def sfx_names() -> list[str]:
    return sorted(p.stem for p in ASSETS.glob("*.m4a"))


def sfx_path(name: str) -> Path:
    p = ASSETS / f"{name}.m4a"
    if not p.exists():
        names = sfx_names()
        raise VidxError(f"unknown sound {name!r} (have {', '.join(names[:8])}{' …' if len(names) > 8 else ''})")
    return p


def joins_of(clips: list[dict]) -> list[float]:
    """Transition length at each join (0 = hard cut); one entry per pair of clips."""
    return [cuts.join_len(c.get("transition")) for c in clips[:-1]]


def out_length(clip: dict) -> float:
    return (clip["out"] - clip["in"]) / float(clip.get("speed") or 1.0)


def out_duration(clips: list[dict]) -> float:
    return sum(out_length(c) for c in clips) - sum(joins_of(clips))


def finish(tl: dict) -> dict:
    """Fill the derived fields every reader relies on."""
    clips = tl["clips"]
    tl["segments"] = [[c["in"], c["out"]] for c in clips]
    tl["joins"] = joins_of(clips)
    tl["speeds"] = [float(c.get("speed") or 1.0) for c in clips]
    tl["out_duration"] = round(out_duration(clips), 3)
    return tl


def clamp(clips: list[dict], warnings: list[str]) -> None:
    """A transition eats into both neighbours, and both must survive it."""
    for i, c in enumerate(clips[:-1]):
        t = c.get("transition")
        d = cuts.join_len(t)
        if d:
            room = min(out_length(c), out_length(clips[i + 1])) / 2 - 0.01
            if d > room:
                warnings.append(f"clip {i + 1}: {t.get('type', 'transition')} {d}s too long for its neighbours; using {max(room, 0.04):.2f}s")
                d = max(room, 0.04)
            c["transition"] = {**t, "dur": round(d, 3)}


def validate(data: dict, default_video: str) -> dict:
    """Parsed JSON -> checked timeline. Errors name the clip and the fix."""
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list) or not data["clips"]:
        raise VidxError('timeline needs a non-empty "clips" list')
    main = data.get("video") or default_video
    v = store.get_video(main)
    clips = []
    for i, raw in enumerate(data["clips"], 1):
        where = f"clip {i}"
        if not isinstance(raw, dict):
            raise VidxError(f"{where}: must be an object")
        extra = set(raw) - CLIP_KEYS
        if extra:
            raise VidxError(f"{where}: unknown key {sorted(extra)[0]!r} (use {', '.join(sorted(CLIP_KEYS))})")
        c = dict(raw)
        src = store.get_video(c.get("video") or main)
        try:
            c["in"], c["out"] = float(c["in"]), float(c["out"])
        except (KeyError, TypeError, ValueError):
            raise VidxError(f'{where}: needs numeric "in" and "out" seconds')
        if not 0 <= c["in"] < c["out"]:
            raise VidxError(f"{where}: in {c['in']} must be before out {c['out']}")
        if c["in"] >= src.duration:
            raise VidxError(f"{where}: in {c['in']} is past the end of {src.id} ({src.duration:.1f}s)")
        c["out"] = min(c["out"], src.duration)
        c["video"] = src.id
        sp = float(c.get("speed") or 1.0)
        if not 0.25 <= sp <= 4:
            raise VidxError(f"{where}: speed must be 0.25-4.0")
        c["speed"] = sp
        if c.get("crop") and not re.fullmatch(r"\d+:\d+", str(c["crop"])):
            raise VidxError(f"{where}: crop is a ratio like 9:16")
        g = c.get("grade") or {}
        if set(g) - GRADE_KEYS:
            raise VidxError(f"{where}: unknown grade key {sorted(set(g) - GRADE_KEYS)[0]!r} (use {', '.join(sorted(GRADE_KEYS))})")
        for cap in c.get("captions") or []:
            if not str(cap.get("text") or "").strip() or "at" not in cap:
                raise VidxError(f'{where}: captions need "text" and "at" (seconds from the clip start)')
        t = c.get("transition")
        if t and i == len(data["clips"]):
            raise VidxError(f"{where}: the last clip has nothing to transition into")
        if t:
            cuts.get(t.get("type", "dissolve"))
        clips.append(c)

    warnings: list[str] = []
    clamp(clips, warnings)
    audio = []
    for j, a in enumerate(data.get("audio") or [], 1):
        kinds = [k for k in ("sfx", "music", "file") if a.get(k)]
        if len(kinds) != 1:
            raise VidxError(f'audio {j}: use exactly one of "sfx", "music", "file"')
        if a.get("sfx"):
            sfx_path(a["sfx"])
        if a.get("music"):
            sfx_path(a["music"])
        if a.get("file") and not Path(a["file"]).expanduser().exists():
            raise VidxError(f"audio {j}: file not found: {a['file']}")
        audio.append(a)
    return finish({"video_id": v.id, "source_duration": round(v.duration, 3), "fps": v.fps, "clips": clips,
                   "audio": audio, "warnings": warnings, "trans": None, "crop": None, "speed": 1.0,
                   "snap": False, "pad": 0})


def load(text: str, default_video: str) -> dict:
    """text is JSON, or @/path/to/timeline.json."""
    text = text.strip()
    if text.startswith("@"):
        p = Path(text[1:]).expanduser()
        if not p.exists():
            raise VidxError(f"timeline file not found: {p}")
        text = p.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise VidxError(f"timeline is not valid JSON: line {e.lineno}: {e.msg}")
    return validate(data, default_video)


def is_timeline(text: str) -> bool:
    t = text.strip()
    return t.startswith("{") or (t.startswith("@") and "\n" not in t)


def export(tl: dict) -> dict:
    """The editable part of a stored timeline, ready to change and pass back to edit()."""
    return {"video": tl["video_id"], "clips": tl["clips"], "audio": tl.get("audio") or []}
