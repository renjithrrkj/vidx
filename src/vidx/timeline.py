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
         "overlays": [{"image": "/path/logo.png", "at": 0.5, "for": 2.0, "pos": "top_right", "width": 0.2,
                       "opacity": 1.0, "fade": 0.3, "motion": "slide_left"}],   # at = seconds from the clip start
         "transition": {"type": "whip_left"}},  # into the NEXT clip: any name from vidx.cuts; omit = hard cut
        {"image": "/path/photo.jpg", "for": 3.0, "motion": "zoom_in", "fit": "cover"}   # a still as a clip
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

import hashlib
import json
import re
from pathlib import Path

from . import config, cuts, store
from .store import VideoInfo, VidxError

ASSETS = Path(__file__).parent / "assets" / "sfx"
CLIP_KEYS = {"in", "out", "video", "speed", "crop", "grade", "captions", "transition", "zoom", "overlays",
             "image", "asset", "for", "motion", "fit"}
GRADE_KEYS = {"brightness", "contrast", "saturation", "match"}
OVERLAY_KEYS = {"image", "asset", "at", "for", "pos", "width", "opacity", "fade", "motion"}
STILL_MOTIONS = ("none", "zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down")
STILL_FITS = ("cover", "contain")
OVERLAY_MOTIONS = ("none", "slide_left", "slide_right", "slide_up", "slide_down")
POSITIONS = ("top_left", "top", "top_right", "left", "center", "right", "bottom_left", "bottom", "bottom_right")
MAX_IMAGE_SIDE = 3840   # larger photos only cost memory: nothing renders above this
MAX_STILL_SECONDS = 600
STILL_SECONDS = 3.0     # a still with no length


def asset_path(asset: str) -> Path:
    return config.HOME / "images" / f"{asset}.png"


def prepare_image(path: str, asset: str | None = None, where: str = "image") -> tuple[str, Path, int, int]:
    """Copy an image into vidx as an upright, size-capped PNG. -> (asset id, its path, width, height).
    Keeping our own copy means phone photos aren't sideways (ffmpeg ignores EXIF rotation), any format Pillow
    reads works, and an edit keeps rendering after the original moves. asset: reuse an earlier copy."""
    from PIL import Image, ImageOps

    src = Path(str(path)).expanduser()
    if not src.exists():
        if asset and asset_path(asset).exists():
            with Image.open(asset_path(asset)) as im:
                return asset, asset_path(asset), *im.size
        raise VidxError(f"{where}: image not found: {src}")
    if not src.is_file():
        raise VidxError(f"{where}: not a file: {src}")
    key = hashlib.sha1(src.read_bytes()).hexdigest()[:16]
    out = asset_path(key)
    if out.exists():
        with Image.open(out) as im:
            return key, out, *im.size
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info else "RGB")
            im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.png")
            im.save(tmp, "PNG")
            tmp.rename(out)
            return key, out, *im.size
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise VidxError(f"{where}: can't read {src.name} as an image (use jpg, png or webp): {e}")


def still(raw: dict, where: str, fps: float) -> dict:
    """A still image as a clip. It joins the timeline as its own silent source, so transitions, captions,
    checks and the cache treat it like any other clip."""
    asset, file, w, h = prepare_image(raw["image"], raw.get("asset"), where)
    try:
        length = float(raw["for"] if raw.get("for") is not None else
                       float(raw["out"]) - float(raw["in"]) if "out" in raw else STILL_SECONDS)
    except (TypeError, ValueError, KeyError):
        raise VidxError(f'{where}: "for" must be seconds')
    if not 0.1 <= length <= MAX_STILL_SECONDS:
        raise VidxError(f'{where}: "for" must be 0.1-{MAX_STILL_SECONDS} seconds')
    if raw.get("crop"):
        raise VidxError(f'{where}: a still has no crop; use "fit": cover or contain')
    motion, fit = raw.get("motion") or "none", raw.get("fit") or "cover"
    if motion not in STILL_MOTIONS:
        raise VidxError(f"{where}: unknown motion {motion!r} (use {', '.join(STILL_MOTIONS)})")
    if fit not in STILL_FITS:
        raise VidxError(f"{where}: unknown fit {fit!r} (use {', '.join(STILL_FITS)})")
    vid = f"img{asset[:10]}"
    store.put_video(VideoInfo(vid, str(file), asset, 86400.0, fps, w, h,
                              {"has_audio": False, "image": True, "source": str(raw["image"])}))
    return {**raw, "image": str(raw["image"]), "asset": asset, "video": vid, "in": 0.0, "out": round(length, 3),
            "for": round(length, 3), "speed": 1.0, "motion": motion, "fit": fit}


def overlay(raw: dict, where: str) -> dict:
    """Check one image layer. `at` may be negative: the layer started in an earlier clip and carries on."""
    if not isinstance(raw, dict) or not raw.get("image"):
        raise VidxError(f'{where}: an overlay needs "image"')
    extra = set(raw) - OVERLAY_KEYS
    if extra:
        raise VidxError(f"{where}: unknown overlay key {sorted(extra)[0]!r} (use {', '.join(sorted(OVERLAY_KEYS))})")
    asset = prepare_image(raw["image"], raw.get("asset"), where)[0]
    try:
        o = {"image": str(raw["image"]), "asset": asset, "at": float(raw.get("at") or 0), "for": float(raw.get("for", 2.0)),
             "pos": raw.get("pos") or "top_right", "width": float(raw.get("width", 0.25)),
             "opacity": float(raw.get("opacity", 1.0)), "fade": float(raw.get("fade") or 0),
             "motion": raw.get("motion") or "none"}
    except (TypeError, ValueError):
        raise VidxError(f"{where}: overlay at, for, width, opacity and fade are numbers")
    pos = o["pos"]
    if isinstance(pos, dict):
        pos = o["pos"] = [pos.get("x"), pos.get("y")]
    if isinstance(pos, str):
        if pos not in POSITIONS:
            raise VidxError(f"{where}: unknown pos {pos!r} (use {', '.join(POSITIONS)}, or [x, y] as 0-1 of the frame)")
    elif not (isinstance(pos, (list, tuple)) and len(pos) == 2 and all(isinstance(p, (int, float)) and 0 <= p <= 1 for p in pos)):
        raise VidxError(f"{where}: pos is a name or [x, y] as 0-1 of the frame")
    if o["for"] <= 0:
        raise VidxError(f'{where}: overlay "for" must be above 0')
    if not 0 < o["width"] <= 1:
        raise VidxError(f"{where}: overlay width is a share of the frame, 0-1")
    if not 0 < o["opacity"] <= 1:
        raise VidxError(f"{where}: overlay opacity must be 0-1")
    if not 0 <= o["fade"] <= o["for"] / 2:
        raise VidxError(f"{where}: overlay fade must fit twice in its {o['for']}s")
    if o["motion"] not in OVERLAY_MOTIONS:
        raise VidxError(f"{where}: unknown overlay motion {o['motion']!r} (use {', '.join(OVERLAY_MOTIONS)})")
    return o


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
        c = still(raw, where, v.fps) if raw.get("image") else dict(raw)
        src = store.get_video(c.get("video") or main)
        try:
            c["in"], c["out"] = float(c["in"]), float(c["out"])
        except (KeyError, TypeError, ValueError):
            raise VidxError(f'{where}: needs numeric "in" and "out" seconds, or "image" for a still')
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
        if c.get("overlays"):
            c["overlays"] = [overlay(o, where) for o in c["overlays"]]
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
