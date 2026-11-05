"""timeline -> finished video, one cached piece at a time.

Each clip body and each transition is rendered once into ~/.vidx/cache, keyed by everything that
affects its pixels and sound. Changing one shot re-encodes only that shot; the final file is a stream
copy of the pieces plus one audio pass (mix, loudness). `preview=True` does the same at 480p.

Pieces hold H.264 + PCM in Matroska. Keeping audio uncompressed until the last step avoids the small
gaps AAC leaves at every segment boundary, which would be audible across hundreds of cuts.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import color, config, cuts, store, timeline, util
from .store import VidxError

PRESETS = {
    # name: (width, height, target LUFS or None, max duration or None)
    "original": (None, None, None, None),
    "reel_9x16": (1080, 1920, -16.0, 90.0),
    "yt_16x9": (1920, 1080, -14.0, None),
}
PREVIEW_SHORT_SIDE = 480
PIECE_FORMAT = 4  # bump when piece encoding changes, so old cache entries aren't reused


def frame_snap(segs: list[list[float]], fps: float) -> list[tuple[float, float]]:
    """Cut on frame boundaries so audio (sample-accurate) and video (frame-accurate) stay in sync."""
    out = []
    for a, b in segs:
        fa, fb = round(a * fps), round(b * fps)
        if fb > fa:
            out.append((fa / fps, fb / fps))
    return out


def crop_expr(ratio: str) -> str:
    rw, rh = map(int, ratio.split(":"))
    return f"crop='trunc(min(iw,ih*{rw}/{rh})/2)*2':'trunc(min(ih,iw*{rh}/{rw})/2)*2'"


def fit_expr(w: int, h: int, src_w: int, src_h: int) -> str:
    """Fit to WxH: center-crop when orientation matches (or for vertical targets), pad when a
    vertical source goes to a horizontal target (cropping would lose ~70% of the frame)."""
    if src_h > src_w and not h > w:
        return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"


def atempo_chain(speed: float) -> str:
    parts, s = [], speed
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


# ---------------------------------------------------------------- planning (pure)


@dataclass
class Part:
    """A slice of one clip. Times are source seconds; `offset` is where it starts in the clip's output."""
    clip: int
    a: float
    b: float
    frames: int
    offset: float


@dataclass
class Piece:
    kind: str  # body | join
    parts: list[Part]
    frames: int
    join: float = 0.0
    effect: str = "fade"
    path: Path | None = None
    fresh: bool = False


def plan_pieces(clips: list[dict], fps: float) -> list[Piece]:
    """Bodies and transitions in output order. A transition of J frames takes the last J frames of
    one clip and the first J of the next, so body frames = clip frames minus both neighbours' joins."""
    n = len(clips)
    frames = [max(1, round(timeline.out_length(c) * fps)) for c in clips]
    joins = [round(cuts.join_len(c.get("transition")) * fps) for c in clips[:-1]] + [0]
    pieces: list[Piece] = []
    for i, c in enumerate(clips):
        sp = float(c.get("speed") or 1.0)
        head = joins[i - 1] if i else 0
        tail = joins[i]
        a0 = round(c["in"] * fps) / fps

        def src(f: int) -> float:  # source time of output frame f of this clip
            return a0 + f / fps * sp

        body = frames[i] - head - tail
        if body > 0:
            pieces.append(Piece("body", [Part(i, src(head), src(head + body), body, head / fps)], body))
        if tail:
            nxt = clips[i + 1]
            nsp = float(nxt.get("speed") or 1.0)
            b0 = round(nxt["in"] * fps) / fps
            pieces.append(Piece("join", [
                Part(i, src(frames[i] - tail), src(frames[i]), tail, (frames[i] - tail) / fps),
                Part(i + 1, b0, b0 + tail / fps * nsp, tail, 0.0)], tail, tail / fps,
                cuts.get(c["transition"].get("type", "dissolve")).xfade))
    return pieces


def output_size(tl: dict, preset: str, preview: bool) -> tuple[int, int]:
    W, H = PRESETS[preset][:2]
    if not W:
        c = tl["clips"][0]
        v = store.get_video(c["video"])
        W, H = v.width, v.height
        ratio = c.get("crop") or tl.get("crop")
        if ratio:
            rw, rh = map(int, ratio.split(":"))
            W, H = min(W, H * rw / rh), min(H, W * rh / rw)
    if preview:
        s = PREVIEW_SHORT_SIDE / min(W, H)
        if s < 1:
            W, H = W * s, H * s
    return int(W) // 2 * 2, int(H) // 2 * 2


# ---------------------------------------------------------------- captions


def caption_png(cache: Path, cap: dict, W: int, H: int) -> Path:
    """Full-frame transparent PNG with the caption drawn on it. Pillow, because many ffmpeg builds
    (Homebrew's included) ship without drawtext."""
    from PIL import Image, ImageDraw

    from .thumbs import caption_font

    text, pos = str(cap["text"]), cap.get("pos", "bottom")
    key = hashlib.sha1(f"{text}|{pos}|{W}x{H}|{cap.get('size')}".encode()).hexdigest()[:16]
    p = cache / f"cap_{key}.png"
    if p.exists():
        return p
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    size = int(cap.get("size") or max(18, min(W, H) // 16))
    font = caption_font(size)
    words, lines, cur = text.split(), [], ""
    for w in words:  # wrap to 86% of the width
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=font) > W * 0.86 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    lines.append(cur)
    lh = int(size * 1.25)
    y = {"top": int(H * 0.08), "center": (H - lh * len(lines)) // 2}.get(pos, int(H * 0.86) - lh * len(lines))
    for ln in lines:
        x = (W - d.textlength(ln, font=font)) / 2
        d.text((x, y), ln, font=font, fill=(255, 255, 255, 255), stroke_width=max(2, size // 12), stroke_fill=(0, 0, 0, 255))
        y += lh
    im.save(p)
    return p


# ---------------------------------------------------------------- filter chains


@dataclass
class Spec:
    W: int
    H: int
    fps: float
    preset: str
    preview: bool
    cache: Path
    videos: dict = field(default_factory=dict)

    def video(self, vid: str):
        if vid not in self.videos:
            v = store.get_video(vid)
            if not os.path.exists(v.path):
                raise VidxError(f"source file missing: {v.path} (re-run index_video on its new path)")
            self.videos[vid] = v
        return self.videos[vid]


def grade_filters(c: dict, spec: Spec) -> list[str]:
    g = c.get("grade") or {}
    out = []
    eq = []
    if g.get("brightness"):
        eq.append(f"brightness={float(g['brightness']):.3f}")
    if g.get("contrast"):
        eq.append(f"contrast={float(g['contrast']):.3f}")
    if g.get("saturation"):
        eq.append(f"saturation={float(g['saturation']):.3f}")
    if eq:
        out.append("eq=" + ":".join(eq))
    m = g.get("match")
    if m:
        ref = spec.video(m.get("video") or c["video"])
        if "from" in m:
            ra, rb = float(m["from"]), float(m["to"])
        else:
            ra, rb = float(m["t"]) - 0.5, float(m["t"]) + 0.5
        def region(v, a, b):
            active = v.meta.get("active") if PRESETS[spec.preset][0] else None
            return {"path": v.path, "a": a, "b": b, "active": active, "size": (v.width, v.height)}

        lut = color.match_lut(spec.cache, region(spec.video(c["video"]), c["in"], c["out"]),
                              region(ref, max(ra, 0), rb), spec.W / spec.H, float(m.get("strength", 0.8)))
        out.append(f"lut3d=file={lut}")
    return out


def video_chain(c: dict, part: Part, spec: Spec, tl: dict) -> str:
    v = spec.video(c["video"])
    sp = float(c.get("speed") or 1.0)
    f = [f"setpts=(PTS-STARTPTS)/{sp:.4f}", f"fps={spec.fps:g}"]
    sw, sh = v.width, v.height
    active = v.meta.get("active")
    if active and PRESETS[spec.preset][0]:  # strip baked-in letterbox bars before reframing
        aw, ah, ax, ay = active
        f.append(f"crop={aw}:{ah}:{ax}:{ay}")
        sw, sh = aw, ah
    ratio = c.get("crop") or tl.get("crop")
    if ratio:
        f.append(crop_expr(ratio))
    if float(c.get("zoom") or 1) > 1:
        z = float(c["zoom"])
        f.append(f"crop='trunc(iw/{z:.4f}/2)*2':'trunc(ih/{z:.4f}/2)*2'")
    f.append(fit_expr(spec.W, spec.H, sw, sh))
    f += grade_filters(c, spec)
    return ",".join(f + ["setsar=1", "settb=AVTB"])


def audio_chain(c: dict, part: Part, has_audio: bool) -> str:
    d = part.frames
    sp = float(c.get("speed") or 1.0)
    f = ["asetpts=PTS-STARTPTS"]
    if abs(sp - 1) > 1e-3:
        f.append(atempo_chain(sp))
    return ",".join(f + ["aresample=48000", "aformat=sample_fmts=fltp:channel_layouts=stereo"])


def piece_inputs_and_graph(piece: Piece, tl: dict, spec: Spec) -> tuple[list[str], str, dict]:
    """-> (ffmpeg input args, filter graph, what the cache key must cover)."""
    args: list[str] = []
    g: list[str] = []
    n_in = 0
    key: dict = {"kind": piece.kind, "frames": piece.frames, "fps": spec.fps, "size": [spec.W, spec.H], "parts": []}
    labels = []
    for pi, part in enumerate(piece.parts):
        c = tl["clips"][part.clip]
        v = spec.video(c["video"])
        args += ["-ss", f"{part.a:.6f}", "-t", f"{part.b - part.a + 0.25:.6f}", "-i", v.path]
        vin = n_in
        n_in += 1
        vc = video_chain(c, part, spec, tl)
        dur = piece.frames / spec.fps
        lv = f"v{pi}"
        g.append(f"[{vin}:v:0]{vc}[{lv}raw]")
        cur = f"{lv}raw"
        caps = []
        for k, cap in enumerate(c.get("captions") or []):
            a = float(cap["at"]) - part.offset
            b = a + float(cap.get("for") or 2.0)
            if b <= 0 or a >= dur:
                continue
            png = caption_png(spec.cache, cap, spec.W, spec.H)
            args += ["-loop", "1", "-t", f"{dur + 0.1:.4f}", "-i", str(png)]
            g.append(f"[{cur}][{n_in}:v]overlay=format=auto:enable='between(t,{max(a, 0):.3f},{b:.3f})'[{lv}c{k}]")
            cur = f"{lv}c{k}"
            n_in += 1
            caps.append([round(a, 3), round(b, 3), cap["text"], cap.get("pos"), cap.get("size")])
        g.append(f"[{cur}]format=yuv420p[{lv}]")
        la = f"a{pi}"
        head, tail = (c.get("_head"), c.get("_tail")) if piece.kind == "body" else (None, None)
        if v.has_audio and (head or tail):  # J/L cut: the sound changes over at a different moment than the picture
            segs_a = []
            hd, td = (head or {}).get("d", 0.0), (tail or {}).get("d", 0.0)
            for tag, side in (("h", head), ("t", tail)):
                if side:
                    sv = spec.video(side["video"])
                    if not sv.has_audio:
                        continue
                    args += ["-ss", f"{side['a']:.6f}", "-t", f"{side['d'] + 0.1:.6f}", "-i", sv.path]
                    g.append(f"[{n_in}:a:0]asetpts=PTS-STARTPTS,aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                             f"atrim=0:{side['d']:.6f},afade=t=in:d=0.02,afade=t=out:st={max(side['d'] - 0.02, 0):.4f}:d=0.02[{la}{tag}]")
                    n_in += 1
                    segs_a.append(tag)
            mid = dur - (hd if "h" in segs_a else 0) - (td if "t" in segs_a else 0)
            g.append(f"[{vin}:a:0]{audio_chain(c, part, True)},atrim={hd if 'h' in segs_a else 0:.6f}:"
                     f"{(hd if 'h' in segs_a else 0) + mid:.6f},asetpts=PTS-STARTPTS,afade=t=in:d=0.02,"
                     f"afade=t=out:st={max(mid - 0.02, 0):.4f}:d=0.02[{la}m]")
            order = (["h"] if "h" in segs_a else []) + ["m"] + (["t"] if "t" in segs_a else [])
            g.append("".join(f"[{la}{x}]" for x in order) + f"concat=n={len(order)}:v=0:a=1,apad=whole_dur={dur:.6f},atrim=0:{dur:.6f}[{la}]")
            key.setdefault("sound", []).append([head, tail])
        elif v.has_audio:
            g.append(f"[{vin}:a:0]{audio_chain(c, part, True)},apad=whole_dur={dur:.6f},atrim=0:{dur:.6f}[{la}]")
        else:
            args += ["-f", "lavfi", "-t", f"{dur:.6f}", "-i", "anullsrc=r=48000:cl=stereo"]
            g.append(f"[{n_in}:a]anull[{la}]")
            n_in += 1
        labels.append((lv, la))
        key["parts"].append({"src": v.sha1, "a": round(part.a, 5), "b": round(part.b, 5), "v": vc, "caps": caps})
    key["effect"] = piece.effect
    if piece.kind == "join":
        (v0, a0), (v1, a1) = labels
        g.append(f"[{v0}][{v1}]xfade=transition={piece.effect}:duration={piece.join:.4f}:offset=0[vout]")
        g.append(f"[{a0}][{a1}]acrossfade=d={piece.join:.4f}:c1=tri:c2=tri[aout]")
    else:
        g.append(f"[{labels[0][0]}]null[vout]")
        g.append(f"[{labels[0][1]}]anull[aout]")
    return args, ";".join(g), key


def encode_args(preview: bool) -> list[str]:
    enc = os.environ.get("VIDX_ENCODER", "libx264")
    if enc != "libx264":
        return ["-c:v", enc, "-b:v", "3M" if preview else "12M"]
    # -bf 0: no reordered frames, so pieces join by stream copy without timestamp games
    return ["-c:v", "libx264", "-preset", "ultrafast" if preview else "veryfast",
            "-crf", "30" if preview else "20", "-bf", "0"]


def render_piece(piece: Piece, tl: dict, spec: Spec) -> None:
    args, graph, key = piece_inputs_and_graph(piece, tl, spec)
    key["enc"] = encode_args(spec.preview)
    key["format"] = PIECE_FORMAT
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
    out = spec.cache / f"{h}.mkv"
    piece.path = out
    if out.exists():
        os.utime(out)
        return
    tmp = out.with_suffix(".tmp.mkv")
    cmd = (["ffmpeg", "-nostdin", "-y", "-hide_banner", "-v", "error"] + args
           + ["-filter_complex", graph, "-map", "[vout]", "-map", "[aout]", "-frames:v", str(piece.frames)]
           + encode_args(spec.preview) + ["-pix_fmt", "yuv420p", "-r", f"{spec.fps:g}", "-c:a", "pcm_s16le", str(tmp)])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise VidxError("render failed: " + " | ".join((p.stderr or "").strip().splitlines()[-3:]))
    tmp.rename(out)
    piece.fresh = True


def audio_mix(tl: dict, lufs: float | None, duration: float, first_input: int) -> tuple[list[str], str | None]:
    """Extra inputs and the graph that mixes sfx/music/files over the edit and sets loudness."""
    args: list[str] = []
    g: list[str] = []
    mix = ["[0:a]"]
    n = first_input
    for a in tl.get("audio") or []:
        gain = float(a.get("gain") or 0)
        if a.get("music"):
            args += ["-stream_loop", "-1", "-i", str(timeline.sfx_path(a["music"]))]
            fade = max(duration - 1.5, 0)
            g.append(f"[{n}:a]volume={gain}dB,atrim=0:{duration:.3f},afade=t=out:st={fade:.3f}:d=1.5,"
                     f"aresample=48000,aformat=channel_layouts=stereo[m{n}]")
        else:
            src = str(timeline.sfx_path(a["sfx"])) if a.get("sfx") else str(Path(a["file"]).expanduser())
            args += ["-i", src]
            ms = int(float(a.get("at") or 0) * 1000)
            g.append(f"[{n}:a]volume={gain}dB,aresample=48000,aformat=channel_layouts=stereo,"
                     f"adelay={ms}|{ms}[m{n}]")
        mix.append(f"[m{n}]")
        n += 1
    tail = f",loudnorm=I={lufs}:TP=-1.5:LRA=11,aresample=48000" if lufs is not None else ""
    if len(mix) > 1:
        g.append("".join(mix) + f"amix=inputs={len(mix)}:duration=first:normalize=0{tail}[aout]")
    elif tail:
        g.append(f"[0:a]{tail[1:]}[aout]")
    else:
        return args, None
    return args, ";".join(g)


def trim_cache(cache: Path, keep: set[Path]) -> None:
    limit = float(os.environ.get("VIDX_CACHE_GB", "5")) * 1e9
    files = sorted((p for p in cache.glob("*.mkv")), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if total <= limit:
            break
        if p not in keep:
            total -= p.stat().st_size
            p.unlink(missing_ok=True)


def default_out(v, version: int, preset: str, preview: bool) -> Path:
    """Next to the source, so a render lands where the user (or a Cowork folder) already looks."""
    name = f"v{version}_{preset}{'_preview' if preview else ''}.mp4"
    base = Path(os.environ.get("VIDX_OUTPUT_DIR") or Path(v.path).parent / "vidx_renders")
    try:
        base.mkdir(parents=True, exist_ok=True)
        if os.access(base, os.W_OK):
            return base / name
    except OSError:
        pass
    return v.dir / "renders" / name


def ensure_clips(tl: dict) -> dict:
    """Versions saved before timeline files only have segments."""
    if tl.get("clips"):
        return tl
    d = (tl.get("trans") or [None, 0])[1]
    segs = tl["segments"]
    tl["clips"] = [{"in": a, "out": b, "video": tl["video_id"], "speed": tl.get("speed") or 1.0,
                    **({"transition": {"type": "dissolve", "dur": d}} if d and i < len(segs) - 1 else {})}
                   for i, (a, b) in enumerate(segs)]
    return timeline.finish(tl)


def render(version_id: str, preset: str = "original", out_path: str | None = None, preview: bool = False) -> str:
    ver = store.get_version(version_id)
    tl = ensure_clips(ver["timeline"])
    if preset not in PRESETS:
        raise VidxError(f"unknown preset {preset!r} (use {', '.join(PRESETS)})")
    v = store.get_video(ver["video_id"])
    fps = float(tl["fps"])
    W, H = output_size(tl, preset, preview)
    cache = config.HOME / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    spec = Spec(W, H, fps, preset, preview, cache)
    out = Path(out_path).expanduser() if out_path else default_out(v, ver["id"], preset, preview)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    expanded, cues = cuts.expand(tl["clips"], fps)  # cut names -> zooms, sound handovers, sfx hits
    tl = {**tl, "clips": expanded, "audio": list(tl.get("audio") or []) + cues}
    pieces = plan_pieces(tl["clips"], fps)
    for piece in pieces:
        render_piece(piece, tl, spec)
    fresh = sum(p.fresh for p in pieces)

    total = sum(p.frames for p in pieces) / fps
    lufs = None if preview else PRESETS[preset][2]
    extra, graph = audio_mix(tl, lufs, total, first_input=1)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("ffconcat version 1.0\n" + "".join(f"file '{p.path}'\n" for p in pieces))
        listing = f.name
    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing] + extra
    cmd += (["-filter_complex", graph, "-map", "0:v", "-map", "[aout]"] if graph else ["-map", "0:v", "-map", "0:a"])
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.6f}",
            "-movflags", "+faststart", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(listing)
    if p.returncode != 0:
        raise VidxError("render failed: " + " | ".join((p.stderr or "").strip().splitlines()[-3:]))
    trim_cache(cache, {x.path for x in pieces})
    dt = time.perf_counter() - t0
    store.set_version_output(ver["id"], str(out), preset + ("+preview" if preview else ""))

    info = util.ffprobe(str(out))
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    dur = float(info["format"]["duration"])
    lines = [f"v{ver['id']} rendered {preset}{' preview' if preview else ''}: {out}",
             f"{util.mmss(dur)} ({dur:.1f}s) {vs['width']}x{vs['height']} {os.path.getsize(out) / 1e6:.1f} MB in {dt:.0f}s"
             f" | pieces: {fresh} encoded, {len(pieces) - fresh} reused"]
    if abs(dur - tl["out_duration"]) > 0.25:
        lines.append(f"note: expected {tl['out_duration']:.1f}s from timeline")
    maxdur = PRESETS[preset][3]
    if maxdur and dur > maxdur:
        lines.append(f"warn: {dur:.0f}s exceeds {preset} limit {maxdur:.0f}s")
    if not preview:
        try:
            from .otio_export import export

            op = export(tl, v.path, out, f"v{ver['id']}")
            if op:
                lines.append(f"otio: {op}")
        except Exception as e:  # export is a convenience; never fail the render over it
            lines.append(f"otio export skipped: {e}")
    lines.append(f"next: check('v{ver['id']}') or peek('v{ver['id']}', 'cuts')")
    return "\n".join(lines)
