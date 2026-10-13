"""timeline -> ffmpeg. trim/atrim per segment, concat (hard cuts) or xfade/acrossfade chains."""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from . import config, store, util
from .store import VidxError

PRESETS = {
    # name: (width, height, target LUFS or None, max duration or None)
    "original": (None, None, None, None),
    "reel_9x16": (1080, 1920, -16.0, 90.0),
    "yt_16x9": (1920, 1080, -14.0, None),
}


def frame_snap(segs: list[list[float]], fps: float) -> list[tuple[float, float]]:
    """Cut on frame boundaries so audio (sample-accurate) and video (frame-accurate) stay in sync."""
    out = []
    for a, b in segs:
        fa, fb = round(a * fps), round(b * fps)
        if fb > fa:
            out.append((fa / fps, fb / fps))
    return out


def runs(segs: list[tuple[float, float]]) -> list[list[int]]:
    """Group segment indices into runs that move forward in source time (one decoder per run)."""
    groups: list[list[int]] = []
    for i, (a, _) in enumerate(segs):
        if groups and a >= segs[groups[-1][-1]][1] - 1e-6:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def crop_expr(ratio: str) -> str:
    rw, rh = map(int, ratio.split(":"))
    return (f"crop='trunc(min(iw,ih*{rw}/{rh})/2)*2':'trunc(min(ih,iw*{rh}/{rw})/2)*2'")


def fit_expr(w: int, h: int, src_w: int, src_h: int) -> str:
    """Fit to WxH: center-crop when orientation matches (or for vertical targets), pad when a
    vertical source goes to a horizontal target (cropping would lose ~70% of the frame)."""
    src_vertical, tgt_vertical = src_h > src_w, h > w
    if src_vertical and not tgt_vertical:
        return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}")


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


def build(timeline: dict, src: str, preset: str, out: str, has_audio: bool,
          src_w: int, src_h: int, active: list[int] | None = None) -> tuple[list[str], str, list[str]]:
    """-> (ffmpeg head argv, filtergraph text, output argv)."""
    if preset not in PRESETS:
        raise VidxError(f"unknown preset {preset!r} (use {', '.join(PRESETS)})")
    W, H, lufs, _ = PRESETS[preset]
    fps = float(timeline["fps"])
    segs = frame_snap(timeline["segments"], fps)
    if not segs:
        raise VidxError("timeline has no segments longer than one frame")
    trans = timeline.get("trans")
    groups = runs(segs)

    args = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-v", "error", "-stats_period", "5"]
    g: list[str] = []
    seg_input: dict[int, tuple[int, float]] = {}
    for r, idxs in enumerate(groups):
        ss = max(0.0, segs[idxs[0]][0] - 1.0)  # fast seek near the run, then exact trims
        args += ["-ss", f"{ss:.3f}", "-i", src]
        for i in idxs:
            seg_input[i] = (r, ss)
        n = len(idxs)
        g.append(f"[{r}:v:0]" + (f"split={n}" if n > 1 else "null") + "".join(f"[sv{i}]" for i in idxs))
        if has_audio:
            g.append(f"[{r}:a:0]" + (f"asplit={n}" if n > 1 else "anull") + "".join(f"[sa{i}]" for i in idxs))

    for i, (a, b) in enumerate(segs):
        _, ss = seg_input[i]
        a0, b0 = a - ss, b - ss
        g.append(f"[sv{i}]trim=start={a0:.6f}:end={b0:.6f},setpts=PTS-STARTPTS,fps={fps:g},"
                 f"format=yuv420p,settb=AVTB[v{i}]")
        if has_audio:
            g.append(f"[sa{i}]atrim=start={a0:.6f}:end={b0:.6f},asetpts=PTS-STARTPTS,"
                     f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")

    n = len(segs)
    if n == 1:
        vlast, alast = "v0", "a0"
    elif trans:
        d = float(trans[1])
        vlast, alast, acc = "v0", "a0", segs[0][1] - segs[0][0]
        for k in range(1, n):
            off = acc - d
            g.append(f"[{vlast}][v{k}]xfade=transition=fade:duration={d:.3f}:offset={off:.4f}[xv{k}]")
            if has_audio:
                g.append(f"[{alast}][a{k}]acrossfade=d={d:.3f}:c1=tri:c2=tri[xa{k}]")
            vlast, alast = f"xv{k}", f"xa{k}"
            acc = acc + (segs[k][1] - segs[k][0]) - d
    else:
        ins = "".join(f"[v{i}]" + (f"[a{i}]" if has_audio else "") for i in range(n))
        g.append(f"{ins}concat=n={n}:v=1:a={1 if has_audio else 0}[cv]" + ("[ca]" if has_audio else ""))
        vlast, alast = "cv", "ca"

    vpost, apost = [], []
    speed = float(timeline.get("speed") or 1.0)
    if abs(speed - 1.0) > 1e-3:
        vpost.append(f"setpts=PTS/{speed:.4f}")
        apost.append(atempo_chain(speed))
    if active and W:  # strip baked-in letterbox bars before reframing for a preset
        aw, ah, ax, ay = active
        vpost.append(f"crop={aw}:{ah}:{ax}:{ay}")
        src_w, src_h = aw, ah
    if timeline.get("crop"):
        vpost.append(crop_expr(timeline["crop"]))
    if W:
        vpost.append(fit_expr(W, H, src_w, src_h))
    vpost.append("setsar=1")
    g.append(f"[{vlast}]" + ",".join(vpost) + "[vout]")
    if has_audio:
        if lufs is not None:
            apost.append(f"loudnorm=I={lufs}:TP=-1.5:LRA=11")
            apost.append("aresample=48000")
        g.append(f"[{alast}]" + (",".join(apost) or "anull") + "[aout]")

    graph = ";\n".join(g)
    enc = os.environ.get("VIDX_ENCODER", "libx264")
    venc = ["-c:v", enc] + (["-preset", "veryfast", "-crf", "20"] if enc == "libx264" else ["-b:v", "12M"])
    tail = ["-map", "[vout]"] + (["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"] if has_audio else [])
    tail += venc + ["-pix_fmt", "yuv420p", "-r", f"{fps:g}", "-movflags", "+faststart", out]
    return args, graph, tail


def render(version_id: str, preset: str = "original", out_path: str | None = None, log=None) -> str:
    ver = store.get_version(version_id)
    tl = ver["timeline"]
    v = store.get_video(ver["video_id"])
    if not os.path.exists(v.path):
        raise VidxError(f"source moved: {v.path} (re-run index_video on the new path)")
    if preset not in PRESETS:
        raise VidxError(f"unknown preset {preset!r} (use {', '.join(PRESETS)})")
    out = Path(out_path).expanduser() if out_path else v.dir / "renders" / f"v{ver['id']}_{preset}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    head, graph, tail = build(tl, v.path, preset, str(out), v.has_audio, v.width, v.height,
                              v.meta.get("active"))
    with tempfile.NamedTemporaryFile("w", suffix=".ffgraph", delete=False) as f:
        f.write(graph)
        gpath = f.name
    cmd = head + ["-/filter_complex", gpath] + tail
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(gpath)
    if p.returncode != 0:
        tailmsg = " | ".join((p.stderr or "").strip().splitlines()[-3:])
        raise VidxError(f"render failed: {tailmsg}")
    dt = time.perf_counter() - t0
    store.set_version_output(ver["id"], str(out), preset)
    info = util.ffprobe(str(out))
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    dur = float(info["format"]["duration"])
    size = os.path.getsize(out)
    _, _, _, maxdur = PRESETS[preset]
    lines = [f"v{ver['id']} rendered {preset}: {out}",
             f"{util.mmss(dur)} ({dur:.1f}s) {vs['width']}x{vs['height']} {size / 1e6:.1f} MB in {dt:.0f}s"]
    if abs(dur - tl["out_duration"]) > 0.25:
        lines.append(f"note: expected {tl['out_duration']:.1f}s from timeline")
    if maxdur and dur > maxdur:
        lines.append(f"warn: {dur:.0f}s exceeds {preset} limit {maxdur:.0f}s")
    try:
        from .otio_export import export

        op = export(tl, v.path, out, f"v{ver['id']}")
        if op:
            lines.append(f"otio: {op}")
    except Exception as e:  # export is a convenience; never fail the render over it
        lines.append(f"otio export skipped: {e}")
    lines.append(f"next: check('v{ver['id']}') or peek('v{ver['id']}', times)")
    return "\n".join(lines)
