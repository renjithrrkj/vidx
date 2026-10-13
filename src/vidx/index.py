"""index_video: probe + run analyzers + one-paragraph summary."""
from __future__ import annotations

import time

from . import store
from .analyzers import Context, run_all
from .analyzers.probe import probe
from .util import mmss, snip


def index_video(path: str, force: bool = False, captions: bool = False, log=None) -> str:
    t0 = time.perf_counter()
    v = probe(path)
    existing = None
    try:
        existing = store.get_video(v.id)
    except store.VidxError:
        pass
    if existing and force:
        store.delete_video_data(v.id)
    if not existing or force:
        store.put_video(v)
    elif existing.path != v.path:  # same file moved/copied: keep data, update path
        store.db().execute("UPDATE videos SET path=? WHERE id=?", (v.path, v.id))
        store.db().commit()
    v = store.get_video(v.id)
    ctx = Context(force=force, captions=captions)
    if log:
        ctx.log = log
    timings = run_all(v, ctx)
    total = time.perf_counter() - t0
    return summary(v.id, timings, total)


def summary(video_id: str, timings: dict[str, float] | None = None, total: float | None = None) -> str:
    v = store.get_video(video_id)
    shots = store.get_shots(video_id)
    words = store.get_words(video_id)
    ev = store.get_events(video_id)
    lang = v.meta.get("language") or "none"
    mix = v.meta.get("language_mix")
    if mix:
        lang += " (script " + ", ".join(f"{k} {int(p * 100)}%" for k, p in sorted(mix.items(), key=lambda x: -x[1])) + ")"
    if v.meta.get("transcript_mode") == "translated-en":
        lang += " [transcript is an ENGLISH TRANSLATION; word times approximate]"
    if v.meta.get("transcript_dropped"):
        lang += f" [{v.meta['transcript_dropped']:.0f}s of unusable ASR dropped]"
    speech = sum(w.end - w.start for w in words)
    sil = sum(e.end - e.start for e in ev if e.kind == "silence")
    counts = {k: sum(e.kind == k for e in ev) for k in ("silence", "loud", "black", "freeze")}
    orient = "vertical" if v.height > v.width else "horizontal" if v.width > v.height else "square"
    lines = [
        f"video_id {v.id} | {mmss(v.duration)} ({v.duration:.1f}s) {v.width}x{v.height} {orient} {v.fps:g}fps | "
        f"{len(shots)} shots | language {lang} | {len(words)} words",
    ]
    if v.meta.get("active"):
        aw, ah = v.meta["active"][:2]
        lines.append(f"letterboxed: picture is {aw}x{ah} inside the frame (reel/yt presets crop the bars)")
    if v.meta.get("lufs") is not None:
        lines.append(
            f"audio: {v.meta['lufs']:.1f} LUFS, speech ~{speech / max(v.duration, 1e-6) * 100:.0f}% of time, "
            f"silence {sil:.1f}s total; events: " + ", ".join(f"{k} {n}" for k, n in counts.items()))
    if words:
        lines.append(f'opens with: "{snip(" ".join(w.text for w in words[:40]), 160)}"')
    tagset: dict[str, float] = {}
    for s in shots:
        if s.tags:
            main = s.tags.split(", ")[0]
            tagset[main] = tagset.get(main, 0) + s.dur
    if tagset:
        top = sorted(tagset.items(), key=lambda x: -x[1])[:3]
        lines.append("mostly: " + "; ".join(f"{t} ({d / v.duration * 100:.0f}%)" for t, d in top))
    if timings:
        lines.append("indexed in " + " ".join(f"{k}={t:.1f}s" for k, t in timings.items())
                     + (f" total={total:.1f}s" if total else ""))
    lines.append(f"next: overview('{v.id}') or query('{v.id}', '<text>')")
    return "\n".join(lines)
