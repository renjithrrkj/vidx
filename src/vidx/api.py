"""The 9 public operations. CLI and MCP server are thin wrappers over these; each returns text
(peek returns images + a caption). Every call's output size is logged to ~/.vidx/usage.log."""
from __future__ import annotations

import functools
import json
import time

from . import config, dsl, store
from .util import mmss


def _usage(fn):
    @functools.wraps(fn)
    def wrap(*a, **kw):
        t0 = time.perf_counter()
        ok = True
        try:
            res = fn(*a, **kw)
            return res
        except Exception:
            ok = False
            raise
        finally:
            try:
                out = locals().get("res")
                chars = len(out) if isinstance(out, str) else len(out[1]) if isinstance(out, tuple) else 0
                imgs = len(out[0]) if isinstance(out, tuple) else 0
                img_tok = 0
                if imgs:  # Claude image cost ~ w*h/750 after resize to <=1568px / ~1.15MP
                    import io as _io
                    import math as _m

                    from PIL import Image as _I

                    for b in out[0]:
                        w, h = _I.open(_io.BytesIO(b)).size
                        sc = min(1.0, 1568 / max(w, h), _m.sqrt(1_150_000 / (w * h)))
                        img_tok += _m.ceil(w * sc * h * sc / 750)
                vid = str(kw.get("video_id") or (a[0] if a else ""))
                if vid[:1] in "vV" and vid[1:].isdigit():  # version -> its video, so reports group per video
                    try:
                        vid = store.get_version(vid)["video_id"]
                    except Exception:
                        pass
                rec = {"t": round(time.time(), 1), "tool": fn.__name__, "target": vid[:40], "chars": chars,
                       "images": imgs, "image_tokens": img_tok, "ok": ok, "secs": round(time.perf_counter() - t0, 2)}
                config.HOME.mkdir(parents=True, exist_ok=True)
                with open(config.HOME / "usage.log", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass
    return wrap


@_usage
def index_video(path: str, force: bool = False, captions: bool = False, log=None) -> str:
    from .index import index_video as _index

    return _index(path, force=force, captions=captions, log=log)


@_usage
def overview(video_id: str, level: str = "scenes", limit: int = config.DEFAULT_LIMIT, offset: int = 0) -> str:
    from .query import overview as _ov

    return _ov(video_id, level, limit, offset)


@_usage
def query(video_id: str, text: str, kind: str = "any", limit: int = 10) -> str:
    from .query import query as _q

    return _q(video_id, text, kind, limit)


@_usage
def get_segment(video_id: str, start, end=None, fields: str = "words,events,shots") -> str:
    from .query import get_segment as _g

    return _g(video_id, start, end, fields)


@_usage
def peek(target: str, times=None, size: int = 384, grid: bool = True) -> tuple[list[bytes], str]:
    from .query import peek as _p

    return _p(target, times, size, grid)


@_usage
def edit(video_id: str, script: str, parent_version: str | None = None) -> str:
    """Parent semantics: the new script is appended to the parent's script (incremental edit),
    unless it has its own `base` line, in which case it stands alone."""
    v = store.get_video(video_id)
    parent_id, sc = None, None
    has_base = any(l.split("#")[0].strip().lower().startswith("base") for l in script.splitlines())
    if parent_version:
        parent = store.get_version(parent_version)
        if parent["video_id"] != v.id:
            raise store.VidxError(f"v{parent['id']} belongs to video {parent['video_id']}, not {v.id}")
        parent_id = parent["id"]
        if not has_base:
            sc = dsl.parse(parent["dsl"], src=f"v{parent_id}")
    sc = dsl.parse(script, sc=sc)
    full = (parent["dsl"].rstrip() + "\n" + script.strip()) if parent_id and not has_base else script.strip()
    tl = dsl.resolve(sc, dsl.IndexCtx(v.id))
    vid = store.put_version(v.id, parent_id, full, tl)
    return describe_version(vid)


def describe_version(version_id) -> str:
    ver = store.get_version(version_id)
    tl = ver["timeline"]
    segs = tl["segments"]
    src = tl["source_duration"]
    delta = tl["out_duration"] - src
    opts = []
    if tl.get("trans"):
        opts.append(f"xfade {tl['trans'][1]}")
    if tl.get("crop"):
        opts.append(f"crop {tl['crop']}")
    if abs((tl.get("speed") or 1) - 1) > 1e-3:
        opts.append(f"speed {tl['speed']}")
    lines = [f"v{ver['id']}" + (f" (from v{ver['parent_id']})" if ver["parent_id"] else "")
             + f": {len(segs)} segments, out {mmss(tl['out_duration'])} ({tl['out_duration']:.1f}s, "
             + f"{delta:+.1f}s vs source)" + (" | " + ", ".join(opts) if opts else "")]
    show = segs[:8]
    lines.append("segs: " + ", ".join(f"{a:.1f}-{b:.1f}" for a, b in show)
                 + (f", … (+{len(segs) - 8})" if len(segs) > 8 else ""))
    for w in tl.get("warnings", [])[:5]:
        lines.append(f"warn: {w}")
    lines.append(f"next: render('v{ver['id']}', preset)")
    return "\n".join(lines)


@_usage
def render(version_id: str, preset: str = "original", out_path: str | None = None) -> str:
    from .render import render as _r

    return _r(version_id, preset, out_path)


@_usage
def check(version_id: str) -> str:
    from .checks import check as _c

    return _c(version_id)


@_usage
def diff(version_a: str, version_b: str) -> str:
    from .diff import diff as _d

    return _d(version_a, version_b)
