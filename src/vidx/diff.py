"""Compare two versions: added / removed / moved source ranges + output duration delta."""
from __future__ import annotations

from . import store
from .dsl import Seg, out_positions
from .store import VidxError
from .util import mmss, rng

MIN_PIECE = 0.05  # ignore sub-frame noise
MAX_LINES = 20


def norm(segs: list[Seg]) -> list[Seg]:
    out: list[Seg] = []
    for a, b in sorted(segs):
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def minus(x: list[Seg], y: list[Seg]) -> list[Seg]:
    """Coverage of x not in y (both arbitrary lists)."""
    res = norm(x)
    for a, b in norm(y):
        nxt = []
        for s, e in res:
            if b <= s or a >= e:
                nxt.append((s, e))
            else:
                if a > s:
                    nxt.append((s, a))
                if b < e:
                    nxt.append((b, e))
        res = nxt
    return [(s, e) for s, e in res if e - s >= MIN_PIECE]


def _pieces(a: list[Seg], b: list[Seg]) -> list[tuple[int, int, float, float]]:
    """Shared content: (index in a, index in b, src start, src end), ordered by position in b."""
    out = []
    for j, (bs, be) in enumerate(b):
        for i, (as_, ae) in enumerate(a):
            s, e = max(as_, bs), min(ae, be)
            if e - s >= MIN_PIECE:
                out.append((i, j, s, e))
    out.sort(key=lambda p: (p[1], p[2]))
    return out


def _lis_keep(keys: list[tuple]) -> set[int]:
    """Indices of a longest non-decreasing subsequence (the pieces that did NOT move)."""
    import bisect

    tails, tails_idx, prev = [], [], [-1] * len(keys)
    for i, k in enumerate(keys):
        pos = bisect.bisect_right(tails, k)
        if pos == len(tails):
            tails.append(k)
            tails_idx.append(i)
        else:
            tails[pos] = k
            tails_idx[pos] = i
        prev[i] = tails_idx[pos - 1] if pos else -1
    keep, i = set(), tails_idx[-1] if tails_idx else -1
    while i != -1:
        keep.add(i)
        i = prev[i]
    return keep


def diff_timelines(ta: dict, tb: dict) -> dict:
    a = [tuple(s) for s in ta["segments"]]
    b = [tuple(s) for s in tb["segments"]]
    pa = out_positions(a, ta.get("speed") or 1, ta.get("trans"))
    pb = out_positions(b, tb.get("speed") or 1, tb.get("trans"))

    sa, sb = ta.get("speed") or 1, tb.get("speed") or 1

    def out_at(t: float, segs, pos, speed) -> float | None:
        for (s, e), (o0, _) in zip(segs, pos):
            if s - 1e-6 <= t <= e + 1e-6:
                return o0 + (t - s) / speed
        return None

    removed = [(s, e, out_at(s, a, pa, sa)) for s, e in minus(a, b)]
    added = [(s, e, out_at(s, b, pb, sb)) for s, e in minus(b, a)]
    pieces = _pieces(a, b)
    keys = [(i, s) for i, _, s, _ in pieces]  # order in a
    stay = _lis_keep(keys)
    moved = []
    for k, (i, j, s, e) in enumerate(pieces):
        if k not in stay:
            moved.append((s, e, out_at(s, a, pa, sa), out_at(s, b, pb, sb)))
    opts = []
    for key in ("trans", "crop", "speed", "pad", "snap"):
        va, vb = ta.get(key), tb.get(key)
        if va != vb:
            fmt = lambda v: "cut" if key == "trans" and not v else (f"xfade {v[1]}" if key == "trans" else str(v))
            opts.append(f"{key} {fmt(va)} -> {fmt(vb)}")
    return {"removed": removed, "added": added, "moved": moved, "opts": opts,
            "dur_a": ta["out_duration"], "dur_b": tb["out_duration"], "n_a": len(a), "n_b": len(b)}


def diff(version_a: str, version_b: str) -> str:
    va, vb = store.get_version(version_a), store.get_version(version_b)
    if va["video_id"] != vb["video_id"]:
        raise VidxError(f"v{va['id']} and v{vb['id']} edit different videos (multi-source diff is v2)")
    d = diff_timelines(va["timeline"], vb["timeline"])
    na, nb = f"v{va['id']}", f"v{vb['id']}"
    lines = [f"{na} -> {nb}: out {mmss(d['dur_a'])} -> {mmss(d['dur_b'])} ({d['dur_b'] - d['dur_a']:+.1f}s), "
             f"segments {d['n_a']} -> {d['n_b']}"]
    body = []
    for s, e, o in d["removed"]:
        body.append(f"removed src {rng(s, e)} ({e - s:.1f}s)" + (f" [was at out {o:.1f}]" if o is not None else ""))
    for s, e, o in d["added"]:
        body.append(f"added src {rng(s, e)} ({e - s:.1f}s)" + (f" [now at out {o:.1f}]" if o is not None else ""))
    for s, e, oa, ob in d["moved"]:
        body.append(f"moved src {rng(s, e)} ({e - s:.1f}s): out {oa:.1f} -> {ob:.1f}")
    if d["opts"]:
        body.append("options: " + "; ".join(d["opts"]))
    if not body:
        body = ["no changes"]
    tot = {k: sum(x[1] - x[0] for x in d[k]) for k in ("removed", "added", "moved")}
    if len(body) > MAX_LINES:
        body = body[: MAX_LINES - 1] + [f"… {len(body) - MAX_LINES + 1} more (removed {tot['removed']:.1f}s, "
                                        f"added {tot['added']:.1f}s, moved {tot['moved']:.1f}s in total)"]
    return "\n".join(lines + body)
