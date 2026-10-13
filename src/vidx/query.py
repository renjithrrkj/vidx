"""Understanding tools: overview / query / get_segment / peek. All return terse text."""
from __future__ import annotations

import difflib
import io
import os
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from . import config, store, thumbs
from .store import Event, Shot, VidxError, Word
from .util import mmss, more, rng, snip

# ---------------------------------------------------------------- helpers


def words_text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def phrases(words: list[Word], gap: float = 0.6, max_len: float = 12.0) -> list[list[Word]]:
    """Split words into phrases at pauses (or every max_len seconds)."""
    out: list[list[Word]] = []
    for w in words:
        if out and w.start - out[-1][-1].end <= gap and w.end - out[-1][0].start <= max_len:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def shot_desc(s: Shot) -> str:
    return s.caption or s.tags or ""


def parse_shot(token: str, shots: list[Shot]) -> Shot:
    m = re.fullmatch(r"s(\d+)", token.strip().lower())
    if not m:
        raise VidxError(f"bad shot id {token!r} (expected like s3)")
    i = int(m.group(1))
    if not 1 <= i <= len(shots):
        raise VidxError(f"unknown shot s{i} (video has s1-s{len(shots)})")
    return shots[i - 1]


def event_marks(evs: list[Event]) -> str:
    """Compact event markers for list lines, e.g. '[silence 2x 3.1s, loud]'."""
    if not evs:
        return ""
    by: dict[str, list[Event]] = {}
    for e in evs:
        by.setdefault(e.kind, []).append(e)
    parts = []
    for k, es in by.items():
        tot = sum(e.end - e.start for e in es)
        parts.append(f"{k}" + (f" {len(es)}x" if len(es) > 1 else "") + (f" {tot:.1f}s" if k != "loud" else ""))
    return " [" + ", ".join(parts) + "]"


# ---------------------------------------------------------------- overview


@dataclass
class Unit:
    start: float
    end: float
    shots: list[int]  # shot idx (1-based)
    emb: np.ndarray | None


def _split_long(s: Shot, words: list[Word], target: float) -> list[tuple[float, float]]:
    """Split one long shot into ~target-length pieces at the biggest pauses between words."""
    n = int(s.dur // target)
    if n < 1:
        return [(s.start, s.end)]
    ws = [w for w in words if s.start <= w.start < s.end]
    cuts = []
    for k in range(1, n + 1):
        ideal = s.start + k * s.dur / (n + 1)
        best, best_score = ideal, -1.0
        for a, b in zip(ws, ws[1:]):
            mid = (a.end + b.start) / 2
            if abs(mid - ideal) < target * 0.35:
                score = (b.start - a.end) - abs(mid - ideal) * 0.02
                if score > best_score:
                    best, best_score = mid, score
        cuts.append(best)
    pts = [s.start] + sorted(c for c in cuts if s.start + 1 < c < s.end - 1) + [s.end]
    return list(zip(pts, pts[1:]))


def scenes(video_id: str, k: int) -> list[Unit]:
    v = store.get_video(video_id)
    shots = store.get_shots(video_id)
    words = store.get_words(video_id)
    emb = store.get_shot_emb(video_id)
    target = max(v.duration / max(k, 1), 1.0)
    units: list[Unit] = []
    for s in shots:
        e = emb[s.idx - 1] if emb is not None and len(emb) >= s.idx else None
        for a, b in _split_long(s, words, target * 1.5):
            units.append(Unit(a, b, [s.idx], e))
    while len(units) > k:
        best_i, best_cost = 0, float("inf")
        for i in range(len(units) - 1):
            a, b = units[i], units[i + 1]
            sim = float(a.emb @ b.emb) if a.emb is not None and b.emb is not None else 0.5
            cost = ((a.end - a.start) + (b.end - b.start)) * (1.6 - sim)
            if cost < best_cost:
                best_i, best_cost = i, cost
        a, b = units[best_i], units[best_i + 1]
        e = None
        if a.emb is not None and b.emb is not None:
            wa, wb = a.end - a.start, b.end - b.start
            e = (a.emb * wa + b.emb * wb)
            e = e / max(np.linalg.norm(e), 1e-6)
        units[best_i: best_i + 2] = [Unit(a.start, b.end, sorted(set(a.shots + b.shots)), e)]
    return units


def overview(video_id: str, level: str = "scenes", limit: int = config.DEFAULT_LIMIT, offset: int = 0) -> str:
    v = store.get_video(video_id)
    shots = store.get_shots(video_id)
    words = store.get_words(video_id)
    events = [e for e in store.get_events(video_id) if not (e.kind == "silence" and e.end - e.start < 1.0)]
    limit = max(1, int(limit))

    def speech(a: float, b: float, n: int = config.SNIPPET_CHARS) -> str:
        ws = [w for w in words if a <= (w.start + w.end) / 2 < b]
        return f' | speech: "{snip(words_text(ws), n)}"' if ws else ""

    def evs(a: float, b: float) -> str:
        return event_marks([e for e in events if e.start < b and e.end > a])

    if level == "shots":
        sel = shots[offset: offset + limit]
        head = f"{v.id} {mmss(v.duration)} | shots {offset + 1}-{offset + len(sel)} of {len(shots)}"
        lines = [head]
        for s in sel:
            d = shot_desc(s)
            lines.append(f"s{s.idx} {rng(s.start, s.end)} ({s.dur:.1f}s)" + (f' "{d}"' if d else "")
                         + speech(s.start, s.end) + evs(s.start, s.end))
        lines += more(len(shots) - offset, len(sel))
        if len(shots) - offset > len(sel):
            lines[-1] += f" (overview level=shots offset={offset + len(sel)})"
        return "\n".join(lines)
    if level != "scenes":
        raise VidxError(f"unknown level {level!r} (use scenes or shots)")

    units = scenes(video_id, limit)
    lines = [f"{v.id} {mmss(v.duration)} | {len(units)} scenes from {len(shots)} shots"
             + (" (scenes merge similar adjacent shots; long shots split at pauses)" if len(units) != len(shots) else "")]
    prev_d = None
    for i, u in enumerate(units, 1):
        # describe a scene by its longest shot; don't repeat an identical description
        main = max((shots[j - 1] for j in u.shots), key=lambda s: min(s.end, u.end) - max(s.start, u.start))
        sid = f"s{u.shots[0]}" if len(u.shots) == 1 else f"s{u.shots[0]}-s{u.shots[-1]}"
        d = shot_desc(main)
        shown = "(same view)" if d and d == prev_d else (f'"{d}"' if d else "")
        prev_d = d
        lines.append(f"c{i} {rng(u.start, u.end)} ({u.end - u.start:.1f}s) {sid}" + (f" {shown}" if shown else "")
                     + speech(u.start, u.end) + evs(u.start, u.end))
    out = "\n".join(lines)
    if len(out) > config.OVERVIEW_MAX_CHARS:  # hard budget guard
        out = out[: config.OVERVIEW_MAX_CHARS - 40].rsplit("\n", 1)[0] + "\n… truncated (lower limit)"
    return out


# ---------------------------------------------------------------- query

EVENT_RE = re.compile(r"^\s*(silence|loud|black|freeze|laugh|beat|ocr)\s*(?:([<>]=?)\s*([\d.]+)\s*s?)?\s*$", re.I)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"[^\w\s']", " ", s).replace("'", "")


def _norm_tokens(words: list[Word]) -> list[str]:
    return [" ".join(_norm(w.text).split()) for w in words]


def speech_hits(words: list[Word], text: str) -> list[tuple[int, int, float]]:
    """(first_word, last_word, score). Exact substring first, then fuzzy windows."""
    q = " ".join(_norm(text).split())
    if not q or not words:
        return []
    toks = _norm_tokens(words)
    joined, starts = "", []
    for t in toks:
        starts.append(len(joined))
        joined += t + " "
    import bisect

    hits: list[tuple[int, int, float]] = []
    pos = joined.find(q)
    while pos != -1:
        a = bisect.bisect_right(starts, pos) - 1
        b = bisect.bisect_right(starts, pos + len(q) - 1) - 1
        hits.append((a, b, 1.0))
        pos = joined.find(q, pos + 1)
    taken = set()
    for a, b, _ in hits:
        taken.update(range(a, b + 1))
    n = max(1, len(q.split()))
    cands = []
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, len(toks) - size + 1):
            if any(j in taken for j in range(i, i + size)):
                continue
            win = " ".join(toks[i: i + size])
            if abs(len(win) - len(q)) > max(4, len(q) * 0.5):
                continue
            r = difflib.SequenceMatcher(None, q, win, autojunk=False).ratio()
            if r >= 0.75:
                cands.append((i, i + size - 1, round(r, 2)))
    cands.sort(key=lambda h: -h[2])
    for a, b, r in cands:
        if not any(j in taken for j in range(a, b + 1)):
            hits.append((a, b, r))
            taken.update(range(a, b + 1))
    hits.sort(key=lambda h: (-h[2], h[0]))
    return hits


def _context(words: list[Word], a: int, b: int, n: int = config.SNIPPET_CHARS) -> str:
    left, right = max(0, a - 4), min(len(words), b + 5)
    s = words_text(words[left:right])
    s = ("…" if left > 0 else "") + s
    return snip(s, n)


def parse_event_query(text: str) -> tuple[str, str | None, float | None] | None:
    m = EVENT_RE.match(text)
    if not m:
        return None
    return m.group(1).lower(), m.group(2), float(m.group(3)) if m.group(3) else None


def event_filter(evs: list[Event], op: str | None, val: float | None) -> list[Event]:
    if op is None:
        return evs
    f = {">": lambda d: d > val, ">=": lambda d: d >= val, "<": lambda d: d < val, "<=": lambda d: d <= val}[op]
    return [e for e in evs if f(e.end - e.start)]


def query(video_id: str, text: str, kind: str = "any", limit: int = 10) -> str:
    v = store.get_video(video_id)
    limit = max(1, int(limit))
    if kind not in ("any", "visual", "speech", "event"):
        raise VidxError(f"unknown kind {kind!r} (use any|visual|speech|event)")
    ev_q = parse_event_query(text)
    if kind == "event" or (kind == "any" and ev_q):
        if not ev_q:
            raise VidxError(f"bad event query {text!r} (try silence>1.5, black, loud, freeze)")
        k, op, val = ev_q
        evs = event_filter(store.get_events(video_id, k), op, val)
        tot = sum(e.end - e.start for e in evs)
        lines = [f"{len(evs)} {k} events" + (f" {op}{val:g}s" if op else "") + f", {tot:.1f}s total"]
        for e in evs[:limit]:
            extra = f" peak {e.value:.0f} LUFS" if k == "loud" and e.value is not None else ""
            lines.append(f"{rng(e.start, e.end)} ({e.end - e.start:.1f}s) {k}{extra}")
        lines += more(len(evs), min(limit, len(evs)))
        return "\n".join(lines)

    out: list[str] = []
    if kind in ("any", "speech"):
        words = store.get_words(video_id)
        hits = speech_hits(words, text)
        n = limit if kind == "speech" else max(1, (limit + 1) // 2)
        out.append(f"speech: {len(hits)} match{'es' if len(hits) != 1 else ''}"
                   + ("" if words else " (no transcript)"))
        for a, b, sc in hits[:n]:
            out.append(f"  {rng(words[a].start, words[b].end)} {'exact' if sc >= 1 else f'fuzzy {sc:.2f}'}"
                       f' "{_context(words, a, b)}"')
        out += ["  " + x for x in more(len(hits), min(n, len(hits)))]
    if kind in ("any", "visual"):
        out += visual_hits(video_id, text, limit if kind == "visual" else max(1, limit // 2))
    return "\n".join(out)


def _conf(sim: float, z: float | None) -> str:
    # ViT-B-32 (laion2b): matching image/text pairs ~0.28-0.35, unrelated ~0.15-0.22
    if z is None:  # too few candidates for a meaningful z-score
        return "strong" if sim > 0.30 else "ok" if sim > 0.25 else "weak"
    return "strong" if sim > 0.27 and z > 1.5 else "weak" if sim < 0.22 or z < 0.5 else "ok"


def visual_hits(video_id: str, text: str, n: int) -> list[str]:
    from .analyzers.visual import encode_text, get_windows

    emb = store.get_shot_emb(video_id)
    shots = store.get_shots(video_id)
    if emb is None or not len(emb):
        return ["visual: no embeddings (re-run index_video)"]
    wemb, widx = get_windows(video_id)
    windowed = set(int(i) for i in widx[:, 0]) if widx is not None and len(widx) else set()
    # candidates: whole short shots + ~5s windows inside long shots
    cands = [(s.idx, s.start, s.end, emb[s.idx - 1]) for s in shots if s.idx not in windowed]
    if windowed:
        cands += [(int(i), float(a), float(b), e) for (i, a, b), e in zip(widx, wemb)]
    qv = encode_text([f"a photo of {text}"])[0]
    sims = np.array([c[3] @ qv for c in cands])
    z = (sims - sims.mean()) / sims.std() if len(sims) >= 8 and sims.std() > 1e-6 else None
    picked: list[list] = []  # [shot, a, b, sim, zscore]
    for k in np.argsort(-sims):
        i, a, b, _ = cands[k]
        zk = float(z[k]) if z is not None else None
        near = next((p for p in picked if p[0] == i and (abs(p[2] - a) < 0.01 or abs(p[1] - b) < 0.01)), None)
        if near and sims[k] > near[3] - 0.02:  # neighbouring window, about as good: widen the hit
            near[1], near[2] = min(near[1], a), max(near[2], b)
            continue
        if len(picked) < n:
            picked.append([i, a, b, float(sims[k]), zk])
    unit = f"{len(cands)} candidates" if windowed else f"{len(shots)} shots"
    lines = [f"visual: top {len(picked)} of {unit} by CLIP similarity"]
    for i, a, b, sim, zk in picked:
        d = shot_desc(shots[i - 1])
        lines.append(f"  s{i} {rng(a, b)} sim {sim:.2f} ({_conf(sim, zk)})"
                     + (f' "{d}"' if d and i not in windowed else ""))
    return lines


# ---------------------------------------------------------------- get_segment

SEGMENT_MAX_CHARS = 4000


def resolve_range(video_id: str, start, end) -> tuple[float, float]:
    v = store.get_video(video_id)
    shots = None

    def one(x, is_end: bool) -> float:
        nonlocal shots
        if isinstance(x, str) and x.strip().lower().startswith("s"):
            shots = shots or store.get_shots(video_id)
            s = parse_shot(x, shots)
            return s.end if is_end else s.start
        try:
            return float(x)
        except (TypeError, ValueError):
            raise VidxError(f"bad time {x!r} (use seconds like 12.5 or a shot id like s3)")

    a = one(start, False)
    b = one(end if end not in (None, "") else start, True)
    if b <= a:
        raise VidxError(f"empty range {rng(a, b)} (end must be after start)")
    if a >= v.duration:
        raise VidxError(f"start {a:.1f} is past the end ({v.duration:.1f}s)")
    return max(0.0, a), min(b, v.duration)


def get_segment(video_id: str, start, end=None, fields: str = "words,events,shots") -> str:
    a, b = resolve_range(video_id, start, end)
    want = {f.strip() for f in fields.split(",") if f.strip()}
    bad = want - {"words", "events", "shots", "wordtimes"}
    if bad:
        raise VidxError(f"unknown fields {','.join(sorted(bad))} (use words,events,shots,wordtimes)")
    lines = [f"{rng(a, b)} ({b - a:.1f}s)"]
    if "shots" in want:
        ss = [s for s in store.get_shots(video_id) if s.start < b and s.end > a]
        for s in ss[:30]:
            d = shot_desc(s)
            lines.append(f"s{s.idx} {rng(s.start, s.end)} ({s.dur:.1f}s) motion {s.motion or 0:.1f}"
                         + (f' "{d}"' if d else ""))
        lines += more(len(ss), min(30, len(ss)))
    if "events" in want:
        evs = store.get_events(video_id, start=a, end=b)
        if evs:
            lines.append("events: " + "; ".join(f"{e.kind} {rng(e.start, e.end)}" for e in evs[:40])
                         + (f"; … {len(evs) - 40} more" if len(evs) > 40 else ""))
    if "words" in want or "wordtimes" in want:
        ws = store.get_words(video_id, a, b)
        if not ws:
            lines.append("words: (none)")
        elif "wordtimes" in want:  # per-word timing for precise cuts
            lines.append("words:")
            lines.append(" ".join(f"{w.text}@{w.start:.2f}" for w in ws))
        else:
            lines.append("words:")
            for ph in phrases(ws):
                lines.append(f"[{ph[0].start:.1f}] {words_text(ph)}")
    out = "\n".join(lines)
    if len(out) > SEGMENT_MAX_CHARS:
        cut = out[:SEGMENT_MAX_CHARS].rsplit("\n", 1)[0]
        last = re.findall(r"^\[([\d.]+)\]", cut, re.M)
        out = cut + f"\n… truncated; continue with get_segment({video_id}, {last[-1] if last else a:.1f}, {b:.1f})"
    return out


# ---------------------------------------------------------------- peek

MAX_TILES = 16


def _peek_source(target: str) -> tuple[str, str | None, float]:
    """-> (media path, video_id or None, duration)."""
    t = str(target).strip()
    if re.fullmatch(r"[vV]\d+", t):
        ver = store.get_version(t)
        p = ver.get("output_path")
        if not p or not os.path.exists(p):
            raise VidxError(f"{t} has no render yet (call render('{t}') first)")
        return p, None, float(ver["timeline"].get("out_duration") or 0)
    v = store.get_video(t)
    return v.path, v.id, v.duration


def peek(target: str, times=None, size: int = 384, grid: bool = True) -> tuple[list[bytes], str]:
    """Returns (jpeg images, caption). One contact sheet by default (max 4x4)."""
    path, video_id, dur = _peek_source(target)
    size = int(min(max(size, 96), 768))
    shots = store.get_shots(video_id) if video_id else []
    if isinstance(times, str):
        times = [x for x in re.split(r"[,\s]+", times) if x]
    times = list(times or [])
    pts: list[tuple[float, str | None]] = []
    if not times:  # default: evenly spread visual summary
        if shots:
            pick = np.linspace(0, len(shots) - 1, min(MAX_TILES, len(shots))).round().astype(int)
            pts = [((shots[i].start + shots[i].end) / 2, f"s{shots[i].idx}") for i in sorted(set(pick))]
            if len(pts) < 4 < dur:
                pts = [(t, None) for t in np.linspace(dur * 0.05, dur * 0.95, 8)]
        else:
            pts = [(t, None) for t in np.linspace(dur * 0.05, dur * 0.95, 8)]
    for x in times:
        if isinstance(x, str) and x.strip().lower().startswith("s"):
            if not shots:
                raise VidxError("shot ids only work with a video_id, not a version (use times)")
            s = parse_shot(x, shots)
            pts.append(((s.start + s.end) / 2, f"s{s.idx}"))
        else:
            try:
                t = float(x)
            except (TypeError, ValueError):
                raise VidxError(f"bad time {x!r} (use seconds or shot ids like s3)")
            if not 0 <= t <= dur + 0.05:
                raise VidxError(f"time {t:.1f} outside 0-{dur:.1f}")
            sid = next((f"s{s.idx}" for s in shots if s.start <= t < s.end), None)
            pts.append((min(t, max(dur - 0.05, 0)), sid))
    note = ""
    if len(pts) > MAX_TILES:
        keep = np.linspace(0, len(pts) - 1, MAX_TILES).round().astype(int)
        note = f" (sampled {MAX_TILES} of {len(pts)})"
        pts = [pts[i] for i in keep]
    frames = thumbs.grab_frames(path, [t for t, _ in pts], long_side=size)
    labels = [thumbs.label(t, s) for t, s in pts]
    imgs = [thumbs.contact_sheet(frames, labels)] if grid else [f for f in frames if f is not None]
    out = []
    for im in imgs:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
        out.append(buf.getvalue())
    cap = f"{len(pts)} frames from {target}{note}: " + ", ".join(labels)
    return out, cap
