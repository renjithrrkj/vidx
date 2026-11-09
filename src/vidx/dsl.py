"""Edit DSL v1: script -> timeline (ordered source segments + global render options).

    base all | base none
    drop 30.0-31.2 | drop s7 | drop s7-s9 | drop silence>1.2 | drop black
    keep 120.0-150.0 | keep s3
    snap words
    pad 0.15
    trans xfade 0.3 | trans cut
    crop 9:16 center | crop off
    speed 1.25
    match s3 0.8                      colour-match every clip to shot s3 (or a time), strength 0-1
    caption 3.0 2.5 Some text         burn in text at output time 3.0 for 2.5s
    still photo.jpg 3.0 at 12.5 motion zoom_in fit cover
                                      a still as a clip, placed at 12.5 of the edit without stills (or at end)
    overlay logo.png 3.0 2.5 pos top_right width 0.2 fade 0.3 opacity 0.9 motion slide_left
                                      an image layer at output time 3.0 for 2.5s (`all` = to the end of the edit)
    sfx impact_01 at 3.2 gain -6      bundled sound at output time 3.2
    music drone_02 gain -18           bundled bed under the whole edit
    cuts hype                         a style pack picks a cut for each join (seed optional: cuts hype 7)
    cut whip_left | cut 3 smash       one cut for every join, or for join 3 (see `vidx cuts`)

Pipeline: base -> keep/drop (line order) -> snap -> pad -> merge (<0.1s gaps) -> discard (<0.3s).
Times accept seconds (12.5) or mm:ss.s (1:02.5).
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from .store import VidxError, Word

Seg = tuple[float, float]

MERGE_GAP = 0.1
MIN_SEG = 0.3
EPS = 1e-6


# ------------------------------------------------------------------ parsing


@dataclass
class Op:
    line: int
    cmd: str
    args: list[str]
    src: str = ""  # "parent" for ops inherited from the parent version


@dataclass
class Script:
    ops: list[Op] = field(default_factory=list)
    base: str = "all"
    snap: bool = False
    pad: float = 0.0
    trans: tuple[str, float] | None = None
    crop: str | None = None
    speed: float = 1.0
    match: tuple[str, float] | None = None
    captions: list = field(default_factory=list)
    stills: list = field(default_factory=list)
    overlays: list = field(default_factory=list)
    audio: list = field(default_factory=list)
    cut_all: str | None = None
    cut_at: dict = field(default_factory=dict)
    pack: tuple[str, int | None] | None = None


def parse_time(tok: str, line: int) -> float:
    tok = tok.strip()
    m = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", tok)
    if m:
        h, mm, ss = m.groups()
        return int(h or 0) * 3600 + int(mm) * 60 + float(ss)
    try:
        v = float(tok)
    except ValueError:
        raise VidxError(f"line {line}: bad time {tok!r} (use 12.5 or 1:02.5)")
    if v < 0:
        raise VidxError(f"line {line}: negative time {tok}")
    return v


def parse(text: str, src: str = "", sc: Script | None = None) -> Script:
    sc = sc or Script()
    seen_base = False
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]
        where = f"{src} line {n}".strip()

        def err(msg: str):
            return VidxError(f"{where}: {msg}")

        if cmd == "base":
            if args not in (["all"], ["none"]):
                raise err("base takes 'all' or 'none'")
            if seen_base:
                raise err("base given twice")
            seen_base = True
            sc.base = args[0]
        elif cmd in ("drop", "keep"):
            if len(args) != 1:
                raise err(f"{cmd} takes one target: a range 1.0-2.5, a shot s3 / s3-s5, or an event like silence>1.2")
            sc.ops.append(Op(n, cmd, args, src))
        elif cmd == "snap":
            if args not in (["words"], ["off"]):
                raise err("snap takes 'words' (or 'off')")
            sc.snap = args[0] == "words"
        elif cmd == "pad":
            if len(args) != 1:
                raise err("pad takes seconds, e.g. pad 0.15")
            sc.pad = parse_time(args[0], n)
            if sc.pad > 2:
                raise err(f"pad {sc.pad} too large (max 2.0)")
        elif cmd == "trans":
            if args == ["cut"]:
                sc.trans = None
            elif len(args) == 2 and args[0] == "xfade":
                d = parse_time(args[1], n)
                if not 0.04 <= d <= 2:
                    raise err("xfade duration must be 0.04-2.0s")
                sc.trans = ("xfade", d)
            else:
                raise err("trans takes 'xfade <sec>' or 'cut'")
        elif cmd == "crop":
            if args == ["off"]:
                sc.crop = None
            elif args and re.fullmatch(r"\d+:\d+", args[0]) and (len(args) == 1 or args[1:] == ["center"]):
                w, h = map(int, args[0].split(":"))
                if not w or not h:
                    raise err("crop ratio must be non-zero, e.g. 9:16")
                sc.crop = args[0]
            else:
                raise err("crop takes a ratio + 'center', e.g. crop 9:16 center (face-tracked crop is v2)")
        elif cmd == "speed":
            if len(args) != 1:
                raise err("speed takes one number, e.g. speed 1.25")
            sc.speed = parse_time(args[0], n)
            if not 0.25 <= sc.speed <= 4:
                raise err("speed must be 0.25-4.0")
        elif cmd == "cuts":
            if len(args) not in (1, 2) or (len(args) == 2 and not args[1].isdigit()):
                raise err("cuts takes a pack and an optional seed: cuts hype 7")
            sc.pack = (args[0], int(args[1]) if len(args) == 2 else None)
        elif cmd == "cut":
            if len(args) == 1:
                sc.cut_all = args[0]
            elif len(args) == 2 and args[0].isdigit():
                sc.cut_at[int(args[0])] = args[1]
            else:
                raise err("cut takes a name, or a join number and a name: cut smash / cut 3 smash")
        elif cmd == "match":
            if len(args) not in (1, 2):
                raise err("match takes a shot or time, and an optional strength: match s3 0.8")
            strength = float(args[1]) if len(args) == 2 else 0.8
            if not 0 < strength <= 1:
                raise err("match strength must be 0-1")
            sc.match = (args[0], strength)
        elif cmd == "caption":
            if len(args) < 3:
                raise err('caption takes time, duration, text: caption 3.0 2.5 Hello there')
            sc.captions.append({"at": parse_time(args[0], n), "for": parse_time(args[1], n), "text": " ".join(args[2:])})
        elif cmd in ("still", "overlay"):
            try:
                toks = shlex.split(line)[1:]  # quotes, for paths with spaces
            except ValueError:
                raise err(f"{cmd}: unmatched quote")
            head = 2 if cmd == "still" else 3
            allowed = {"at", "motion", "fit"} if cmd == "still" else {"pos", "width", "fade", "opacity", "motion"}
            opts = dict(zip(toks[head::2], toks[head + 1::2]))
            if len(toks) < head or (len(toks) - head) % 2 or set(opts) - allowed:
                raise err("still takes file, seconds, then at/motion/fit: still photo.jpg 3.0 at 12.5 motion zoom_in" if cmd == "still"
                          else "overlay takes file, time, seconds or all, then pos/width/fade/opacity/motion: "
                               "overlay logo.png 3.0 2.5 pos top_right width 0.2")
            if cmd == "still":
                at = opts.pop("at", "0")
                sc.stills.append({"image": toks[0], "for": parse_time(toks[1], n),
                                  "at": float("inf") if at == "end" else parse_time(at, n), **opts})
            else:
                if "pos" in opts and "," in opts["pos"]:
                    try:
                        opts["pos"] = [float(p) for p in opts["pos"].split(",")]
                    except ValueError:
                        raise err("overlay pos is a name or x,y as 0-1 of the frame")
                try:
                    nums = {k: float(opts[k]) for k in ("width", "fade", "opacity") if k in opts}
                except ValueError:
                    raise err("overlay width, fade and opacity are numbers")
                sc.overlays.append({**opts, **nums, "image": toks[0], "at": parse_time(toks[1], n),
                                    "for": None if toks[2] == "all" else parse_time(toks[2], n)})
        elif cmd in ("sfx", "music"):
            opts = dict(zip(args[1::2], args[2::2]))
            if not args or len(args) % 2 == 0 or set(opts) - {"at", "gain"}:
                raise err(f"{cmd} takes a name, then 'at <sec>' / 'gain <dB>': {cmd} impact_01 at 3.2 gain -6")
            entry = {cmd: args[0], "gain": float(opts.get("gain", -6 if cmd == "sfx" else -18))}
            if cmd == "sfx":
                entry["at"] = parse_time(opts.get("at", "0"), n)
            sc.audio.append(entry)
        else:
            raise err(f"unknown command {cmd!r} (base, drop, keep, snap, pad, trans, crop, speed, match, caption, still, overlay, sfx, music, cuts, cut)")
    return sc


# ------------------------------------------------------------------ segment math (pure)


def subtract(segs: list[Seg], a: float, b: float) -> list[Seg]:
    out: list[Seg] = []
    for s, e in segs:
        if b <= s or a >= e:
            out.append((s, e))
            continue
        if a > s:
            out.append((s, a))
        if b < e:
            out.append((b, e))
    return out


def union_sorted(segs: list[Seg], a: float, b: float) -> list[Seg]:
    allsegs = sorted(segs + [(a, b)])
    out: list[Seg] = []
    for s, e in allsegs:
        if out and s <= out[-1][1] + EPS:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def merge_close(segs: list[Seg], gap: float = MERGE_GAP) -> list[Seg]:
    """Merge consecutive segments that continue forward in source time with a gap < `gap`."""
    out: list[Seg] = []
    for s, e in segs:
        if out and out[-1][1] - EPS <= s < out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        elif out and s < out[-1][1] and s >= out[-1][0] and e > out[-1][1] - EPS and gap > 0:
            out[-1] = (out[-1][0], e)  # overlapping forward (e.g. after pad)
        else:
            out.append((s, e))
    return out


def drop_short(segs: list[Seg], min_len: float = MIN_SEG) -> tuple[list[Seg], list[Seg]]:
    keep = [s for s in segs if s[1] - s[0] >= min_len - EPS]
    gone = [s for s in segs if s[1] - s[0] < min_len - EPS]
    return keep, gone


def _word_at(words: list[Word], t: float, tol: float = 0.02) -> int | None:
    lo, hi = 0, len(words)
    while lo < hi:  # first word with end > t
        mid = (lo + hi) // 2
        if words[mid].end <= t:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(words) and words[lo].start + tol < t < words[lo].end - tol:
        return lo
    return None


def acoustic_words(words: list[Word], silences: list[Seg]) -> list[Word]:
    """Clip ASR word intervals to where there is actually sound.

    Whisper word stamps can drift into measured silence (worse in translate mode); silencedetect is
    acoustic ground truth, so a word's interval is trimmed to its non-silent part and a word lying
    entirely inside a silence is dropped. Used by snap, pad and the mid-word check.
    """
    if not silences:
        return words
    sil = sorted(silences)
    out, j = [], 0
    for w in words:
        a, b = w.start, w.end
        while j < len(sil) and sil[j][1] <= a:
            j += 1
        k = j
        while k < len(sil) and sil[k][0] < b:
            s0, s1 = sil[k]
            if s0 <= a and s1 >= b:  # fully inside silence: phantom word
                a = b
                break
            if s0 <= a < s1:
                a = s1
            elif s0 < b <= s1:
                b = s0
            k += 1
        if b - a > 0.01:
            out.append(Word(a, b, w.text))
    return out


def snap_point(t: float, role: str, words: list[Word], duration: float, handle: float = 0.15) -> float:
    """Move a cut point that falls inside a word into the nearest word gap.

    role='start': the kept segment begins here -> sit just before the following word.
    role='end'  : the kept segment ends here   -> sit just after the preceding word.
    """
    i = _word_at(words, t)
    if i is None:
        return t
    w = words[i]
    if t - w.start <= w.end - t:  # nearer to the word's start: use the gap before it
        g0 = words[i - 1].end if i > 0 else max(0.0, w.start - 0.3)
        g1 = w.start
    else:  # use the gap after it
        g0 = w.end
        g1 = words[i + 1].start if i + 1 < len(words) else min(duration, w.end + 0.3)
    if g1 < g0:  # overlapping word stamps: split the difference
        return (g0 + g1) / 2
    mid = (g0 + g1) / 2
    return max(mid, g1 - handle) if role == "start" else min(mid, g0 + handle)


def snap_segments(segs: list[Seg], words: list[Word], duration: float) -> list[Seg]:
    out = []
    for s, e in segs:
        s2 = s if s <= EPS else snap_point(s, "start", words, duration)
        e2 = e if e >= duration - EPS else snap_point(e, "end", words, duration)
        if e2 > s2 + EPS:
            out.append((s2, e2))
    return out


def pad_segments(segs: list[Seg], pad: float, duration: float, words: list[Word] | None = None) -> list[Seg]:
    """Add handles; never extend into a word that lies outside the segment (no half-syllables)."""
    out = []
    for s, e in segs:
        s2, e2 = max(0.0, s - pad), min(duration, e + pad)
        if words:
            for w in words:
                if w.end <= s + 0.02 and w.end > s2:  # word ends inside the new head handle
                    s2 = min(s, max(s2, w.end + 0.02))
                if w.start >= e - 0.02 and w.start < e2:  # word starts inside the tail handle
                    e2 = max(e, min(e2, w.start - 0.02))
        out.append((s2, e2))
    return out


def out_duration(segs: list[Seg], speed: float = 1.0, trans: tuple[str, float] | None = None) -> float:
    d = sum(e - s for s, e in segs) / speed
    if trans and len(segs) > 1:
        d -= trans[1] * (len(segs) - 1)
    return d


def _per_seg(speed, n: int) -> list[float]:
    return list(speed) if isinstance(speed, (list, tuple)) else [speed] * n


def _per_join(trans, n: int) -> list[float]:
    """trans: None, ("xfade", d) for every join, or an explicit list of join lengths."""
    if not trans:
        return [0.0] * max(n - 1, 0)
    if isinstance(trans[0], str):
        return [float(trans[1])] * max(n - 1, 0)
    return [float(x) for x in trans]


def out_positions(segs: list[Seg], speed=1.0, trans=None) -> list[tuple[float, float]]:
    """Output (start, end) of each source segment. A transition overlaps its neighbours."""
    speeds, joins = _per_seg(speed, len(segs)), _per_join(trans, len(segs))
    out, t = [], 0.0
    for k, (s, e) in enumerate(segs):
        start = t if k == 0 else t - joins[k - 1]
        end = start + (e - s) / speeds[k]
        out.append((start, end))
        t = end
    return out


def src_to_out(t_src: float, segs: list[Seg], speed=1.0, trans=None) -> float | None:
    speeds = _per_seg(speed, len(segs))
    for k, ((s, e), (o0, _)) in enumerate(zip(segs, out_positions(segs, speed, trans))):
        if s - EPS <= t_src <= e + EPS:
            return o0 + (t_src - s) / speeds[k]
    return None


def out_to_src(t_out: float, segs: list[Seg], speed=1.0, trans=None) -> float | None:
    speeds = _per_seg(speed, len(segs))
    for k, ((s, e), (o0, o1)) in enumerate(zip(segs, out_positions(segs, speed, trans))):
        if o0 - EPS <= t_out <= o1 + EPS:
            return s + (t_out - o0) * speeds[k]
    return None


# ------------------------------------------------------------------ resolve (needs the index)


def _resolve_target(tok: str, line: int, ctx) -> list[Seg]:
    """tok -> source ranges. ctx gives duration, shots, events()."""
    t = tok.strip().lower()
    m = re.fullmatch(r"s(\d+)(?:-s(\d+))?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        n = len(ctx.shots)
        for i in (a, b):
            if not 1 <= i <= n:
                raise VidxError(f"line {line}: unknown shot s{i} (video has s1-s{n})")
        if b < a:
            raise VidxError(f"line {line}: shot range s{a}-s{b} is backwards")
        return [(ctx.shots[a - 1].start, ctx.shots[b - 1].end)]
    from .query import event_filter, parse_event_query

    ev = parse_event_query(t)
    if ev:
        kind, op, val = ev
        return [(e.start, e.end) for e in event_filter(ctx.events(kind), op, val)]
    m = re.fullmatch(r"([\d:.]+)-([\d:.]+)", t)
    if m:
        a, b = parse_time(m.group(1), line), parse_time(m.group(2), line)
        if b <= a:
            raise VidxError(f"line {line}: empty range {tok} (end must be after start)")
        if a >= ctx.duration:
            raise VidxError(f"line {line}: {tok} starts after the end ({ctx.duration:.1f}s)")
        return [(a, min(b, ctx.duration))]
    raise VidxError(f"line {line}: bad target {tok!r} (use 1.0-2.5, s3, s3-s5, silence>1.2, black)")


def resolve(sc: Script, ctx) -> dict:
    D = ctx.duration
    warnings: list[str] = []
    segs: list[Seg] = [(0.0, D)] if sc.base == "all" else []
    for op in sc.ops:
        ranges = _resolve_target(op.args[0], op.line, ctx)
        if not ranges and op.cmd == "drop":
            warnings.append(f"line {op.line}: '{op.cmd} {op.args[0]}' matched nothing")
        for a, b in ranges:
            if op.cmd == "drop":
                segs = subtract(segs, a, b)
            elif sc.base == "all":
                segs = union_sorted(segs, a, b)
            else:
                segs.append((a, b))
    if sc.base == "none" and not any(op.cmd == "keep" for op in sc.ops):
        raise VidxError("base none needs at least one keep line")
    words = ctx.words() if (sc.snap or sc.pad) else []
    if sc.snap:
        if not words:
            warnings.append("snap words: video has no transcript, skipped")
        segs = snap_segments(segs, words, D)
    if sc.pad:
        segs = pad_segments(segs, sc.pad, D, words)
    segs = merge_close(segs)
    segs, gone = drop_short(segs)
    if gone:
        warnings.append(f"discarded {len(gone)} segment(s) shorter than {MIN_SEG}s: "
                        + ", ".join(f"{a:.1f}-{b:.1f}" for a, b in gone[:5]) + (" …" if len(gone) > 5 else ""))
    if not segs:
        raise VidxError("edit leaves nothing to render (check drops / keeps)")
    trans = sc.trans
    if trans and len(segs) > 1:
        shortest = min(e - s for s, e in segs)
        if trans[1] > shortest / 2:
            new = round(max(0.04, shortest / 2 - 0.01), 2)
            warnings.append(f"xfade {trans[1]}s longer than half the shortest segment ({shortest:.1f}s); using {new}s")
            trans = (trans[0], new)
    tl = {
        "video_id": ctx.video_id,
        "source_duration": round(D, 3),
        "fps": ctx.fps,
        "trans": list(trans) if trans else None,
        "crop": sc.crop,
        "speed": sc.speed,
        "snap": sc.snap,
        "pad": sc.pad,
        "warnings": warnings,
        "audio": sc.audio,
    }
    tl["clips"] = build_clips(segs, sc, trans, ctx)
    from .timeline import clamp, finish

    clamp(tl["clips"], warnings)
    return finish(tl)


def build_clips(segs: list[Seg], sc: Script, trans, ctx) -> list[dict]:
    grade = {}
    if sc.match:
        (a, b), = _resolve_target(sc.match[0] if "-" in sc.match[0] or sc.match[0].lower().startswith("s")
                                  else f"{sc.match[0]}-{float(parse_time(sc.match[0], 0)) + 1}", 0, ctx)
        grade = {"match": {"from": a, "to": b, "strength": sc.match[1], "video": ctx.video_id}}
    clips = []
    for i, (a, b) in enumerate(segs):
        c: dict = {"in": round(a, 3), "out": round(b, 3), "video": ctx.video_id, "speed": sc.speed}
        if sc.crop:
            c["crop"] = sc.crop
        if grade:
            c["grade"] = grade
        clips.append(c)
    if sc.stills:
        clips = insert_stills(clips, sc.stills, trans, grade, ctx.fps)
    if trans:
        for c in clips[:-1]:
            c["transition"] = {"type": "dissolve", "dur": trans[1]}
    if sc.pack or sc.cut_all or sc.cut_at:
        from . import cuts

        pack, seed = sc.pack or (None, None)
        cuts.apply(clips, sc.cut_all, sc.cut_at, pack, seed, ctx)
    from .timeline import joins_of, overlay

    pos = out_positions([(c["in"], c["out"]) for c in clips], [c["speed"] for c in clips], joins_of(clips))
    for cap in sc.captions:  # output time -> the clip showing at that moment
        k = next((i for i, (o0, o1) in enumerate(pos) if o0 <= cap["at"] < o1), len(pos) - 1)
        clips[k].setdefault("captions", []).append({**cap, "at": round(cap["at"] - pos[k][0], 3)})
    total = pos[-1][1]
    for n, ov in enumerate(sc.overlays, 1):
        if ov["at"] >= total:
            raise VidxError(f"overlay {n}: starts at {ov['at']:.1f}, but the edit ends at {total:.1f}")
        length = ov["for"] if ov["for"] is not None else total - ov["at"]
        o = overlay({**ov, "for": length}, f"overlay {n}")
        for k, (o0, o1) in enumerate(pos):  # every clip it touches gets the same layer, so fades happen only at its ends
            if o["at"] < o1 and o["at"] + length > o0:
                clips[k].setdefault("overlays", []).append({**o, "at": round(o["at"] - o0, 3)})
    return clips


def insert_stills(clips: list[dict], stills: list[dict], trans, grade: dict, fps: float) -> list[dict]:
    """Place each still at its time in the edit without stills, splitting the clip it lands in. Working from the
    last still back keeps the earlier times valid; stills sharing a time keep their script order."""
    from .timeline import still

    out = list(clips)
    for n, s in sorted(enumerate(stills, 1), key=lambda x: (x[1]["at"], x[0]), reverse=True):
        clip = still({k: v for k, v in s.items() if k != "at"}, f"still {n}", fps)
        if grade:
            clip["grade"] = grade
        pos = out_positions([(c["in"], c["out"]) for c in out], [c["speed"] for c in out], trans)
        k = next((i for i, (_, o1) in enumerate(pos) if s["at"] < o1), None)
        if k is None:
            out.append(clip)
            continue
        c, (o0, o1) = out[k], pos[k]
        if s["at"] - o0 < MERGE_GAP:
            out.insert(k, clip)
        elif o1 - s["at"] < MERGE_GAP:
            out.insert(k + 1, clip)
        else:
            cut = round(c["in"] + (s["at"] - o0) * c["speed"], 3)
            out[k: k + 1] = [{**c, "out": cut}, clip, {**c, "in": cut}]
    return out


class IndexCtx:
    """Lazy view of the index used by resolve()."""

    def __init__(self, video_id: str):
        from . import store

        self.store = store
        v = store.get_video(video_id)
        self.video_id, self.duration, self.fps = v.id, v.duration, v.fps
        self.shots = store.get_shots(video_id)
        self._words = None

    def events(self, kind: str):
        return self.store.get_events(self.video_id, kind)

    def words(self):
        if self._words is None:
            sil = [(e.start, e.end) for e in self.store.get_events(self.video_id, "silence")]
            self._words = acoustic_words(self.store.get_words(self.video_id), sil)
        return self._words
