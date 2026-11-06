"""The cut library: how one clip gets to the next, and style packs that choose per join.

An edit that repeats one transition feels mechanical, and more wipes don't fix it. Editors vary the
cut by what is happening at the join, so a pack looks at each join (same shot? speech crossing it? fast
motion? a big colour change?) and picks accordingly. Hard cuts stay the most common choice on purpose.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from . import config
from .store import VidxError


@dataclass(frozen=True)
class Cut:
    kind: str                  # hard | xfade | audio
    note: str
    xfade: str = ""            # ffmpeg xfade transition
    dur: float = 0.0           # default length (for audio cuts: how far the sound leads or trails)
    sfx: str = ""              # sound played at the join
    gain: float = -8.0
    zoom: float = 0.0          # punch the next clip in by this factor
    audio: str = ""            # lead (J-cut) or lag (L-cut)


LIBRARY: dict[str, Cut] = {
    "hard": Cut("hard", "straight cut"),
    "smash": Cut("hard", "hard cut with an impact hit: a sudden change of energy", sfx="impact_05"),
    "punch_in": Cut("hard", "hard cut that zooms in 15%, hides a jump cut in one take", zoom=1.15),
    "punch_big": Cut("hard", "hard cut that zooms in 30%, for emphasis", zoom=1.3),
    "j_cut": Cut("audio", "next clip's sound starts before its picture: pulls the viewer forward", dur=0.5, audio="lead"),
    "l_cut": Cut("audio", "this clip's sound carries over the next picture: a reaction shot feel", dur=0.5, audio="lag"),
    "dissolve": Cut("xfade", "soft blend: time passing", "fade", 0.5),
    "dissolve_slow": Cut("xfade", "long blend: a reflective beat", "fadeslow", 1.0),
    "dip_black": Cut("xfade", "fade through black: end of a chapter", "fadeblack", 0.6),
    "dip_white": Cut("xfade", "fade through white: memory, dream", "fadewhite", 0.5),
    "flash": Cut("xfade", "quick white flash: a camera pop, an impact", "fadewhite", 0.13),
    "whip_left": Cut("xfade", "fast slide with a whoosh", "slideleft", 0.22, sfx="whoosh_02", gain=-12),
    "whip_right": Cut("xfade", "fast slide with a whoosh", "slideright", 0.22, sfx="whoosh_02", gain=-12),
    "whip_up": Cut("xfade", "fast vertical slide with a whoosh", "slideup", 0.22, sfx="whoosh_02", gain=-12),
    "zoom": Cut("xfade", "zoom through into the next shot", "zoomin", 0.3, sfx="whoosh_01", gain=-14),
    "blur": Cut("xfade", "horizontal smear", "hblur", 0.3),
    "wipe": Cut("xfade", "classic wipe", "wipeleft", 0.35),
    "smooth": Cut("xfade", "soft-edged wipe", "smoothleft", 0.45),
    "cover": Cut("xfade", "next shot slides over this one", "coverleft", 0.3),
    "reveal": Cut("xfade", "this shot slides away to reveal the next", "revealleft", 0.3),
    "iris": Cut("xfade", "circle opening", "circleopen", 0.45),
    "radial": Cut("xfade", "clock wipe", "radial", 0.45),
    "squeeze": Cut("xfade", "picture squeezes away", "squeezeh", 0.25),
    "slices": Cut("xfade", "horizontal slices", "hlslice", 0.35),
    "wind": Cut("xfade", "ragged wind edge", "hrwind", 0.35),
    "pixel": Cut("xfade", "digital break-up", "pixelize", 0.3),
    "gray_fade": Cut("xfade", "drains to gray, then back", "fadegrays", 0.6),
}

PACKS: dict[str, dict[str, float]] = {
    "clean": {"hard": 7, "j_cut": 1.5, "l_cut": 1.5, "dissolve": 0.5},
    "documentary": {"hard": 5, "j_cut": 2, "l_cut": 2, "dissolve": 1, "dip_black": 0.4},
    "comedy": {"hard": 5, "punch_in": 3, "punch_big": 1, "smash": 1, "j_cut": 0.7},
    "hype": {"hard": 3, "smash": 1.5, "punch_in": 1.5, "whip_left": 1, "whip_right": 1, "zoom": 1, "flash": 1,
             "blur": 0.5, "cover": 0.4},
    "trailer": {"hard": 3, "dip_black": 2, "flash": 1.5, "smash": 1.5, "l_cut": 0.7},
}


def get(name: str) -> Cut:
    if name in ("xfade", "fade"):  # older timelines
        name = "dissolve"
    if name not in LIBRARY:
        raise VidxError(f"unknown cut {name!r} (use {', '.join(sorted(LIBRARY))})")
    return LIBRARY[name]


def describe() -> str:
    lines = [f"{n:<14}{c.note}" for n, c in LIBRARY.items()]
    lines.append("")
    lines += [f"pack {n:<13}" + ", ".join(k for k in w) for n, w in PACKS.items()]
    return "\n".join(lines)


def join_len(t: dict | None) -> float:
    """How much two clips overlap in the output at this transition."""
    if not t:
        return 0.0
    c = get(t.get("type", "dissolve"))  # a bare {"dur": 0.3} has always meant a dissolve
    return float(t.get("dur") or c.dur) if c.kind == "xfade" else 0.0


# ---------------------------------------------------------------- choosing per join


@dataclass
class JoinInfo:
    jump: bool = False        # same shot on both sides: a jump cut
    speech: bool = False      # someone is talking across the join
    motion: float = 0.0       # 0-1 frame change around the join
    delta: float = 0.0        # colour distance across the join


def join_info(a: dict, b: dict, ctx) -> JoinInfo:
    """What is happening at the join between clips a and b. Missing analysis just means less to go on."""
    info = JoinInfo()
    if a.get("image") or b.get("image"):  # a still's times aren't times in the analysed video
        return info
    fps = float(ctx.fps)
    same = a.get("video") == b.get("video")
    if same:
        for s in getattr(ctx, "shots", []):
            if s.start <= a["out"] - 0.05 <= s.end and s.start <= b["in"] + 0.05 <= s.end:
                info.jump = True
                break
    words = ctx.words() if same else []
    info.speech = any(a["out"] - 0.3 <= w.end <= a["out"] + 0.3 or b["in"] - 0.3 <= w.start <= b["in"] + 0.3 for w in words)
    try:
        base = config.video_dir(ctx.video_id)
        mo = np.load(base / "motion.npy").astype(np.float32)
        co = np.load(base / "color.npy").astype(np.float32)
        i, j = int(a["out"] * fps), int(b["in"] * fps)
        info.motion = float(np.concatenate([mo[max(0, i - 8): i], mo[j: j + 8]]).mean())
        if same and i < len(co) and j < len(co):
            info.delta = float(np.linalg.norm(co[max(0, i - 2), :3] - co[min(len(co) - 1, j + 2), :3]))
    except (OSError, ValueError):
        pass
    return info


def weigh(pack: dict[str, float], info: JoinInfo) -> dict[str, float]:
    w = dict(pack)

    def boost(names: tuple, by: float) -> None:
        for n in names:
            if n in w:
                w[n] *= by

    if info.jump:   # a jump cut needs hiding; a dissolve between identical framing looks like a mistake
        boost(("punch_in", "punch_big"), 5)
        boost(("dissolve", "dip_black", "whip_left", "whip_right", "zoom"), 0.2)
    else:
        for n in ("punch_in", "punch_big"):
            w.pop(n, None)  # a zoom means nothing across unrelated shots
    if info.speech:  # sound bridging a cut is what makes talk flow
        boost(("j_cut", "l_cut"), 3)
        boost(("flash", "smash", "dip_black"), 0.3)
    else:
        for n in ("j_cut", "l_cut"):
            w.pop(n, None)
    if info.motion > 0.06:  # movement already carries energy across the cut
        boost(("whip_left", "whip_right", "zoom", "blur", "smash"), 3)
    if info.delta > 60:     # a big visual change wants acknowledging, or a clean break
        boost(("dip_black", "flash", "smash"), 2)
    return w


def choose(clips: list[dict], pack_name: str, ctx, seed: int | None = None) -> list[str]:
    """One cut per join. Same inputs give the same edit, so re-renders stay cached."""
    if pack_name not in PACKS:
        raise VidxError(f"unknown cut pack {pack_name!r} (use {', '.join(PACKS)})")
    rng = random.Random(seed if seed is not None else hash_clips(clips))
    picks: list[str] = []
    for a, b in zip(clips, clips[1:]):
        w = weigh(PACKS[pack_name], join_info(a, b, ctx))
        if picks and picks[-1] != "hard":
            w.pop(picks[-1], None)            # never the same effect twice running
        if picks[-2:] == ["hard", "hard"] and len(w) > 1:
            w.pop("hard", None)               # nor a long unbroken run of straight cuts
        room = min((a["out"] - a["in"]) / float(a.get("speed") or 1), (b["out"] - b["in"]) / float(b.get("speed") or 1))
        w = {k: v for k, v in w.items() if LIBRARY[k].kind != "xfade" or LIBRARY[k].dur * 2.2 < room} or {"hard": 1.0}
        names = sorted(w)
        picks.append(rng.choices(names, [w[n] for n in names])[0])
    return picks


def hash_clips(clips: list[dict]) -> int:
    return sum(int(c["in"] * 100) * 31 + int(c["out"] * 100) * 17 for c in clips)


def apply(clips: list[dict], default: str | None, at: dict[int, str], pack: str | None, seed: int | None, ctx) -> None:
    """Set each clip's transition in place: a pack first, then `cut x` for all, then `cut 3 x` per join."""
    n = len(clips) - 1
    picks = choose(clips, pack, ctx, seed) if pack else [None] * n
    for k in range(n):
        name = at.get(k + 1) or default or picks[k]
        if name:
            get(name)
            clips[k]["transition"] = {"type": name}
    for k in at:
        if not 1 <= k <= n:
            raise VidxError(f"cut {k}: this edit has joins 1-{n}")


def expand(clips: list[dict], fps: float) -> tuple[list[dict], list[dict]]:
    """Turn cut names into render instructions. -> (clips with private _ fields, extra audio cues)."""
    from . import dsl, timeline

    out = [dict(c) for c in clips]
    cues: list[dict] = []
    segs = [(c["in"], c["out"]) for c in out]
    pos = dsl.out_positions(segs, [float(c.get("speed") or 1) for c in out], timeline.joins_of(out))
    zoomed = 0.0
    for k in range(len(out) - 1):
        t = out[k].get("transition")
        cut = get(t.get("type", "dissolve")) if t else LIBRARY["hard"]
        nxt = out[k + 1]
        if cut.zoom:  # alternate in and out, so repeated punches don't keep creeping closer
            zoomed = 0.0 if zoomed else cut.zoom
            if zoomed:
                nxt["zoom"] = zoomed
        elif zoomed and out[k].get("video") == nxt.get("video"):
            zoomed = 0.0
        if cut.sfx:
            cues.append({"sfx": cut.sfx, "at": round(max(pos[k + 1][0] - (0.12 if cut.kind == "xfade" else 0), 0), 3),
                         "gain": cut.gain})
        if cut.kind == "audio":
            d = float(t.get("dur") or cut.dur)
            d = min(d, (out[k]["out"] - out[k]["in"]) / 2, (nxt["out"] - nxt["in"]) / 2)
            if cut.audio == "lead" and nxt["in"] >= d:      # tail of this clip plays the next clip's sound
                out[k]["_tail"] = {"video": nxt["video"], "a": nxt["in"] - d, "d": d}
            elif cut.audio == "lag":                         # head of the next clip keeps this clip's sound
                nxt["_head"] = {"video": out[k]["video"], "a": out[k]["out"], "d": d}
    return out, cues
