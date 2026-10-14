"""Automated QA on a rendered version. <=15 lines, 'OK' when clean."""
from __future__ import annotations

import os
import subprocess

from . import store, util
from .analyzers.audio import detect_cmd, parse_detect_log
from .dsl import acoustic_words, out_to_src, src_to_out
from .render import PRESETS, frame_snap
from .store import VidxError, Word

MAX_LINES = 15
POP_WIN = 0.03  # seconds of audio compared on each side of a join
POP_DB = 12.0
WORD_TOL = 0.05


def word_at(words: list[Word], t: float, tol: float = WORD_TOL) -> Word | None:
    for w in words:
        if w.start + tol < t < w.end - tol:
            return w
        if w.start > t:
            break
    return None


def mid_word_cuts(segs, words: list[Word]) -> list[tuple[int, float, Word]]:
    """Joins (k = index of the segment after the join) where an out- or in-point lands inside a word."""
    out = []
    for k in range(1, len(segs)):
        for t in (segs[k - 1][1], segs[k][0]):
            w = word_at(words, t)
            if w:
                out.append((k, t, w))
                break
    return out


def audio_pops(segs, samples, sr: int) -> list[tuple[int, float]]:
    """Hard-cut joins where RMS differs by more than POP_DB across the join."""
    out = []
    n = int(POP_WIN * sr)
    for k in range(1, len(segs)):
        e = int(segs[k - 1][1] * sr)
        s = int(segs[k][0] * sr)
        before = util.rms_db(samples[max(0, e - n): e])
        after = util.rms_db(samples[s: s + n])
        diff = abs(before - after)
        if diff > POP_DB and max(before, after) > -45:  # ignore jumps between two near-silences
            out.append((k, round(diff, 1)))
    return out


def check(version_id: str) -> str:
    ver = store.get_version(version_id)
    vname = f"v{ver['id']}"
    path, preset = ver.get("output_path"), ver.get("preset") or "original"
    if not path or not os.path.exists(path):
        raise VidxError(f"{vname} not rendered (call render('{vname}') first)")
    tl = ver["timeline"]
    v = store.get_video(ver["video_id"])
    segs = frame_snap(tl["segments"], tl["fps"])
    speed = float(tl.get("speed") or 1.0)
    trans = tl.get("trans")

    def o(t_src: float) -> float:  # output time of a source time
        r = src_to_out(t_src, segs, speed, trans)
        return r if r is not None else 0.0

    def join_out(k: int) -> float:
        return o(segs[k][0])

    info = util.ffprobe(path)
    dur = float(info["format"]["duration"])
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    p = subprocess.run(detect_cmd(path, has_audio, hwaccel=False), capture_output=True, text=True)
    events, meta = parse_detect_log(p.stderr, dur)
    src_freezes = store.get_events(v.id, "freeze")
    src_blacks = store.get_events(v.id, "black")

    issues: list[str] = []
    for e in events:
        if e.kind == "black":
            s = out_to_src((e.start + e.end) / 2, segs, speed, trans)
            in_src = s is not None and any(b.start - 0.1 <= s <= b.end + 0.1 for b in src_blacks)
            fix = (f"drop {s - (e.end - e.start) / 2:.1f}-{s + (e.end - e.start) / 2:.1f}" if s is not None else "check joins")
            issues.append(f"black {util.rng(e.start, e.end)} ({e.end - e.start:.1f}s)"
                          + (" from source" if in_src else "") + f" -> {fix}")
        elif e.kind == "freeze":
            s = out_to_src((e.start + e.end) / 2, segs, speed, trans)
            if s is not None and any(f.start - 0.2 <= s <= f.end + 0.2 for f in src_freezes):
                continue  # the source already froze here (static slide, paused screen)
            issues.append(f"freeze {util.rng(e.start, e.end)} ({e.end - e.start:.1f}s) -> check the join near"
                          f" {e.start:.1f}" + (f" (src {s:.1f})" if s is not None else ""))

    sil = [(e.start, e.end) for e in store.get_events(v.id, "silence")]
    words = acoustic_words(store.get_words(v.id), sil)
    mw = mid_word_cuts(segs, words)
    for k, t, w in mw[:4]:
        issues.append(f'mid_word_cut at {join_out(k):.1f} (src {t:.1f} inside "{w.text.strip()}" '
                      f'{w.start:.1f}-{w.end:.1f}) -> add `snap words`')
    if len(mw) > 4:
        issues.append(f"… {len(mw) - 4} more mid-word cuts -> add `snap words`")

    if not trans and v.has_audio and len(segs) > 1:
        wav = v.dir / "audio16k.wav"
        if not wav.exists():
            util.extract_audio16k(v.path, wav)
        reader = util.WavReader(wav)  # reads only the 30 ms around each join
        pops = audio_pops(segs, reader, reader.sr)
        fix = "`trans xfade 0.08`" if (tl.get("pad") or 0) >= 0.1 else "`pad 0.1` or `trans xfade 0.08`"
        for k, d in pops[:3]:
            issues.append(f"audio_pop at {join_out(k):.1f} ({d:.0f} dB jump) -> {fix}")
        if len(pops) > 3:
            issues.append(f"… {len(pops) - 3} more audio jumps -> `trans xfade 0.08`")

    W, H, target, maxdur = PRESETS.get(preset, PRESETS["original"])
    lufs = meta.get("lufs")
    if has_audio and lufs is not None and target is not None and abs(lufs - target) > 2:
        issues.append(f"loudness {lufs:.1f} LUFS, target {target:.0f}±2 for {preset} -> re-render {preset}")
    elif has_audio and lufs is not None and lufs < -30:
        issues.append(f"loudness {lufs:.1f} LUFS is very quiet -> render reel_9x16 or yt_16x9 to normalize")
    if maxdur and dur > maxdur:
        issues.append(f"duration {dur:.1f}s exceeds {preset} limit {maxdur:.0f}s -> drop "
                      f"{dur - maxdur:.1f}s (e.g. `drop silence>0.6` or drop shots)")
    if abs(dur - tl["out_duration"]) > 0.3:
        issues.append(f"duration {dur:.1f}s differs from timeline {tl['out_duration']:.1f}s -> re-render")

    stats = f"{util.mmss(dur)} ({dur:.1f}s), {len(segs)} segments" + (f", {lufs:.1f} LUFS" if lufs is not None else "")
    if not issues:
        return f"OK {vname} {preset}: {stats}"
    head = f"{vname} {preset}: {len(issues)} issue{'s' if len(issues) != 1 else ''} | {stats}"
    lines = [head] + issues
    if len(lines) > MAX_LINES:
        lines = lines[: MAX_LINES - 1] + [f"… {len(lines) - MAX_LINES + 1} more"]
    return "\n".join(lines)
