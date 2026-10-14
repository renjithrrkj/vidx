"""Heuristic laughter / crowd-reaction detector (pluggable-analyzer demo, no ML).

A window counts as laughter-like when it is:
  * not covered by transcribed words (whisper doesn't transcribe laughs/applause),
  * reasonably energetic (within 12 dB of the loud-speech level),
  * noisy / broadband (high spectral flatness), and
  * bursty with a 3-8 Hz energy modulation (the "ha-ha-ha" syllable rate).
Good enough to find audience reactions in stand-up; expect false positives on music and wind.
"""
from __future__ import annotations

import numpy as np

from .. import store
from ..store import Event, VideoInfo
from . import AnalyzerResult, Context

HOP = 0.01   # 10 ms frames
WIN = 1.0    # analysis window
STEP = 0.25


def _frames(x: np.ndarray, sr: int, hop: float = HOP) -> np.ndarray:
    n = int(sr * hop)
    m = len(x) // n
    return x[: m * n].reshape(m, n)


def frame_features(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-10ms-frame RMS and spectral flatness."""
    fr = _frames(x, sr)
    if not len(fr):
        return np.zeros(0), np.zeros(0)
    rms = np.sqrt((fr.astype(np.float64) ** 2).mean(1) + 1e-12)
    spec = np.abs(np.fft.rfft(fr * np.hanning(fr.shape[1]), axis=1)) + 1e-9
    flat = np.exp(np.log(spec).mean(1)) / spec.mean(1)
    return rms, flat


def window_scores(rms: np.ndarray, flat: np.ndarray, speech_mask: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """-> (window start times, score 0..1) from per-frame features."""
    if len(rms) < 10:
        return np.zeros(0), np.zeros(0)
    db = 20 * np.log10(rms)
    ref = np.percentile(db, 95)
    wn, st = int(WIN / HOP), int(STEP / HOP)
    times, scores = [], []
    freqs = np.fft.rfftfreq(wn, HOP)
    band = (freqs >= 3) & (freqs <= 8)
    for i in range(0, len(db) - wn, st):
        seg = db[i: i + wn]
        if seg.mean() < ref - 12:
            score = 0.0
        else:
            env = rms[i: i + wn] - rms[i: i + wn].mean()
            mod = np.abs(np.fft.rfft(env)) ** 2
            mod_ratio = mod[band].sum() / (mod[1:].sum() + 1e-12)
            fl = np.median(flat[i: i + wn])
            score = float(np.clip((mod_ratio - 0.25) * 2, 0, 1) * np.clip((fl - 0.05) * 8, 0, 1))
        if speech_mask is not None and speech_mask[i: i + wn].mean() > 0.3:
            score *= 0.2
        times.append(i * HOP)
        scores.append(score)
    return np.asarray(times), np.asarray(scores)


def laugh_scores(x: np.ndarray, sr: int, speech_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """In-memory convenience wrapper (short clips / tests)."""
    return window_scores(*frame_features(x, sr), speech_mask)


def to_events(times: np.ndarray, scores: np.ndarray, thr: float = 0.35, min_dur: float = 0.7) -> list[Event]:
    ev, start, peak = [], None, 0.0
    for t, s in list(zip(times, scores)) + [(times[-1] + WIN if len(times) else 0, 0.0)]:
        if s >= thr:
            if start is None:
                start, peak = t, s
            peak = max(peak, s)
        elif start is not None:
            end = t - STEP + WIN
            if ev and start <= ev[-1].end + 0.25:  # windows overlap: extend the previous event
                prev = ev[-1]
                ev[-1] = Event("laugh", prev.start, round(end, 2), max(prev.value, round(peak, 2)))
            elif end - start >= min_dur:
                ev.append(Event("laugh", round(start, 2), round(end, 2), round(peak, 2)))
            start = None
    return ev


class Laugh:
    name = "laugh"
    optional = False

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        from .. import util

        wav = ctx.audio16k(video)
        if wav is None:
            return AnalyzerResult(skipped="no audio stream")
        reader = util.WavReader(wav)
        # stream 60s chunks (a multiple of HOP, so frames line up); keep only per-frame features
        feats = [frame_features(x, reader.sr) for _, x in reader.chunks(60.0)]
        rms = np.concatenate([f[0] for f in feats]) if feats else np.zeros(0)
        flat = np.concatenate([f[1] for f in feats]) if feats else np.zeros(0)
        mask = np.zeros(len(rms) + 1)
        for w in store.get_words(video.id):
            mask[int(w.start / HOP): int(w.end / HOP) + 1] = 1
        t, s = window_scores(rms, flat, mask)
        ev = to_events(t, s)
        store.put_events(video.id, ["laugh"], ev)
        return AnalyzerResult(info={"laugh": len(ev)})
