"""Label each stretch of audio as speech / music / sfx, find beats, and add loudness per shot.

Speech comes from Silero VAD, which ships inside faster-whisper (offline). Everything else is signal
processing: sustained tonal sound is music, short noisy bursts are sfx. Silence stays with the audio
analyzer. Audio is read in chunks, so memory stays flat on long files.
"""
from __future__ import annotations

import numpy as np

from .. import store, util
from ..store import Event, VideoInfo
from . import AnalyzerResult, Context

KINDS = ["speech", "music", "sfx", "beat"]
HOP = 0.02              # 20 ms analysis frames
WIN = 0.5               # labelling resolution
SILENT_DB = -50.0
BPM_RANGE = (70, 180)


def frame_features(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame RMS, spectral flatness and positive spectral flux (the onset signal)."""
    n = int(sr * HOP)
    m = len(x) // n
    if m < 2:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    fr = x[: m * n].reshape(m, n).astype(np.float32)
    rms = np.sqrt((fr.astype(np.float64) ** 2).mean(1) + 1e-12)
    spec = np.abs(np.fft.rfft(fr * np.hanning(n).astype(np.float32), axis=1)) + 1e-7
    flat = np.exp(np.log(spec).mean(1)) / spec.mean(1)
    logs = np.log1p(spec * 100)
    flux = np.maximum(np.diff(logs, axis=0, prepend=logs[:1]), 0).mean(1)
    return rms, flat, flux


def label_windows(rms: np.ndarray, flat: np.ndarray, flux: np.ndarray, speech: np.ndarray) -> list[str]:
    """One label per WIN seconds. speech: per-frame bool from the VAD."""
    per = int(WIN / HOP)
    labels = []
    for i in range(0, len(rms) - per + 1, per):
        sl = slice(i, i + per)
        db = 20 * np.log10(rms[sl].mean() + 1e-9)
        if speech[sl].mean() > 0.5:
            labels.append("speech")
        elif db < SILENT_DB:
            labels.append("silence")
        elif np.median(flat[sl]) < 0.12:
            labels.append("music")   # energy concentrated in tones
        else:
            labels.append("sfx")     # noisy / transient
    return labels


def smooth(labels: list[str], min_run: int = 3) -> list[str]:
    """Short sfx islands inside music or speech are usually noise in the classifier, except a real
    hit, which survives because it stands between different neighbours."""
    out = list(labels)
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1] == out[i]:
            j += 1
        if j - i + 1 < min_run and 0 < i and j + 1 < len(out) and out[i - 1] == out[j + 1]:
            for k in range(i, j + 1):
                out[k] = out[i - 1]
        i = j + 1
    return out


def stretches(labels: list[str]) -> list[Event]:
    ev: list[Event] = []
    for i, l in enumerate(labels):
        if l == "silence":
            continue
        a, b = i * WIN, (i + 1) * WIN
        if ev and ev[-1].kind == l and abs(ev[-1].end - a) < 1e-6:
            ev[-1].end = b
        else:
            ev.append(Event(l, a, b, None))
    for e in ev:
        e.value = round(e.end - e.start, 2)
    return ev


def track_beats(flux: np.ndarray) -> tuple[float, float, np.ndarray]:
    """-> (bpm, strength 0-1, beat times). Autocorrelation picks the tempo, then dynamic programming
    (Ellis 2007) places beats on strong onsets at that spacing."""
    if len(flux) < int(8 / HOP):
        return 0.0, 0.0, np.zeros(0)
    env = flux - flux.mean()
    env = np.maximum(env, 0)
    sd = env.std() or 1.0
    env = env / sd
    n = len(env)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(env, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:n]
    lo, hi = int(60 / BPM_RANGE[1] / HOP), int(60 / BPM_RANGE[0] / HOP)
    if hi >= n or ac[0] <= 0:
        return 0.0, 0.0, np.zeros(0)
    lag = lo + int(np.argmax(ac[lo:hi]))
    strength = float(ac[lag] / ac[0])
    score = env.copy()
    back = np.full(n, -1)
    span = np.arange(-2 * lag, -lag // 2)
    pen = -100 * np.log(-span / lag) ** 2 / 100
    for i in range(int(lag * 2), n):
        idx = i + span
        cand = score[idx] + pen
        k = int(np.argmax(cand))
        score[i] += cand[k]
        back[i] = idx[k]
    i = int(np.argmax(score[-lag * 2:])) + n - lag * 2
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    return round(60 / (lag * HOP), 1), round(strength, 2), np.asarray(beats[::-1]) * HOP


class AudioMap:
    name = "audiomap"
    optional = False

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        wav = ctx.audio16k(video)
        if wav is None:
            return AnalyzerResult(skipped="no audio stream")
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        reader = util.WavReader(wav)
        sr = reader.sr
        parts = []
        speech_spans: list[tuple[float, float]] = []
        for t0, x in reader.chunks(300.0):  # 300 s is a multiple of HOP, so frames line up
            parts.append(frame_features(x, sr))
            for s in get_speech_timestamps(x, VadOptions(min_silence_duration_ms=300)):
                speech_spans.append((t0 + s["start"] / sr, t0 + s["end"] / sr))
            del x
        rms, flat, flux = (np.concatenate([p[i] for p in parts]) for i in range(3))
        speech = np.zeros(len(rms), bool)
        for a, b in speech_spans:
            speech[int(a / HOP): int(b / HOP) + 1] = True
        events = stretches(smooth(label_windows(rms, flat, flux, speech)))

        music_s = sum(e.end - e.start for e in events if e.kind == "music")
        bpm, strength, beats = track_beats(flux)
        keep_beats = strength >= 0.25 and music_s >= 0.1 * reader.duration
        if keep_beats:
            events += [Event("beat", round(float(t), 2), round(float(t) + 0.05, 2), None) for t in beats]
        store.put_events(video.id, KINDS, events)
        store.update_meta(video.id, bpm=bpm if keep_beats else None, beat_strength=strength)

        db = 20 * np.log10(rms + 1e-9)
        updates = {}
        for s in store.get_shots(video.id):
            seg = db[int(s.start / HOP): max(int(s.start / HOP) + 1, int(s.end / HOP))]
            if len(seg):
                updates[s.idx] = {"audio": {"db": round(float(seg.mean()), 1), "peak_db": round(float(seg.max()), 1)}}
        store.merge_shot_detail(video.id, updates)
        counts = {k: sum(e.kind == k for e in events) for k in KINDS}
        return AnalyzerResult(info={**counts, "bpm": bpm if keep_beats else None})
