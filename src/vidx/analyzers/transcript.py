"""faster-whisper transcript with word timestamps. Language auto-detected.

Small multilingual Whisper models produce degenerate output (repeated glyphs, wrong script) for
several Indic languages, Malayalam included, and the temperature-fallback loop then makes that
3x slower than realtime. So:
  * decoding is greedy at temperature 0 (bounded cost), degenerate segments are dropped;
  * for languages the chosen model handles badly, the transcript is an English translation
    (task="translate"), flagged as such in the summary. Override with VIDX_WHISPER_TASK or a
    bigger model (VIDX_WHISPER_MODEL=medium / large-v3).
"""
from __future__ import annotations

import os
import re

import numpy as np

from .. import config, store, util
from ..store import Event, VideoInfo, Word
from . import AnalyzerResult, Context

_models: dict[str, object] = {}

# languages where tiny/base/small give unusable transcripts (verified for ml; others per Whisper paper WER)
WEAK_ON_SMALL = {"ml", "si", "ta", "te", "kn", "or", "as", "pa", "ne", "my", "km", "lo", "am", "sd", "gu", "mr", "bn"}
SMALL_MODELS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en"}

SCRIPTS = {  # unicode blocks, for script sanity + mix reporting
    "latin": (0x0041, 0x024F), "malayalam": (0x0D00, 0x0D7F), "tamil": (0x0B80, 0x0BFF),
    "devanagari": (0x0900, 0x097F), "kannada": (0x0C80, 0x0CFF), "telugu": (0x0C00, 0x0C7F),
    "sinhala": (0x0D80, 0x0DFF), "tibetan": (0x0F00, 0x0FFF), "bengali": (0x0980, 0x09FF),
    "arabic": (0x0600, 0x06FF), "cjk": (0x4E00, 0x9FFF),
}
LANG_SCRIPT = {"ml": "malayalam", "ta": "tamil", "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
               "kn": "kannada", "te": "telugu", "si": "sinhala", "bn": "bengali", "ar": "arabic", "ur": "arabic",
               "zh": "cjk", "ja": "cjk"}


def script_of(text: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in text:
        o = ord(ch)
        for name, (a, b) in SCRIPTS.items():
            if a <= o <= b:
                counts[name] = counts.get(name, 0) + 1
                break
    return max(counts, key=counts.get) if counts else None


def degenerate(text: str, compression_ratio: float, lang: str | None, task: str) -> bool:
    if compression_ratio > 2.4 or "�" in text:
        return True
    if re.search(r"(\S{1,4})( ?\1){7,}", text):  # the same short token 8+ times in a row
        return True
    sc = script_of(text)
    allowed = {"latin"} if task == "translate" else {"latin", LANG_SCRIPT.get(lang or "", "latin")}
    return sc is not None and sc not in allowed


def load_whisper(name: str):
    if name not in _models:
        from faster_whisper import WhisperModel

        dev = config.device()
        if dev == "cuda":
            opts = {"device": "cuda", "compute_type": "float16"}
        else:  # ctranslate2 has no MPS backend; int8 on CPU is the fast path on Apple silicon
            opts = {"device": "cpu", "compute_type": "int8", "cpu_threads": os.cpu_count() or 4}
        try:  # never touch the network when the weights are already here
            _models[name] = WhisperModel(name, local_files_only=True, **opts)
        except Exception:
            _models[name] = WhisperModel(name, **opts)
    return _models[name]


def choose_task(lang: str | None, model: str) -> str:
    forced = os.environ.get("VIDX_WHISPER_TASK", "auto")
    if forced in ("transcribe", "translate"):
        return forced
    if lang and lang in WEAK_ON_SMALL and model in SMALL_MODELS:
        return "translate"
    return "transcribe"


CHUNK_S = 300.0   # target chunk length
SEARCH_S = 30.0   # look this far back from the target for the quietest moment to cut at

# Second-pass recovery: a false start + retake can get merged into one segment, leaving a long
# gap with one word stretched across it. See DECISIONS.md for why these numbers.
SPEECH_HOP_S = 0.1       # resolution of the per-frame speech mask used to judge "is this a wordless
                         # gap of real speech" -- a wordless gap full of music or ambient tone must
                         # not count, so this comes from Silero VAD (same model faster-whisper's
                         # vad_filter and audiomap use), not a raw energy threshold.
GAP_MIN_S = 1.5          # a wordless stretch longer than this, while VAD says it's speech, is suspect
SPEECH_FRAC = 0.5        # fraction of a gap's frames that must be VAD-speech to count as suspect
MAX_S_PER_CHAR = 0.5     # implausible-word bound: even slow, deliberate speech is well under this
MIN_IMPLAUSIBLE_S = 1.2  # never flag a short word just for being brief
PAD_S = 0.3              # padding added on each side before re-transcribing a span
MAX_SPAN_S = 60.0        # hard cap on one suspect span. A real merged retake is a few seconds; but
                         # if the first pass's degenerate() drops almost everything (wrong/weak-
                         # language detection), word_gaps() can return one gap spanning nearly the
                         # whole file, and if VAD calls that stretch speech-like, "the suspect span"
                         # would otherwise be "the whole video". Capping keeps every re-transcribe
                         # call bounded regardless of how bad the first pass was (see DECISIONS.md).


def word_gaps(words: list[Word], duration: float) -> list[tuple[float, float]]:
    """Wordless stretches: before the first word, between words, and after the last."""
    gaps = []
    prev_end = 0.0
    for w in sorted(words, key=lambda w: w.start):
        if w.start > prev_end:
            gaps.append((prev_end, w.start))
        prev_end = max(prev_end, w.end)
    if duration > prev_end:
        gaps.append((prev_end, duration))
    return gaps


def implausible_words(words: list[Word], max_s_per_char: float = MAX_S_PER_CHAR,
                       min_s: float = MIN_IMPLAUSIBLE_S) -> list[Word]:
    """Words that last far longer than their length plausibly takes to say."""
    return [w for w in words if w.end - w.start > max(min_s, max_s_per_char * len(w.text.strip()))]


def _speech_frac_in(speech: np.ndarray, hop: float, a: float, b: float) -> float:
    i0, i1 = int(a / hop), max(int(a / hop) + 1, int(round(b / hop)))
    seg = speech[i0:i1]
    return float(seg.mean()) if len(seg) > 0 else 0.0


def suspect_spans(words: list[Word], speech: np.ndarray, hop: float, duration: float,
                   gap_min: float = GAP_MIN_S, speech_frac: float = SPEECH_FRAC,
                   max_span: float = MAX_SPAN_S) -> list[tuple[float, float]]:
    """Stretches worth transcribing again: a long wordless gap VAD says is real speech, or a word
    stretched implausibly long (a merged false-start + retake). `speech` is a per-`hop` bool mask
    (see `speech_profile`), not raw energy: background music or room tone during a pause has
    plenty of energy but isn't speech, and must never be flagged. No padding here; that's added
    only when a span is actually re-transcribed, so this stays easy to test against made-up word
    lists and speech masks. Spans longer than `max_span` are split into contiguous `max_span`-sized
    pieces -- a degenerate first pass can leave one wordless gap covering nearly the whole file,
    and that must never become one unbounded re-transcribe."""
    spans = [(a, b) for a, b in word_gaps(words, duration)
              if b - a >= gap_min and _speech_frac_in(speech, hop, a, b) >= speech_frac]
    spans += [(w.start, w.end) for w in implausible_words(words)]
    spans.sort()
    merged: list[tuple[float, float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    capped: list[tuple[float, float]] = []
    for a, b in merged:
        t = a
        while b - t > max_span:
            capped.append((t, t + max_span))
            t += max_span
        capped.append((t, b))
    return capped


def merge_recovered(words: list[Word], span: tuple[float, float], recovered: list[Word]) -> list[Word]:
    """Drop existing words entirely inside the span (what made it suspect in the first place), add
    recovered words that don't land on audio something is already transcribed for, keep time order."""
    a, b = span
    kept = [w for w in words if not (a <= w.start and w.end <= b)]
    out = kept + [nw for nw in recovered if not any(w.start < nw.end and nw.start < w.end for w in kept)]
    out.sort(key=lambda w: w.start)
    return out


def speech_profile(reader: "util.WavReader", hop: float = SPEECH_HOP_S) -> np.ndarray:
    """Per-`hop`-second bool, True where Silero VAD (the model that ships inside faster-whisper and
    already backs `vad_filter` and `audiomap`'s speech label) hears speech. Streamed in chunks so
    memory stays flat on long files. This -- not raw energy -- is what tells a merged false-start
    retake apart from background music or ambient tone during an ordinary pause, neither of which
    VAD calls speech."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    speech = np.zeros(int(reader.duration / hop) + 2, dtype=bool)
    for t0, x in reader.chunks(300.0):
        for s in get_speech_timestamps(x, VadOptions(min_silence_duration_ms=300)):
            a, b = t0 + s["start"] / reader.sr, t0 + s["end"] / reader.sr
            speech[int(a / hop): int(b / hop) + 1] = True
        del x
    return speech


def transcribe_span(model, lang: str | None, task: str, audio: np.ndarray) -> list[Word]:
    """One whisper call on a short span of audio, same degenerate filtering as the main pass.
    Returns words with times relative to the start of `audio`."""
    segments, _ = model.transcribe(
        audio, language=lang, task=task, word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500}, beam_size=1, temperature=[0.0],
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    for seg in segments:
        if degenerate(seg.text, seg.compression_ratio, lang, task):
            continue
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(round(w.start, 3), round(max(w.end, w.start + 0.01), 3), text))
    return words


def _uncovered(words: list[Word], a: float, b: float) -> list[tuple[float, float]]:
    """Sub-ranges of [a, b) that no word (existing or just recovered) covers at all -- a span is
    only as recovered as the words that actually landed in it, not "any word, anywhere in it"."""
    ivs = sorted((max(w.start, a), min(w.end, b)) for w in words if w.start < b and w.end > a)
    gaps, cur = [], a
    for s, e in ivs:
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e)
    if cur < b:
        gaps.append((cur, b))
    return gaps


def recover_suspect_spans(reader, words: list[Word], spans: list[tuple[float, float]], transcribe,
                           pad: float = PAD_S) -> tuple[list[Word], list[Event], float, float]:
    """Re-run `transcribe` (chunk-relative samples -> chunk-relative Words) on each suspect span, with
    a little padding so a word isn't clipped at the edge. A span longer than `CHUNK_S` is itself
    read and transcribed in `chunk_bounds`-sized pieces, never in one call, so memory stays flat
    even if a span is as long as `MAX_SPAN_S` allows. Merges recovered words into `words` by time,
    without duplicating what's already there. Only the part of a span actually covered by a word
    counts as recovered; whatever sub-range is still wordless after the retry becomes its own
    `unclear` event, even if the rest of the span was recovered. Returns (new words, unclear
    events, seconds recovered, seconds still unclear)."""
    out = list(words)
    unclear: list[Event] = []
    recovered_s = unclear_s = 0.0
    for a, b in spans:
        pa, pb = max(0.0, a - pad), min(reader.duration, b + pad)
        # suspect_spans caps spans at MAX_SPAN_S, but pad (or a future larger cap) could still push
        # a span past a single safe read; chunk it the same way the main pass chunks the whole file,
        # so a long suspect span is never read or transcribed in one piece.
        bounds = chunk_bounds(reader, size=CHUNK_S, search=SEARCH_S, start=pa, end=pb) if pb - pa > CHUNK_S else [(pa, pb)]
        recovered: list[Word] = []
        for c0, c1 in bounds:
            recovered += [Word(round(w.start + c0, 3), round(w.end + c0, 3), w.text) for w in transcribe(reader.seconds(c0, c1))]
        out = merge_recovered(out, (a, b), recovered)
        gaps = _uncovered(out, a, b)
        still = sum(g1 - g0 for g0, g1 in gaps)
        recovered_s += (b - a) - still
        unclear_s += still
        unclear += [Event("unclear", round(g0, 2), round(g1, 2), None) for g0, g1 in gaps]
    return out, unclear, round(recovered_s, 1), round(unclear_s, 1)


def chunk_bounds(reader: "util.WavReader", size: float = CHUNK_S, search: float = SEARCH_S,
                  start: float = 0.0, end: float | None = None) -> list[tuple[float, float]]:
    """Split [start, end) -- the whole file by default -- into ~size-second chunks, each cut at the
    quietest 0.5s near the boundary so a word is never split between two whisper calls. `start`/
    `end` let `recover_suspect_spans` reuse this to chunk a single long suspect span the same way,
    instead of reading it into memory in one piece."""
    D = reader.duration if end is None else end
    out, cur = [], start
    while D - cur > size + search:
        lo, hi = cur + size - search, cur + size
        x = reader.seconds(lo, hi)
        hop = int(reader.sr * 0.1)
        n = len(x) // hop
        rms = np.sqrt((x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-12)
        smooth = np.convolve(rms, np.ones(5) / 5, mode="same")  # 0.5s windows
        cut = lo + (int(np.argmin(smooth)) + 0.5) * 0.1
        out.append((cur, cut))
        cur = cut
    out.append((cur, D))
    return out


class Transcript:
    name = "transcript"
    optional = False

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        wav = ctx.audio16k(video)
        if wav is None:
            store.put_words(video.id, [])
            return AnalyzerResult(info={"language": None, "words": 0}, skipped="no audio stream")
        reader = util.WavReader(wav)
        model = load_whisper(config.WHISPER_MODEL)
        lang = config.WHISPER_LANGUAGE
        prob = None
        if lang is None:
            # detect on voiced windows spread through the file (not just the first 30s), using
            # three 60s samples so memory stays small on long files
            D = reader.duration
            probe = np.concatenate([reader.seconds(t, t + 60) for t in (D * 0.1, D * 0.5, D * 0.85)])
            lang, prob, _ = model.detect_language(probe, vad_filter=True, language_detection_segments=3)
        task = choose_task(lang, config.WHISPER_MODEL)
        words: list[Word] = []
        dropped = 0.0
        # Transcribe in ~5 min chunks cut at quiet moments. faster-whisper builds the spectrogram of
        # whatever it is given in one go: a whole hour peaked at several GB on an 8 GB machine.
        for c0, c1 in chunk_bounds(reader):
            audio = reader.seconds(c0, c1)
            segments, _ = model.transcribe(
                audio, language=lang, task=task, word_timestamps=True, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500}, beam_size=1, temperature=[0.0],
                condition_on_previous_text=False,
            )
            for seg in segments:
                if degenerate(seg.text, seg.compression_ratio, lang, task):
                    dropped += seg.end - seg.start
                    continue
                for w in seg.words or []:
                    text = w.word.strip()
                    if text:
                        ws, we = c0 + w.start, c0 + max(w.end, w.start + 0.01)
                        words.append(Word(round(ws, 3), round(we, 3), text))
            del audio

        # A false start + retake can get merged into one segment: a long gap of real speech with
        # one word stretched across it. Re-transcribe just those seconds (see DECISIONS.md). The
        # VAD pass is itself only worth running when there's a long-enough wordless gap or an
        # implausible word to check in the first place.
        has_candidate = bool(implausible_words(words)) or any(
            b - a >= GAP_MIN_S for a, b in word_gaps(words, reader.duration))
        spans = suspect_spans(words, speech_profile(reader), SPEECH_HOP_S, reader.duration) if has_candidate else []
        recovered_s = unclear_s = 0.0
        unclear_events: list[Event] = []
        if spans:
            words, unclear_events, recovered_s, unclear_s = recover_suspect_spans(
                reader, words, spans, lambda audio: transcribe_span(model, lang, task, audio))
        store.put_events(video.id, ["unclear"], unclear_events)

        store.put_words(video.id, words)
        _models.clear()  # ~1 GB; reloading takes seconds, holding it in a long-lived server doesn't pay
        scripts: dict[str, float] = {}
        for w in words:
            sc = script_of(w.text)
            if sc:
                scripts[sc] = scripts.get(sc, 0) + (w.end - w.start)
        tot = sum(scripts.values()) or 1
        mix = {k: round(v / tot, 2) for k, v in scripts.items()} if len(scripts) > 1 else None
        mode = "translated-en" if task == "translate" else "transcribed"
        meta = dict(language=lang, language_prob=prob, language_mix=mix,
                    transcript_mode=mode, transcript_dropped=round(dropped, 1))
        info = {"language": lang, "mode": mode, "words": len(words),
                "dropped_s": round(dropped, 1), "model": config.WHISPER_MODEL}
        if spans:  # keep output unchanged on clips with nothing to recover
            meta["transcript_recovered"] = recovered_s
            meta["transcript_unclear"] = unclear_s
            info["recovered_s"] = recovered_s
            info["unclear_s"] = unclear_s
        store.update_meta(video.id, **meta)
        return AnalyzerResult(info=info)
