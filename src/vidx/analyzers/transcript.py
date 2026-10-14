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
from ..store import VideoInfo, Word
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
            _models[name] = WhisperModel(name, device="cuda", compute_type="float16")
        else:  # ctranslate2 has no MPS backend; int8 on CPU is the fast path on Apple silicon
            _models[name] = WhisperModel(name, device="cpu", compute_type="int8",
                                         cpu_threads=os.cpu_count() or 4)
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


def chunk_bounds(reader: "util.WavReader", size: float = CHUNK_S, search: float = SEARCH_S) -> list[tuple[float, float]]:
    """Split the file into ~size-second chunks, each cut at the quietest 0.5s near the boundary
    so a word is never split between two whisper calls."""
    D = reader.duration
    out, start = [], 0.0
    while D - start > size + search:
        lo, hi = start + size - search, start + size
        x = reader.seconds(lo, hi)
        hop = int(reader.sr * 0.1)
        n = len(x) // hop
        rms = np.sqrt((x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-12)
        smooth = np.convolve(rms, np.ones(5) / 5, mode="same")  # 0.5s windows
        cut = lo + (int(np.argmin(smooth)) + 0.5) * 0.1
        out.append((start, cut))
        start = cut
    out.append((start, D))
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
        scripts: dict[str, float] = {}
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
                        sc = script_of(text)
                        if sc:
                            scripts[sc] = scripts.get(sc, 0) + (we - ws)
            del audio
        store.put_words(video.id, words)
        tot = sum(scripts.values()) or 1
        mix = {k: round(v / tot, 2) for k, v in scripts.items()} if len(scripts) > 1 else None
        mode = "translated-en" if task == "translate" else "transcribed"
        store.update_meta(video.id, language=lang, language_prob=prob, language_mix=mix,
                          transcript_mode=mode, transcript_dropped=round(dropped, 1))
        return AnalyzerResult(info={"language": lang, "mode": mode, "words": len(words),
                                    "dropped_s": round(dropped, 1), "model": config.WHISPER_MODEL})
