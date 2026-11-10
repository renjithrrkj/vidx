import wave

import numpy as np

from vidx.analyzers.audio import active_area, dominant_crop, parse_detect_log
from vidx.analyzers.laugh import to_events
from vidx.analyzers.transcript import choose_task, degenerate
from vidx.store import Word


def test_laugh_windows_merge_into_one_event():
    t = np.arange(0, 10, 0.25)
    s = np.zeros_like(t)
    s[(t >= 2) & (t < 3.5)] = 0.8  # overlapping 1s windows -> one event
    ev = to_events(t, s)
    assert len(ev) == 1 and ev[0].start == 2.0 and ev[0].end >= 4.0 and ev[0].value == 0.8


def test_dominant_crop_ignores_full_frame_titles():
    log = "\n".join(["crop=320:136:0:22"] * 80 + ["crop=320:180:0:0"] * 20)
    assert dominant_crop(log) == [320, 136, 0, 22]
    assert active_area([320, 136, 0, 22], 1280, 720) == [1280, 544, 0, 88]
    assert active_area([320, 180, 0, 0], 1280, 720) is None  # full frame
    assert dominant_crop("crop=320:136:0:22\n" * 3) is None  # too few samples


def test_parse_detect_log():
    log = """[silencedetect @ 0x1] silence_start: 6.02
[silencedetect @ 0x1] silence_end: 8.01 | silence_duration: 1.99
[blackdetect @ 0x2] black_start:14 black_end:15 black_duration:1
[freezedetect @ 0x3] lavfi.freezedetect.freeze_start: 3.5
[silencedetect @ 0x1] silence_start: 19.5
Integrated loudness:
    I:         -18.3 LUFS"""
    ev, info = parse_detect_log(log, 20.0)
    kinds = {(e.kind, round(e.start, 1), round(e.end, 1)) for e in ev}
    assert ("silence", 6.0, 8.0) in kinds and ("black", 14.0, 15.0) in kinds
    assert ("freeze", 3.5, 20.0) in kinds and ("silence", 19.5, 20.0) in kinds  # open-ended -> EOF
    assert info["lufs"] == -18.3


def test_degenerate_asr_output():
    assert degenerate("༼ ༼ ༼ ༼ ༼ ༼ ༼ ༼ ༼ ༼", 11.7, "ml", "transcribe")
    assert degenerate("ස්වාපෙග තාවත්තාව්", 1.2, "ml", "transcribe")  # Sinhala script for Malayalam
    assert not degenerate("Chicken was juicy. Ambience was awesome.", 1.4, "ml", "translate")
    assert not degenerate("ഞാൻ വീട്ടിൽ പോയി", 1.3, "ml", "transcribe")
    assert degenerate("ഞാൻ വീട്ടിൽ പോയി", 1.3, "ml", "translate")  # translate must be Latin


def test_choose_task(monkeypatch):
    monkeypatch.delenv("VIDX_WHISPER_TASK", raising=False)
    assert choose_task("ml", "small") == "translate"
    assert choose_task("ml", "large-v3") == "transcribe"
    assert choose_task("en", "small") == "transcribe"
    monkeypatch.setenv("VIDX_WHISPER_TASK", "transcribe")
    assert choose_task("ml", "small") == "transcribe"


def test_chunk_bounds_cut_in_quiet_gaps(tmp_path):
    import wave

    from vidx.analyzers.transcript import chunk_bounds
    from vidx.util import WavReader

    sr = 16000
    t = np.arange(sr * 22) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    for a, b in ((4.0, 4.6), (9.2, 9.8)):  # quiet gaps near the 5s / 10s boundaries
        x[int(a * sr): int(b * sr)] = 0
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    r = WavReader(p)
    assert abs(r.duration - 22) < 1e-6 and len(r[0:sr]) == sr
    b = chunk_bounds(r, size=5, search=2)
    assert b[0][0] == 0 and b[-1][1] == r.duration
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))  # contiguous
    assert 4.0 <= b[0][1] <= 4.6 and 9.2 <= b[1][1] <= 9.8


def _thumbs(pattern_a, pattern_b, n=60, cut=30):
    return np.stack([pattern_a if i < cut else pattern_b for i in range(n)]).astype(np.uint8)


def test_cuts_snap_to_exact_frame_and_reject_flashes():
    from vidx.analyzers.shots import cuts_from_scores, is_flash

    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 255, (18, 32)), rng.integers(0, 255, (18, 32))
    th = _thumbs(a, b)
    motion = np.full(60, 0.004, np.float32)
    motion[30] = 0.3
    scores = np.zeros(60)
    scores[28:30] = 0.9  # the network's peak sits two frames early
    assert cuts_from_scores(scores, motion, th) == [30]
    assert cuts_from_scores(np.zeros(60), motion, th) == [30]  # jump alone is enough

    # brightness flash: same picture, brighter
    fl = np.stack([a if not 30 <= i < 33 else np.clip(a * 0.6 + 90, 0, 255) for i in range(60)]).astype(np.uint8)
    assert is_flash(fl, 30)
    # scene revealed for a moment, then the old picture returns
    rv = np.stack([a if not 30 <= i < 36 else b for i in range(60)]).astype(np.uint8)
    assert is_flash(rv, 30)
    assert not is_flash(th, 30)


def test_sustained_motion_is_not_a_cut():
    from vidx.analyzers.shots import jump_candidates

    m = np.full(100, 0.12, np.float32) + np.random.default_rng(1).random(100).astype(np.float32) * 0.03
    assert jump_candidates(m) == []  # a fight: constant big movement


def test_watermark_vs_text_clustering():
    from vidx.analyzers.detail import cluster_text

    hits = [(t, 0.9, 0.05, 0.08, 0.03, "ACME") for t in range(0, 10)] + [(4, 0.3, 0.8, 0.4, 0.05, "Chapter one")]
    rows = cluster_text(hits, 10, 1.0)
    kinds = {r[7]: r[0] for r in rows}
    assert kinds == {"ACME": "watermark", "Chapter one": "text"}


def test_static_logo_found_but_letterbox_ignored():
    from vidx.analyzers.detail import static_logo_boxes

    rng = np.random.default_rng(2)
    maps = []
    for _ in range(8):
        e = (rng.random((100, 200)) > 0.97).astype(np.uint8) * 255
        e[5:15, 170:190] = 255   # logo, same place every time
        e[0:1, :] = 255          # letterbox edge, reaches frame edges
        maps.append(e)
    boxes = static_logo_boxes(maps)
    assert len(boxes) == 1 and 0.8 < boxes[0][0] < 0.9 and boxes[0][1] < 0.1


def test_glitch_needs_sustained_spike_above_shot_baseline():
    from vidx.analyzers.glitch import find_suspect

    r = np.full(60, 3.0, np.float32) + np.random.default_rng(3).random(60).astype(np.float32)
    assert find_suspect(r) is None
    r2 = r.copy()
    r2[30:34] = 30
    assert find_suspect(r2)[0] == 30
    r3 = r.copy()
    r3[30] = 30  # one frame is noise
    assert find_suspect(r3) is None


def test_suspect_spans_finds_wordless_speech_but_not_silence_or_normal_pause():
    from vidx.analyzers.transcript import suspect_spans

    hop = 0.1
    words = [Word(0.0, 1.0, "hi"), Word(1.8, 2.4, "there"), Word(7.0, 7.5, "ok")]
    duration = 7.5  # ends exactly at the last word: no trailing tail gap to confuse the cases below

    # the 2.4 -> 7.0 gap (4.6s) is VAD-speech: a merged false-start + retake
    speech_gap = np.zeros(int(duration / hop), bool)
    speech_gap[int(2.4 / hop): int(7.0 / hop)] = True
    assert suspect_spans(words, speech_gap, hop, duration) == [(2.4, 7.0)]

    real_silence = np.zeros(int(duration / hop), bool)
    assert suspect_spans(words, real_silence, hop, duration) == []  # same gap, but VAD hears nothing

    close_words = [Word(0.0, 1.0, "hi"), Word(1.6, 2.2, "there")]
    pause_speech = np.ones(int(2.2 / hop), bool)
    assert suspect_spans(close_words, pause_speech, hop, 2.2) == []  # a normal 0.6s pause, not 1.5s+


def test_suspect_spans_ignores_loud_non_speech_like_music_in_a_pause():
    from vidx.analyzers.transcript import suspect_spans

    hop = 0.1
    words = [Word(0.0, 1.0, "hi"), Word(7.0, 7.5, "ok")]
    duration = 7.5

    # the 1.0 -> 7.0 gap is plenty loud (e.g. underscore music or room tone) but VAD never calls it
    # speech: energy alone can't tell this apart from a missed retake, VAD can.
    not_speech = np.zeros(int(duration / hop), bool)
    assert suspect_spans(words, not_speech, hop, duration) == []


def test_suspect_spans_flags_implausibly_long_word():
    from vidx.analyzers.transcript import suspect_spans

    hop, duration = 0.1, 6.6
    speech = np.ones(int(duration / hop), bool)
    # "there" stretched across a false-start + retake: 6s for a 5-letter word
    words = [Word(0.0, 0.4, "hi"), Word(0.6, 6.6, "there")]
    assert suspect_spans(words, speech, hop, duration) == [(0.6, 6.6)]

    normal = [Word(0.0, 0.4, "hi"), Word(0.6, 1.4, "wonderful")]  # a long word, plausibly spoken
    assert suspect_spans(normal, speech, hop, 1.4) == []


def test_recover_suspect_spans_merges_without_duplicates_and_keeps_order():
    from vidx.analyzers.transcript import recover_suspect_spans

    class StubReader:
        duration = 10.0

        def seconds(self, a, b):
            return np.zeros(1)

    words = [Word(0.0, 0.5, "hello"), Word(0.5, 1.0, "world"), Word(1.0, 7.0, "bridge"), Word(7.2, 7.6, "end")]

    def stub_transcribe(audio):
        # span-relative times (pa = 1.0 - pad = 0.7): together these cover the whole 1.0-7.0 span,
        # and "end" lands on audio already covered by the existing word at 7.2-7.6, so it must not
        # be duplicated
        return [Word(0.3, 3.0, "take"), Word(3.0, 6.3, "two"), Word(6.6, 7.0, "end")]

    new_words, unclear, recovered_s, unclear_s = recover_suspect_spans(
        StubReader(), words, [(1.0, 7.0)], stub_transcribe, pad=0.3)
    assert [w.text for w in new_words] == ["hello", "world", "take", "two", "end"]
    assert [w.start for w in new_words] == sorted(w.start for w in new_words)
    assert recovered_s == 6.0 and unclear_s == 0.0
    assert unclear == []


def test_recover_suspect_spans_counts_only_the_covered_sliver_as_recovered():
    from vidx.analyzers.transcript import recover_suspect_spans

    class StubReader:
        duration = 20.0

        def seconds(self, a, b):
            return np.zeros(1)

    words = [Word(0.0, 1.0, "hi")]
    # a 10s suspect span; the retry only recovers 0.5s near the start, the rest stays wordless
    new_words, unclear, recovered_s, unclear_s = recover_suspect_spans(
        StubReader(), words, [(2.0, 12.0)], lambda audio: [Word(0.0, 0.5, "um")], pad=0.0)
    assert recovered_s == 0.5 and unclear_s == 9.5
    assert [(e.start, e.end) for e in unclear] == [(2.5, 12.0)]


def test_wordless_span_without_recovery_becomes_unclear_event():
    from vidx.analyzers.transcript import recover_suspect_spans

    class StubReader:
        duration = 10.0

        def seconds(self, a, b):
            return np.zeros(1)

    words = [Word(0.0, 1.0, "hi")]
    new_words, unclear, recovered_s, unclear_s = recover_suspect_spans(
        StubReader(), words, [(2.0, 5.0)], lambda audio: [])
    assert new_words == words  # nothing to replace, nothing recovered
    assert [e.kind for e in unclear] == ["unclear"]
    assert (unclear[0].start, unclear[0].end) == (2.0, 5.0)
    assert recovered_s == 0.0 and unclear_s == 3.0


class _FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _FakeSeg:
    def __init__(self, words, text="ok", compression_ratio=1.0):
        self.text, self.compression_ratio, self.words = text, compression_ratio, words


class _FakeModel:
    """Stands in for faster-whisper's WhisperModel: pops the next queued segment list on each
    .transcribe() call, in the order Transcript.run() makes them (main pass, then each retry)."""

    def __init__(self, calls):
        self._calls = list(calls)

    def transcribe(self, audio, **kw):
        return self._calls.pop(0), None


def _write_silence_wav(path, seconds, sr=16000):
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))


def _test_video(tmp_path, duration):
    from vidx import store
    from vidx.store import VideoInfo

    wav = tmp_path / "audio16k.wav"
    _write_silence_wav(wav, duration)
    v = VideoInfo(id="v1", path=str(tmp_path / "clip.mp4"), sha1="x", duration=duration,
                  fps=30.0, width=640, height=360, meta={})
    store.put_video(v)
    return v, wav


def _patch_store(tmp_path, monkeypatch):
    from vidx import config, store

    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "vidx.db")
    monkeypatch.setattr(store, "_conn", None)
    monkeypatch.setattr(config, "WHISPER_LANGUAGE", "en")  # skip detect_language in the stub model


def test_transcript_run_with_no_suspect_spans_indexes_exactly_as_before(tmp_path, monkeypatch):
    from vidx import store
    from vidx.analyzers import Context
    from vidx.analyzers.transcript import Transcript

    _patch_store(tmp_path, monkeypatch)
    video, wav = _test_video(tmp_path, 5.0)
    # five words, back to back with only short (<1.5s) pauses: nothing here is suspect
    words = [_FakeWord("hi", 0.0, 1.0), _FakeWord("there", 1.2, 2.2), _FakeWord("friend", 2.4, 3.4),
             _FakeWord("today", 3.6, 4.6), _FakeWord("ok", 4.6, 5.0)]
    model = _FakeModel([[_FakeSeg(words)]])
    monkeypatch.setattr("vidx.analyzers.transcript.load_whisper", lambda name: model)

    def _no_vad(*a, **kw):
        raise AssertionError("speech_profile (the VAD pass) must not run with no candidate gap")
    monkeypatch.setattr("vidx.analyzers.transcript.speech_profile", _no_vad)

    ctx = Context()
    ctx.audio16k = lambda v: wav
    result = Transcript().run(video, ctx)

    assert result.info["words"] == 5
    assert "recovered_s" not in result.info and "unclear_s" not in result.info
    stored = store.get_video(video.id)
    assert "transcript_recovered" not in stored.meta and "transcript_unclear" not in stored.meta
    assert store.get_events(video.id, "unclear") == []


def test_transcript_run_recovers_partial_speech_and_records_remaining_unclear(tmp_path, monkeypatch):
    from vidx import store
    from vidx.analyzers import Context
    from vidx.analyzers.transcript import Transcript

    _patch_store(tmp_path, monkeypatch)
    video, wav = _test_video(tmp_path, 10.0)
    # "hi" .. 7.5s wordless speech-like gap .. "ok": a merged false-start + retake
    first_pass = [_FakeSeg([_FakeWord("hi", 0.0, 1.0), _FakeWord("ok", 8.5, 8.9)])]
    # span-relative words (pad=0.3 -> absolute 1.0-4.0); only the first part of the gap recovers
    second_pass = [_FakeSeg([_FakeWord("retake", 0.3, 3.3)])]
    model = _FakeModel([first_pass, second_pass])
    monkeypatch.setattr("vidx.analyzers.transcript.load_whisper", lambda name: model)
    monkeypatch.setattr("vidx.analyzers.transcript.speech_profile",
                         lambda reader, hop=0.1: np.ones(int(reader.duration / hop) + 2, bool))

    ctx = Context()
    ctx.audio16k = lambda v: wav
    result = Transcript().run(video, ctx)

    assert result.info["recovered_s"] == 3.0 and result.info["unclear_s"] == 4.5
    stored = store.get_video(video.id)
    assert stored.meta["transcript_recovered"] == 3.0 and stored.meta["transcript_unclear"] == 4.5
    unclear = store.get_events(video.id, "unclear")
    assert len(unclear) == 1 and (unclear[0].start, unclear[0].end) == (4.0, 8.5)
    assert [w.text for w in store.get_words(video.id)] == ["hi", "retake", "ok"]


def test_audio_labels_and_beats():
    from vidx.analyzers.audiomap import frame_features, label_windows, smooth, stretches, track_beats

    sr = 16000
    t = np.arange(sr * 6) / sr
    tone = 0.3 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 330 * t))
    noise = 0.3 * np.random.default_rng(4).standard_normal(sr * 2)
    x = np.concatenate([tone, np.zeros(sr * 2), noise]).astype(np.float32)
    rms, flat, flux = frame_features(x, sr)
    ev = stretches(smooth(label_windows(rms, flat, flux, np.zeros(len(rms), bool))))
    assert [e.kind for e in ev] == ["music", "sfx"]
    assert ev[0].end == 6.0 and ev[1].start == 8.0

    clicks = np.zeros(sr * 20, np.float32)  # 120 bpm
    for k in np.arange(0, 20, 0.5):
        i = int(k * sr)
        clicks[i: i + 160] = np.random.default_rng(int(k * 2)).standard_normal(160) * 0.8
    bpm, strength, beats = track_beats(frame_features(clicks, sr)[2])
    assert abs(bpm - 120) < 4 and strength > 0.25
    assert abs(np.median(np.diff(beats)) - 0.5) < 0.03
