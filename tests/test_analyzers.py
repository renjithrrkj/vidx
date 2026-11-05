import numpy as np

from vidx.analyzers.audio import active_area, dominant_crop, parse_detect_log
from vidx.analyzers.laugh import to_events
from vidx.analyzers.transcript import choose_task, degenerate


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
