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
