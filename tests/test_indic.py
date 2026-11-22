"""IndicConformer backend: real Malayalam (and other Indian-language) transcripts instead of an
English translation from Whisper small."""
import numpy as np
import pytest

from vidx.analyzers import indic


def test_words_from_ctc_tokens_groups_on_word_markers():
    # CTC tokens with (start, end) times; ▁ marks a word start. The first token often has no marker,
    # and a lone ▁ is just a space.
    tokens = [("ട", 0.96, 1.04), ("▁ബ", 1.20, 1.28), ("ക്ക്", 1.36, 1.44), ("▁", 5.04, 5.12),
              ("ഓ", 5.12, 5.20), ("ഫ", 5.20, 5.28), ("ർ", 5.36, 5.44)]
    words = indic.words_from_tokens(tokens)
    assert [(w.text, w.start, w.end) for w in words] == [
        ("ട", 0.96, 1.04), ("ബക്ക്", 1.2, 1.44), ("ഓഫർ", 5.12, 5.44)]


def test_words_from_ctc_tokens_offsets_and_never_returns_zero_length():
    words = indic.words_from_tokens([("▁എ", 2.0, 2.0)], offset=10.0)
    assert words[0].start == 12.0 and words[0].end > words[0].start


def test_use_for_needs_an_indian_language_and_the_model(monkeypatch):
    monkeypatch.delenv("VIDX_INDIC", raising=False)
    monkeypatch.delenv("VIDX_WHISPER_TASK", raising=False)
    monkeypatch.setattr(indic, "available", lambda: True)
    assert indic.use_for("ml") and indic.use_for("ta")
    assert not indic.use_for("en") and not indic.use_for(None)
    monkeypatch.setattr(indic, "available", lambda: False)
    assert not indic.use_for("ml")


def test_use_for_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(indic, "available", lambda: True)
    monkeypatch.setenv("VIDX_INDIC", "0")
    assert not indic.use_for("ml")
    monkeypatch.delenv("VIDX_INDIC")
    monkeypatch.setenv("VIDX_WHISPER_TASK", "translate")  # asking for English keeps Whisper
    assert not indic.use_for("ml")


def _setup(tmp_path, monkeypatch, lang):
    from test_analyzers import _patch_store, _test_video
    from vidx import config
    _patch_store(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "WHISPER_LANGUAGE", lang)
    monkeypatch.delenv("VIDX_INDIC", raising=False)
    monkeypatch.delenv("VIDX_WHISPER_TASK", raising=False)
    return _test_video(tmp_path, 5.0)


def test_malayalam_is_transcribed_by_indic_when_installed(tmp_path, monkeypatch):
    from vidx import store
    from vidx.analyzers import Context
    from vidx.analyzers.transcript import Transcript
    from vidx.store import Word

    video, wav = _setup(tmp_path, monkeypatch, "ml")

    class _NoWhisper:
        def transcribe(self, *a, **kw):
            raise AssertionError("whisper must not transcribe Malayalam when IndicConformer is installed")
    monkeypatch.setattr("vidx.analyzers.transcript.load_whisper", lambda name: _NoWhisper())
    monkeypatch.setattr(indic, "available", lambda: True)
    seen = []

    def fake(audio, lang):
        seen.append(lang)
        return [Word(0.5, 1.0, "ഓഫർ"), Word(1.1, 1.6, "ബുക്ക്")]
    monkeypatch.setattr(indic, "transcribe", fake)

    ctx = Context()
    ctx.audio16k = lambda v: wav
    result = Transcript().run(video, ctx)

    assert seen == ["ml"]
    assert result.info["mode"] == "transcribed" and result.info["model"] == indic.NAME
    assert [w.text for w in store.get_words(video.id)] == ["ഓഫർ", "ബുക്ക്"]
    assert store.get_video(video.id).meta["transcript_mode"] == "transcribed"


def test_malayalam_without_indic_still_translates_and_says_how_to_fix(tmp_path, monkeypatch):
    from test_analyzers import _FakeModel, _FakeSeg, _FakeWord
    from vidx.analyzers import Context
    from vidx.analyzers.transcript import Transcript

    video, wav = _setup(tmp_path, monkeypatch, "ml")
    model = _FakeModel([[_FakeSeg([_FakeWord("offer", 0.5, 1.0)])]])
    monkeypatch.setattr("vidx.analyzers.transcript.load_whisper", lambda name: model)
    monkeypatch.setattr(indic, "available", lambda: False)
    # the 4 s pause after the word would trigger the VAD pass (faster-whisper, absent in CI): no speech there
    monkeypatch.setattr("vidx.analyzers.transcript.speech_profile",
                        lambda reader, hop=0.1: np.zeros(int(reader.duration / hop) + 2, bool))

    ctx = Context()
    ctx.audio16k = lambda v: wav
    result = Transcript().run(video, ctx)

    assert result.info["mode"] == "translated-en"
    assert "setup-indic" in result.info["hint"]


def test_idle_unload_frees_the_indic_model():
    from vidx import idle
    indic._model = object()
    assert idle.unload_models() is True
    assert indic._model is None


def test_cli_setup_indic_downloads_the_model(monkeypatch):
    pytest.importorskip("typer")
    from typer.testing import CliRunner
    from vidx import cli
    called = []
    monkeypatch.setattr(indic, "install", lambda: called.append(True) or "/models/indic")
    result = CliRunner().invoke(cli.app, ["setup-indic"])
    assert result.exit_code == 0, result.output
    assert called == [True]


@pytest.mark.slow
def test_real_model_returns_nothing_for_silence():
    if not indic.available():
        pytest.skip("IndicConformer not installed (vidx setup-indic)")
    assert indic.transcribe(np.zeros(16000 * 2, np.float32), "ml") == []
    indic.unload()
