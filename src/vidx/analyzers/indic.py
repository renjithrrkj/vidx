"""AI4Bharat IndicConformer 600M (MIT) for Indian languages that Whisper small can't transcribe.

Whisper small turns Malayalam into broken script, so vidx used to fall back to an English translation,
which can change the meaning ("offer" heard as "friend"). IndicConformer writes the language's own
script, with word times from its CTC decoder. It is optional (about 2.4 GB): `vidx setup-indic`.

vidx runs the model's ONNX and TorchScript files itself rather than the repo's Python code, so no
remote code runs and transformers isn't needed. The files are kept as real files, not Hugging Face
cache symlinks, because recent onnxruntime refuses external weights reached through a symlink.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .. import config
from ..store import Word

NAME = "indic-conformer-600m"
REPO = "ai4bharat/indic-conformer-600m-multilingual"
REVISION = "2a77d305038dfc502fd9464e7d028b90381976cb"  # pinned: the files vidx was tested with
LANGS = {"as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or",
         "pa", "sa", "sat", "sd", "ta", "te", "ur"}
FRAME_S = 0.08   # one encoder frame (10 ms features, 8x subsampling)
BLANK = 256      # CTC blank id in each language's 257-token vocabulary
CHUNK_S = 30.0   # attention cost grows with length; 30 s chunks keep memory near 2.7 GB
SEARCH_S = 5.0
SILENT_RMS = 10 ** (-60 / 20)  # below -60 dBFS there is no speech to transcribe

_model = None


def model_dir() -> Path:
    return Path(os.environ.get("VIDX_INDIC_DIR", Path.home() / ".cache" / "vidx" / NAME)).expanduser()


def available() -> bool:
    a = model_dir() / "assets"
    return all((a / f).exists() for f in ("encoder.onnx", "ctc_decoder.onnx", "preprocessor.ts",
                                           "vocab.json", "language_masks.json"))


def use_for(lang: str | None) -> bool:
    if os.environ.get("VIDX_INDIC", "auto").lower() in ("0", "off", "false", "no"):
        return False
    if os.environ.get("VIDX_WHISPER_TASK") == "translate":  # the user asked for English
        return False
    return bool(lang) and lang in LANGS and available()


def install() -> Path:
    """Download the pinned model (gated: accept its terms on Hugging Face and log in first) and copy
    it out of the cache as real files."""
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(REPO, revision=REVISION,
                                 ignore_patterns=["*.py", "README.md", "assets/joint_*", "assets/rnnt_*"]))
    dst = model_dir()
    tmp = dst.with_name(dst.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, symlinks=False)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return dst


class _Model:
    def __init__(self, d: Path):
        import onnxruntime as ort
        import torch

        a = d / "assets"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
        self.torch = torch
        self.pre = torch.jit.load(str(a / "preprocessor.ts"), map_location="cpu")
        self.encoder = ort.InferenceSession(str(a / "encoder.onnx"), opts, providers=["CPUExecutionProvider"])
        self.ctc = ort.InferenceSession(str(a / "ctc_decoder.onnx"), opts, providers=["CPUExecutionProvider"])
        self.vocab = json.loads((a / "vocab.json").read_text())
        self.masks = json.loads((a / "language_masks.json").read_text())

    def tokens(self, audio: np.ndarray, lang: str) -> list[tuple[str, float, float]]:
        """Greedy CTC: (token, start, end) per emitted token, times relative to `audio`."""
        torch = self.torch
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            feats, length = self.pre(input_signal=wav, length=torch.tensor([wav.shape[-1]]))
        enc, enc_len = self.encoder.run(["outputs", "encoded_lengths"],
                                        {"audio_signal": feats.numpy(), "length": length.numpy()})
        logits = self.ctc.run(["logprobs"], {"encoder_output": enc})[0][0]
        logits = logits[: int(enc_len[0]), np.array(self.masks[lang], dtype=bool)]
        path = logits.argmax(-1)
        vocab = self.vocab[lang]
        out: list[tuple[str, float, float]] = []
        cur, start = None, 0
        for f, tok in enumerate(list(path) + [BLANK]):
            tok = int(tok)
            if tok != cur:
                if cur is not None and cur != BLANK:
                    out.append((vocab[cur], start * FRAME_S, f * FRAME_S))
                cur, start = tok, f
        return out


def words_from_tokens(tokens: list[tuple[str, float, float]], offset: float = 0.0) -> list[Word]:
    """Join CTC tokens into words: a token starting with ▁ begins a new word."""
    groups: list[list] = []
    for tok, t0, t1 in tokens:
        text = tok.replace("▁", "")
        if tok.startswith("▁") or not groups:
            groups.append([text, t0, t1] if text else ["", None, None])
        elif text:
            g = groups[-1]
            g[0] += text
            g[1] = t0 if g[1] is None else g[1]
            g[2] = t1
    words = []
    for text, t0, t1 in groups:
        if text and t0 is not None:
            s = round(offset + t0, 3)
            words.append(Word(s, round(max(offset + t1, s + 0.01), 3), text))
    return words


def load() -> _Model:
    global _model
    if _model is None:
        _model = _Model(model_dir())
    return _model


def unload() -> None:
    global _model
    _model = None


def transcribe(audio: np.ndarray, lang: str) -> list[Word]:
    """Words with times relative to the start of `audio` (16 kHz mono float32, at most ~CHUNK_S)."""
    if len(audio) < 1600:  # under 0.1 s: nothing to hear, and the encoder needs a few frames
        return []
    if float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < SILENT_RMS:
        return []  # CTC emits a stray letter on digital silence
    return words_from_tokens(load().tokens(audio, lang))
