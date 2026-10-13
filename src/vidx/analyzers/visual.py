"""CLIP ViT-B-32 shot embeddings (+ cheap zero-shot tags so overview lines say what's on screen)."""
from __future__ import annotations

import hashlib
import threading

import numpy as np

from .. import config, store, thumbs
from ..store import VideoInfo
from . import AnalyzerResult, Context

# Zero-shot vocabulary. Grouped so a shot gets at most one tag per group (framing, subject, setting).
TAG_GROUPS: dict[str, list[str]] = {
    "subject": [
        "a person talking to the camera", "two people talking", "a group of people", "a crowd or audience",
        "a comedian on stage with a microphone", "a person giving a presentation", "people fighting or sparring",
        "a person cooking food", "a plate of food", "a person driving a car", "a car on a road",
        "a computer screen or software interface", "a phone screen", "slides with text", "a document or text on paper",
        "a title card with large text", "a black screen", "a 3D rendered scene", "a cartoon or animation",
        "a dog or cat", "a building", "a landscape with mountains or fields", "a city street", "the sea or a beach",
        "a sports game", "a musician playing an instrument", "hands working on something close up",
        "a logo or graphic", "a chart or diagram", "trees and greenery", "a room interior with no people",
    ],
    "framing": ["a close-up of a face", "a medium shot of a person", "a wide shot", "an aerial drone shot"],
    "setting": ["indoors", "outdoors in daylight", "at night", "on a stage with spotlights"],
}

_lock = threading.Lock()


def _cached_weights() -> str | None:
    """Path of already-downloaded CLIP weights, or None (then open_clip downloads them)."""
    try:
        import open_clip
        from huggingface_hub import hf_hub_download

        cfg = open_clip.pretrained.get_pretrained_cfg(config.CLIP_MODEL, config.CLIP_PRETRAINED)
        repo = (cfg or {}).get("hf_hub", "").strip("/")
        for fn in ("open_clip_model.safetensors", "open_clip_pytorch_model.bin"):
            try:
                return hf_hub_download(repo, fn, local_files_only=True)
            except Exception:
                continue
    except Exception:
        pass
    return None
_model = None


def load_clip():
    """Lazy, process-wide CLIP model (the MCP server keeps it warm for text queries)."""
    global _model
    with _lock:
        if _model is None:
            import open_clip
            import torch

            dev = config.device()
            pretrained = _cached_weights() or config.CLIP_PRETRAINED  # local path: no Hub round-trip
            model, _, preprocess = open_clip.create_model_and_transforms(
                config.CLIP_MODEL, pretrained=pretrained, device=dev)
            tok = open_clip.get_tokenizer(config.CLIP_MODEL)
            model.eval()
            _model = (model, preprocess, tok, dev, torch)
    return _model


def encode_images(images) -> np.ndarray:
    model, preprocess, _, dev, torch = load_clip()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = torch.stack([preprocess(im) for im in images[i:i + 32]]).to(dev)
            f = model.encode_image(batch).float()
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def encode_text(texts: list[str]) -> np.ndarray:
    model, _, tok, dev, torch = load_clip()
    with torch.no_grad():
        f = model.encode_text(tok(texts).to(dev)).float()
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()


def _vocab() -> tuple[list[tuple[str, str]], np.ndarray]:
    items = [(g, t) for g, ts in TAG_GROUPS.items() for t in ts]
    digest = hashlib.sha1("|".join(t for _, t in items).encode()).hexdigest()[:8]
    key = f"{config.CLIP_MODEL}-{config.CLIP_PRETRAINED}-{digest}"
    p = config.HOME / f"clip_vocab_{key}.npy"
    if p.exists():
        return items, np.load(p)
    emb = encode_text([f"a photo of {t}" for _, t in items])
    np.save(p, emb)
    return items, emb


def short_tag(t: str) -> str:
    for pre in ("a photo of ", "a ", "an ", "the "):
        if t.startswith(pre):
            t = t[len(pre):]
    return t


def tag_shots(emb: np.ndarray) -> list[str]:
    items, vemb = _vocab()
    sims = emb @ vemb.T * 100  # CLIP logit scale
    tags = []
    for row in sims:
        chosen = []
        for g in TAG_GROUPS:
            idx = [i for i, (gg, _) in enumerate(items) if gg == g]
            probs = np.exp(row[idx] - row[idx].max())
            probs /= probs.sum()
            j = int(np.argmax(probs))
            # keep confident picks only; subject is always shown
            if g == "subject" or probs[j] > 0.5:
                chosen.append(short_tag(items[idx[j]][1]))
        tags.append(", ".join(chosen))
    return tags


def win_path(video_id: str):
    return config.video_dir(video_id) / "win_emb.npy"


def get_windows(video_id: str):
    p = win_path(video_id)
    if not p.exists():
        return None, None
    return np.load(p), np.load(p.with_suffix(".idx.npy"))


def sample_times(start: float, end: float) -> list[float]:
    mid = (start + end) / 2
    if end - start > 4:
        inset = min(1.0, (end - start) * 0.1)
        return [start + inset, mid, end - inset]
    return [mid]


WIN_MIN_SHOT = 15.0  # shots longer than this also get window embeddings
WIN_STEP = 5.0


def windows(shots) -> list[tuple[int, float, float]]:
    """(shot idx, start, end) windows of ~WIN_STEP seconds inside long shots."""
    out = []
    for s in shots:
        if s.dur > WIN_MIN_SHOT:
            n = int(round(s.dur / WIN_STEP))
            edges = np.linspace(s.start, s.end, n + 1)
            out += [(s.idx, float(a), float(b)) for a, b in zip(edges, edges[1:])]
    return out


class Visual:
    name = "visual"
    optional = False

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        shots = store.get_shots(video.id)
        times, owner = [], []
        for k, s in enumerate(shots):
            for t in sample_times(s.start, s.end):
                times.append(t)
                owner.append(k)
        frames = thumbs.grab_frames(video.path, times, long_side=256)
        good = [(o, f) for o, f in zip(owner, frames) if f is not None]
        emb_frames = encode_images([f for _, f in good])
        dim = emb_frames.shape[1] if len(good) else 512
        emb = np.zeros((len(shots), dim), np.float32)
        for (o, _), e in zip(good, emb_frames):
            emb[o] += e
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-6)
        np.save(store.emb_path(video.id), emb)
        # long takes: one extra embedding per ~5s window so visual search can point inside the shot
        wins = windows(shots)
        if wins:
            wframes = thumbs.grab_frames(video.path, [(a + b) / 2 for _, a, b in wins], long_side=256)
            ok = [(w, f) for w, f in zip(wins, wframes) if f is not None]
            wemb = encode_images([f for _, f in ok])
            np.save(win_path(video.id), wemb)
            np.save(win_path(video.id).with_suffix(".idx.npy"), np.array([[i, a, b] for (i, a, b), _ in ok]))
        tags = tag_shots(emb)
        store.update_shot_field(video.id, "tags", {s.idx: t for s, t in zip(shots, tags)})
        return AnalyzerResult(info={"frames": len(good), "windows": len(wins), "device": config.device()})
