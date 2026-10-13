"""Optional one-line shot captions with Florence-2-base (`index_video(captions=True)`).

Install extras first: `uv sync --extra captions`. Captions take precedence over CLIP tags in
overview lines. One middle frame per shot.
"""
from __future__ import annotations

from .. import config, store, thumbs
from ..store import VideoInfo
from . import AnalyzerResult, Context

MODEL_ID = "florence-community/Florence-2-base"  # native transformers (>=4.56) port of microsoft/Florence-2-base
LEGACY_ID = "microsoft/Florence-2-base"           # trust_remote_code version for older transformers


def _load():
    import torch
    import transformers

    dev = config.device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    if hasattr(transformers, "Florence2ForConditionalGeneration"):
        model = transformers.Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype)
        proc = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(LEGACY_ID, trust_remote_code=True, torch_dtype=dtype)
        proc = transformers.AutoProcessor.from_pretrained(LEGACY_ID, trust_remote_code=True)
    return model.to(dev).eval(), proc, dev, dtype, torch


def _clean(text: str) -> str:
    t = text.replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
    for pre in ("The image shows ", "The image is ", "In this image we can see ", "This image shows "):
        if t.startswith(pre):
            t = t[len(pre):]
    t = t.split(". ")[0].rstrip(".")
    return t[:80]


class Captions:
    name = "captions"
    optional = True  # runs only when ctx.captions is True

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult:
        try:
            model, proc, dev, dtype, torch = _load()
        except ImportError as e:
            return AnalyzerResult(skipped=f"captions need extras: uv sync --extra captions ({e.name})")
        shots = store.get_shots(video.id)
        frames = thumbs.grab_frames(video.path, [(s.start + s.end) / 2 for s in shots], long_side=512)
        caps = {}
        task = "<CAPTION>"
        with torch.no_grad():
            for s, im in zip(shots, frames):
                if im is None:
                    continue
                inp = proc(text=task, images=im, return_tensors="pt").to(dev)
                inp["pixel_values"] = inp["pixel_values"].to(dtype)
                ids = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                     max_new_tokens=40, num_beams=3)
                caps[s.idx] = _clean(proc.batch_decode(ids, skip_special_tokens=True)[0])
        store.update_shot_field(video.id, "caption", caps)
        return AnalyzerResult(info={"captions": len(caps), "model": MODEL_ID})
