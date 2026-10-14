"""Measure how many tokens vidx hands an agent vs naive ways of giving an LLM a video.

Usage: uv run python scripts/token_report.py <video_id> [<video_id> ...] > report.md

Token estimates (no API calls, nothing leaves the machine):
  text  ~ chars / 4
  image ~ width * height / 750, after Claude's resize (long edge <= 1568 px, <= ~1.15 MP)
"""
from __future__ import annotations

import io
import math
import sys

import numpy as np
from PIL import Image

from vidx import api, store
from vidx.index import summary
from vidx.query import phrases, words_text


def text_tok(s: str) -> int:
    return math.ceil(len(s) / 4)


def img_tok(w: int, h: int) -> int:
    s = min(1.0, 1568 / max(w, h), math.sqrt(1_150_000 / (w * h)))
    return math.ceil((w * s) * (h * s) / 750)


def jpeg_tok(b: bytes) -> tuple[int, tuple[int, int]]:
    im = Image.open(io.BytesIO(b))
    return img_tok(*im.size), im.size


def frame_tok(v, long_side: int = 384) -> int:
    s = long_side / max(v.width, v.height)
    return img_tok(round(v.width * s), round(v.height * s))


def baselines(vid: str) -> list[tuple[str, int, str]]:
    v = store.get_video(vid)
    words = store.get_words(vid)
    shots = store.get_shots(vid)
    word_dump = "\n".join(f"{w.start:.2f}-{w.end:.2f} {w.text}" for w in words)
    phrase_dump = "\n".join(f"[{p[0].start:.1f}] {words_text(p)}" for p in phrases(words))
    ft = frame_tok(v)
    return [
        ("Transcript, word-level timestamps (what most ASR tools return)", text_tok(word_dump),
         f"{len(words)} words, {len(word_dump):,} chars"),
        ("Transcript, phrase lines with timestamps (already compacted)", text_tok(phrase_dump),
         f"{len(phrase_dump):,} chars"),
        ("Frames at 1 fps, 384 px", math.ceil(v.duration) * ft, f"{math.ceil(v.duration)} frames x {ft} tok"),
        ("Frames at 1 per 5 s, 384 px", math.ceil(v.duration / 5) * ft, f"{math.ceil(v.duration / 5)} frames"),
        ("One frame per shot, 384 px", len(shots) * ft, f"{len(shots)} shots"),
    ]


def measure(label: str, fn, *a, **kw) -> tuple[str, int, str]:
    try:
        out = fn(*a, **kw)
    except store.VidxError as e:  # e.g. source file deleted: index text still measurable, frames not
        return label, 0, f"n/a ({e})"
    if isinstance(out, tuple):  # peek -> (images, caption)
        imgs, cap = out
        toks, dims = zip(*(jpeg_tok(b) for b in imgs)) if imgs else ((), ())
        return label, text_tok(cap) + sum(toks), f"{len(imgs)} image {dims[0][0]}x{dims[0][1]} + caption" if imgs else cap
    return label, text_tok(out), f"{len(out):,} chars"


def vidx_calls(vid: str, version: str | None) -> list[tuple[str, int, str]]:
    v = store.get_video(vid)
    shots = store.get_shots(vid)
    mid = v.duration / 2
    rows = [
        measure("index_video summary", lambda: summary(vid)),
        measure("overview (scenes, default limit 20)", api.overview, vid),
        measure("overview (all shots, paged to the end)",
                lambda: "\n".join(api.overview(vid, "shots", 20, o) for o in range(0, len(shots), 20))),
        measure("query speech", api.query, vid, "mother", "speech", 5),
        measure("query visual", api.query, vid, "two people talking", "visual", 5),
        measure("query event (silence>1.0)", api.query, vid, "silence>1.0", "event", 10),
        measure("get_segment 30 s", api.get_segment, vid, mid, mid + 30),
        measure("peek 4 frames", api.peek, vid, [v.duration * f for f in (0.2, 0.4, 0.6, 0.8)]),
        measure("peek 16-tile visual summary", api.peek, vid),
        measure("edit (silence cleanup)", api.edit, vid, "drop silence>1.0\nsnap words\npad 0.1"),
    ]
    if version:
        rows += [measure("check (rendered version)", api.check, version)]
    return rows


def table(rows, head=("What the agent receives", "~tokens", "detail")) -> str:
    out = [f"| {head[0]} | {head[1]} | {head[2]} |", "|---|---:|---|"]
    out += [f"| {a} | {b:,} | {c} |" for a, b, c in rows]
    return "\n".join(out)


def main(vids: list[str]) -> None:
    rendered = {r[1]: f"v{r[0]}" for r in store.db().execute(
        "SELECT id, video_id FROM versions WHERE output_path IS NOT NULL ORDER BY id")}
    for vid in vids:
        v = store.get_video(vid)
        base = baselines(vid)
        calls = vidx_calls(vid, rendered.get(vid))
        c = {k: t for k, t, _ in calls}
        b = {k: t for k, t, _ in base}
        print(f"\n## {vid}: {v.duration / 60:.1f} min, {v.width}x{v.height}, {len(store.get_shots(vid))} shots\n")
        print("### Naive baselines\n")
        print(table(base, ("Naive approach", "~tokens", "detail")))
        print("\n### vidx tool calls (measured output)\n")
        print(table(calls))
        understand = c["index_video summary"] + c["overview (scenes, default limit 20)"]
        naive_understand = b["Transcript, word-level timestamps (what most ASR tools return)"] + b["One frame per shot, 384 px"]
        task_cut = (understand + c["query event (silence>1.0)"] + c["edit (silence cleanup)"]
                    + c.get("check (rendered version)", 0))
        task_find = understand + c["query speech"] + c["query visual"] + c["get_segment 30 s"] + c["peek 4 frames"]
        naive_find = naive_understand
        print("\n### Task totals\n")
        rows = [
            ("Get the gist of the whole video", naive_understand, understand),
            ("Find a moment (speech + visual), read it, look at it", naive_find, task_find),
            ("Cut dead air, render, QA", b["Transcript, word-level timestamps (what most ASR tools return)"], task_cut),
        ]
        print("| Task | naive ~tokens | vidx ~tokens | saving |\n|---|---:|---:|---:|")
        for name, n, x in rows:
            print(f"| {name} | {n:,} | {x:,} | {n / max(x, 1):.0f}x |")


if __name__ == "__main__":
    main(sys.argv[1:] or [v.id for v in store.list_videos()])
