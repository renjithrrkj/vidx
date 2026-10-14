"""The "without vidx" way of editing a video with an LLM agent, instrumented the same way as vidx.

Every command prints what an agent would read and appends {tool: "naive:<cmd>", chars, images}
to ~/.vidx/usage.log, so `vidx usage` / scripts/ab_report.py can compare both ways fairly.

  uv run python scripts/naive.py transcript <video>            # full word-timestamped transcript
  uv run python scripts/naive.py frames <video> --every 5      # frames as images (counted, saved to dir)
  uv run python scripts/naive.py probe <video>                 # ffprobe summary
  uv run python scripts/naive.py cut <video> <out> "0-12.5,14-30"   # hand-written cut list -> ffmpeg
  uv run python scripts/naive.py silences <video>              # raw ffmpeg silencedetect log

Uses the same faster-whisper model as vidx (cached transcript if the video is indexed) so the
comparison isolates the *interface*, not ASR quality.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import typer

from vidx import config, store, thumbs, util

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _log(cmd: str, target: str, text: str, images: int = 0, image_tokens: int = 0) -> None:
    rec = {"t": round(time.time(), 1), "tool": f"naive:{cmd}", "target": target, "chars": len(text),
           "images": images, "image_tokens": image_tokens, "ok": True, "secs": 0}
    with open(config.HOME / "usage.log", "a") as f:
        f.write(json.dumps(rec) + "\n")


def _video_id(path: str) -> str:
    from vidx.analyzers.probe import probe

    return probe(path).id


@app.command()
def transcript(path: str):
    """Word-level transcript, the way ASR tools usually hand it over."""
    vid = _video_id(path)
    try:
        words = store.get_words(vid)
    except store.VidxError:
        words = []
    if not words:  # not indexed: run whisper exactly like vidx would
        from vidx.index import index_video

        index_video(path)
        words = store.get_words(vid)
    text = "\n".join(f"{w.start:.2f}-{w.end:.2f} {w.text}" for w in words)
    _log("transcript", vid, text)
    print(text)


@app.command()
def frames(path: str, every: float = 5.0, size: int = 384, out: Path = Path("naive_frames")):
    """Frames every N seconds as separate images (what an agent 'watching' the video would read)."""
    v = util.ffprobe(path)
    dur = float(v["format"]["duration"])
    times = [t for t in [i * every for i in range(int(dur // every) + 1)] if t < dur]
    out.mkdir(parents=True, exist_ok=True)
    imgs = thumbs.grab_frames(path, times, long_side=size)
    tok = 0
    for t, im in zip(times, imgs):
        if im is not None:
            im.save(out / f"t{t:07.1f}.jpg", quality=82)
            tok += round(im.width * im.height / 750)
    caption = f"{len(times)} frames every {every}s in {out}/ (t<seconds>.jpg)"
    _log("frames", _video_id(path), caption, images=len(times), image_tokens=tok)
    print(caption + f" ~{tok} image tokens if all are read")


@app.command()
def probe(path: str):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", path],
                       capture_output=True, text=True)
    _log("probe", _video_id(path), p.stdout)
    print(p.stdout)


@app.command()
def silences(path: str, noise: str = "-35dB", d: float = 0.5):
    p = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", path, "-vn", "-af",
                        f"silencedetect=n={noise}:d={d}", "-f", "null", "-"], capture_output=True, text=True)
    text = "\n".join(l for l in p.stderr.splitlines() if "silence_" in l)
    _log("silences", _video_id(path), text)
    print(text)


@app.command()
def cut(path: str, out: str, ranges: str, reel: bool = False):
    """Hand-written keep list -> one ffmpeg trim/concat command (no snapping, padding or QA)."""
    segs = [tuple(map(float, r.split("-"))) for r in ranges.split(",") if r.strip()]
    parts, ins = [], ""
    for i, (a, b) in enumerate(segs):
        parts.append(f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS[v{i}];[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS[a{i}]")
        ins += f"[v{i}][a{i}]"
    g = ";".join(parts) + f";{ins}concat=n={len(segs)}:v=1:a=1[v][a]"
    if reel:
        g += ";[v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[v2]"
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", path, "-filter_complex", g,
           "-map", "[v2]" if reel else "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-c:a", "aac", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = f"{'ok' if p.returncode == 0 else 'failed'}: {out}\n{p.stderr[-500:]}"
    _log("cut", _video_id(path), " ".join(cmd) + "\n" + text)
    print(text)


if __name__ == "__main__":
    sys.exit(app())
