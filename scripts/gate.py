#!/usr/bin/env python3
"""The gate: decides whether a change to vidx is good enough to show a human.

  python scripts/gate.py --fast                 tests only, about 15s (what a builder runs while working)
  python scripts/gate.py --target DIR           everything, against the code in DIR
  python scripts/gate.py --rebaseline           a human accepts today's results as the new normal

Saved results and the private test clips live in ~/.vidx-harness, outside the repo, so a change under
test cannot edit what it is measured against. Reports name clips only as a, b, c: they may end up on a
public pull request, so no file paths, words from the footage or frames ever go in them.

Standard library only, so it runs before a checkout's dependencies exist. The golden edits run through
the target's own environment in a separate process, with a throwaway VIDX_HOME.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HARNESS_HOME = Path(os.environ.get("VIDX_HARNESS_HOME", Path.home() / ".vidx-harness")).expanduser()
BASELINE = HARNESS_HOME / "baseline.json"
CLIPS = HARNESS_HOME / "clips"

# Fixed edits, one per clip. They cover the cheap clean-up path, a full-quality reel, and stills with layers.
GOLDEN = {
    "a": {"preset": "reel_9x16", "preview": False,
          "script": "drop silence>1.0\nsnap words\npad 0.1\ncrop 9:16 center"},
    "b": {"preset": "original", "preview": True,
          "script": "base none\nkeep 5-25\nkeep 40-60\ntrans xfade 0.3\ncaption 2 2 Test caption\nmusic drone_01 gain -20"},
    "c": {"preset": "yt_16x9", "preview": True,
          "script": "base none\nkeep 10-30\nkeep 60-80\nstill {image} 2 at 8 motion zoom_in\n"
                    "overlay {image} 1 5 pos top_right width 0.2 fade 0.2"},
}

# Limits. "hard" problems always fail. "drift" means the output changed: fine only if the issue meant to.
TOKEN_GROWTH = 1.05
TIME_GROWTH, TIME_SLACK = 1.5, 15.0
MEMORY_GROWTH = 1.5  # unchanged code swung 1270 -> 1662 MB between runs, so peaks are noisy
MEMORY_CEILING_MB = 3500  # an 8 GB Mac also has to run everything else
PICTURE_TOLERANCE = 6.0   # mean difference per pixel, out of 255, on small grey frames
LOUDNESS_TOLERANCE = 1.0  # dB
LENGTH_TOLERANCE = 0.15   # seconds


def compare(base: dict, now: dict) -> tuple[list[str], list[str]]:
    """One clip's results against its baseline. Returns (hard failures, drift)."""
    hard, drift = [], []
    if now["size"] != base["size"]:
        hard.append(f"frame size {now['size']} (was {base['size']})")
    if now["check_issues"] > base["check_issues"]:
        hard.append(f"check found {now['check_issues']} problems (was {base['check_issues']})")
    if now["secs"] > base["secs"] * TIME_GROWTH + TIME_SLACK:
        hard.append(f"took {now['secs']:.0f}s (was {base['secs']:.0f}s)")
    if now["peak_mb"] > max(base["peak_mb"] * MEMORY_GROWTH, 1):
        hard.append(f"peak memory {now['peak_mb']:.0f} MB (was {base['peak_mb']:.0f} MB)")
    if now["peak_mb"] > MEMORY_CEILING_MB:
        hard.append(f"peak memory {now['peak_mb']:.0f} MB is over the {MEMORY_CEILING_MB} MB ceiling")
    if now["tokens"] > base["tokens"] * TOKEN_GROWTH:
        drift.append(f"tool output {now['tokens']} tokens (was {base['tokens']})")
    if abs(now["length"] - base["length"]) > LENGTH_TOLERANCE:
        drift.append(f"length {now['length']:.2f}s (was {base['length']:.2f}s)")
    if abs(now["loudness"] - base["loudness"]) > LOUDNESS_TOLERANCE:
        drift.append(f"loudness {now['loudness']:.1f} dB (was {base['loudness']:.1f} dB)")
    picture = frame_difference(base["frames"], now["frames"])
    if picture > PICTURE_TOLERANCE:
        drift.append(f"picture changed by {picture:.1f} (limit {PICTURE_TOLERANCE})")
    return hard, drift


def frame_difference(a: list[list[int]], b: list[list[int]]) -> float:
    """Mean per-pixel difference between two lists of small grey frames. Different shapes count as maximal."""
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)) or not a:
        return 255.0
    return max(sum(abs(p - q) for p, q in zip(x, y)) / max(len(x), 1) for x, y in zip(a, b))


def passed_count(pytest_output: str) -> int:
    m = re.search(r"(\d+) passed", pytest_output)
    return int(m.group(1)) if m else 0


def run_tests(target: Path, marker: str) -> tuple[bool, int, float, str]:
    t0 = time.time()
    p = subprocess.run(["uv", "run", "pytest", "-q", "-m", marker], cwd=target, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode == 0, passed_count(p.stdout), time.time() - t0, p.stdout[-3000:]


def run_clip(target: Path, name: str, keep: Path | None) -> dict:
    """One golden edit in a separate process, with its memory measured from outside."""
    home = Path(tempfile.mkdtemp(prefix=f"vidx-gate-{name}-"))
    try:
        job = {"name": name, "clip": str(clip_path(name)), "out": str(home / "out.mp4"), **GOLDEN[name]}
        env = {**os.environ, "VIDX_HOME": str(home), "PYTHONPATH": str(target / "src")}
        cmd = ["/usr/bin/time", "-l", "uv", "run", "python", str(Path(__file__).resolve()), "--worker", json.dumps(job)]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=target, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        secs = time.time() - t0
        if p.returncode != 0:
            # Only the exception's type: its message can hold paths or words from the footage.
            kind = re.findall(r"^(\w+(?:Error|Exception))\b", p.stderr, re.M)
            raise RuntimeError(f"clip {name} crashed ({kind[-1] if kind else 'unknown error'})")
        res = json.loads(p.stdout.strip().splitlines()[-1])
        peak = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
        res.update(secs=round(secs, 1), peak_mb=round(int(peak.group(1)) / 1e6) if peak else 0)
        if keep:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy(job["out"], keep / f"{name}.mp4")
        return res
    finally:
        shutil.rmtree(home, ignore_errors=True)


def clip_path(name: str) -> Path:
    found = sorted(CLIPS.glob(f"{name}.*"))
    if not found:
        raise FileNotFoundError(f"test clip {name} is missing from the harness folder")
    return found[0]


def worker(job: dict) -> None:
    """Runs inside the target's environment: index, edit, render, check, then measure the result."""
    from PIL import Image

    from vidx import api, store, util

    image = Path(job["out"]).with_name("card.png")
    Image.new("RGB", (800, 600), (200, 40, 40)).save(image)
    vid = api.index_video(job["clip"]).split()[1]
    api.overview(vid)
    api.overview(vid, "shots")
    api.query(vid, "silence>1")
    api.query(vid, "people talking")
    api.get_segment(vid, 0, 10)
    version = api.edit(vid, job["script"].format(image=image)).split(":")[0].split()[0]
    api.render(version, preset=job["preset"], out_path=job["out"], preview=job["preview"])
    report = api.check(version)
    imgs, _ = api.peek(version, ["cuts"])

    info = util.ffprobe(job["out"])
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    length = float(info["format"]["duration"])
    usage = [json.loads(x) for x in (Path(os.environ["VIDX_HOME"]) / "usage.log").read_text().splitlines()]
    tokens = sum(r["chars"] // 4 + r["image_tokens"] for r in usage if r["tool"] != "index_video")
    print(json.dumps({
        "size": [video["width"], video["height"]],
        "length": round(length, 3),
        "tokens": tokens,
        "check_issues": len([x for x in report.splitlines()[1:] if x.strip()]),
        "loudness": loudness(job["out"]),
        "frames": sample_frames(job["out"], length),
    }))


def sample_frames(path: str, length: float, points: int = 6) -> list[list[int]]:
    frames = []
    for i in range(points):
        at = length * (i + 0.5) / points
        p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{at:.3f}", "-i", path, "-frames:v", "1",
                            "-vf", "scale=32:18,format=gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE, check=True)
        frames.append(list(p.stdout))
    return frames


def loudness(path: str) -> float:
    p = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
                       text=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    m = re.search(r"mean_volume: (-?[\d.]+) dB", p.stderr)
    return float(m.group(1)) if m else -91.0


def measure(target: Path, keep: Path | None) -> tuple[dict, list[str]]:
    results: dict = {"clips": {}}
    hard: list[str] = []
    for marker, key in (("not slow", "fast_tests"), ("slow", "e2e_tests")):
        ok, count, secs, tail = run_tests(target, marker)
        results[key] = count
        print(f"{key}: {count} passed in {secs:.0f}s", flush=True)
        if not ok:
            hard.append(f"{key} failed")
            print(tail)
            return results, hard
    for name in GOLDEN:
        try:
            results["clips"][name] = run_clip(target, name, keep)
            r = results["clips"][name]
            print(f"clip {name}: {r['secs']:.0f}s, {r['peak_mb']} MB, {r['tokens']} tokens", flush=True)
        except (RuntimeError, FileNotFoundError) as e:
            hard.append(str(e))
    return results, hard


def judge(base: dict, now: dict) -> tuple[list[str], list[str]]:
    hard, drift = [], []
    for key in ("fast_tests", "e2e_tests"):
        if now.get(key, 0) < base.get(key, 0):
            hard.append(f"{key}: {now.get(key, 0)} passed, fewer than the {base[key]} before")
    for name, b in base["clips"].items():
        if name in now["clips"]:
            h, d = compare(b, now["clips"][name])
            hard += [f"clip {name}: {x}" for x in h]
            drift += [f"clip {name}: {x}" for x in d]
    return hard, drift


def scorecard(now: dict, hard: list[str], drift: list[str], allow_drift: bool) -> str:
    ok = not hard and (allow_drift or not drift)
    lines = [f"**Gate: {'PASSED' if ok else 'FAILED'}**", "",
             f"Tests: {now.get('fast_tests', 0)} fast, {now.get('e2e_tests', 0)} end-to-end", ""]
    if now.get("clips"):
        lines += ["| Clip | Time | Peak memory | Tool output | Check problems |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {v['secs']:.0f}s | {v['peak_mb']} MB | {v['tokens']} tokens | {v['check_issues']} |"
                  for k, v in now["clips"].items()]
    if hard:
        lines += ["", "Failures:"] + [f"- {x}" for x in hard]
    if drift:
        lines += ["", "Output changed" + (" (this issue allows it):" if allow_drift else ":")] + [f"- {x}" for x in drift]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=".")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--rebaseline", action="store_true")
    ap.add_argument("--allow-drift", action="store_true")
    ap.add_argument("--keep", help="folder for this run's renders")
    ap.add_argument("--report", help="write the scorecard here")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
        return 0

    target = Path(args.target).resolve()
    if args.fast:
        ok, count, secs, tail = run_tests(target, "not slow")
        print(tail if not ok else f"{count} passed in {secs:.0f}s")
        return 0 if ok else 1

    keep = Path(args.keep) if args.keep else None
    if args.rebaseline:
        now, hard = measure(target, HARNESS_HOME / "baseline")
        if hard:
            print("Not saved, these failed:\n" + "\n".join(hard))
            return 1
        BASELINE.write_text(json.dumps(now))
        print("Baseline saved.")
        return 0

    if not BASELINE.exists():
        print("No baseline yet. A human needs to run --rebaseline first.")
        return 2
    now, hard = measure(target, keep)
    extra_hard, drift = judge(json.loads(BASELINE.read_text()), now)
    hard += extra_hard
    card = scorecard(now, hard, drift, args.allow_drift)
    print(card)
    if args.report:
        Path(args.report).write_text(card)
    return 0 if not hard and (args.allow_drift or not drift) else 1


if __name__ == "__main__":
    sys.exit(main())
