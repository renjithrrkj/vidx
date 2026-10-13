"""Probe: sha1 + ffprobe -> VideoInfo. Runs before every other analyzer (it defines the video id)."""
from __future__ import annotations

import hashlib
import os

from .. import store, util
from ..store import VideoInfo, VidxError


def sha1_file(path: str, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _cached_sha1(path: str, size: int, mtime: float) -> str | None:
    """Skip re-hashing a multi-GB file when path/size/mtime are unchanged."""
    for r in store.db().execute("SELECT id FROM videos WHERE path=?", (path,)):
        v = store.get_video(r["id"])
        if v.meta.get("size") == size and v.meta.get("mtime") == mtime:
            return v.sha1
    return None


def probe(path: str) -> VideoInfo:
    util.need_ffmpeg()
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise VidxError(f"file not found: {path}")
    st = os.stat(path)
    sha1 = _cached_sha1(path, st.st_size, st.st_mtime) or sha1_file(path)

    info = util.ffprobe(path)
    vs = [s for s in info["streams"] if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not vs:
        raise VidxError(f"no video stream in {os.path.basename(path)}")
    v = vs[0]
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    if not 1 <= fps <= 240:
        fps = 30.0
    dur = float(info["format"].get("duration") or v.get("duration") or 0)
    w, h = int(v["width"]), int(v["height"])
    # phone footage: honour rotation metadata so 9:16 clips are reported as 9:16
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(v.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w
    return VideoInfo(
        id=sha1[:8], path=path, sha1=sha1, duration=dur, fps=round(fps, 3), width=w, height=h,
        meta={"has_audio": has_audio, "size": st.st_size, "mtime": st.st_mtime,
              "vcodec": v.get("codec_name")},
    )
