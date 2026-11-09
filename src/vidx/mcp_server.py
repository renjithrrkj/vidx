"""MCP server: the 9 vidx tools over stdio. Register with:

    claude mcp add vidx -- uv run --directory /path/to/vidx vidx-mcp
"""
from __future__ import annotations

import asyncio
import functools
import sys

from mcp.server.mcpserver import Image, MCPServer

from . import api, store

INSTRUCTIONS = """vidx turns a video into a compact text index so you can understand and edit it cheaply.
Workflow: index_video(path) -> overview(id) -> query/get_segment for detail -> peek only when you must SEE
frames -> edit(id, dsl) -> render(version) -> check(version) -> fix with edit(parent_version=...) -> diff.
Times are seconds. Shots are s1..sN. Versions are v1..vN.
DSL (one per line): base all|none; drop 30.0-31.2 | drop s7 | drop s3-s5 | drop silence>1.2 | drop black;
keep 120-150 | keep s3 (with base none, order = output order); snap words; pad 0.15; trans xfade 0.3 | trans cut;
crop 9:16 center; speed 1.1; match s3 0.8 (colour-match all clips to a reference shot);
caption 3.0 2.5 Some text; sfx impact_01 at 3.2 gain -6; music drone_02 gain -18;
still /abs/photo.jpg 3.0 at 12.5 motion zoom_in fit cover (a still as a clip, at a time in the edit without stills, or
at end; motions none zoom_in zoom_out pan_left pan_right pan_up pan_down; fit cover|contain; stills are silent, so add music);
overlay /abs/logo.png 3.0 2.5 pos top_right width 0.2 fade 0.3 opacity 0.9 motion slide_left (an image layer at output
time 3.0 for 2.5s, or `all` to the end; pos top_left top top_right left center right bottom_left bottom bottom_right or
x,y as 0-1; width = share of the frame; motions none slide_left slide_right slide_up slide_down). Quote paths with spaces.
Cuts vary the feel of an edit: "cuts hype" picks a cut per join from what is happening there (packs: clean,
documentary, comedy, hype, trailer; add a number to reroll), "cut 3 smash" sets one join, "cut whip_left" sets all.
Names: hard smash punch_in punch_big j_cut l_cut dissolve dissolve_slow dip_black dip_white flash whip_left whip_right
whip_up zoom blur wipe smooth cover reveal iris radial squeeze slices wind pixel gray_fade.
Typical clean-up: "drop silence>1.0\\nsnap words\\npad 0.1".
For per-clip control pass a timeline instead of DSL: JSON (or @/path/file.json) with clips [{in, out, video?,
speed?, crop?, zoom?, grade?{brightness,contrast,saturation,match}, captions?[{at,for,text,pos}],
overlays?[{image,at,for,pos,width,opacity,fade,motion}], transition?{type,dur?}}, or a still {image,for,motion,fit,...}]
and audio [{sfx|music|file, at?, gain?}]. Every edit writes its timeline file; change that file and pass it back.
Rendering caches each clip, so changing one shot re-encodes only that shot. Use render(preview=true) while iterating,
then peek(version, ["cuts"]) to review every cut in one image.
Sounds: drone_01-06, impact_01-06, riser_01-05, whoosh_01-03.
Render presets: original, reel_9x16 (1080x1920, -16 LUFS, <=90s), yt_16x9 (1920x1080, -14 LUFS).
Renders land in vidx_renders/ next to the source unless out_path is given."""

mcp = MCPServer("vidx", instructions=INSTRUCTIONS)


def _errors(fn):
    """VidxErrors become one-line tool results (with the fix), not stack traces."""

    @functools.wraps(fn)
    async def wrap(*a, **kw):
        try:
            return await fn(*a, **kw)
        except store.VidxError as e:
            return f"error: {e}"

    return wrap


async def _run(fn, *a, **kw):
    # analyzers / ffmpeg are blocking; keep the event loop free
    return await asyncio.to_thread(fn, *a, **kw)


@mcp.tool()
@_errors
async def index_video(path: str, force: bool = False, captions: bool = False, glitch: bool = False) -> str:
    """Index a video file: neural shot cuts, keyframes, motion and colour per shot, on-screen text and
    watermarks, transcript, speech/music/sfx stretches, beats, events, CLIP embeddings. Re-indexing the same
    file only runs what is missing unless force=true. captions=true adds one-line shot captions (slower).
    glitch=true also scans AI-generated clips for morphing; hits are suspects, so confirm them with peek.
    Returns video_id, duration, shot count, language and a short summary."""
    return await _run(api.index_video, path, force, captions, glitch)


@mcp.tool()
@_errors
async def overview(video_id: str, level: str = "scenes", limit: int = 20, offset: int = 0) -> str:
    """Table of contents. level="scenes" covers the whole video in <=limit lines (similar shots merged,
    long shots split at pauses); level="shots" lists shots s1..sN (page with offset)."""
    return await _run(api.overview, video_id, level, limit, offset)


@mcp.tool()
@_errors
async def query(video_id: str, text: str, kind: str = "any", limit: int = 10) -> str:
    """Find time ranges. kind: speech (exact+fuzzy transcript match), visual (CLIP text->shot), text (on-screen
    words), event (e.g. "silence>1.5", "black", "loud", "freeze", "laugh", "glitch", "music", "sfx", "speech",
    "beat", or any detector trained with `vidx teach`), any (event syntax, else text+speech+visual)."""
    return await _run(api.query, video_id, text, kind, limit)


@mcp.tool()
@_errors
async def get_segment(video_id: str, start: str, end: str = "",
                      fields: str = "words,events,shots") -> str:
    """Detail for one range: start/end in seconds or shot ids (start="s3" alone = that shot).
    fields: any of words,events,shots,wordtimes,detail. wordtimes = per-word timestamps for precise cuts;
    detail = per-shot motion curve and peak, colour stats, audio level, keyframe times, glitch flag, text boxes."""
    return await _run(api.get_segment, video_id, start, end or None, fields)


@mcp.tool()
async def peek(target: str, times: list[str] | None = None, size: int = 384, grid: bool = True) -> list:
    """SEE frames (costs image tokens; use sparingly). target = video_id or a rendered version id (v3).
    times = seconds or shot ids, e.g. ["12.5", "s3"]; omit for an even visual summary. Returns one
    contact sheet (max 4x4 tiles, each labeled with its timestamp); size = longest side of a tile.
    times=["cuts"] on a rendered version shows the frames either side of each cut (worst colour jumps
    first when there are many) plus a colour strip of every clip: a whole edit in one look."""
    try:
        imgs, cap = await _run(api.peek, target, times, size, grid)
    except store.VidxError as e:
        return [f"error: {e}"]
    return [cap] + [Image(data=b, format="jpeg") for b in imgs]


@mcp.tool()
@_errors
async def edit(video_id: str, dsl: str, parent_version: str = "") -> str:
    """Create a version from an edit script (see server instructions for the DSL). With parent_version,
    the script is appended to the parent's script (incremental fix) unless it has its own `base` line.
    Returns version id, segment count, output duration and warnings."""
    return await _run(api.edit, video_id, dsl, parent_version or None)


@mcp.tool()
@_errors
async def render(version_id: str, preset: str = "original", out_path: str = "", preview: bool = False) -> str:
    """Render a version. preset: original | reel_9x16 | yt_16x9. preview=true is a fast 480p draft.
    Clips are cached, so re-rendering after a small change only encodes what changed. Writes next to the
    source in vidx_renders/ (or to out_path). Returns path, duration, size, pieces encoded vs reused."""
    return await _run(api.render, version_id, preset, out_path or None, preview)


@mcp.tool()
@_errors
async def check(version_id: str) -> str:
    """QA a rendered version: black frames, new freezes, mid-word cuts, audio jumps at joins, loudness vs
    preset target, duration limits. Each finding has a time and a DSL fix. Returns OK when clean."""
    return await _run(api.check, version_id)


@mcp.tool()
@_errors
async def diff(version_a: str, version_b: str) -> str:
    """What changed between two versions: added/removed/moved source ranges, option changes,
    output duration delta."""
    return await _run(api.diff, version_a, version_b)


def main() -> None:
    print("vidx MCP server on stdio", file=sys.stderr)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
