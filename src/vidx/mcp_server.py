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
crop 9:16 center; speed 1.1. Typical clean-up: "drop silence>1.0\\nsnap words\\npad 0.1".
Render presets: original, reel_9x16 (1080x1920, -16 LUFS, <=90s), yt_16x9 (1920x1080, -14 LUFS)."""

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
async def index_video(path: str, force: bool = False, captions: bool = False) -> str:
    """Index a video file (shots, transcript, audio/visual events, CLIP embeddings). Re-indexing the same
    file is a no-op unless force=true. captions=true adds one-line shot captions (slower).
    Returns video_id, duration, shot count, language and a short summary."""
    return await _run(api.index_video, path, force, captions)


@mcp.tool()
@_errors
async def overview(video_id: str, level: str = "scenes", limit: int = 20, offset: int = 0) -> str:
    """Table of contents. level="scenes" covers the whole video in <=limit lines (similar shots merged,
    long shots split at pauses); level="shots" lists shots s1..sN (page with offset)."""
    return await _run(api.overview, video_id, level, limit, offset)


@mcp.tool()
@_errors
async def query(video_id: str, text: str, kind: str = "any", limit: int = 10) -> str:
    """Find time ranges. kind: speech (exact+fuzzy transcript match), visual (CLIP text->shot),
    event (e.g. "silence>1.5", "black", "loud", "freeze"), any (event syntax, else speech+visual)."""
    return await _run(api.query, video_id, text, kind, limit)


@mcp.tool()
@_errors
async def get_segment(video_id: str, start: str, end: str = "",
                      fields: str = "words,events,shots") -> str:
    """Detail for one range: start/end in seconds or shot ids (start="s3" alone = that shot).
    fields: any of words,events,shots,wordtimes (wordtimes = per-word timestamps for precise cuts)."""
    return await _run(api.get_segment, video_id, start, end or None, fields)


@mcp.tool()
async def peek(target: str, times: list[str] | None = None, size: int = 384, grid: bool = True) -> list:
    """SEE frames (costs image tokens; use sparingly). target = video_id or a rendered version id (v3).
    times = seconds or shot ids, e.g. ["12.5", "s3"]; omit for an even visual summary. Returns one
    contact sheet (max 4x4 tiles, each labeled with its timestamp); size = longest side of a tile."""
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
async def render(version_id: str, preset: str = "original", out_path: str = "") -> str:
    """Render a version with ffmpeg. preset: original | reel_9x16 | yt_16x9. Returns path, duration, size."""
    return await _run(api.render, version_id, preset, out_path or None)


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
