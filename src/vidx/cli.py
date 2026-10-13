"""CLI: thin wrapper over vidx.api (prints exactly what the MCP tools return)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from . import api, store

app = typer.Typer(add_completion=False, no_args_is_help=True, help="vidx: token-efficient video index + edit")


def _out(fn, *a, **kw):
    try:
        res = fn(*a, **kw)
    except store.VidxError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(res)


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


@app.command()
def index(path: str, force: bool = False, captions: bool = False):
    """Index a video (shots, transcript, events, embeddings)."""
    _out(api.index_video, path, force=force, captions=captions, log=_log)


@app.command("list")
def list_():
    """List indexed videos."""
    from .util import mmss

    for v in store.list_videos():
        typer.echo(f"{v.id} {mmss(v.duration)} {v.width}x{v.height} {v.path}")


@app.command()
def overview(video_id: str, level: str = "scenes", limit: int = 20, offset: int = 0):
    """Table of contents (scenes or shots)."""
    _out(api.overview, video_id, level, limit, offset)


@app.command()
def query(video_id: str, text: str, kind: str = "any", limit: int = 10):
    """Search speech / visuals / events."""
    _out(api.query, video_id, text, kind, limit)


@app.command()
def segment(video_id: str, start: str, end: Optional[str] = typer.Argument(None), fields: str = "words,events,shots"):
    """Detail for a time range (seconds or shot ids)."""
    _out(api.get_segment, video_id, start, end, fields)


@app.command()
def peek(target: str, times: Optional[str] = typer.Argument(None, help="comma list: 12.5,s3,40"),
         size: int = 384, out: Path = Path("peek.jpg"), no_grid: bool = False):
    """Write a labeled contact sheet (or separate frames with --no-grid)."""
    try:
        imgs, cap = api.peek(target, times, size, not no_grid)
    except store.VidxError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    paths = []
    for i, b in enumerate(imgs):
        p = out if len(imgs) == 1 else out.with_name(f"{out.stem}_{i + 1}{out.suffix}")
        p.write_bytes(b)
        paths.append(str(p))
    typer.echo(cap + "\n" + "\n".join(paths))


@app.command()
def edit(video_id: str, script: str = typer.Argument(..., help="DSL text, or @file.vdsl"),
         parent: Optional[str] = None):
    """Create a version from an edit script."""
    if script.startswith("@"):
        script = Path(script[1:]).read_text()
    _out(api.edit, video_id, script.replace("\\n", "\n").replace(";", "\n"), parent)


@app.command()
def render(version_id: str, preset: str = "original", out: Optional[str] = None):
    """Render a version with ffmpeg."""
    _out(api.render, version_id, preset, out)


@app.command()
def check(version_id: str):
    """QA report on a rendered version."""
    _out(api.check, version_id)


@app.command()
def diff(version_a: str, version_b: str):
    """Compare two versions."""
    _out(api.diff, version_a, version_b)


@app.command()
def usage(video: Optional[str] = typer.Option(None, help="only this video_id / version"), last: int = 0):
    """Token report: characters returned per tool call, totalled per video and tool (~4 chars/token)."""
    import json
    from collections import defaultdict

    from .config import HOME

    p = HOME / "usage.log"
    if not p.exists():
        typer.echo("no usage yet")
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if video:
        rows = [r for r in rows if r["target"].startswith(video)]
    if last:
        for r in rows[-last:]:
            typer.echo(f"{r['tool']:<12} {r['chars']:>6} chars ~{r['chars'] // 4:>5} tok"
                       + (f" +{r['images']} img" if r.get("images") else "") + f"  {r['secs']}s  {r['target']}")
        return
    agg: dict = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # target -> tool -> [calls, chars, max, imgs]
    for r in rows:
        a = agg[r["target"] or "-"][r["tool"]]
        a[0] += 1
        a[1] += r["chars"]
        a[2] = max(a[2], r["chars"])
        a[3] += r.get("images", 0)
    for target, tools in agg.items():
        tot = sum(t[1] for t in tools.values())
        typer.echo(f"{target}: {sum(t[0] for t in tools.values())} calls, {tot} chars (~{tot // 4} tokens)")
        for tool, (n, ch, mx, im) in sorted(tools.items(), key=lambda x: -x[1][1]):
            typer.echo(f"  {tool:<12} {n:>3} calls  {ch:>7} chars  avg {ch // n:>5}  max {mx:>5}"
                       + (f"  {im} images" if im else ""))


if __name__ == "__main__":
    app()
