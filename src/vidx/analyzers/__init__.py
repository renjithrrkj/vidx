"""Pluggable analyzers. Add a new one by appending to ANALYZERS (order matters)."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .. import store, util
from ..store import VideoInfo


@dataclass
class Context:
    force: bool = False
    captions: bool = False
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)
    timings: dict[str, float] = field(default_factory=dict)

    def audio16k(self, video: VideoInfo) -> Path | None:
        if not video.has_audio:
            return None
        return util.extract_audio16k(video.path, video.dir / "audio16k.wav")


@dataclass
class AnalyzerResult:
    info: dict = field(default_factory=dict)  # small facts, stored in analyzer_runs.info
    skipped: str | None = None


class Analyzer(Protocol):
    name: str
    optional: bool  # only runs when its flag is on (e.g. captions)

    def run(self, video: VideoInfo, ctx: Context) -> AnalyzerResult: ...


def registry() -> list[Analyzer]:
    from .audio import AudioEvents
    from .captions import Captions
    from .laugh import Laugh
    from .shots import Shots
    from .transcript import Transcript
    from .visual import Visual

    # shots before visual/captions (per shot); laugh after transcript (uses words as a speech mask)
    return [Shots(), Visual(), Transcript(), AudioEvents(), Laugh(), Captions()]


def run_all(video: VideoInfo, ctx: Context) -> dict[str, float]:
    for a in registry():
        if a.optional and not getattr(ctx, a.name, False):
            continue
        if not ctx.force and store.analyzer_done(video.id, a.name):
            ctx.log(f"  {a.name:<11} cached")
            continue
        t0 = time.perf_counter()
        res = a.run(video, ctx)
        dt = time.perf_counter() - t0
        ctx.timings[a.name] = dt
        store.mark_analyzer(video.id, a.name, dt, res.info)
        ctx.log(f"  {a.name:<11} {dt:6.1f}s" + (f"  ({res.skipped})" if res.skipped else ""))
    return ctx.timings
