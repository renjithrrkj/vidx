"""Unload the speech and vision models when the MCP server has been idle.

A Claude session can stay open for days. Without this, a server that indexed one video keeps
about 850 MB of models in memory until the session ends. They reload on the next call that needs them.
"""
from __future__ import annotations

import contextlib
import gc
import os
import threading
import time

IDLE_UNLOAD_S = float(os.environ.get("VIDX_IDLE_UNLOAD_S", "600"))  # 0 turns it off


def unload_models() -> bool:
    """Drop the cached whisper and CLIP models. Returns True if anything was loaded."""
    from vidx.analyzers import indic, transcript, visual

    had = bool(transcript._models) or visual._model is not None or indic._model is not None
    with visual._lock:
        visual._model = None
    transcript._models.clear()
    indic.unload()
    if had:
        gc.collect()
    return had


class Tracker:
    def __init__(self, after_s: float = IDLE_UNLOAD_S, clock=time.monotonic):
        self.after_s = after_s
        self.clock = clock
        self.last = clock()
        self.active = 0
        self._lock = threading.Lock()

    def touch(self) -> None:
        with self._lock:
            self.last = self.clock()

    @contextlib.contextmanager
    def busy(self):
        with self._lock:
            self.active += 1
            self.last = self.clock()
        try:
            yield
        finally:
            with self._lock:
                self.active -= 1
                self.last = self.clock()

    def maybe_unload(self) -> bool:
        if self.after_s <= 0:
            return False
        with self._lock:
            if self.active or self.clock() - self.last < self.after_s:
                return False
            self.last = self.clock()  # don't retry every tick
        return unload_models()

    def start(self, every_s: float = 60.0) -> None:
        """Check in a daemon thread, so it never keeps the process alive."""
        if self.after_s <= 0:
            return

        def loop():
            while True:
                time.sleep(every_s)
                self.maybe_unload()

        threading.Thread(target=loop, name="vidx-idle", daemon=True).start()
