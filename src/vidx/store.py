"""SQLite store. One DB for all videos; per-video artifacts live in ~/.vidx/<video_id>/."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos(
  id TEXT PRIMARY KEY, path TEXT, sha1 TEXT, duration REAL, fps REAL,
  width INTEGER, height INTEGER, created_at REAL, meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS shots(
  video_id TEXT, idx INTEGER, start REAL, end REAL, caption TEXT, motion REAL, tags TEXT,
  PRIMARY KEY(video_id, idx)
);
CREATE TABLE IF NOT EXISTS words(
  video_id TEXT, start REAL, end REAL, text TEXT, speaker TEXT
);
CREATE INDEX IF NOT EXISTS words_vid ON words(video_id, start);
CREATE TABLE IF NOT EXISTS events(
  video_id TEXT, kind TEXT, start REAL, end REAL, value REAL
);
CREATE INDEX IF NOT EXISTS events_vid ON events(video_id, kind, start);
CREATE TABLE IF NOT EXISTS versions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT, parent_id INTEGER,
  dsl TEXT, timeline_json TEXT, output_path TEXT, preset TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS analyzer_runs(
  video_id TEXT, name TEXT, seconds REAL, info TEXT, done_at REAL,
  PRIMARY KEY(video_id, name)
);
"""


@dataclass
class VideoInfo:
    id: str
    path: str
    sha1: str
    duration: float
    fps: float
    width: int
    height: int
    meta: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return config.video_dir(self.id)

    @property
    def has_audio(self) -> bool:
        return bool(self.meta.get("has_audio", True))


@dataclass
class Shot:
    idx: int  # 1-based, shown as s{idx}
    start: float
    end: float
    caption: str | None = None
    motion: float | None = None
    tags: str | None = None

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Event:
    kind: str
    start: float
    end: float
    value: float | None = None


class VidxError(Exception):
    """User-facing error: one line, includes the fix."""


_conn: sqlite3.Connection | None = None


def db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.HOME.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
    return _conn


# ---------- videos ----------

def put_video(v: VideoInfo) -> None:
    c = db()
    c.execute(
        "INSERT OR REPLACE INTO videos VALUES(?,?,?,?,?,?,?,?,?)",
        (v.id, v.path, v.sha1, v.duration, v.fps, v.width, v.height, time.time(), json.dumps(v.meta)),
    )
    c.commit()


def update_meta(video_id: str, **kv) -> None:
    v = get_video(video_id)
    v.meta.update(kv)
    db().execute("UPDATE videos SET meta=? WHERE id=?", (json.dumps(v.meta), video_id))
    db().commit()


def get_video(video_id: str) -> VideoInfo:
    r = db().execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
    if r is None:
        known = [x["id"] for x in db().execute("SELECT id FROM videos ORDER BY created_at DESC LIMIT 5")]
        hint = f" (known: {', '.join(known)})" if known else " (run index_video first)"
        raise VidxError(f"unknown video {video_id}{hint}")
    return VideoInfo(r["id"], r["path"], r["sha1"], r["duration"], r["fps"], r["width"], r["height"],
                     json.loads(r["meta"] or "{}"))


def list_videos() -> list[VideoInfo]:
    return [get_video(r["id"]) for r in db().execute("SELECT id FROM videos ORDER BY created_at")]


def delete_video_data(video_id: str) -> None:
    c = db()
    for t in ("shots", "words", "events"):
        c.execute(f"DELETE FROM {t} WHERE video_id=?", (video_id,))
    c.execute("DELETE FROM analyzer_runs WHERE video_id=?", (video_id,))
    c.commit()


# ---------- analyzer bookkeeping ----------

def analyzer_done(video_id: str, name: str) -> bool:
    return db().execute("SELECT 1 FROM analyzer_runs WHERE video_id=? AND name=?", (video_id, name)).fetchone() is not None


def mark_analyzer(video_id: str, name: str, seconds: float, info: dict | None = None) -> None:
    db().execute("INSERT OR REPLACE INTO analyzer_runs VALUES(?,?,?,?,?)",
                 (video_id, name, seconds, json.dumps(info or {}), time.time()))
    db().commit()


def analyzer_runs(video_id: str) -> dict[str, dict]:
    rows = db().execute("SELECT * FROM analyzer_runs WHERE video_id=?", (video_id,))
    return {r["name"]: {"seconds": r["seconds"], **json.loads(r["info"] or "{}")} for r in rows}


# ---------- shots / words / events ----------

def put_shots(video_id: str, shots: list[Shot]) -> None:
    c = db()
    c.execute("DELETE FROM shots WHERE video_id=?", (video_id,))
    c.executemany("INSERT INTO shots VALUES(?,?,?,?,?,?,?)",
                  [(video_id, s.idx, s.start, s.end, s.caption, s.motion, s.tags) for s in shots])
    c.commit()


def update_shot_field(video_id: str, column: str, values: dict[int, str | None]) -> None:
    assert column in ("caption", "tags")
    c = db()
    c.executemany(f"UPDATE shots SET {column}=? WHERE video_id=? AND idx=?",
                  [(v, video_id, i) for i, v in values.items()])
    c.commit()


def get_shots(video_id: str) -> list[Shot]:
    rows = db().execute("SELECT * FROM shots WHERE video_id=? ORDER BY idx", (video_id,))
    return [Shot(r["idx"], r["start"], r["end"], r["caption"], r["motion"], r["tags"]) for r in rows]


def put_words(video_id: str, words: list[Word]) -> None:
    c = db()
    c.execute("DELETE FROM words WHERE video_id=?", (video_id,))
    c.executemany("INSERT INTO words VALUES(?,?,?,?,NULL)", [(video_id, w.start, w.end, w.text) for w in words])
    c.commit()


def get_words(video_id: str, start: float | None = None, end: float | None = None) -> list[Word]:
    q, args = "SELECT start,end,text FROM words WHERE video_id=?", [video_id]
    if start is not None:
        q += " AND end>?"
        args.append(start)
    if end is not None:
        q += " AND start<?"
        args.append(end)
    return [Word(r[0], r[1], r[2]) for r in db().execute(q + " ORDER BY start", args)]


def put_events(video_id: str, kinds: list[str], events: list[Event]) -> None:
    """Replace all events of the given kinds."""
    c = db()
    c.executemany("DELETE FROM events WHERE video_id=? AND kind=?", [(video_id, k) for k in kinds])
    c.executemany("INSERT INTO events VALUES(?,?,?,?,?)",
                  [(video_id, e.kind, e.start, e.end, e.value) for e in events])
    c.commit()


def get_events(video_id: str, kind: str | None = None, start: float | None = None,
               end: float | None = None) -> list[Event]:
    q, args = "SELECT kind,start,end,value FROM events WHERE video_id=?", [video_id]
    if kind:
        q += " AND kind=?"
        args.append(kind)
    if start is not None:
        q += " AND end>?"
        args.append(start)
    if end is not None:
        q += " AND start<?"
        args.append(end)
    return [Event(*r) for r in db().execute(q + " ORDER BY start", args)]


# ---------- embeddings ----------

def emb_path(video_id: str) -> Path:
    return config.video_dir(video_id) / "shot_emb.npy"


def get_shot_emb(video_id: str) -> np.ndarray | None:
    p = emb_path(video_id)
    return np.load(p) if p.exists() else None


# ---------- versions ----------

def put_version(video_id: str, parent_id: int | None, dsl: str, timeline: dict) -> int:
    c = db()
    cur = c.execute("INSERT INTO versions(video_id,parent_id,dsl,timeline_json,created_at) VALUES(?,?,?,?,?)",
                    (video_id, parent_id, dsl, json.dumps(timeline), time.time()))
    c.commit()
    return int(cur.lastrowid)


def set_version_output(version_id: int, path: str, preset: str) -> None:
    db().execute("UPDATE versions SET output_path=?, preset=? WHERE id=?", (path, preset, version_id))
    db().commit()


def parse_version_id(v: str | int) -> int:
    s = str(v).strip().lower().lstrip("v")
    if not s.isdigit():
        raise VidxError(f"bad version id {v!r} (expected like v3)")
    return int(s)


def get_version(version_id: str | int) -> dict:
    vid = parse_version_id(version_id)
    r = db().execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
    if r is None:
        last = db().execute("SELECT max(id) FROM versions").fetchone()[0]
        raise VidxError(f"unknown version v{vid}" + (f" (latest is v{last})" if last else " (create one with edit)"))
    d = dict(r)
    d["timeline"] = json.loads(d.pop("timeline_json"))
    return d
