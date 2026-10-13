# vidx

Token-efficient video understanding + editing for LLM agents. A video becomes a compact,
queryable text index (shots, transcript, audio/visual events, CLIP embeddings). An agent can then
understand it, edit it with a tiny line-based script, render it with ffmpeg, and QA the result.
**It never reads full frames by default**: pixels reach the agent only through `peek`, as one small
labeled contact sheet.

Everything runs locally. No paid APIs.

## Install

```bash
brew install ffmpeg            # or apt install ffmpeg
cd vidx && uv sync             # Python 3.12, torch, faster-whisper, open_clip, scenedetect, mcp
```

The first index downloads Whisper `small` (~460 MB) and CLIP ViT-B-32 (~600 MB) into the HF cache.

## Use from Claude Code (MCP)

```bash
claude mcp add vidx -- uv run --directory /path/to/vidx vidx-mcp
```

Then just ask, e.g. *"index ~/Movies/talk.mp4, cut the dead air and make a 9:16 reel of the
part where I talk about pricing"*. The tools:

| Tool | What it returns |
|---|---|
| `index_video(path, force, captions)` | video_id, duration, shots, language, short summary |
| `overview(video_id, level, limit, offset)` | table of contents (scenes cover the whole video in ≤limit lines) |
| `query(video_id, text, kind, limit)` | ranked time ranges: speech / visual (CLIP) / event (`silence>1.5`) |
| `get_segment(video_id, start, end, fields)` | words, events, shots for one range (`fields=wordtimes` for per-word stamps) |
| `peek(target, times, size, grid)` | **image**: one contact sheet, ≤4×4 labeled tiles |
| `edit(video_id, dsl, parent_version)` | version id + segment count, output duration, warnings |
| `render(version_id, preset, out_path)` | output path, duration, size |
| `check(version_id)` | QA report ≤15 lines, `OK` when clean |
| `diff(version_a, version_b)` | added / removed / moved source ranges, duration delta |

## Edit DSL

```
base all              # start from the full video (default) | base none
drop 30.0-31.2        # remove a time range (also 1:02.5 style)
drop s7               # remove a shot (or s7-s9)
drop silence>1.2      # remove every silence longer than 1.2s (also: black, freeze, loud)
keep 120.0-150.0      # add a range (with base none; line order = output order)
keep s3
snap words            # move cut points out of words into the nearest gap
pad 0.15              # handles around kept segments (never into a neighbouring word)
trans xfade 0.3       # crossfade at every join (default: hard cut)
crop 9:16 center
speed 1.1
```

Typical dead-air cleanup: `drop silence>1.0` / `snap words` / `pad 0.1`.
`edit(..., parent_version="v3")` appends to v3's script, so fixes from `check` are one-liners.

Presets: `original` (as is), `reel_9x16` (1080×1920, −16 LUFS, ≤90 s), `yt_16x9` (1920×1080, −14 LUFS).

## CLI (same output as the tools)

```bash
uv run vidx index ~/Movies/talk.mp4
uv run vidx overview 5450c7a6
uv run vidx query 5450c7a6 "whiteboard diagram" --kind visual
uv run vidx segment 5450c7a6 120 150
uv run vidx peek 5450c7a6 "12.5,s3,40" --out sheet.jpg
uv run vidx edit 5450c7a6 "drop silence>1.0;snap words;pad 0.1"
uv run vidx render v1 --preset reel_9x16
uv run vidx check v1
uv run vidx diff v1 v2
uv run vidx usage            # characters returned per call (~tokens)
```

## Optional extras

- `uv sync --extra captions` → `index_video(..., captions=true)` adds Florence-2 one-line shot captions.
- `uv sync --extra otio` → every render also writes a `.otio` timeline (opens in Resolve / Premiere).
- `laugh` events (heuristic audience-reaction detector) run by default: `query(id, "laugh")`.

## Measured (Apple M1, 8 GB, CPU whisper + MPS CLIP)

| Video | Index time | Overview size |
|---|---|---|
| 10:00 1080p vertical stand-up set (1 take, Malayalam) | shots 71 s, visual 23 s, transcript 120 s, audio 64 s ≈ **4.6 min** | 2.6k chars (~650 tok) |
| 7:51 720p short film (124 shots, letterboxed, Malayalam) | **2.6 min** (+5 min with captions) | 3.0k chars (~760 tok) |

A whole session on the 10-min video (23 calls: overviews, queries, segments, edits, 2 peeks)
returned ~9.4k chars, roughly 2.3k tokens of text. Rendering the 9:41 silence-cut at 1080×1920
takes ~3 min, a 54 s reel ~10 s.

Malayalam and other languages that Whisper-small transcribes badly are indexed as an **English
translation** (flagged in the summary). Set `VIDX_WHISPER_MODEL=large-v3` for native-script transcripts.

## Config (env vars)

`VIDX_HOME` (default `~/.vidx`), `VIDX_WHISPER_MODEL` (`small`; try `medium` for Malayalam),
`VIDX_WHISPER_LANGUAGE` (force e.g. `ml`), `VIDX_WHISPER_TASK` (`auto`/`transcribe`/`translate`), `VIDX_DEVICE` (`cuda`/`mps`/`cpu`),
`VIDX_ENCODER` (`libx264` or `h264_videotoolbox`), `VIDX_SHOT_THRESHOLD`, `VIDX_SILENCE_DB`.

## Layout

```
src/vidx/
  analyzers/   probe, shots, visual (CLIP + tags + windows), transcript, audio events (+letterbox),
               laugh, captions (optional)  — pluggable list
  store.py     SQLite (~/.vidx/vidx.db) + per-video .npy embeddings
  query.py     overview / query / get_segment / peek
  dsl.py       edit script -> timeline (pure segment math)
  render.py    timeline -> ffmpeg
  checks.py    QA      diff.py  version compare
  api.py       the 9 operations (+ usage log)   cli.py / mcp_server.py thin wrappers
  otio_export.py  .otio next to renders
```

Adding an analyzer: write a class with `name`, `optional`, and `run(video, ctx) -> AnalyzerResult`
that writes rows to `shots/words/events` (or files in `~/.vidx/<id>/`), then append it in
`analyzers/__init__.py:registry()`.

## Tests

```bash
uv run pytest -m "not slow"   # DSL, segment math, snap/pad, diff, checks (no video needed)
uv run pytest                 # + end-to-end on a generated testsrc clip
```

See `DECISIONS.md` for the choices made along the way.
