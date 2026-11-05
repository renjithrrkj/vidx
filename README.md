# vidx

[![tests](https://github.com/renjithrrkj/vidx/actions/workflows/tests.yml/badge.svg)](https://github.com/renjithrrkj/vidx/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Let AI agents edit video without burning tokens on frames.**

For an agent to "watch" a 10-minute video at 1 frame per second costs ~67,000 tokens; an hour is ~400,000.
vidx watches it once, locally, and gives the agent a compact text map instead: exact shots, keyframes,
motion and colour per shot, on-screen text and watermarks, a transcript, speech/music/sfx stretches,
beats, and text-searchable visuals. The agent reads a ~600-token overview, searches what it needs,
edits with a few lines of script or a timeline file, previews at 480p, renders only what changed, and
runs automatic QA. Pixels reach the agent only when it explicitly asks to look (`peek`).

Runs 100% locally. No paid APIs. Works with any MCP agent (Claude Code, Claude Desktop, …).

| Measured on 3 real videos (8, 10, 60 min) | transcript + frames | vidx |
|---|---:|---:|
| Get the gist of the video | 7k–17k tokens | **~650–930** |
| Find a moment, read it, look at it | 7k–17k | **~1.4k–1.8k** |
| Cut dead air, render, QA | 3k–8k | **~0.8k–1k** |
| Overview of a 10-min vs 60-min video | grows with length | **507 vs 564** |

6–18× fewer tokens in these tests, and the gap widens with longer video. Details, method and the
cases where it does *not* win: [TOKEN_REPORT.md](TOKEN_REPORT.md).

## Quickstart (Claude Code)

You need [ffmpeg](https://ffmpeg.org) and [uv](https://docs.astral.sh/uv/):

```bash
brew install ffmpeg uv          # Linux: apt install ffmpeg, then curl -LsSf https://astral.sh/uv/install.sh | sh
claude mcp add vidx -- uvx --from git+https://github.com/renjithrrkj/vidx vidx-mcp
```

No clone needed. Then ask your agent something like *"index ~/Movies/talk.mp4, cut the dead air
and make a 9:16 reel of the part where I talk about pricing"*.

The first index downloads Whisper `small` (~460 MB) and CLIP ViT-B-32 (~600 MB); run `vidx setup` to
fetch them ahead of time. After that vidx never touches the network. The shot detector, OCR and speech
detector ship inside their packages. Set `VIDX_MODEL_DIR` to keep all weights in one folder you can copy.
Indexing costs CPU time, not tokens, and stays around 1 GB of RAM.

**Claude Desktop / other MCP clients:** add to your MCP config (desktop apps don't inherit your shell
PATH, so give full paths):

```json
"mcpServers": {
  "vidx": {
    "command": "/full/path/to/uvx",
    "args": ["--from", "git+https://github.com/renjithrrkj/vidx", "vidx-mcp"],
    "env": { "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" }
  }
}
```

**From source (to hack on it):**

```bash
git clone https://github.com/renjithrrkj/vidx && cd vidx && uv sync
claude mcp add vidx -- uv run --directory "$PWD" vidx-mcp
```

Tested on macOS (Apple silicon). Linux should work (CPU/CUDA); Windows is untested.

## The tools

| Tool | What it returns |
|---|---|
| `index_video(path, force, captions, glitch)` | video_id, duration, shots, language, short summary |
| `overview(video_id, level, limit, offset)` | table of contents (scenes cover the whole video in ≤limit lines) |
| `query(video_id, text, kind, limit)` | ranked time ranges: speech / visual (CLIP) / on-screen text / event (`silence>1.5`, `music`, `sfx`, `beat`, `laugh`, `glitch`) |
| `get_segment(video_id, start, end, fields)` | words, events, shots for one range (`wordtimes` = per-word stamps; `detail` = motion curve and peak, colour, audio level, keyframes, text boxes) |
| `peek(target, times, size, grid)` | **image**: one contact sheet, ≤4×4 labeled tiles. `times=["cuts"]` on a version = both sides of every cut + a colour strip of all clips |
| `edit(video_id, dsl, parent_version)` | DSL or a JSON timeline; version id, segment count, output duration, warnings, timeline file path |
| `render(version_id, preset, out_path, preview)` | output path, duration, size, pieces encoded vs reused. `preview=true` = fast 480p |
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
match s3 0.8          # colour-match every clip to shot s3 (or a time), strength 0-1
caption 3.0 2.5 Some text           # output time, duration, text
sfx impact_01 at 3.2 gain -6        # bundled sound at output time 3.2
music drone_02 gain -18             # bundled bed under the whole edit
```

Typical dead-air cleanup: `drop silence>1.0` / `snap words` / `pad 0.1`.
`edit(..., parent_version="v3")` appends to v3's script, so fixes from `check` are one-liners.

## Cuts

So an edit doesn't feel like one trick repeated, vidx has 27 named cuts (`vidx cuts` lists them):

- **Editorial:** `hard`, `smash` (with an impact hit), `punch_in` / `punch_big` (zoom that hides a jump cut),
  `j_cut` / `l_cut` (sound leads or trails the picture).
- **Transitions:** `dissolve`, `dissolve_slow`, `dip_black`, `dip_white`, `flash`, `whip_left/right/up`, `zoom`, `blur`,
  `wipe`, `smooth`, `cover`, `reveal`, `iris`, `radial`, `squeeze`, `slices`, `wind`, `pixel`, `gray_fade`.

```
cuts hype        # a style pack picks each cut from what happens at that join
cuts hype 7      # same pack, different roll
cut 3 smash      # set one join
cut whip_left    # set all joins
```

Packs: `clean`, `documentary`, `comedy`, `hype`, `trailer`. They look at each join: same shot on both sides
(a jump cut, so punch in), speech crossing it (bridge with a J or L cut), fast motion (whip), a big colour
change (dip or flash). They never repeat an effect twice in a row, and hard cuts stay the most common.

## Train your own detector

Teach vidx a moment it doesn't know, from time ranges you label:

```bash
uv run vidx teach knockdown --pos abcd1234:12-14,abcd1234:88-90,ef012345:30-33 --neg abcd1234:50-70,ef012345:0-20,ef012345:40-60
```

This trains a small new layer (about 500 weights) on frozen CLIP embeddings plus motion, in under a minute.
It reports precision and recall on held-out ranges. Then `query(video_id, "knockdown")` finds it in any video.
When it gets something wrong, add that range to `--neg` and teach again. Use several videos: a detector taught
on one video may learn that video rather than the idea. It sees one frame per second plus motion, so it suits
things visible in a frame or a burst of movement, not a split-second contact.

## Timeline files

Every edit, DSL or not, compiles to one JSON timeline, saved as `~/.vidx/<id>/v<N>.timeline.json`. Change
it and pass it back with `edit(id, "@/path/to/file.json")`, or write one directly for per-clip control:

```json
{"clips": [
   {"in": 12.0, "out": 18.5, "captions": [{"at": 0.5, "for": 2, "text": "Hello"}]},
   {"in": 40.0, "out": 44.0, "video": "other_video_id", "speed": 1.2,
    "grade": {"contrast": 1.05, "match": {"from": 12, "to": 18}},
    "transition": {"type": "xfade", "dur": 0.3}},
   {"in": 90.0, "out": 95.0}],
 "audio": [{"sfx": "impact_01", "at": 3.2, "gain": -6}, {"music": "drone_02", "gain": -18}]}
```

Rendering caches each clip and each transition in `~/.vidx/cache` (5 GB, `VIDX_CACHE_GB`). Change one shot
and only that shot re-encodes; the rest is a stream copy. Renders land in `vidx_renders/` next to the
source (`VIDX_OUTPUT_DIR` to override).

Sounds: `drone_01`-`06`, `impact_01`-`06`, `riser_01`-`05`, `whoosh_01`-`03`. All synthesized by
`scripts/make_sfx.py`, so they ship under this repo's MIT license.

Presets: `original` (as is), `reel_9x16` (1080×1920, −16 LUFS, ≤90 s), `yt_16x9` (1920×1080, −14 LUFS).

## CLI (same output as the tools)

```bash
uv run vidx index ~/Movies/talk.mp4
uv run vidx overview 5450c7a6
uv run vidx query 5450c7a6 "whiteboard diagram" --kind visual
uv run vidx segment 5450c7a6 120 150
uv run vidx peek 5450c7a6 "12.5,s3,40" --out sheet.jpg
uv run vidx edit 5450c7a6 "drop silence>1.0;snap words;pad 0.1"
uv run vidx render v1 --preset reel_9x16 --preview
uv run vidx render v1 --preset reel_9x16
uv run vidx peek v1 cuts --out cuts.jpg
uv run vidx cuts             # the cut library
uv run vidx check v1
uv run vidx diff v1 v2
uv run vidx usage            # characters returned per call (~tokens)
```

## Optional extras

- `uv sync --extra captions` → `index_video(..., captions=true)` adds Florence-2 one-line shot captions.
- `uv sync --extra otio` → every render also writes a `.otio` timeline (opens in Resolve / Premiere).
- `laugh` events (heuristic audience-reaction detector) run by default: `query(id, "laugh")`.
- `index_video(..., glitch=true)` scans AI-generated clips for morphing and reports the last clean frame.
  These are suspects to confirm with `peek`: expect about one false alarm per handheld phone clip.

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
  analyzers/   probe, shots (TransNetV2 + curves), detail (keyframes, OCR, watermarks), visual, transcript,
               audio events (+letterbox), audiomap (speech/music/sfx, beats), laugh, glitch, captions
               — pluggable list
  store.py     SQLite (~/.vidx/vidx.db) + per-video .npy embeddings
  query.py     overview / query / get_segment / peek
  dsl.py       edit script -> timeline (pure segment math)
  timeline.py  the JSON edit format and its validation
  render.py    timeline -> cached pieces -> ffmpeg
  color.py     Reinhard colour match as a 3D LUT
  cuts.py      the cut library and style packs
  detect.py    trainable detectors on frozen CLIP
  assets/sfx/  20 bundled sounds
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

## License

MIT, see [LICENSE](LICENSE).
