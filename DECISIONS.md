# Decisions

Choices made where the spec left room, or where reality differed from it.

## Environment / dependencies
- **Python 3.12** pinned via `.python-version` (spec says 3.11+; 3.14 on this machine is too new for reliable torch / ctranslate2 wheels).
- **MCP SDK 2.x**: `FastMCP` was renamed to `MCPServer` (`mcp.server.mcpserver`). Same decorator API, same `Image` helper. Targeting 2.x rather than pinning `mcp<2`.
- **scenedetect 0.7** hard-depends on `opencv-python` (not headless). Installing both variants corrupts `cv2/`, so we only use `opencv-python`. On macOS this prints harmless `objc: Class AVF… is implemented in both` warnings because cv2 and PyAV (from faster-whisper) both bundle libavdevice.
- **Apple silicon**: CLIP runs on MPS when available; faster-whisper (ctranslate2) has no MPS backend, so it runs CPU int8 using all cores. CUDA is used for both when present.

## Storage
- Extra columns/tables beyond the spec: `videos.meta` (JSON: has_audio, language, language_mix, lufs, size/mtime), `shots.tags`, `versions.preset`, and `analyzer_runs(video_id, name, seconds, info)` which is how `index_video` skips analyzers already done and reports timings.
- Re-hashing a multi-GB file is skipped when path + size + mtime match a known video.
- Version ids are global integers shown as `v3` (short, unambiguous across videos).
- `audio16k.wav` (mono 16 kHz) is cached per video; it feeds whisper and the `audio_pop` check.

## Analyzers
- **Shots**: instead of scenedetect's OpenCV video backend, frames come from an ffmpeg pipe (VideoToolbox hw decode on macOS, scaled to 256px long side, CFR) fed to `ContentDetector.process_frame`. Much faster on 1080p. `motion` = mean ContentDetector frame score within the shot.
- **Visual tags**: without the optional captioner, overview lines would have no visual description. Each shot gets cheap CLIP zero-shot tags from a ~40-phrase vocabulary (subject + optional framing/setting when confident). Stored in `shots.tags`; a real caption (M6) takes precedence.
- **Transcript (biggest real-world finding)**: on a 10-min Malayalam stand-up set, Whisper `small` detected `ml` correctly but produced degenerate output (repeated glyphs, compression ratio 10-18, even Sinhala script), and the temperature-fallback loop re-generated each bad window up to 6x at max tokens: **1297 s for 600 s of audio**. Now:
  - greedy decoding at temperature 0 only (bounded cost), `condition_on_previous_text=False`, VAD on;
  - language detected on 3 voiced windows across the file (not just the first 30 s);
  - segments are dropped when degenerate (compression ratio > 2.4, 8+ repeated tokens, U+FFFD, or a script that doesn't match the language);
  - for languages small models handle badly (`ml, ta, te, kn, si, bn, mr, gu, …`) with tiny/base/small, the transcript is an **English translation** (`task="translate"`). The summary says so loudly, since word times are then approximate. Override with `VIDX_WHISPER_TASK=transcribe` or `VIDX_WHISPER_MODEL=large-v3`.
  - Result on the same file: **120 s**, coherent, searchable English (e.g. a fuzzy query for "my mom gave me a smart phone" finds "my mother gave me a new smartphone").
  - `multilingual=True` was dropped: it gave inconsistent per-window translate/transcribe behaviour, and faster-whisper doesn't expose per-segment language anyway. `language_mix` now reports the *script* mix of the transcribed words.
- **Audio events**: one ffmpeg pass computes silencedetect (−35 dB, ≥0.5 s), ebur128 (momentary + integrated), blackdetect (≥0.3 s) and freezedetect (≥1.0 s). `loud` = momentary loudness ≥ integrated + 10 LU for ≥0.3 s (relative so it works for quiet and hot recordings).
- Rotation metadata is honoured so phone footage reports its display orientation.
- **Letterbox detection**: the events pass also runs per-frame `cropdetect`; height and width are voted separately (each must cover ≥60% of frames), so full-frame title cards don't cancel a film's 2.35:1 bars. Stored as `meta.active`; `reel_9x16`/`yt_16x9` crop to it before reframing (otherwise a 9:16 reel from a letterboxed film keeps big black bars, as found on a real short film).
- **Window embeddings**: shots longer than 15 s also get a CLIP embedding per ~5 s window (`win_emb.npy`). A one-take 10-min talk is a single shot, and visual search would otherwise only ever return "the whole video". The per-shot embedding the spec asks for is unchanged; windows are extra. Adjacent windows of similar score are merged into one hit.
- CLIP weights are loaded from the local HF cache path once downloaded (no Hub round-trip or warnings per query).
- **laugh** (M6 plugin demo): DSP heuristic, no model. Non-speech (outside word intervals), energetic, spectrally flat, with 3-8 Hz energy modulation. On the stand-up set it marks reactions after punchlines and coincides with `loud` events; expect false positives on music/wind.
- **captions** (M6, optional): Florence-2-base through transformers' native `Florence2ForConditionalGeneration` (`florence-community/Florence-2-base`; transformers 5.x breaks the old remote-code path, which stays as a fallback). ~2.4 s/shot on M1 MPS, which is why it's opt-in. Captions override CLIP tags in overview lines.

## Query / output
- `overview(level="scenes")` always covers the whole video in ≤ `limit` lines: adjacent shots are merged agglomeratively (cost = combined duration × visual dissimilarity), and long single shots are first split at the largest speech pauses (otherwise a one-take talking head is a single useless line). Scene ids are `c1..cN` and each line lists the shot ids it spans.
- `overview(level="shots")` pages with an extra `offset` arg.
- Repeated identical visual descriptions on consecutive overview lines print as `(same view)`.
- Overview lines carry compact event markers (`[silence 2x 3.1s, loud]`); silences < 1 s are omitted there to save tokens.
- `query(kind="any")` returns separate `speech:` and `visual:` sections (scores aren't comparable across modalities, so no fake merged ranking). An event-syntax query (`silence>1.5`) is routed to events automatically.
- Visual hits are labelled `strong / ok / weak` from raw CLIP similarity + z-score, so the agent knows when to `peek` instead of trusting the hit.
- `get_segment` accepts shot ids as start/end, caps output at ~4000 chars with a continuation hint, and supports `fields=wordtimes` for per-word timestamps.
- `peek` returns JPEG (not PNG): same image tokens, ~10x smaller payload. `size` is the longest side of each tile. With no times it returns an evenly spread visual summary.
- Every api call logs `{tool, target video, chars, images, secs}` to `~/.vidx/usage.log`; version targets are resolved to their video. `vidx usage [--video id]` totals per video and tool.

## DSL
- Op order: base → drop/keep in line order → `snap words` → `pad` → merge (<0.1 s gaps, forward only) → discard (<0.3 s, with warning).
- With `base all`, `keep` unions back into source order; with `base none`, keeps append in line order (reordering allowed). Drops always subtract.
- Extra target forms: shot ranges `s3-s5`, any event kind (`drop black`, `drop freeze`), `mm:ss.s` times.
- `snap words`: a cut point inside a word moves to the nearer word gap; a segment *start* sits just before the following word, an *end* just after the preceding word (≤0.15 s from the word), so `pad` then adds handles into silence.
- `pad` never extends into a word lying outside the kept segment (no half-syllables).
- **Acoustic word clipping**: before snap / pad / the mid-word check, ASR word intervals are clipped to non-silent audio (silencedetect events), and a word entirely inside a silence is dropped. Found on real footage: Whisper stamped a phantom word inside a measured 1.3 s silence, so `snap words` moved the cut onto real speech onset and `pad` refused to add a handle. Audio pops went 8 → 2 (→ 0 with the `check`-suggested `trans xfade 0.08`).
- `trans xfade d` is clamped to half the shortest segment (with a warning).
- `edit(parent_version=…)` appends the new script to the parent's script (incremental fixes like "drop s7"); a script with its own `base` line stands alone. Lineage is kept either way.

## Render
- Segment edges are snapped to frame boundaries so frame-accurate video and sample-accurate audio stay in sync across hundreds of cuts.
- Segments are grouped into forward-moving runs; each run gets its own (fast-seeked) input, so reordered edits don't make ffmpeg buffer raw frames. Within a run, `split`/`asplit` + `trim`/`atrim` + `concat` (or `xfade`/`acrossfade` chains).
- Filter graphs go through a script file (`-/filter_complex file`) — silence removal can produce hundreds of segments.
- Presets: `reel_9x16` center-crops to 1080×1920 and loudnorms to −16 LUFS; `yt_16x9` crops to 1920×1080 for horizontal sources but **pads** vertical sources (cropping 9:16 → 16:9 would lose ~70% of the frame); loudnorm −14 LUFS. `original` keeps size and loudness.
- Encoder: libx264 veryfast CRF 20 + AAC 192k; override with `VIDX_ENCODER=h264_videotoolbox`.

## Export
- OTIO export runs after every render when `opentimelineio` is installed (`uv sync --extra otio`), giving `<render>.otio` next to the mp4 with video + audio tracks. Dissolves consume handles in OTIO rather than overlapping clips, so each clip is shortened by d/2 at every transition to match the rendered duration. Crop and speed go in timeline metadata.

## Checks
- `mid_word_cut` flags a join only when it is more than 50 ms inside a word (whisper word stamps are themselves ±~100 ms).
- `audio_pop` suggests `trans xfade 0.08` instead of `pad` when the script already pads ≥ 0.1 s.
- `audio_pop` compares 30 ms RMS on each side of hard-cut joins in the source audio; jumps between two near-silences (both < −45 dB) are ignored.
- Black frames that already existed in the source are still reported (marked "from source") — they are usually unwanted in an edit. Freezes present in the source are skipped, per spec.
