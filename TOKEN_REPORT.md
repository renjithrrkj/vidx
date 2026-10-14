# vidx token report

*Measured 2026-10-14 on three real videos. Reproduce with `uv run python scripts/token_report.py <video_id>...`.*

## Bottom line

- To understand, search and edit a video, an agent using vidx reads **6–18x fewer tokens**
  than one handed a timestamped transcript plus frames, measured on 8–60 min videos.
- **The saving comes from pixels.** Looking at a video costs 10k–400k tokens in frames; vidx's
  overview describes it in **~500–760 tokens**, and frames are fetched only when asked for.
- **The overview doesn't grow with video length.** It was 507 tokens for 10 min and 564 for
  60 min, while frames and transcripts grow linearly. Longer video, bigger saving.
- **It is not a win everywhere.** For a short, dialogue-light film, a compact phrase-level
  transcript (924 tok) costs about the same as the vidx overview (762 tok). Listing every shot of
  a 124-shot film (3.3k tok) costs more than reading its transcript.

## Method

- **vidx side:** the real tool functions were called and their output measured exactly.
- **Naive baselines:** built from the same index (same ASR text, same frames), so only the
  *interface* differs, not the ASR quality.
- **Text tokens** ≈ chars / 4. **Image tokens** ≈ w×h / 750 after Claude's resize (≤1568 px,
  ≈1.15 MP), i.e. ~111 tok for a 384 px frame. Estimates only; nothing was sent to an API.

| id | video | length | shape | notes |
|---|---|---|---|---|
| 5450c7a6 | stand-up set (Malayalam → English translation) | 10.0 min | 1080×1920, 1 take | dense speech |
| ca833113 | short film (Malayalam → English) | 7.9 min | 1280×720, 124 shots | letterboxed, light dialogue |
| 1a04b524 | Riverside recording | 59.7 min | 1280×720, 1 take | speech ~19% of time, 465 s ASR unusable |

## Task totals (what the agent reads)

"Naive" here means a word-level transcript + one frame per shot (for the gist and find tasks),
and the word-level transcript alone (for the cut task).

| Task | 10-min stand-up | 8-min film | 60-min recording |
|---|---|---|---|
| Get the gist of the whole video | 7,416 → **644** (12x) | 16,610 → **931** (18x) | 8,293 → **730** (11x) |
| Find a moment (speech + visual), read it, look at it | 7,416 → **1,439** (5x)\* | 16,610 → **1,823** (9x) | 8,293 → **1,437** (6x) |
| Cut dead air, render, QA | 7,305 → **786** (9x) | 2,846 → **1,023** (3x) | 8,182 → **877** (9x) |

\* The stand-up's source file has since been deleted, so its `peek` (~467 tok) is taken from
yesterday's measurement.

The naive baselines are generous to the naive side. "One frame per shot" is only 1 frame for
a one-take video; an agent that really watched it would sample many more (next table).

## How each side grows with length

| ~tokens | 8-min film | 10-min stand-up | 60-min recording |
|---|---:|---:|---:|
| Frames, 1 fps | 52,392 | 66,711 | **397,713** |
| Frames, 1 per 5 s | 10,545 | 13,431 | **79,587** |
| Transcript, word-level | 2,846 | 7,305 | 8,182 † |
| Transcript, phrase lines | 924 | 2,235 | 2,501 † |
| **vidx overview** (whole video, ≤20 lines) | **762** | **507** | **564** |
| vidx index summary | 169 | 137 | 166 |

† The 60-min recording has little speech. A talk-dense hour at the stand-up's rate (~156
words/min) would be **~44k tokens** word-level and **~13k** as phrases. That is an
extrapolation, not a measurement. The overview would stay ~20 lines either way.

Raw silence detection on the 60-min recording found 847 silences: about 8k tokens if dumped
raw. `query("silence>1.0")` returns a count, the total and the top 10 in **73 tokens**, and
`drop silence>1.0` acts on all of them without the agent ever reading the list.

## Individual tool calls (typical size)

| call | ~tokens |
|---|---:|
| index_video summary | 137–169 |
| overview, scenes | 507–762 |
| query (speech / visual / event) | 22–116 |
| get_segment, 30 s | 162–251 |
| edit, check, diff | 14–74 |
| peek, 4 frames on one sheet | ~467 |
| peek, 16-tile summary | 1,039–1,592 |

A full working session on the stand-up (23 calls: overviews, queries, segments, edits,
2 peeks) returned **~9.4k chars ≈ 2.3k tokens** of text in total.

## Where vidx does *not* save tokens

- **`peek` costs what the pixels cost.** A 4-tile sheet (~467 tok) is about the same as 4
  separate frames (~444). The saving is in not sending frames by default, not in compression.
- **Short dialogue-light video vs a compact transcript:** roughly even (924 vs 762).
- **`overview(level="shots")` paged to the end** grows with the shot count (3.3k for 124
  shots). The default scene view is the cheap one.
- **Compute is the real cost instead of tokens.** Indexing takes CPU time once per video:
  ~4.6 min for 10 min of 1080p, ~9.7 min for 60 min of 720p on an M1 with 8 GB, at
  **≤951 MB peak memory** after the streaming fix below. Re-queries are free.

## Context carry-over (why the totals understate the gap)

Whatever an agent reads stays in its context and is re-sent on every later turn. The largest
single result is what the agent carries around:

| largest single result | naive | vidx |
|---|---:|---:|
| 10-min stand-up | 7,305 (word transcript) | 507 text / ~1,039 image |
| 60-min recording | 8,182 (≈44k for a talk-dense hour) or 79k+ in frames | 564 text / ~1,039 image |

Over a 20-turn editing session, carrying a 7k-token transcript re-bills it ~20 times (unless
prompt caching applies). A 500-token overview plus small answers stays small.

## Fixed along the way

Indexing the 60-min video first exhausted memory on this 8 GB machine. Whisper was given the
whole hour at once and built its full spectrogram in RAM. Audio is now streamed in ~5-min
chunks cut at quiet moments (and the laugh detector and pop check stream too). Peak memory for
the same file is **951 MB**. A file whose source was deleted also crashed `peek` with
unbounded recursion; it now fails with a one-line error.

## Caveats

- These are estimates (chars/4 plus the image formula), not billed usage.
- Both Malayalam videos are indexed as **English translations** (Whisper-small's Malayalam is
  unusable), which tends to make transcripts shorter than native text would be.
- This measures **what the agent reads**, not edit quality. The real test is the A/B protocol
  (`AB_PROTOCOL.md`): the same brief edited both ways on new videos, comparing tokens, calls,
  wall time and `check` results.
