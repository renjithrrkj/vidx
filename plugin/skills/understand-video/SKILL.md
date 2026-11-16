---
name: understand-video
description: Understand, search, trim or check a video file with vidx. Use when the user gives a local video or audio file and wants to know what is in it, find a moment, cut dead air or bad takes, make a reel or short, or check an edit before posting.
---

vidx reads the video once on the user's machine and answers in short text. Read before you look, and look before you cut.

1. `index_video` with the absolute path. The first run downloads speech and vision models, so warn the user it can take a few minutes.
2. `overview` for the whole video. Each line is a stretch of time with what is said and shown.
3. Narrow down with `query` (words, visuals, on-screen text, or events such as `silence>1.0`, `laugh`, `unclear`) and `get_segment` (`wordtimes` for exact cut points).
4. If the index reports unclear speech, run `query(video, "unclear")` and `peek` those times before cutting near them. Speech recognition can miss a sentence, and a gap in the words is not always silence.
5. Use `peek` only when you must see frames. It costs image tokens.
6. To edit, write a short script with `edit`, then `render` with `preview=true` while iterating.
7. Always run `check` on the render and fix what it reports with `edit(parent_version=...)`. Then `peek(version, ["cuts"])` to look at every join in one image.

Tell the user what you kept and why, with times from the source. If you are unsure whether a stretch is a bad take or a deliberate pause, ask.

vidx can also hand off: understand and plan with vidx, cut in another editor if the user prefers one, then index the exported file and run `check` style queries on it.
