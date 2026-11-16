# vidx

Video understanding for Claude, in small text.

vidx indexes a video on your own computer and gives Claude a compact map of it: what is said, what is
shown, shots, on-screen text, silences, laughs, music, and stretches where speech could not be
transcribed. Claude reads a few hundred tokens instead of thousands of frames, finds the moment you
mean, makes the cut, and checks the result for mid-word cuts, black frames, freezes and loudness before
you post.

## Use it

Ask Claude about a video file on your computer, for example:

- "What happens in ~/Movies/interview.mov? Give me the key moments with timestamps."
- "Cut the dead air and the false starts out of ~/Desktop/take3.mov and make a 9:16 reel under 40 seconds."
- "Check ~/Desktop/final.mp4 before I post it."

The plugin adds the vidx MCP server (nine tools: index, overview, query, segment, peek, edit, render,
check, diff) and a skill that tells Claude how to use them in the right order.

## Requirements

- Claude Code, or a Cowork session running on your computer. The claude.ai web chat can't start a local
  server, so the tools don't load there.
- [ffmpeg](https://ffmpeg.org) and [uv](https://docs.astral.sh/uv/) installed. On a Mac:
  `brew install ffmpeg uv`.
- About 2 GB of free disk space. Tested on Apple-silicon Macs with 8 GB of memory. Linux should work.
  Windows has not been tested.

## What it runs and downloads

- The plugin starts the server with `uvx --python 3.12 --from vidx==0.2.0 vidx-mcp`, which installs the
  pinned `vidx` package and its dependencies from PyPI the first time.
- The first index downloads open model weights: Whisper `small` for speech (about 460 MB) and CLIP
  ViT-B-32 for visual search (about 600 MB), both from Hugging Face, cached on your machine. The
  shot-detection and OCR models are already inside their Python packages.
- After that, vidx makes no network requests.

## Privacy policy

- **Data collection:** vidx collects nothing about you. It reads the video files you point it at.
- **Use and storage:** the index (transcript, shot list, small keyframe thumbnails, search embeddings) is
  stored in `~/.vidx` on your computer. Renders are written next to the source video in `vidx_renders/`,
  or to the path you give.
- **Third-party sharing:** none. Video, audio, transcripts and renders never leave your computer. The only
  network traffic is the one-time package and model downloads described above. Text that tools return
  goes to Claude as part of your conversation, as with any tool.
- **Retention:** kept until you delete it. Remove `~/.vidx` to delete every index.
- **Contact:** open an issue at https://github.com/renjithrrkj/vidx/issues.

## Licence

MIT. Source and full documentation: https://github.com/renjithrrkj/vidx
