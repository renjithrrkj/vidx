# A/B protocol: editing a video with vidx vs without

Goal: for each new video, do the *same* edit two ways and report what each cost and how good the
result is. "Without" is what an agent does today with no video index: read the whole transcript,
look at frames, hand-write ffmpeg.

## Setup (once per machine)

```bash
brew install ffmpeg          # or apt install ffmpeg (Cowork VM is Linux)
cd vidx && uv sync
```
vidx tools come from the MCP server (`vidx-mcp`) or the CLI (`uv run vidx ...`). The naive tools
are `uv run python scripts/naive.py ...`. Both log every call to `~/.vidx/usage.log`.

## Per video

1. **Write the brief first** (same for both ways), e.g. "remove dead air and make a 60 s 9:16 reel
   of the funniest bit". Note the start time: `date +%s`.
2. **Way A, naive.** Allowed: `naive.py probe`, `transcript`, `silences`, `frames --every N`
   (and reading those frame images), `cut`. No vidx tools. Pick ranges by reading.
3. **Way B, vidx.** Allowed: the 9 vidx tools only. Indexing is done once and shared (it is the
   same ASR model; its cost is compute, not tokens).
4. **Quality on both outputs**, same yardstick:
   - `vidx check` on B. For A, index the naive output as a video and run `query silence>1.0`,
     plus listen/peek at every join.
   - Did the result meet the brief? (duration limit, chosen moment, no mid-word cuts, no black.)
5. **Report** with `uv run python scripts/ab_report.py <video_id> --since <start>`:
   tokens in (text + images), calls, wall time, issues found, retries needed.

## What to report

| metric | how |
|---|---|
| tokens the agent read | ab_report.py (chars/4 + image tokens) |
| context carried per turn | the largest single tool result, since it stays in context every later turn |
| calls / retries | ab_report.py call list |
| wall time | start/end timestamps |
| output quality | check findings, brief met y/n, notes |

## Caveats to state in every report

- Tool-output tokens are only part of session cost. Whatever an agent reads stays in context and is
  re-sent every later turn, so one 7k-token transcript dump costs more over a 20-turn session than
  twenty 300-token answers. Report the largest single result as well as the sum.
- If the Cowork UI shows actual session token usage, record that too; it beats our estimate.
- The naive agent is not strawmanned: it may use a compact phrase transcript and sparse frames if it
  chooses. Record what it actually chose.
- Run the naive way first (otherwise knowledge from the vidx run leaks into the naive choices), or
  run each in a fresh session.
