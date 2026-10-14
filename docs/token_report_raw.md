
## 5450c7a6: 10.0 min, 1080x1920, 1 shots

### Naive baselines

| Naive approach | ~tokens | detail |
|---|---:|---|
| Transcript, word-level timestamps (what most ASR tools return) | 7,305 | 1561 words, 29,219 chars |
| Transcript, phrase lines with timestamps (already compacted) | 2,235 | 8,939 chars |
| Frames at 1 fps, 384 px | 66,711 | 601 frames x 111 tok |
| Frames at 1 per 5 s, 384 px | 13,431 | 121 frames |
| One frame per shot, 384 px | 111 | 1 shots |

### vidx tool calls (measured output)

| What the agent receives | ~tokens | detail |
|---|---:|---|
| index_video summary | 137 | 548 chars |
| overview (scenes, default limit 20) | 507 | 2,028 chars |
| overview (all shots, paged to the end) | 58 | 229 chars |
| query speech | 93 | 370 chars |
| query visual | 54 | 213 chars |
| query event (silence>1.0) | 78 | 309 chars |
| get_segment 30 s | 181 | 723 chars |
| peek 4 frames | 0 | n/a (source file deleted before this run; see TOKEN_REPORT.md) |
| peek 16-tile visual summary | 0 | n/a (source file deleted before this run; see TOKEN_REPORT.md) |
| edit (silence cleanup) | 49 | 193 chars |
| check (rendered version) | 15 | 57 chars |

### Task totals

| Task | naive ~tokens | vidx ~tokens | saving |
|---|---:|---:|---:|
| Get the gist of the whole video | 7,416 | 644 | 12x |
| Find a moment (speech + visual), read it, look at it | 7,416 | 972 | 8x |
| Cut dead air, render, QA | 7,305 | 786 | 9x |

## ca833113: 7.9 min, 1280x720, 124 shots

### Naive baselines

| Naive approach | ~tokens | detail |
|---|---:|---|
| Transcript, word-level timestamps (what most ASR tools return) | 2,846 | 623 words, 11,382 chars |
| Transcript, phrase lines with timestamps (already compacted) | 924 | 3,696 chars |
| Frames at 1 fps, 384 px | 52,392 | 472 frames x 111 tok |
| Frames at 1 per 5 s, 384 px | 10,545 | 95 frames |
| One frame per shot, 384 px | 13,764 | 124 shots |

### vidx tool calls (measured output)

| What the agent receives | ~tokens | detail |
|---|---:|---|
| index_video summary | 169 | 673 chars |
| overview (scenes, default limit 20) | 762 | 3,046 chars |
| overview (all shots, paged to the end) | 3,307 | 13,226 chars |
| query speech | 57 | 225 chars |
| query visual | 116 | 464 chars |
| query event (silence>1.0) | 40 | 158 chars |
| get_segment 30 s | 251 | 1,002 chars |
| peek 4 frames | 468 | 1 image 772x436 + caption |
| peek 16-tile visual summary | 1,592 | 1 image 1548x876 + caption |
| edit (silence cleanup) | 38 | 152 chars |
| check (rendered version) | 14 | 56 chars |

### Task totals

| Task | naive ~tokens | vidx ~tokens | saving |
|---|---:|---:|---:|
| Get the gist of the whole video | 16,610 | 931 | 18x |
| Find a moment (speech + visual), read it, look at it | 16,610 | 1,823 | 9x |
| Cut dead air, render, QA | 2,846 | 1,023 | 3x |

## 1a04b524: 59.7 min, 1280x720, 1 shots

### Naive baselines

| Naive approach | ~tokens | detail |
|---|---:|---|
| Transcript, word-level timestamps (what most ASR tools return) | 8,182 | 1620 words, 32,727 chars |
| Transcript, phrase lines with timestamps (already compacted) | 2,501 | 10,004 chars |
| Frames at 1 fps, 384 px | 397,713 | 3583 frames x 111 tok |
| Frames at 1 per 5 s, 384 px | 79,587 | 717 frames |
| One frame per shot, 384 px | 111 | 1 shots |

### vidx tool calls (measured output)

| What the agent receives | ~tokens | detail |
|---|---:|---|
| index_video summary | 166 | 664 chars |
| overview (scenes, default limit 20) | 564 | 2,256 chars |
| overview (all shots, paged to the end) | 60 | 238 chars |
| query speech | 22 | 88 chars |
| query visual | 56 | 223 chars |
| query event (silence>1.0) | 73 | 291 chars |
| get_segment 30 s | 162 | 646 chars |
| peek 4 frames | 467 | 1 image 772x436 + caption |
| peek 16-tile visual summary | 1,039 | 1 image 1160x656 + caption |
| edit (silence cleanup) | 74 | 296 chars |

### Task totals

| Task | naive ~tokens | vidx ~tokens | saving |
|---|---:|---:|---:|
| Get the gist of the whole video | 8,293 | 730 | 11x |
| Find a moment (speech + visual), read it, look at it | 8,293 | 1,437 | 6x |
| Cut dead air, render, QA | 8,182 | 877 | 9x |
