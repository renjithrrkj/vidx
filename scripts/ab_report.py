"""Compare the two ways of editing the same video from ~/.vidx/usage.log.

  uv run python scripts/ab_report.py <video_id> [--since <unix time>]

"vidx" = calls to the 9 vidx tools; "naive" = scripts/naive.py commands. Tokens are estimated as
chars/4 for text plus logged image tokens. Wall time and output quality are reported separately
(quality = `vidx check` run on both outputs).
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

from vidx import config


def load(video: str, since: float = 0) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for line in (config.HOME / "usage.log").read_text().splitlines():
        r = json.loads(line)
        if r["target"].startswith(video) and r["t"] >= since:
            by["naive" if r["tool"].startswith("naive:") else "vidx"].append(r)
    return by


def main(video: str, since: float = 0) -> None:
    by = load(video, since)
    print(f"| way | calls | text ~tok | images | image ~tok | total ~tok |\n|---|---:|---:|---:|---:|---:|")
    totals = {}
    for way in ("vidx", "naive"):
        rows = by.get(way, [])
        txt = sum(math.ceil(r["chars"] / 4) for r in rows)
        imgs = sum(r.get("images", 0) for r in rows)
        itok = sum(r.get("image_tokens", 0) for r in rows)
        totals[way] = txt + itok
        print(f"| {way} | {len(rows)} | {txt:,} | {imgs} | {itok:,} | {txt + itok:,} |")
    if totals.get("vidx") and totals.get("naive"):
        print(f"\nnaive / vidx = {totals['naive'] / totals['vidx']:.1f}x")
    for way in ("vidx", "naive"):
        print(f"\n{way} calls:")
        agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in by.get(way, []):
            agg[r["tool"]][0] += 1
            agg[r["tool"]][1] += math.ceil(r["chars"] / 4) + r.get("image_tokens", 0)
        for tool, (n, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
            print(f"  {tool:<18} {n:>3} calls  ~{t:,} tok")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[a.index("--since") + 1]) if "--since" in a else 0)
