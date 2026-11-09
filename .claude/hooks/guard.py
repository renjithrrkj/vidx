#!/usr/bin/env python3
"""Runs before every tool call in this repo. Exit 2 blocks the call and tells Claude why.

This catches honest mistakes and keeps the rules out of reach of the session they apply to. It is not a
sandbox: code run by tests can still do anything the user can. scripts/harness.py repeats the protected-file
check on the finished diff, and branch protection on main is the last line.
"""
from __future__ import annotations

import json
import os
import re
import sys

# The rules, the measurements and their tooling. Only a human changes these.
PROTECTED = (
    "scripts/gate.py", "scripts/harness.py", "tests/test_harness.py", "CLAUDE.md", ".claude/", ".github/",
)
PRIVATE = (".vidx-harness", "/.vidx/")

BLOCKED_COMMANDS = (
    (r"\bgit\s+push\b", "pushing is the runner's job"),
    (r"\bgit\s+(reset\s+--hard|clean\b|filter-branch)", "this discards work"),
    (r"\bgh\s+pr\s+merge\b", "only a human merges"),
    (r"\bgh\s+api\b", "repo settings are a human's job"),
    (r"(^|[\s;&|])rm\s+-[a-z]*[rf]", "recursive deletes are not allowed"),
    (r"--no-verify\b", "hooks stay on"),
    (r"--rebaseline\b", "only a human accepts a new baseline"),
)


def is_protected(path: str, root: str) -> bool:
    full = os.path.normpath(os.path.join(root, path))
    rel = os.path.relpath(full, os.path.normpath(root))
    return any(rel == p or rel.startswith(p) for p in PROTECTED)


def outside(path: str, root: str) -> bool:
    full = os.path.normpath(os.path.join(root, path))
    return not (full + os.sep).startswith(os.path.normpath(root) + os.sep)


def reason_to_block(tool: str, args: dict, root: str) -> str | None:
    if tool in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        path = args.get("file_path") or args.get("notebook_path") or ""
        if outside(path, root):
            return f"{path} is outside this checkout"
        if is_protected(path, root):
            return f"{path} is protected. Only a human changes the rules and the gate"
        return None
    if tool == "Bash":
        cmd = args.get("command", "")
        for pattern, why in BLOCKED_COMMANDS:
            if re.search(pattern, cmd):
                return why
        if any(p in cmd for p in PRIVATE):
            return "the harness folder and the real vidx data are off limits"
        # Shell writes dodge the Edit check, so look for a protected path next to anything that writes.
        if re.search(r">|\b(sed\s+-i|tee|mv|cp|rm|chmod|truncate|git\s+(checkout|restore|rm))\b", cmd):
            if any(p in cmd for p in PROTECTED):
                return "that command touches a protected file"
    return None


def main() -> int:
    try:
        call = json.load(sys.stdin)
    except ValueError:
        print("guard: could not read the tool call", file=sys.stderr)
        return 2
    root = os.environ.get("CLAUDE_PROJECT_DIR") or call.get("cwd") or os.getcwd()
    why = reason_to_block(call.get("tool_name", ""), call.get("tool_input") or {}, root)
    if why:
        print(f"Blocked: {why}.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
