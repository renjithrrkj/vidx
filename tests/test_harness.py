"""The harness's safety logic. A hole here means a change could reach a human unchecked, so each rule is tested."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


guard = load(ROOT / ".claude/hooks/guard.py")
gate = load(ROOT / "scripts/gate.py")
harness = load(ROOT / "scripts/harness.py")

R = "/work/vidx"


@pytest.mark.parametrize("path", ["scripts/gate.py", f"{R}/scripts/gate.py", "CLAUDE.md", ".claude/settings.json",
                                  ".github/workflows/tests.yml", "tests/test_harness.py", "src/../CLAUDE.md"])
def test_guard_blocks_edits_to_the_rules(path):
    assert guard.reason_to_block("Edit", {"file_path": path}, R)


@pytest.mark.parametrize("path", ["/etc/hosts", "/Users/x/.vidx-harness/baseline.json", "../other/file.py", f"{R}-evil/a.py"])
def test_guard_blocks_edits_outside_the_checkout(path):
    assert guard.reason_to_block("Write", {"file_path": path}, R)


@pytest.mark.parametrize("path", ["src/vidx/render.py", f"{R}/tests/test_dsl.py", "README.md"])
def test_guard_allows_normal_work(path):
    assert guard.reason_to_block("Edit", {"file_path": path}, R) is None


@pytest.mark.parametrize("cmd", [
    "git push origin main", "cd x && git push", "gh pr merge 3", "gh api repos/a/b", "rm -rf src", "git reset --hard",
    "git commit --no-verify -m x", "python3 scripts/gate.py --rebaseline", "cat ~/.vidx-harness/baseline.json",
    "ls ~/.vidx/cache", "echo x > scripts/gate.py", "sed -i '' s/a/b/ CLAUDE.md", "git checkout main -- .claude/settings.json",
])
def test_guard_blocks_commands(cmd):
    assert guard.reason_to_block("Bash", {"command": cmd}, R)


@pytest.mark.parametrize("cmd", ["uv run pytest -q", "git diff", "python3 scripts/gate.py --fast", "cat scripts/gate.py"])
def test_guard_allows_commands(cmd):
    assert guard.reason_to_block("Bash", {"command": cmd}, R) is None


def test_guard_exit_codes_are_what_claude_code_expects():
    def call(payload):
        return subprocess.run([sys.executable, str(ROOT / ".claude/hooks/guard.py")], input=payload, text=True,
                              capture_output=True).returncode

    blocked = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git push"}, "cwd": R})
    allowed = json.dumps({"tool_name": "Read", "tool_input": {"file_path": "a"}, "cwd": R})
    assert call(blocked) == 2 and call(allowed) == 0
    assert call("not json") == 2  # a broken call fails closed


def issue(**kw):
    return {"state": "OPEN", "author": {"login": harness.OWNER}, "labels": [{"name": "ready"}], **kw}


def test_only_the_owners_ready_issues_are_worked():
    assert harness.issue_problem(issue()) is None
    assert harness.issue_problem(issue(author={"login": "stranger"}))
    assert harness.issue_problem(issue(labels=[]))
    assert harness.issue_problem(issue(state="CLOSED"))
    assert harness.issue_problem(issue(labels=[{"name": "ready"}, {"name": "blocked"}]))


def test_finished_changes_are_stopped_for_the_right_reasons():
    def problems(files, patch="", sizes=None):
        return harness.diff_problems(files, patch, sizes or {})

    assert problems(["src/vidx/api.py"], "+x = 1") == []
    assert problems(["scripts/gate.py"])
    assert problems([".claude/hooks/guard.py"])
    assert problems(["tests/data/clip.MOV"])
    assert problems(["a.bin"], sizes={"a.bin": 3_000_000})
    assert problems(["a.py"], "+p = '/Users/someone/video.mov'")
    assert problems(["a.py"], "+mail = 'someone@gmail.com'")
    assert problems(["t.py"], "+@pytest.mark.skip(reason='later')")
    assert problems(["t.py"], "-p = '/Users/removed'") == []  # removing a path is fine
    assert problems(["t.py"], "+@pytest.mark.xfail")
    assert problems(["t.py"], "+@pytest.mark.skipif(True, reason='x')")
    # guarding a test that needs ffmpeg is normal, and wrongly blocked the first real run
    assert problems(["t.py"], "+needs_ffmpeg = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='no ffmpeg')") == []


CLIP = {"size": [1080, 1920], "length": 30.0, "tokens": 1000, "check_issues": 0, "loudness": -16.0, "secs": 60.0,
        "peak_mb": 1200, "frames": [[100] * 576] * 6}


def test_identical_results_pass():
    assert gate.compare(CLIP, dict(CLIP)) == ([], [])


@pytest.mark.parametrize("change", [{"size": [720, 1280]}, {"check_issues": 1}, {"secs": 200.0}, {"peak_mb": 1900}])
def test_broken_results_are_hard_failures(change):
    hard, _ = gate.compare(CLIP, {**CLIP, **change})
    assert hard


@pytest.mark.parametrize("change", [{"tokens": 1100}, {"length": 31.0}, {"loudness": -12.0}, {"frames": [[140] * 576] * 6}])
def test_changed_output_is_drift_not_a_failure(change):
    hard, drift = gate.compare(CLIP, {**CLIP, **change})
    assert drift and not hard


def test_small_noise_is_ignored():
    near = {**CLIP, "tokens": 1040, "length": 30.1, "loudness": -16.5, "secs": 80.0, "peak_mb": 1700, "frames": [[103] * 576] * 6}
    assert gate.compare(CLIP, near) == ([], [])


def test_memory_ceiling_holds_even_if_the_baseline_was_already_high():
    high = {**CLIP, "peak_mb": 3400}
    assert gate.compare(high, {**high, "peak_mb": 3600})[0]


def test_one_bad_frame_is_not_averaged_away():
    frames = [[100] * 576] * 5 + [[200] * 576]
    assert gate.frame_difference(CLIP["frames"], frames) == 100
    assert gate.frame_difference(CLIP["frames"], []) == 255.0


def test_deleting_tests_fails_the_gate():
    base = {"fast_tests": 100, "e2e_tests": 1, "clips": {}}
    assert gate.judge(base, {"fast_tests": 99, "e2e_tests": 1, "clips": {}})[0]
    assert not gate.judge(base, {"fast_tests": 101, "e2e_tests": 1, "clips": {}})[0]


def test_scorecard_is_safe_to_post():
    now = {"fast_tests": 1, "e2e_tests": 1, "clips": {"a": {**CLIP}}}
    card = gate.scorecard(now, ["clip a: frame size"], ["clip a: length"], False)
    assert "FAILED" in card and not harness.PRIVATE_TEXT.search(card)
    assert "PASSED" in gate.scorecard(now, [], ["clip a: length"], True)
    assert "FAILED" in gate.scorecard(now, [], ["clip a: length"], False)


def test_only_time_and_memory_failures_get_a_second_measurement():
    assert gate.noisy_clips(["clip a: took 90s (was 50s)", "clip b: peak memory 1741 MB (was 1105 MB)"]) == ["a", "b"]
    assert gate.noisy_clips(["clip a: took 90s (was 50s)", "clip a: frame size [1, 1] (was [2, 2])"]) == []
    assert gate.noisy_clips(["clip a: peak memory 3600 MB is over the 3500 MB ceiling"]) == []
    assert gate.noisy_clips(["fast_tests failed"]) == []


def test_builders_can_read_and_run_the_quick_gate():
    tools = harness.BUILDER_TOOLS
    for rule in ("Bash(python3 scripts/gate.py --fast:*)", "Bash(uv run python3:*)", "Bash(cat:*)"):
        assert rule in tools
    assert "git push" not in tools and "gh " not in tools
