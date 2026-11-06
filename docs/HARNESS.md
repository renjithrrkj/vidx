# The development harness

Claude builds a GitHub issue, a fixed gate checks the result, a separate session reviews it, and a pull request
waits for a human. Nothing merges by itself.

## Run an issue

1. Write the issue and add the `ready` label. Add `baseline-change` too if the output is meant to change.
2. Run it:

   ```bash
   python3 scripts/harness.py 12
   ```

3. Read the pull request, watch the renders in `~/.vidx-harness/runs/`, and merge if you are happy.

If it can't finish in 3 attempts, the issue gets the `blocked` label and the run log says why.

## What the gate checks

`scripts/gate.py` runs the tests, then three fixed edits on private clips. It compares each result with the
saved baseline: frame size, `check` problems, time, memory, tool output, length, loudness and picture.

Broken results always fail. Changed output fails unless the issue has `baseline-change`.

After merging a change that adds tests or meant to change output, accept the new normal:

```bash
python3 scripts/gate.py --rebaseline
```

## What keeps it honest

- Only `ready` issues written by the owner are worked on, and issue comments are ignored.
- Claude sessions can edit files and run tests. The runner does all git and GitHub work.
- A hook blocks changes to the rules, and the runner checks the finished diff again.
- A change is refused if it skips a test, adds media, or contains a local path or email address.
- Clips and baselines stay in `~/.vidx-harness`, outside the repo. Renders never go to GitHub.
- One job runs at a time, on mains power. It won't start below 25% free memory and stops below 8%.

This is not a sandbox. Code run by tests can do anything your account can, which is why only your own issues
are accepted and why `main` should be protected.
