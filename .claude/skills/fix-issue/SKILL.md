---
name: fix-issue
description: End-to-end workflow for fixing GitHub issues on the nf-metro repo. Use when the user references a GitHub issue (by number, URL, or description) and wants it fixed. Handles worktree setup, environment creation, implementation, testing, visual review, and PR creation. Trigger on phrases like "fix issue #N", "address #N", "work on issue N", or any request to fix a bug or implement a feature that references an issue.
---

# Fix Issue

Structured workflow for fixing nf-metro GitHub issues in an isolated worktree. Delegates visual regression checking to the companion `/render-topologies` skill.

## Phase 1: Understand the Issue

```bash
gh issue view <N> --repo pinin4fjords/nf-metro
```

Summarize the problem and proposed approach. Wait for user confirmation before proceeding.

## Phase 2: Worktree + Environment Setup

```bash
# Worktree
cd /Users/jonathan.manning/projects/nf-metro
git fetch origin main
git worktree add /tmp/nf-metro-fix-<N> -b fix/<N>-<slug> origin/main

# Fix environment
ulimit -n 1000000 && export CONDA_OVERRIDE_OSX=15.0 && /opt/homebrew/bin/micromamba create -n nf-metro-fix-<N> python=3.11 cairo -y
source ~/.local/bin/mm-activate nf-metro-fix-<N>
pip install -e "/tmp/nf-metro-fix-<N>[dev]" && pip install cairosvg

# Baseline environment (shared across issues, create only if missing)
if [ ! -d "/Users/jonathan.manning/micromamba/envs/nf-metro-main" ]; then
  ulimit -n 1000000 && export CONDA_OVERRIDE_OSX=15.0 && /opt/homebrew/bin/micromamba create -n nf-metro-main python=3.11 cairo -y
fi
cd /Users/jonathan.manning/projects/nf-metro && git checkout main && git pull origin main
source ~/.local/bin/mm-activate nf-metro-main
pip install -e "/Users/jonathan.manning/projects/nf-metro[dev]" && pip install cairosvg
```

All subsequent work happens inside `/tmp/nf-metro-fix-<N>`.

## Phase 3: Implement the Fix

The shell cwd resets after each Bash call. Always chain `cd` into the worktree:

```bash
source ~/.local/bin/mm-activate nf-metro-fix-<N> && cd /tmp/nf-metro-fix-<N> && ruff format src/ tests/ && ruff check src/ tests/ && pytest
```

Fix any failures before proceeding.

## Phase 4: Visual Review

### 4a: Motivating example before/after

If the issue references a specific `.mmd` file, render from both environments:

```bash
# Before (main, unmodified)
source ~/.local/bin/mm-activate nf-metro-main
python -m nf_metro render <file.mmd> -o /tmp/<name>_before.svg
python -c "import cairosvg; cairosvg.svg2png(url='/tmp/<name>_before.svg', write_to='/tmp/<name>_before.png', scale=2)"

# After (worktree with fix)
source ~/.local/bin/mm-activate nf-metro-fix-<N>
cd /tmp/nf-metro-fix-<N> && python -m nf_metro render <file.mmd> -o /tmp/<name>_after.svg
python -c "import cairosvg; cairosvg.svg2png(url='/tmp/<name>_after.svg', write_to='/tmp/<name>_after.png', scale=2)"

open /tmp/<name>_before.png /tmp/<name>_after.png
```

Each env must point at a **different source directory** (main repo vs worktree). Do NOT use `git stash` to switch within one env - Python's bytecode cache serves stale code.

**STOP and ask the user to review.** If identical or non-reproducing, discuss before proceeding.

### 4b: Full regression check

Ask the user to run `/render-topologies` for a pixel-level diff of all `.mmd` renders against main. That skill handles baseline rendering, branch rendering, diffing, and opening only the changed BEFORE/AFTER pairs.

**STOP and wait for the user to confirm no regressions.** If they spot problems, return to Phase 3.

## Phase 5: Commit and PR

Once the user approves:

```bash
cd /tmp/nf-metro-fix-<N>
gh pr create --repo pinin4fjords/nf-metro --base main --title "<title>" --body "$(cat <<'EOF'
## Summary
<bullets>

Fixes #<N>

## Test plan
- [ ] pytest passes
- [ ] ruff check clean
- [ ] Visual review of all topology renders

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## Phase 6: Cleanup

Offer to clean up (only if user agrees):

```bash
cd /Users/jonathan.manning/projects/nf-metro
git worktree remove /tmp/nf-metro-fix-<N>
/opt/homebrew/bin/micromamba env remove -n nf-metro-fix-<N> -y
```

The `nf-metro-main` env is shared across issues - do NOT delete it.
