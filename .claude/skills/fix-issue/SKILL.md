---
name: fix-issue
description: End-to-end workflow for fixing GitHub issues on the nf-metro repo. Use when the user references a GitHub issue (by number, URL, or description) and wants it fixed. Handles worktree setup, environment creation, implementation, testing, visual review, and PR creation. Trigger on phrases like "fix issue #N", "address #N", "work on issue N", or any request to fix a bug or implement a feature that references an issue.
---

# Fix Issue

Structured workflow for fixing nf-metro GitHub issues in an isolated worktree with a dedicated environment.

## Phase 1: Understand the Issue

Fetch the issue details:

```bash
gh issue view <NUMBER> --repo pinin4fjords/nf-metro
```

Read the full issue description and comments. Summarize the problem and proposed approach to the user before proceeding.

## Phase 2: Set Up Isolated Worktree

Create a branch and worktree so the main checkout stays clean:

```bash
cd /Users/jonathan.manning/projects/nf-metro
git fetch origin main
# Branch name: fix/<issue-number>-<short-slug>
git worktree add /tmp/nf-metro-fix-<NUMBER> -b fix/<NUMBER>-<slug> origin/main
```

All subsequent work happens inside `/tmp/nf-metro-fix-<NUMBER>`.

## Phase 3: Create Micromamba Environments

Create two environments: one for the worktree (fix branch) and one for the main repo (before/after comparison).

```bash
# Fix environment (worktree)
ulimit -n 1000000 && export CONDA_OVERRIDE_OSX=15.0 && /opt/homebrew/bin/micromamba create -n nf-metro-fix-<NUMBER> python=3.11 cairo -y
source ~/.local/bin/mm-activate nf-metro-fix-<NUMBER>
pip install -e "/tmp/nf-metro-fix-<NUMBER>[dev]" && pip install cairosvg

# Main environment (baseline for before/after comparison)
# First ensure the main repo is on the latest main branch
ulimit -n 1000000 && export CONDA_OVERRIDE_OSX=15.0 && /opt/homebrew/bin/micromamba create -n nf-metro-main-<NUMBER> python=3.11 cairo -y
cd /Users/jonathan.manning/projects/nf-metro && git checkout main && git pull origin main
source ~/.local/bin/mm-activate nf-metro-main-<NUMBER>
pip install -e "/Users/jonathan.manning/projects/nf-metro[dev]" && pip install cairosvg
```

## Phase 4: Implement the Fix

Work inside the worktree. After making changes:

**IMPORTANT:** The shell cwd resets after each Bash tool call, so ruff/pytest
MUST be run with an explicit `cd` into the worktree in the same command.
Running `ruff check src/ tests/` without `cd` will silently check the main repo
instead of the worktree, masking lint errors.

```bash
source ~/.local/bin/mm-activate nf-metro-fix-<NUMBER> && cd /tmp/nf-metro-fix-<NUMBER> && ruff format src/ tests/ && ruff check src/ tests/ && pytest
```

Fix any failures before proceeding.

## Phase 5: Before/After Comparison and Visual Review

### 5a: Before/after for the motivating example

If the issue references a specific `.mmd` file or reproduction command, render it from **both** environments to produce a before/after comparison. This uses the main env (pointing at the unmodified main repo) vs the fix env (pointing at the worktree with your changes).

```bash
# "Before" -- render from main env (unmodified code)
source ~/.local/bin/mm-activate nf-metro-main-<NUMBER>
python -m nf_metro render <path-to-example.mmd> -o /tmp/<name>_before.svg
python -c "import cairosvg; cairosvg.svg2png(url='/tmp/<name>_before.svg', write_to='/tmp/<name>_before.png', scale=2)"

# "After" -- render from fix env (worktree with changes)
source ~/.local/bin/mm-activate nf-metro-fix-<NUMBER>
cd /tmp/nf-metro-fix-<NUMBER> && python -m nf_metro render <path-to-example.mmd> -o /tmp/<name>_after.svg
python -c "import cairosvg; cairosvg.svg2png(url='/tmp/<name>_after.svg', write_to='/tmp/<name>_after.png', scale=2)"

# Open both in one Preview session for side-by-side comparison
open /tmp/<name>_before.png /tmp/<name>_after.png
```

**IMPORTANT:** Editable installs cache `.pyc` files. Each env must point at a **different source directory** (main repo vs worktree) -- do NOT use `git stash` to switch code within a single env, as Python's bytecode cache will serve stale code.

**STOP and ask the user to review the before/after.** If they look identical or the issue doesn't reproduce, discuss with the user before proceeding.

### 5b: Full render review

Render ALL .mmd files in the repo from the fix env to check for regressions:

```bash
source ~/.local/bin/mm-activate nf-metro-fix-<NUMBER>

# Render everything (creates a unique /tmp/nf_metro_renders_XXXXX/ dir)
cd /tmp/nf-metro-fix-<NUMBER> && python scripts/render_topologies.py

# Open all PNGs in one Preview session (use the dir printed by the script)
open /tmp/nf_metro_renders_*/*.png
```

**STOP and ask the user to review the renders.** Do NOT proceed until the user confirms they look correct. If they spot problems, return to Phase 4 and iterate.

## Phase 6: Commit and PR

Once the user approves:

1. Stage and commit changes in the worktree (follow repo commit style).
2. Push the branch.
3. Create a PR against `main`:

```bash
cd /tmp/nf-metro-fix-<NUMBER>
gh pr create --repo pinin4fjords/nf-metro --base main --title "<title>" --body "$(cat <<'EOF'
## Summary
<bullets>

Fixes #<NUMBER>

## Test plan
- [ ] pytest passes
- [ ] ruff check clean
- [ ] Visual review of all topology renders

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## Phase 7: Cleanup

After the PR is created, offer to clean up:

```bash
cd /Users/jonathan.manning/projects/nf-metro
git worktree remove /tmp/nf-metro-fix-<NUMBER>
/opt/homebrew/bin/micromamba env remove -n nf-metro-fix-<NUMBER> -y
/opt/homebrew/bin/micromamba env remove -n nf-metro-main-<NUMBER> -y
```

Only clean up if the user agrees.
