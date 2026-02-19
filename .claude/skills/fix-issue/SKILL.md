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

## Phase 3: Create Micromamba Environment

```bash
ulimit -n 1000000 && export CONDA_OVERRIDE_OSX=15.0 && /opt/homebrew/bin/micromamba create -n nf-metro-fix-<NUMBER> python=3.11 cairo -y
source ~/.local/bin/mm-activate nf-metro-fix-<NUMBER>
pip install -e "/tmp/nf-metro-fix-<NUMBER>[dev]" && pip install cairosvg
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

## Phase 5: Render and Visual Review

Render ALL topology fixtures (including nextflow ones) and examples to PNG, then open for user review.

```bash
source ~/.local/bin/mm-activate nf-metro-fix-<NUMBER>

# Clean previous renders
rm -rf /tmp/nf_metro_topology_renders/

# Render topologies + examples (via repo batch script)
python /tmp/nf-metro-fix-<NUMBER>/scripts/render_topologies.py

# Render nextflow fixtures (not in the batch script)
for f in /tmp/nf-metro-fix-<NUMBER>/tests/fixtures/nextflow/*.mmd; do
  name=$(basename "$f" .mmd)
  python -m nf_metro render "$f" -o "/tmp/nf_metro_topology_renders/${name}.svg"
  python -c "import cairosvg; cairosvg.svg2png(url='/tmp/nf_metro_topology_renders/${name}.svg', write_to='/tmp/nf_metro_topology_renders/${name}.png', scale=2)"
done

# Open all PNGs in one Preview session
open /tmp/nf_metro_topology_renders/*.png
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
```

Only clean up if the user agrees.
