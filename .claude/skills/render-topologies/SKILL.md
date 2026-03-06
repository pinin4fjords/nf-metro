---
name: render-topologies
description: Render all .mmd files to PNG, pixel-diff against main, and open only changed renders as BEFORE/AFTER pairs in Preview. Use after layout or rendering changes to check for visual regressions. Works in worktree mode (fix branch vs main) or standalone mode (current working tree vs main). Companion to the fix-issue skill, which delegates full regression checks here.
disable-model-invocation: true
allowed-tools: Bash(rm -rf *), Bash(python *), Bash(open *), Bash(cd *), Bash(git *), Bash(source *), Bash(pip *), Bash(cp *)
---

# Render Topologies

Pixel-diff all `.mmd` renders between the current branch and `origin/main`. Opens only changed renders as numbered BEFORE/AFTER pairs in Preview.

## Step 1: Detect context

Determine the working mode:

- **Worktree mode**: A worktree exists at `/tmp/nf-metro-fix-<N>` with a matching `nf-metro-fix-<N>` env. Use the worktree path and env for branch renders.
- **Standalone mode**: Working from the main repo at `/Users/jonathan.manning/projects/nf-metro`. Use the `nf-metro` env for branch renders. Stash or commit any uncommitted changes first.

## Step 2: Render baseline from main

Update the main repo checkout and render using the shared `nf-metro-main` baseline environment:

```bash
cd /Users/jonathan.manning/projects/nf-metro && git fetch origin main && git checkout main && git pull origin main
source ~/.local/bin/mm-activate nf-metro-main && pip install -e "/Users/jonathan.manning/projects/nf-metro[dev]" -q
cd /Users/jonathan.manning/projects/nf-metro && python scripts/render_topologies.py
# Note the output directory → MAIN_DIR
```

## Step 3: Render from the current branch

Switch back to the branch first (standalone mode) or use the worktree path:

```bash
# Standalone: switch back to the branch
cd /Users/jonathan.manning/projects/nf-metro && git checkout <branch-name>
source ~/.local/bin/mm-activate nf-metro && pip install -e ".[dev]" -q
python scripts/render_topologies.py
# Note the output directory → BRANCH_DIR

# Worktree: use worktree path and env
source ~/.local/bin/mm-activate nf-metro-fix-<N> && pip install -e "/tmp/nf-metro-fix-<N>[dev]" -q
cd /tmp/nf-metro-fix-<N> && python scripts/render_topologies.py
# Note the output directory → BRANCH_DIR
```

## Step 4: Diff and open changed renders

Clean previous comparison files, diff each PNG pair, copy changed renders with sequential numbering (BEFORE/AFTER adjacent when flipping with arrow keys):

```python
from PIL import Image, ImageChops
import os, glob, shutil

# Clean previous comparison files
for f in glob.glob("/tmp/*_BEFORE.png") + glob.glob("/tmp/*_AFTER.png"):
    os.remove(f)

main_dir = "<MAIN_DIR>"
branch_dir = "<BRANCH_DIR>"

pngs = sorted(f for f in os.listdir(branch_dir) if f.endswith('.png'))
changed = []
for name in pngs:
    path_main = os.path.join(main_dir, name)
    if not os.path.exists(path_main):
        changed.append(name)  # new file
        continue
    im_b = Image.open(os.path.join(branch_dir, name))
    im_m = Image.open(path_main)
    if im_b.size != im_m.size or ImageChops.difference(im_b, im_m).getbbox():
        changed.append(name)

print(f"{len(changed)} changed, {len(pngs) - len(changed)} unchanged")
for i, name in enumerate(changed):
    stem = name.replace('.png', '')
    idx = i * 2 + 1
    main_path = os.path.join(main_dir, name)
    if os.path.exists(main_path):
        shutil.copy(main_path, f"/tmp/{idx:02d}_{stem}_BEFORE.png")
    shutil.copy(os.path.join(branch_dir, name), f"/tmp/{idx+1:02d}_{stem}_AFTER.png")
    print(f"  {name}")
```

**IMPORTANT**: Write the diff script to a file (`/tmp/diff_renders.py`) and run it with `python3`, rather than inlining Python in the shell. Inline Python with `!=` gets mangled by shell escaping.

```bash
# Open all pairs sorted so BEFORE/AFTER interleave correctly (skip if zero changed)
ls /tmp/[0-9][0-9]_*_BEFORE.png /tmp/[0-9][0-9]_*_AFTER.png | sort | xargs open -a Preview
```

The numbered prefixes (`01_`, `02_`, ...) ensure Preview orders them correctly when using arrow keys: `01_foo_BEFORE.png`, `02_foo_AFTER.png`, `03_bar_BEFORE.png`, `04_bar_AFTER.png`, etc.

## Step 5: Report

- Count of changed vs unchanged renders
- List changed file names
- If zero changed, say so and skip Preview

## Notes

- Render script: `scripts/render_topologies.py` (discovers all `.mmd` files under project root).
- Nextflow fixtures (`tests/fixtures/nextflow/*.mmd`) are auto-detected and converted before rendering.
- Baseline always uses `nf-metro-main` env + main repo updated to `origin/main`.
- To render a single file: `python -m nf_metro render <file.mmd> -o /tmp/output.svg`
