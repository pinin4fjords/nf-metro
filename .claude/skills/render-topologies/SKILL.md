---
name: render-topologies
description: Re-render all .mmd files in the repo to PNG, diff against main, and open only the changed renders (BEFORE/AFTER pairs) in Preview. Use when the user wants to check renders after layout or rendering changes.
disable-model-invocation: true
allowed-tools: Bash(rm -rf *), Bash(python *), Bash(open *), Bash(cd *), Bash(git *), Bash(source *), Bash(pip *), Bash(cp *)
---

# Render Topologies

Re-render all `.mmd` files from the current branch and from `origin/main`, then open only the renders that differ as BEFORE/AFTER pairs in a single Preview session.

## Workflow

### Step 1: Render baseline from main

Use the dedicated `nf-metro-main` micromamba environment and the main repo checkout at `/Users/jonathan.manning/projects/nf-metro`. First update it to match `origin/main`:

```bash
cd /Users/jonathan.manning/projects/nf-metro && git fetch origin main && git checkout main && git pull origin main
source ~/.local/bin/mm-activate nf-metro-main && pip install -e "/Users/jonathan.manning/projects/nf-metro[dev]" -q
cd /Users/jonathan.manning/projects/nf-metro && python scripts/render_topologies.py
# Note the output directory (MAIN_DIR)
```

### Step 2: Render from the fix branch

Use the worktree's own micromamba environment:

```bash
source ~/.local/bin/mm-activate <env> && pip install -e "<worktree>[dev]" -q
cd <worktree> && python scripts/render_topologies.py
# Note the output directory (FIX_DIR)
```

### Step 3: Diff renders (pixel-level)

Use Pillow to compare each PNG pair. Only report files where the pixels or dimensions actually differ:

```python
from PIL import Image, ImageChops
import os

main_dir = "<MAIN_DIR>"
fix_dir = "<FIX_DIR>"

changed = []
for name in sorted(os.listdir(fix_dir)):
    if not name.endswith('.png'):
        continue
    im_fix = Image.open(os.path.join(fix_dir, name))
    im_main = Image.open(os.path.join(main_dir, name))
    if im_fix.size != im_main.size:
        changed.append(name)
    else:
        diff = ImageChops.difference(im_fix, im_main)
        if diff.getbbox():
            changed.append(name)

print(f"{len(changed)} changed, {len(os.listdir(fix_dir)) - len(changed)} unchanged")
for c in changed:
    print(f"  {c}")
```

### Step 4: Open changed renders as BEFORE/AFTER pairs

Copy each changed render into `/tmp/` with numbered BEFORE/AFTER names so they sort together in Preview, then open all in one session:

```bash
# For each changed file, e.g. with index N:
cp "$MAIN_DIR/<name>.png" "/tmp/<N>_<short_name>_BEFORE.png"
cp "$FIX_DIR/<name>.png"  "/tmp/<N+1>_<short_name>_AFTER.png"

# Open all pairs in one Preview session
open /tmp/*_BEFORE.png /tmp/*_AFTER.png
```

Use sequential numbering (1, 2, 3, 4, ...) so BEFORE and AFTER for each render are adjacent when flipping with arrow keys.

### Step 5: Report results

- List how many renders changed vs unchanged
- List the changed file names
- If zero changed, say so and skip opening Preview

## Notes

- The render script is at `scripts/render_topologies.py` in the repo root.
- It automatically discovers all `.mmd` files under the project root (examples, test fixtures, topologies).
- Nextflow fixtures (`tests/fixtures/nextflow/*.mmd`) are auto-detected and converted before rendering.
- Output filenames include the path (e.g. `examples_guide_01_minimal.png`) to avoid collisions.
- **Baseline renders** always use the `nf-metro-main` env with the main repo at `/Users/jonathan.manning/projects/nf-metro`, updated to `origin/main`.
- **Branch renders** use the worktree's own `nf-metro-fix-<N>` env.
- If the user asks to render a specific file, use the CLI directly instead:

```bash
python -m nf_metro render <file.mmd> -o /tmp/output.svg
```
