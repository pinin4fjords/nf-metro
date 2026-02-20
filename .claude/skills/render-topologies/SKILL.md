---
name: render-topologies
description: Re-render all .mmd files in the repo to PNG and open them in Preview for visual review. Use when the user wants to check renders after layout or rendering changes.
disable-model-invocation: true
allowed-tools: Bash(rm -rf *), Bash(python *), Bash(open *)
---

# Render Topologies

Re-render all `.mmd` files in the repository and open the results for visual review.

## Workflow

1. Clean the output directory:

```bash
rm -rf /tmp/nf_metro_topology_renders/
```

2. Run the batch render script (ensure the project is installed in the active Python environment with cairosvg available):

```bash
python scripts/render_topologies.py
```

3. Open all rendered PNGs in Preview:

```bash
open /tmp/nf_metro_topology_renders/*.png
```

4. Report the results: list which files rendered successfully and flag any failures.

## Notes

- The render script is at `scripts/render_topologies.py` in the repo root.
- It automatically discovers all `.mmd` files under the project root (examples, test fixtures, topologies).
- Nextflow fixtures (`tests/fixtures/nextflow/*.mmd`) are auto-detected and converted before rendering.
- Output filenames include the path (e.g. `examples_guide_01_minimal.png`) to avoid collisions.
- Outputs go to `/tmp/nf_metro_topology_renders/` as PNGs.
- The Python environment must have nf-metro installed (editable mode is fine) along with cairosvg for SVG-to-PNG conversion.
- If the user asks to render a specific file, use the CLI directly instead:

```bash
python -m nf_metro render <file.mmd> -o /tmp/output.svg
```
