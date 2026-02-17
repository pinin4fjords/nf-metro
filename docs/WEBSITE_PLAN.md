# nf-metro Showcase Website Plan

A single-page gallery site deployed via GitHub Pages. Each example shows the CLI command, the `.mmd` source, and the rendered SVG - all scrollable in a clean layout.

## Tech Choice: mkdocs-material

- Zero frontend code to write; just markdown + config
- Built-in syntax highlighting for the Mermaid source blocks
- Navigation sidebar auto-generated from headings (easy to scroll between examples)
- Dark mode by default (matches the nf-core theme renders)
- GitHub Actions deploys automatically on push to `main`
- Versioned docs via `mike`: "dev" from main, release versions on publish

## File Structure

```
docs/
  WEBSITE_PLAN.md       <-- this file (remove before shipping)
  index.md              <-- landing page: what nf-metro is, install, quick example
  gallery/
    index.md            <-- auto-generated gallery with all examples
  assets/
    renders/            <-- pre-rendered SVGs (generated at build time)
mkdocs.yml             <-- mkdocs-material config
scripts/
  build_gallery.py      <-- generates gallery/index.md + assets/renders/
.github/
  workflows/
    docs.yml            <-- GitHub Actions: build + deploy to gh-pages
```

## Step-by-Step

### 1. Add mkdocs-material as a docs dependency

In `pyproject.toml`, add a `docs` optional-dependencies group alongside the existing `dev` group:

```toml
[project.optional-dependencies]
docs = [
    "mkdocs-material>=9.0",
    "cairosvg>=2.5",
]
```

### 2. Create `mkdocs.yml`

```yaml
site_name: nf-metro
site_description: Metro-map-style SVG diagrams from Mermaid definitions
site_url: https://pinin4fjords.github.io/nf-metro/
repo_url: https://github.com/pinin4fjords/nf-metro
repo_name: pinin4fjords/nf-metro

theme:
  name: material
  palette:
    scheme: slate
    primary: teal
    accent: light green
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.top
    - navigation.sections
    - content.code.copy
    - toc.follow

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.details
  - admonitions
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Gallery: gallery/index.md
```

### 3. Create `docs/index.md` (landing page)

Content drawn from the existing README:
- What nf-metro does (one paragraph)
- Hero SVG render (rnaseq animated light theme, already committed at `examples/rnaseq_light_animated.svg`)
- Install command (`pip install nf-metro`)
- Quick start (render / validate / info commands)
- Link to the gallery
- Link to the input format reference (README anchors or a future dedicated page)

### 4. Create `scripts/build_gallery.py`

This script auto-generates `docs/gallery/index.md` and SVG renders in `docs/assets/renders/`. It:

1. Collects `.mmd` files from `examples/` (top-level) and `examples/topologies/`
2. For each file:
   a. Reads the `.mmd` source
   b. Renders it to SVG via the nf-metro Python API
   c. Saves the SVG to `docs/assets/renders/<name>.svg`
   d. Appends a section to the gallery markdown with heading, description, CLI command, collapsible source, and rendered SVG

The gallery is **one long scrollable page** with a heading per example. mkdocs-material's right-hand TOC provides jump-links.

**Example ordering:**

Main examples first, then topologies grouped by category:

1. `simple_pipeline` (simplest, good intro)
2. `rnaseq_auto` (real-world, auto-layout)
3. `rnaseq_sections` (real-world, manual grid)

Simple topologies:
4. `single_section`
5. `deep_linear`
6. `parallel_independent`

Fan-out and fan-in:
7. `wide_fan_out`
8. `wide_fan_in`
9. `section_diamond`

Branching and multipath:
10. `asymmetric_tree`
11. `complex_multipath`

Multi-line bundles:
12. `multi_line_bundle`
13. `mixed_port_sides`

Realistic pipelines:
14. `rnaseq_lite`
15. `variant_calling`

Fold topologies:
16. `fold_fan_across`
17. `fold_double`
18. `fold_stacked_branch`

Descriptions are derived from:
- `%%metro title:` directive in the `.mmd` file (for main examples)
- The descriptions in `examples/topologies/README.md` (for topologies)

### 5. Create `.github/workflows/docs.yml`

GitHub Actions workflow using `mike` for versioned deploys:

- **On push to `main`**: deploys as `dev` version
- **On GitHub release**: deploys as the release version (e.g., `0.2.0`) with a `latest` alias, and sets `latest` as the default redirect
- **On `workflow_dispatch`**: deploys as `dev` (manual trigger)

`mike` pushes versioned builds to the `gh-pages` branch. Each version lives in its own subdirectory. The mkdocs-material version selector dropdown lets users switch between versions.

### 6. Enable GitHub Pages

In the repo settings: Settings > Pages > Source: "Deploy from a branch", branch: `gh-pages`.

### 7. Local preview workflow

```bash
pip install -e ".[docs]"
python scripts/build_gallery.py
mkdocs serve
# Open http://localhost:8000
```

## Nice-to-haves (later)

- **Theme toggle**: Light/dark palette toggle rendering each example in both themes with tabs to switch.
- **Animated versions**: `--animate` renders as a toggle per example.
- **Interactive editor**: Textarea for pasting `.mmd` with live render (would need WASM or server-side API).
- **Input format reference**: Dedicated page with the directive reference (currently in README).
