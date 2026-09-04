---
title: "Embedding guide"
description: "How to embed nf-metro SVG maps in host applications: sizing, theming, and driving maps at runtime."
---

:::note[Stable as of nf-metro 1.0]
The embedding surface (the `--nfm-*` properties, the `data-*` contract, and
the manifest schema) is a public, versioned surface. The manifest schema
version (`MANIFEST_SCHEMA_VERSION`) and driver contract version
(`DRIVER_CONTRACT_VERSION`) are both `1.0` and change under `major.minor`
semantics; see [Versioning and stability](#versioning-and-stability) below.
:::

This guide is for someone putting a rendered nf-metro map into **their own**
page or application, such as a docs site, an internal dashboard or a pipeline
run viewer. You do not need to read `src/` to follow it. It covers how to
produce an embed-friendly file, how to size and theme it from the host page,
and how to drive it from live state so that nodes light up as a job runs.

:::tip[Just want a picture?]
Skip straight to [Static embed](#a-static-embed). If you want a panel that reacts to a running pipeline, read on to [Interactive and progress embeds](#interactive-and-progress-embeds).
:::

## Choosing an output

nf-metro renders two shapes. Which one you want depends on what the host needs.

| You want                                                    | Use                                           | Why                                                              |
| ----------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------- |
| A static picture (thumbnail, README, slide)                 | `render` → **SVG**                            | One self-contained file; scales crisply; no scripts.             |
| A live, interactive panel (pan/zoom, line filtering, hover) | `render --format html`                        | A self-contained page with the driver and styling already wired. |
| A progress overlay driven by your own app                   | **SVG** + the [manifest](/nf-metro/manifest/) | You read the embedded manifest and draw your own status layer.   |

The SVG carries a machine-readable [manifest](/nf-metro/manifest/) and a stable
[`data-*` contract](/nf-metro/embed/) either way, so a static embed can become an
interactive one later without re-rendering.

## Render options for embedding

These flags shape the SVG for life inside someone else's page. They apply to
`--format svg`. The interactive HTML page already handles sizing, scoping and
chrome itself (see [Interactive and progress embeds](#interactive-and-progress-embeds)).

### Responsive sizing - `--responsive`

By default the `<svg>` carries fixed `width`/`height` attributes. With
`--responsive` it emits **only** a `viewBox` (plus `preserveAspectRatio`), so
the host sizes it with CSS:

```bash
nf-metro render pipeline.mmd -o pipeline.svg --responsive
```

```css
.metro-map svg {
  width: 100%;
  height: auto;
}
```

Use this for any fluid layout. The `viewBox` stays `0 0 <width> <height>`, so
overlays built from the manifest still line up (see
[Progress overlays](#progress-overlays)).

### Font portability - `--embed-font` / `--text-to-paths`

By default the SVG references a system font family, which renders differently,
or falls back entirely, on a host that lacks that font. Two flags make the file
self-contained:

| Flag              | What it does                                              | Keeps selectable text?        | Trade-off                                            |
| ----------------- | --------------------------------------------------------- | ----------------------------- | ---------------------------------------------------- |
| `--embed-font`    | Inlines a subset of Inter as a base64 `@font-face` block. | Yes (and `data-*` on labels). | Larger file.                                         |
| `--text-to-paths` | Converts every glyph to a vector `<path>`.                | No.                           | Smallest dependency surface; needs `nf-metro[font]`. |

```bash
nf-metro render pipeline.mmd -o pipeline.svg --embed-font      # portable, still selectable
nf-metro render pipeline.mmd -o pipeline.svg --text-to-paths   # zero font dependency
```

Use `--embed-font` when you want labels to stay selectable and searchable. Use
`--text-to-paths` when the consumer is a strict renderer, or when you need
pixel fidelity with no font handling at all.

### Bare fragment - `--bare`

`--bare` drops the title and the outer right padding so the canvas hugs the
content. Use it when the host supplies its own frame and heading:

```bash
nf-metro render pipeline.mmd -o pipeline.svg --bare
```

The `viewBox` origin stays at `0 0` and coordinates stay absolute, so the
[manifest](/nf-metro/manifest/) and any overlay still align. Bare mode **keeps**
the attribution watermark (see [Attribution](#attribution)).

### Theming from the host - `--nfm-map-*` properties

Chrome colors, meaning the background, title, labels, section boxes and legend,
are emitted as CSS custom properties with the theme color as the fallback, as in
`fill: var(--nfm-map-bg, light-dark(#f5f5f5, #2b2b2b))`. A host recolors the map
**without re-rendering** by setting these on a wrapping element:

```css
.metro-map {
  --nfm-map-bg: #ffffff;
  --nfm-map-title-color: #222;
  --nfm-map-label-color: #333;
  --nfm-map-section-fill: #f4f4f4;
  --nfm-map-section-stroke: #ddd;
  --nfm-map-section-label-color: #555;
  --nfm-map-legend-bg: #fafafa;
  --nfm-map-legend-text-color: #333;
  --nfm-map-marker-stroke: #333;
  --nfm-map-muted-color: #999;
}
```

| Property                                              | Recolors                                                                            |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `--nfm-map-bg`                                        | Background rectangle, and the knockout halo behind station labels                   |
| `--nfm-map-title-color`                               | Title text                                                                          |
| `--nfm-map-label-color`                               | Station labels and terminus icon captions                                           |
| `--nfm-map-section-fill` / `--nfm-map-section-stroke` | Section box fill / border                                                           |
| `--nfm-map-section-label-color`                       | Section names, group labels, group underlines                                       |
| `--nfm-map-legend-bg` / `--nfm-map-legend-text-color` | Legend background / text                                                            |
| `--nfm-map-marker-stroke`                             | Marker station outlines and the legend marker key                                   |
| `--nfm-map-muted-color`                               | Labels, captions and marker outlines greyed by [`--inactive-lines`](/nf-metro/cli/) |

The muted state has its own property so the two states can be themed apart. Set
`--nfm-map-label-color` and full-strength labels follow it, while greyed ones
stay grey.

Line and route colors are **not** recolorable. They carry meaning, so they stay
baked in as presentation attributes.

The fallback behind each property is a `light-dark()` pair rather than a single
color, so the map already adapts to the viewer's `color-scheme` before any host
override. See [Theming](/nf-metro/theming/) for how that mechanism works and how
to reuse it in your own SVGs.

### Multiple maps on one page - `--svg-class-prefix`

Two inline SVGs on the same page share class names such as `nf-metro-station`,
so host CSS or the dark-mode block from one can bleed into the other. Give each
a distinct prefix:

```bash
nf-metro render a.mmd -o a.svg --svg-class-prefix mapA
nf-metro render b.mmd -o b.svg --svg-class-prefix mapB
```

`mapA-nf-metro-station`, `mapB-nf-metro-station`, and so on stay independent.
`data-*` attributes and the manifest element id are never prefixed, so the
[contract](/nf-metro/embed/) is unchanged.

### Following your page's own theme toggle - `--no-self-color-scheme`

By default the map's root `<svg>` declares its own `color-scheme: light dark`,
so it follows the **viewer's OS or browser** preference regardless of what your
page does. If your page has its own light/dark toggle, pass
`--no-self-color-scheme` so the map inherits `color-scheme` from your page
instead:

```bash
nf-metro render pipeline.mmd -o pipeline.svg --no-self-color-scheme
```

Your page then has to set `color-scheme` somewhere the map can inherit it. Use
a class or `data-theme` attribute toggled by your theme switch, each setting
`color-scheme: light` or `color-scheme: dark` on an ancestor. Set a single
value, not `light dark`. See [Theming](/nf-metro/theming/) for why this flag
exists and how the docs site itself uses it.

### Dark-mode opt-out - `--no-dark-mode-css`

When a theme has a transparent background, the SVG injects a
`@media (prefers-color-scheme: dark)` block so labels stay readable on a dark
host page. If your host manages its own theme and that media query fights it,
suppress it:

```bash
nf-metro render pipeline.mmd -o pipeline.svg --no-dark-mode-css
```

This block is a separate, coarser fallback from the `--nfm-*` custom properties
above. It exists because a transparent background has no color of its own to
carry a `light-dark()` pair. See [Theming](/nf-metro/theming/) for why the two
mechanisms differ.

### Raster export (PNG) - `--mode` and `--no-chrome-css`

Two independent settings control correct PNG output:

**Palette (`--mode`).** Always pass `--mode light` or `--mode dark` explicitly.
Without it you get the default palette, which may not match your intent. The
flag also pins `color-scheme` on the SVG root, so CSS-aware rasterizers resolve
`light-dark()` to the right values regardless of the host OS color scheme.

**CSS variables (`--no-chrome-css`).** The `--nfm-*` properties above use CSS
`var()`, which many rasterizers, **cairosvg** among them, cannot parse and abort
on. Add `--no-chrome-css` to bake the concrete theme colors instead. The map
looks identical, and the only thing you lose is live host recoloring:

```bash
nf-metro render pipeline.mmd -o pipeline.svg --no-chrome-css --mode light
python -c "import cairosvg; cairosvg.svg2png(url='pipeline.svg', write_to='pipeline.png', scale=2)"
```

A rasterizer that understands CSS custom properties, such as `resvg`,
`rsvg-convert` or headless Chromium, resolves `var()` and `light-dark()`
natively, so skip `--no-chrome-css`. Still pass `--mode` to pin the palette:

```bash
nf-metro render pipeline.mmd -o pipeline.svg --mode light
resvg pipeline.svg pipeline.png
```

## Sizing and placement

Everything in an nf-metro SVG lives in one coordinate space: `viewBox="0 0 w h"`
with no outer transform. That keeps the host's job simple:

- **Size** the SVG with CSS (`width: 100%; height: auto`). Use `--responsive` so
  there are no fixed dimensions to override.
- **Stack** a base render and an overlay by giving both the **same `viewBox`**
  and absolutely positioning them in the same box. Coordinates are absolute and
  share the origin, so a marker the overlay draws at a node's manifest `(x, y)`
  lands exactly on that node.

```html
<div class="metro-map" style="position: relative;">
  <!-- base render, sized by CSS -->
  <object data="pipeline.svg" type="image/svg+xml" style="width:100%;"></object>
  <!-- overlay, same viewBox, on top -->
  <svg
    viewBox="0 0 1509 759"
    style="position:absolute; inset:0; width:100%; pointer-events:none;"
  >
    <!-- status markers at manifest coordinates -->
  </svg>
</div>
```

The manifest's `width`/`height` fields give the exact `viewBox` to reuse.

## The embed contract

Each part of the stable surface has one authoritative page. This guide links to
them rather than restating them:

- **[Embed contract](/nf-metro/embed/)** covers the `data-node-*`,
  `data-station-*` and `data-section-*` attribute vocabulary, and the driver API
  (`attachMetroMap`, `highlightLine`, `selectNode`, `getManifest` and the rest).
- **[Data manifest](/nf-metro/manifest/)** covers the manifest JSON schema, its
  version, the matching semantics (`patterns` → runtime names) and the
  `overlay_svg` helper.

The join key across all of it is the node `id`, which equals `data-node-id` on
the drawn element and `node.id` in the manifest JSON.

## A static embed

This is the minimum needed to put a map on a page. Render a portable, fluid SVG
and inline it:

```bash
nf-metro render pipeline.mmd -o pipeline.svg --responsive --embed-font
```

```html
<div class="metro-map" style="max-width: 1000px;">
  <!-- paste the contents of pipeline.svg here, or: -->
  <object data="pipeline.svg" type="image/svg+xml" style="width:100%;"></object>
</div>
```

GitHub READMEs strip `<script>`, so a static SVG is the right choice there. Most
static-site generators and wikis accept the inline SVG unchanged.

## Interactive and progress embeds

### The self-contained interactive page

`render --format html` produces a complete page with the SVG, driver and styling
inlined and no network access needed. Its **Embed…** modal offers an inline
`<div>` snippet that keeps interactivity without an iframe, an iframe one-liner,
and a static-SVG fallback. The page is already responsive and scopes each map
independently, so the SVG-only sizing and namespacing flags above do not apply
to it, and the CLI warns if you pass them with `--format html`. Font portability
**does** reach the inlined SVG, so an embeddable page can carry its own fonts:

```bash
nf-metro render pipeline.mmd --format html -o pipeline.html --embed-font
```

To wire the driver onto a page yourself rather than copy the modal snippet, see
the [driver API](/nf-metro/embed/#driver-api) and `nf-metro embed-script`.

### Progress overlays

To light up nodes as a pipeline runs, keep the base map static and redraw a thin
**overlay** layer on each state change. The base SVG is the durable map, and the
overlay is a cheap, disposable status layer. Three coordinate-space rules make
that work:

- The base SVG and overlay share `viewBox="0 0 w h"` (origin `0 0`).
- The manifest's `width`/`height` match the base render's dimensions.
- Each node's `x`/`y`/`r` are absolute units in that space, so an overlay
  marker at `(x, y)` lands on the node.

The recipe is always the same three steps: `read_manifest` on the committed SVG,
`match_node_ids` to map each runtime event to a node, and `overlay_svg()` to
redraw a status layer over the base. The manifest tutorial,
**[Light up a diagram as a job runs](/nf-metro/manifest/#tutorial-light-up-a-diagram-as-a-job-runs)**,
works through it in about 50 lines of Python. The matching semantics and the
node state model are documented alongside it on the
[Data manifest](/nf-metro/manifest/) page.

For a ready-made server that does all of this for a live Nextflow run with no
code to write, see [Live progress](/nf-metro/live/).

## Calling the Python API directly

The CLI wraps parse and layout errors into a clean `click.ClickException`
message. An embedder calling `nf_metro.render_string()`, or `prepare_graph()`
plus `render_graph()`, directly from Python gets the pipeline's typed errors
raw, and can decide for itself how to present a rejected input to its own users.

Every specific parse and layout error type below subclasses
`nf_metro.NfMetroError`, so one `except` clause covers all of them without
naming each type:

`render_string` also takes `source_dir`. Pass the directory the map was read
from, or its `%%metro logo:` paths will only resolve when the working directory
happens to match.

```python
from nf_metro import render_string, NfMetroError

try:
    svg = render_string(mmd_path.read_text(), source_dir=str(mmd_path.parent))
except NfMetroError as e:
    # e.g. show the author their `.mmd` was rejected, with str(e) as the reason
    ...
except ValueError as e:
    # a grammar/directive syntax error - not an NfMetroError, still worth
    # catching separately if you want the same "bad input" handling for it
    ...
```

| Raised when...                                                                   | Type                                                                     | When                 | Also a...    |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------- | ------------ |
| The `.mmd` grammar or a directive is malformed                                   | plain `ValueError` (**not** an `NfMetroError`, see below)                | parsing              | -            |
| The source parses to no stations at all                                          | `nf_metro.EmptyGraphError`                                               | layout               | `ValueError` |
| An edge or port survives parsing with a dangling reference                       | `nf_metro.parser.UnresolvedEndpointError` / `UnresolvedPortSectionError` | parsing/layout       | `ValueError` |
| The station graph has a cycle                                                    | `nf_metro.parser.CyclicGraphError`                                       | layout               | `ValueError` |
| An inter-section edge would have to flow backward                                | `nf_metro.layout.BackwardFlowError`                                      | layout               | `ValueError` |
| One section is entered from more than one direction                              | `nf_metro.layout.MixedEntryDirectionError`                               | layout               | `ValueError` |
| A layout-engine self-check fails mid-layout                                      | `nf_metro.layout.PhaseInvariantError`                                    | layout               | -            |
| A user-set `fold_threshold` compresses the grid past what the router can resolve | `nf_metro.layout.FoldThresholdError`                                     | **render step only** | `ValueError` |

The first row sits outside the hierarchy deliberately. The parser raises a plain
`ValueError` ad hoc for most grammar and directive problems rather than through
a dedicated type, so `except ValueError` is the right catch-all for "the `.mmd`
text itself doesn't parse". `except NfMetroError` covers every problem detected
_after_ parsing succeeds: a graph that parsed fine but cannot be laid out, or,
for `FoldThresholdError`, cannot be drawn honestly.

Catch a specific row instead of the base class when the distinction matters, for
example to offer "fix your fold threshold" only for `FoldThresholdError`, or to
fall back to `%%metro permissive: true` semantics only for
`PhaseInvariantError`.

**Not** part of this hierarchy: `render_string()`'s render step also runs a
handful of self-checks (`CurveInvariantError`, `BridgeInvariantError`,
`SectionHeaderClashError`, `SectionHeaderOverflowError`,
`SectionHeaderBandError`, `OffsetAnchorError`)
that indicate a defect in nf-metro's own drawing rather than a problem with
your input, so they are left out of `NfMetroError` on purpose. See the
`render_string` docstring for the full list and the rationale. Report one if you
hit it. Only a broad `except Exception` shields a host page from them, and that
also masks genuine nf-metro bugs.

## Versioning and stability

The manifest schema and the driver contract are versioned independently, and
both are `1.0` today. The stable surface keyed to those versions covers the
`data-*` attribute names, the manifest fields, the `0 0 w h` coordinate rule and
the driver method names. That surface and the `major.minor` rules for changing
it are specified under [Versioning](/nf-metro/embed/#versioning) on the Embed
contract page. It is stable as of nf-metro 1.0, so within a major version it
only grows in backward-compatible ways and **consumers must ignore unknown
fields**. Pin to a specific nf-metro release only if you depend on the exact
bytes of the output.

## Attribution

:::note[Please keep the watermark]
Rendered maps carry a small `created with nf-metro` watermark in the corner, including in `--bare` mode. It is a small credit that helps people find the project, and keeping it is the easiest way to support nf-metro. There is no convenience flag to remove it, because removal is reserved for specific functionality rather than offered as a toggle. This is a friendly ask, not a license restriction.
:::
