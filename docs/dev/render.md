---
title: "Render"
description: How a laid-out MetroGraph becomes an SVG, including animation, bridges, legends, and themes.
sidebar:
  order: 9
---

Rendering has two steps. First, `build_render_plan` copies a laid-out
`MetroGraph` and finishes the render-specific geometry. It stores the result in
an immutable `RenderPlan`. Second, an emitter turns that plan into SVG or HTML.

This split prevents rendering from changing the caller's graph. It also gives
validators and metrics the exact geometry used to draw the output.
The plan stores the resolved theme and all settings that affect geometry.
Emitters accept only output options such as animation and responsive sizing.

The main entry point is `render_svg` in
[`src/nf_metro/render/svg.py`](https://github.com/seqeralabs/nf-metro/blob/main/src/nf_metro/render/svg.py),
which performs both steps and returns an SVG string. Callers that need to reuse
a plan can call `build_render_plan`, `emit_render_plan`, or
`emit_render_plan_html` directly.

The internal
[candidate executor](/nf-metro/dev/candidate_execution/) observes the route
plan used by the final `RenderPlan`. This matters when render-time geometry
settling causes a reroute: acceptance evidence must describe the SVG that was
actually emitted, not an earlier routing pass.

`build_render_plan` makes one deep copy of the graph. Render-specific routing,
label placement, and section adjustments run on this private copy. This is
safer than changing the caller's graph and trying to restore it after an error.
Across three representative maps, the copy took 1.8 to 2.2 ms. That was 1.6%
to 2.8% of the combined plan-build and SVG-emission time.

## Deterministic text metrics

`src/nf_metro/text_metrics.py` owns text advances, ink bounds, line heights,
and reserved widths. Each measurement carries a semantic role, such as station
label, section header, legend entry, or icon caption. This keeps the safety
margin for each use explicit while sharing one deterministic measurement path.

The default SVG mode retains the existing Helvetica-family output and its
conservative per-role reservations. Its proportional advance table is bundled
in the package, so layout never searches the host for an installed font.
`--embed-font` and `--text-to-paths` instead select exact Inter metrics from
generated tables shipped in `src/nf_metro/_inter_metrics.py`. Those tables come
from the same bundled Inter Regular and Bold WOFF2 files used by the output;
weights 600, 700, and `bold` all select Inter Bold. Unsupported characters use
the visible `?` replacement's advance, bounds, and outline.

The runtime path has no FontTools dependency. Maintainers can regenerate the
tables after intentionally replacing the bundled fonts with:

```bash
python scripts/build_text_metrics.py
```

The generator requires FontTools with WOFF2 support. Commit the generated table
with the font files so metrics and portable output cannot drift apart.

## SVG generation (`svg.py`)

`render_svg(graph, theme, ...)` is the top-level call. It:

1. Scales theme fonts by `graph.font_scale` (set by the `%%metro font_scale:`
   directive or the `--font-scale` CLI flag). It also scales stroke widths and
   station pills by `graph.stroke_scale`. Layout reserves space with the same
   scale values.
2. Builds a `RenderPlan` from a private graph copy.
3. Calls `emit_render_plan`, which draws the plan with `drawsvg`.
4. If `graph.animate` is set (or `--animate` was passed), calls
   `render_animation` from `animate.py` to add travelling balls.
5. If requested, embeds the plan's node, group, region, marker, and canvas
   geometry as a manifest.

`apply_route_offsets(routes, station_offsets)` lives in
`layout/routing/common.py`. It separates a route bundle into parallel tracks.
It uses the per-station offsets from `compute_station_offsets` in
`routing/offsets.py`. Animation uses the same offset paths.

### What gets drawn

`emit_render_plan` draws in layers:

1. **Section boxes** - rounded rectangles with optional section labels and
   tick marks for group labels.
2. **Edges** - polylines from `RoutedPath.points`, with quadratic Bézier
   curves at corners (radius computed by `routing/corners.py`). Where
   `compute_bridges` identifies a non-merging crossing, `_render_bridged_edge`
   draws the under-route with a gap (see [Bridges](#bridges-bridgespy) below).
3. **Station markers** - pill-shaped rectangles (or circles/squares for
   alternative marker styles). Rail-mode interchange stations span multiple
   rails and are drawn by `_render_rail_pill`.
4. **Icons** - file, files, and folder icons for off-track input nodes (drawn
   by `icons.py`).
5. **Labels** - placed by `layout/labels.py` and rendered with optional
   line-wrapping; positioned above or beside their station.
6. **Legend** - drawn by `legend.py`, auto-positioned to avoid overlapping
   section boxes and routes. Position can be overridden via
   `%%metro legend_position:` or set to `"none"` (suppressed) for the HTML
   output mode.

### Canvas sizing

The canvas is a first-quadrant frame: its `viewBox` always starts at `0 0`, so
overlays can share it without an outer transform (see
[`manifest/__init__.py`](https://github.com/seqeralabs/nf-metro/blob/main/src/nf_metro/manifest/__init__.py)).
Width and height come from `_compute_canvas_bounds` plus a margin -
`CANVAS_PADDING` on the right, the watermark band at the bottom - so the far
edges grow to hold whatever the render draws.

The near edges cannot grow, so the map moves instead.
`_settle_clear_of_the_canvas_margins` measures the ink that lands outside the
section-box envelope on the left or top - an inter-row return band wrapping
around the first box of a row, a bundle rising over the top of one - and moves
the whole laid-out graph away from the edge by the shortfall (`translate_graph`
in `layout/phases/canvas.py`, which owns the full set of absolute coordinates a
graph carries). Routing is then re-derived on the moved copy, because where a
run lands is only known once it is routed. A map that draws nothing outside its
box envelope never moves.

`_content_origin` reports the left and top edges that move settles on - the box
envelope, carried outwards by any run drawn past it. A decoration the author
left unpinned is placed against those edges rather than against a box, so the
legend sits flush with the content whether a box or a run defines the boundary.
An authored pin (`legend: x,y`, `| canvas`, `| dx,dy`) is placed as written.

## Bridges (`bridges.py`)

Two distinct metro lines may cross at a point that is not a shared station,
port, junction, or merge. Drawn naively that reads as an interchange.
`compute_bridges` resolves the ambiguity by inserting a short gap in the
under-route where it passes beneath the over-route.

`compute_bridges(graph, routes)` takes the assembled polylines (with offsets
already applied) and:

1. Identifies all genuine pairwise crossings: ignores crossings between the
   same line, crossings at shared endpoints, and crossings within
   `BRIDGE_NODE_TOLERANCE` of any node.
2. For same-line crossings, distinguishes a fan-in/out (two legs that share a
   common ancestor and rejoin at a common descendant) from an independent
   self-crossing that genuinely needs a bridge.
3. Groups nearby crossings into clusters and assigns "over" and "under" by
   2-colouring the cluster graph.
4. Returns a list of `BridgeBreak` objects, one per under-route segment,
   recording the `t_start`/`t_end` parametric range on that segment to omit.

The drawing half lives in `svg.py`: `_render_bridged_edge` splits the
polyline at the gap span and renders each piece separately.

## Interactive HTML (`html.py`)

`render_html(graph, theme, ...)` builds a plan without an SVG legend, because
the HTML side panel replaces it. It then passes the plan to
`emit_render_plan_html` and returns a complete HTML page.

The HTML output has two delivery modes:

- **Standalone page** (`_STANDALONE_TEMPLATE`): a full `<!DOCTYPE html>`
  document with a two-column layout (canvas left, legend panel right).
  Supports pan/zoom, per-line focus (click a line chip to dim everything
  else and zoom to visible), station tooltips, and an embed modal that
  generates copy-paste snippets.
- **Inline embed snippet** (`_INLINE_TEMPLATE`, via `_build_inline_snippet`):
  a `<div>` with scoped CSS and an IIFE - paste into any HTML host (MkDocs,
  Confluence, blog templates) without hosting a separate file.

Both modes use the same driver from `get_driver_js`, so the
interaction behaviour is identical. The embed snippet scopes CSS under a
per-render hash (`.nfmm-<sha1[:8]>`) so multiple maps can coexist on one
page.

## Manifest (`manifest.py`)

Plan construction maps the final render geometry onto the
[embedded-manifest standard](/nf-metro/manifest/): stations become nodes,
sections become groups, and visual regions (section bboxes) become regions.
The manifest is serialised to JSON and injected as a `<metadata>` element
inside the SVG, keyed by `MANIFEST_ELEMENT_ID`.

The tool-neutral serialisation/deserialisation logic lives in
`nf_metro.manifest` (a dependency-free package built to be lifted into its
own distribution). `render/manifest.py` is the thin nf-metro-specific
adapter: it imports from `nf_metro.manifest` and re-exports the public API
so that existing `nf_metro.render.manifest` import paths keep working.

`manifest_metadata_svg(manifest)` returns the raw SVG `<metadata>` XML string
for cases where the caller assembles the SVG element manually.

## Render-geometry validation (`validate.py`)

Layout guards run before rendering applies line offsets and final label
adjustments. Some problems can appear only in the finished SVG. For example,
an offset can move a line across a label.

`validate_render(svg)` checks the finished output. It reads station markers
from the embedded manifest, routes from the drawn path elements, and labels
from the drawn text elements. It then runs three checks:

- **label-strike** finds lines drawn through station-label text. It uses
  `segment_strikes_label` at the rendered font size.
- **marker-cross** flags a route segment through a non-consumer station's
  marker. A station that carries the line is exempt. Rail interchanges are
  also exempt because lines are meant to pass through their markers.
- **offset-collapse** flags two distinct lines drawn flush where the offset
  plan placed them at least one `OFFSET_STEP` apart. Lines assigned to the same
  slot may share a track by design.

The label and marker checks need only the SVG. They run in
`validate-svg --geometry` and `render --validate`. The offset check also needs
the original `RenderPlan`, so call `validate_render(svg, plan=plan)` to enable
it.

## Animation (`animate.py`)

`render_animation(d, graph, routes, station_offsets, theme)` appends animated
`<circle>` elements to an existing `drawsvg.Drawing`. CSS `offset-path` and
`@keyframes` move each circle along its line. `emit_render_plan` calls it when
`animate` is `True`.

CSS animation is used rather than SMIL `<animateMotion>` because SMIL does
not run when an SVG is injected into a host page via `innerHTML` (the
playground preview, the inline embed snippet, any host inlining an exported
map): the timeline advances but the motion is never sampled, freezing every
ball at its path start. CSS `offset-path` animates whether the SVG is opened
standalone, referenced from `<img>`, or inlined.

Each metro line gets one ball. All balls are synchronised to the same
cycle duration (`max_dur`, chosen so the slowest ball just finishes one
lap per cycle): a shorter line covers its path in the first `move_frac` of
the cycle and holds at the terminus for the rest (a 3-stop `@keyframes`),
so no ball restarts while another is mid-track.

## Theming (`style.py`)

`Theme` is a frozen dataclass of visual properties: colours, font sizes,
line widths, station radii, animation speed, and legend layout.

Built-in themes live in `src/nf_metro/themes/`:

- `nfcore.py` - dark theme (default), matching nf-core visual style.
- `light.py` - light theme variant.

To add a theme: create a `Theme` instance and register it in
`themes/__init__.py`'s `THEMES` dict under a string key; it then becomes
selectable via `%%metro style: <key>` or `--style`.

## Module map

| Module         | Responsibility                                                                                                                     |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `plan.py`      | Immutable `RenderPlan` data and conversion helpers                                                                                 |
| `svg.py`       | `render_svg`, plan construction, SVG output, and drawing passes                                                                    |
| `bridges.py`   | `compute_bridges` - detects genuine non-merging crossings and returns `BridgeBreak` gap spans; drawing is in `svg.py`              |
| `html.py`      | `render_html` - standalone HTML page and inline embed snippet around the SVG                                                       |
| `manifest.py`  | nf-metro adapter for the embedded-manifest standard; `build_manifest`, `manifest_metadata_svg`                                     |
| `validate.py`  | `validate_render` - render-geometry guards that read the drawn SVG (markers, route ink, label ink) as their own oracle             |
| `animate.py`   | `render_animation` - animated balls via CSS `offset-path` + `@keyframes`                                                           |
| `style.py`     | `Theme` dataclass                                                                                                                  |
| `legend.py`    | `render_legend`, `compute_legend_dimensions`                                                                                       |
| `icons.py`     | `render_file_icon`, `render_files_icon`, `render_folder_icon`                                                                      |
| `constants.py` | render magic numbers (canvas padding, legend sizing, animation params, debug overlay); theme-dependent values remain in `style.py` |
