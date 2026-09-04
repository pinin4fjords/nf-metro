---
title: "CLI reference"
description: "Full reference for the nf-metro command-line interface: every command and every option."
---

nf-metro ships eleven commands. This page documents every one of them and every option they accept.

| Command                                    | What it does                                                   |
| ------------------------------------------ | -------------------------------------------------------------- |
| [`render`](#nf-metro-render)               | Render one or more `.mmd` files to SVG or interactive HTML     |
| [`render-many`](#nf-metro-render-many)     | Render a JSON manifest of render jobs in one process           |
| [`convert`](#nf-metro-convert)             | Convert a Nextflow `-with-dag` mermaid file to nf-metro format |
| [`validate`](#nf-metro-validate)           | Check a `.mmd` file for errors without producing output        |
| [`info`](#nf-metro-info)                   | Show what nf-metro parsed and derived from a map               |
| [`explain`](#nf-metro-explain)             | Show _why_ the layout engine made each decision                |
| [`serve`](#nf-metro-serve)                 | Serve a live-progress view of one map                          |
| [`serve-multi`](#nf-metro-serve-multi)     | Run a persistent live server many pipelines can report into    |
| [`check-mapping`](#nf-metro-check-mapping) | Check a map's `%%metro process:` mapping against a pipeline    |
| [`validate-svg`](#nf-metro-validate-svg)   | Validate a rendered SVG's embedded manifest (and its ink)      |
| [`embed-script`](#nf-metro-embed-script)   | Print the embed driver JS for a host page                      |

`nf-metro --version` prints the installed version. Every command also takes `--help`.

Most `render` options have a `%%metro` directive twin; an explicitly-passed flag overrides the directive.

## `nf-metro render`

Render a Mermaid metro map definition to SVG or interactive HTML.

```bash frame="terminal"
nf-metro render [OPTIONS] INPUT_FILE...
```

Accepts one or more `INPUT_FILE`s. With more than one, all render within the
same process (amortising interpreter and import startup across the batch) and
each write to their own sibling `<input>.<format>`; every file is attempted
even if an earlier one fails, successful outputs are kept, and the command
exits non-zero if any failed.

A rejected input, and any other failure, surfaces as a plain error message
rather than a traceback; set `NF_METRO_DEBUG=1` to re-raise the original
exception instead. An empty file, or one whose `graph` block holds no
stations, is rejected by name rather than drawn.

Most of the options below also have a `%%metro` directive twin; an explicitly-passed flag overrides the directive (see the [precedence table](/nf-metro/guide/#cli-flags-and-directive-precedence) in the guide).

### Output and source

| Option                 | Default            | Description                                                            |
| ---------------------- | ------------------ | ---------------------------------------------------------------------- |
| `-o`, `--output PATH`  | `<input>.<format>` | Output file path (only valid with a single `INPUT_FILE`)               |
| `--format [svg\|html]` | `svg`              | Output format: `svg`, or `html` for an interactive self-contained page |
| `--from-nextflow`      | off                | Convert Nextflow `-with-dag` mermaid input before rendering            |
| `--debug / --no-debug` | off                | Show the debug overlay (ports, hidden stations, edge waypoints)        |

### Theme and branding

| Option                                                                                  | Default                      | Description                                                                                                                                                   |
| --------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--theme [nfcore\|nfcore-light\|nfcore-dark\|seqera\|seqera-light\|seqera-dark\|light]` | from `style:`, else `nfcore` | Visual theme. A bare brand name (`nfcore`, `seqera`) takes the mode from `--mode`; the suffixed names pin a mode. Directive twin: `%%metro style:`            |
| `--mode [light\|dark]`                                                                  | from `mode:`, else `dark`    | Display mode, independent of the brand. Bakes the chosen mode's palette, so use it for light or dark PNG export. Directive twin: `%%metro mode:`              |
| `--logo PATH`                                                                           | none                         | Logo image path (must exist; errors on a bad path). Directive twin: `%%metro logo:`                                                                           |
| `--title TEXT`                                                                          | from `title:`                | Pipeline title. Directive twin: `%%metro title:`                                                                                                              |
| `--caption TEXT`                                                                        | none                         | Free-text caption or attribution line rendered bottom-left of the map (e.g. `Adapted from Author et al., Journal (Year)`). Directive twin: `%%metro caption:` |

`--theme light` is the transparent embed theme rather than a brand: it has no
light/dark pair, so `--mode` does not apply to it.

### Legend and logo

| Option                      | Default | Description                                                                                                                                           |
| --------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--legend TEXT`             | auto    | Position the legend+logo block: keyword (`bl`/`br`/`tl`/`tr`/`bottom`/`right`/`none`), `<keyword> \| canvas`, `<keyword> \| dx,dy`, or absolute `x,y` |
| `--logo-scale FLOAT`        | 1.0     | Scale the logo within the legend block (1.0 = default auto-size)                                                                                      |
| `--legend-min-height FLOAT` | 0       | Minimum legend content height in pixels (useful for single-line maps where the logo would otherwise be tiny)                                          |
| `--legend-logo-gap FLOAT`   | auto    | Horizontal gap in pixels between the logo and the legend entries                                                                                      |

### Layout

| Option                                     | Default           | Description                                                                                                                                                                                                                                                                         |
| ------------------------------------------ | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--line-spread [bundle\|centered\|rails]`  | `bundle`          | How lines sharing a station relate vertically: `bundle` merges them onto one trunk, `centered` balances the bundle about the midline, `rails` draws parallel rails with interchange stations. Overrides the graph-wide directive; per-section `%%metro line_spread:` overrides stay |
| `--x-spacing FLOAT`                        | auto              | Horizontal spacing between layers (auto widens from 60 only when wide labels would otherwise collide)                                                                                                                                                                               |
| `--y-spacing FLOAT`                        | auto              | Vertical spacing between tracks (auto is derived from the map's content so captioned icons and dense labels don't collide)                                                                                                                                                          |
| `--section-x-gap FLOAT`                    | 50                | Horizontal gap between sections                                                                                                                                                                                                                                                     |
| `--section-y-gap FLOAT`                    | 50                | Vertical gap between sections                                                                                                                                                                                                                                                       |
| `--track-gap FLOAT`                        | 1                 | Visual gap in pixels (0 to 3) between adjacent line strokes in a bundle, edge to edge rather than centre to centre. 0 means the lines touch; values above 3 are rejected                                                                                                            |
| `--fold-threshold INTEGER`                 | 15                | Max station-columns a section row may reach before the auto-layout wraps it onto the next row. Raise it to keep a long horizontal trunk on one row                                                                                                                                  |
| `--diamond-style [straight\|symmetric]`    | `straight`        | Fork-join (diamond) layout: `straight` keeps the top branch on the main track, `symmetric` fans the branches evenly                                                                                                                                                                 |
| `--line-order [definition\|span]`          | `definition`      | Line ordering for track assignment: `definition` preserves `.mmd` order, `span` gives longest-spanning lines inner tracks                                                                                                                                                           |
| `--center-ports / --no-center-ports`       | off               | Centre inter-section ports on the shorter of the two connected sections, so lines enter and exit at the visual midpoint                                                                                                                                                             |
| `--compact-offsets / --no-compact-offsets` | off               | Size each station only for the lines actually passing through it, rather than reserving a slot for every declared line                                                                                                                                                              |
| `--label-angle FLOAT`                      | theme default (0) | Angle in degrees for station labels (0 = horizontal). Useful for dense trunks where horizontal labels collide                                                                                                                                                                       |
| `--font-scale FLOAT`                       | 1.0               | Scale every text size and the label-width metrics that drive layout spacing                                                                                                                                                                                                         |
| `--stroke-scale FLOAT`                     | 1.0               | Scale track stroke weight and station pill size, widening bundle spacing, marker clearance, and rail pitch to match                                                                                                                                                                 |
| `--width INTEGER`                          | auto              | Output width in pixels                                                                                                                                                                                                                                                              |
| `--height INTEGER`                         | auto              | Output height in pixels                                                                                                                                                                                                                                                             |

Spacings, scales, `--fold-threshold` and output dimensions must be greater
than 0; the section gaps, `--track-gap`, `--legend-min-height` and
`--legend-logo-gap` also accept 0. A value outside an option's range is
rejected by the flag and by its `%%metro` directive alike.

### Line styling

| Option                             | Default                 | Description                                                                                                                                                                                                                                                                                                               |
| ---------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--inactive-lines TEXT`            | from `line:` directives | Comma-separated line IDs to render inactive: their strokes, chevrons, and legend swatches grey out, as do the stations, labels, and terminus icons touched only by inactive lines. Unknown IDs error. Fully replaces the map's `inactive`-marked lines; an empty value forces every line active. Does not edit the `.mmd` |
| `--animate / --no-animate`         | off                     | Add animated balls traveling along the metro lines                                                                                                                                                                                                                                                                        |
| `--directional / --no-directional` | off                     | Draw static chevrons along each route pointing in the flow direction (source to target)                                                                                                                                                                                                                                   |

### Live-progress metadata

These carry into the rendered SVG's manifest and drive [live progress](/nf-metro/live/); they do not change the drawn map.

| Option                               | Default | Description                                                                                                                                                                                                                                                                                                      |
| ------------------------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--auto-process / --no-auto-process` | off     | Map each station to its own id as a default process pattern when it has no explicit `%%metro process:` directive, so a map whose station ids already name their Nextflow processes lights up live with no per-station mapping. Explicit directives override the default                                          |
| `--process-scope TEXT`               | none    | Common fully-qualified-name prefix shared by the pipeline's processes (e.g. `NFCORE_RNASEQ:RNASEQ`). Each `%%metro process:` value is then the tail under this scope, joined as `<scope>:<tail>` and matched literally, so a pasted process path needs no regex. Without a scope, `process:` values stay regexes |

### Guard behaviour

| Option                           | Default | Description                                                                                                                                                                                                                                                                         |
| -------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--validate`                     | off     | After rendering, fail if the render-geometry guards find a defect in the produced SVG: a route drawn through a station's label or marker, or two lines collapsed onto one stroke. Tier-A layout-invariant violations stay warnings here; `--strict` fails on those. SVG output only |
| `--strict / --no-strict`         | off     | Treat a Tier-A layout-invariant violation on the rendered geometry as an error (non-zero exit) instead of a warning                                                                                                                                                                 |
| `--permissive / --no-permissive` | off     | Downgrade layout and render guard failures to warnings and render best-effort on whatever geometry was computed, instead of aborting with no output. Overrides `--strict`                                                                                                           |

### Warnings

A map that parses with complaints (an unknown `%%metro` directive, a
non-LR primary direction) or a layout that widens a gap to fit its routing
reports each one as a bulleted `Warnings:` block on stderr, and renders. The
map is still written; the block says what nf-metro ignored or adjusted.

### Embedding options

Flags for producing an SVG to embed in another page or application. The [Embedding guide](/nf-metro/embedding/) explains when to use each.

| Option                                 | Default | Description                                                                                                                                                                                                                                                 |
| -------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--responsive / --no-responsive`       | off     | Emit `viewBox` only (no fixed `width`/`height`) for CSS-scalable embedding                                                                                                                                                                                  |
| `--embed-font / --no-embed-font`       | off     | Inline a subset of Inter as a base64 `@font-face` block so the SVG renders identically on any host regardless of installed fonts                                                                                                                            |
| `--text-to-paths / --no-text-to-paths` | off     | Convert all text to vector paths, removing font dependencies entirely. Loses selectable text; requires `fonttools[woff]`                                                                                                                                    |
| `--bare / --no-bare`                   | off     | Omit the title and outer padding so the canvas hugs the diagram content (the attribution watermark is kept)                                                                                                                                                 |
| `--svg-class-prefix TEXT`              | none    | Prefix every SVG presentation class with this string (e.g. `myapp` produces `myapp-nf-metro-station`). Use distinct prefixes for each map on a shared page. No effect on the interactive HTML output, which already scopes each map                         |
| `--no-self-color-scheme`               | off     | Omit `color-scheme: light dark` from the root `<svg>`. Use when inlining into a host page that owns the theme: the SVG then inherits the page's `color-scheme`, so a manual toggle drives `light-dark()` resolution rather than the viewer's OS preference  |
| `--no-dark-mode-css`                   | off     | Suppress the `prefers-color-scheme: dark` `<style>` block when a host page manages its own theme and the injected media query would conflict                                                                                                                |
| `--no-chrome-css`                      | off     | Omit the chrome `--nfm-*` CSS custom-property `<style>` block. Colors still render (they are baked as presentation attributes); only live host recoloring is dropped. Needed for raster export, since cairosvg and similar rasterizers cannot parse `var()` |
| `--manifest / --no-manifest`           | on      | Embed the machine-readable [data manifest](/nf-metro/manifest/) (the `<metadata>` block and per-node `data-node-*` attributes) in the SVG. On by default; `--no-manifest` emits the drawn map only. Directive twin: `%%metro manifest:`                     |

### Interactive HTML output

`--format html` produces a self-contained `.html` file with the SVG inlined plus a small JS/CSS layer (no external dependencies, no network):

```bash frame="terminal"
nf-metro render pipeline.mmd --format html -o pipeline.html
```

The page supports drag-to-pan, scroll-to-zoom, station hover tooltips, and a clickable line legend. Clicking a line isolates it: stations and sections not carrying that line are hidden and the view zooms to the bounding box of what remains. Click again, hit `Esc`, or use the Reset button to restore.

The **Embed&hellip;** button opens a panel with copyable inline-HTML, iframe, and static-SVG snippets. The [Embedding guide](/nf-metro/embedding/) explains when to reach for each, plus responsive sizing, font portability, host theming, and progress overlays.

### Validating the rendered geometry

Pass `--validate` to check the _drawn_ SVG after rendering and fail (non-zero exit) if a route is drawn through a station's label or marker, or two distinct lines collapse into one stroke where they should run parallel. This reads the geometry as it ends up on the page (after the per-line offsets and label shifts the layout applies), catching defects the pre-render checks cannot see:

```bash frame="terminal"
nf-metro render pipeline.mmd -o pipeline.svg --validate
```

`--validate` covers those drawn-geometry guards only. A Tier-A layout-invariant violation (two stations landing on the same coordinate, say) is reported as a warning and still renders; pass `--strict` to exit non-zero on one, or use [`nf-metro validate --with-layout`](#nf-metro-validate) to catch it before rendering at all.

To run the same geometry checks on an already-rendered SVG, use [`nf-metro validate-svg --geometry`](#nf-metro-validate-svg).

## `nf-metro render-many`

Render multiple metro maps from a JSON manifest in one process, amortising interpreter and import startup across the whole corpus. Output directories are created as needed; on partial failure, successful outputs are kept and the command exits non-zero.

```bash frame="terminal"
nf-metro render-many MANIFEST_FILE
```

`MANIFEST_FILE` is a JSON array of render jobs. Each job is an object with `input` and `output` (both required) plus any subset of the `render` options, expressed as JSON keys:

| Key                    | Description                                                                                                                                                                  |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input`                | Path to the source `.mmd` file (required)                                                                                                                                    |
| `output`               | Path for the output file (required)                                                                                                                                          |
| `format`               | `"svg"` (default) or `"html"`                                                                                                                                                |
| `theme`                | Theme name (`nfcore`, `light`, `seqera`, and the mode-suffixed variants)                                                                                                     |
| `mode`                 | `"light"` or `"dark"`; bakes a concrete palette                                                                                                                              |
| `debug`                | Show the debug overlay (default `false`)                                                                                                                                     |
| `logo`                 | Logo image path (overrides `%%metro logo:`)                                                                                                                                  |
| `line_spread`          | `"bundle"`, `"centered"`, or `"rails"`                                                                                                                                       |
| `legend`               | Legend position keyword or coordinate                                                                                                                                        |
| `from_nextflow`        | Convert from a Nextflow DAG first (default `false`)                                                                                                                          |
| `title`                | Pipeline title override                                                                                                                                                      |
| `responsive`           | Emit viewBox-only SVG (default `false`)                                                                                                                                      |
| `embed_font`           | Inline the Inter `@font-face` subset (default `false`)                                                                                                                       |
| `text_to_paths`        | Convert text to vector paths (default `false`)                                                                                                                               |
| `svg_class_prefix`     | Prefix for SVG presentation classes                                                                                                                                          |
| `no_self_color_scheme` | Omit `color-scheme` on the root `<svg>` (default `false`)                                                                                                                    |
| `no_dark_mode_css`     | Suppress the `prefers-color-scheme` block (default `false`)                                                                                                                  |
| `no_chrome_css`        | Omit the chrome CSS custom properties (default `false`)                                                                                                                      |
| `bare`                 | Omit the title and outer padding (default `false`)                                                                                                                           |
| `validate`             | Run the render-geometry guards (default `false`)                                                                                                                             |
| `inactive_lines`       | Line IDs to render inactive, as a comma-separated string or a JSON list. Omit the key to use the map's own inactive-by-directive lines; give `[]` to force every line active |
| `layout_options`       | Object of layout overrides, e.g. `{"manifest": false, "x_spacing": 60}`                                                                                                      |

```json
[
  { "input": "examples/rnaseq_auto.mmd", "output": "out/rnaseq.svg" },
  {
    "input": "examples/sarek.mmd",
    "output": "out/sarek.svg",
    "mode": "light",
    "layout_options": { "x_spacing": 60 }
  }
]
```

## `nf-metro convert`

Convert a Nextflow `-with-dag` mermaid file to nf-metro `.mmd` format. The output can then be rendered with `nf-metro render` or hand-tuned first.

```bash frame="terminal"
nf-metro convert [OPTIONS] INPUT_FILE
```

| Option                | Default | Description                             |
| --------------------- | ------- | --------------------------------------- |
| `-o`, `--output PATH` | stdout  | Output `.mmd` file path                 |
| `--title TEXT`        | none    | Pipeline title for the converted output |

See [Importing from Nextflow](/nf-metro/nextflow/) for details and examples.

## `nf-metro validate`

Check a `.mmd` file for errors without producing output. The bare command runs graph-semantic checks: every edge references a defined line, every section points at stations that exist, and the graph is acyclic.

```bash frame="terminal"
nf-metro validate [OPTIONS] INPUT_FILE
```

| Option          | Default | Description                                                                                                               |
| --------------- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--with-layout` | off     | Also run the layout engine with its full invariant suite, reporting any layout failure as an error instead of a traceback |
| `--strict`      | off     | Treat warnings (e.g. a non-LR primary direction) as errors                                                                |

## `nf-metro info`

Show information about a parsed map: sections, lines, stations, and edges. The default output is a stable human summary.

```bash frame="terminal"
nf-metro info [OPTIONS] INPUT_FILE
```

| Option      | Default | Description                                                                                                                            |
| ----------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `--json`    | off     | Emit the full introspection as JSON, for scripting                                                                                     |
| `--verbose` | off     | Add the section dependency graph, per-line routes, inferred auto-layout defaults, and synthetic ports and junctions to the text output |

Parse warnings print as a `Warnings:` block on stderr, keeping the summary on
stdout clean. `--verbose` and `--json` carry them in the report itself instead.

`Style:` reports the theme the map resolves to, which is the name `render --theme` accepts.

## `nf-metro explain`

Explain _why_ nf-metro made each layout decision: the rule that fired for each inferred choice (section direction, port sides, fold and row layout) and each synthetic element the engine inserted (fan-out junctions, bypass-V stations). It pairs with `nf-metro info`, which shows _what_ was built.

```bash frame="terminal"
nf-metro explain [OPTIONS] INPUT_FILE
```

| Option                 | Default | Description                                         |
| ---------------------- | ------- | --------------------------------------------------- |
| `--json`               | off     | Emit the full explanation as JSON                   |
| `--section SECTION_ID` | none    | Restrict output to decisions involving this section |
| `--station STATION_ID` | none    | Restrict output to decisions involving this station |

## `nf-metro serve`

Serve a live-progress view of a metro map. `INPUT_FILE` may be a `.mmd` source or an already-rendered nf-metro SVG. The map is rendered once and served at `http://HOST:PORT/`; point a Nextflow run's weblog at the events endpoint to light up stations as tasks run.

```bash frame="terminal"
nf-metro serve [OPTIONS] INPUT_FILE [-- LAUNCH_CMD...]
```

Stations are tied to processes with `%%metro process:` directives in the map, so only mapped stations change state. Everything about the event format, the overlay styles, and the endpoints is covered in [Live progress](/nf-metro/live/).

| Option                                                                                  | Default       | Description                                                                                            |
| --------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------ |
| `--port INTEGER`                                                                        | 8080          | Port to listen on                                                                                      |
| `--host TEXT`                                                                           | `127.0.0.1`   | Interface to bind. The default is local only; use `0.0.0.0` to accept connections from other hosts     |
| `--theme [nfcore\|nfcore-light\|nfcore-dark\|seqera\|seqera-light\|seqera-dark\|light]` | from `style:` | Visual theme, the same choices as `render --theme`                                                     |
| `--overlay [ring\|pulse\|dot\|led]`                                                     | `ring`        | Status-overlay style shown until a viewer picks another in the page                                    |
| `--token TEXT`                                                                          | none          | If set, `/events` POSTs must supply `?token=...` or an `X-Metro-Token` header                          |
| `--open`                                                                                | off           | Open the live page in a browser                                                                        |
| `--shutdown-after-complete`                                                             | off           | Stop the server shortly after the run's completed or error event (or after the launched command exits) |
| `--shutdown-grace FLOAT`                                                                | 10            | Seconds to keep the map up after the run finishes, with `--shutdown-after-complete`                    |

With an SVG input the map is served exactly as drawn, so `--theme` applies only to a `.mmd` input.

Passing a `LAUNCH_CMD` after `--` starts the run in one step with the weblog wired up automatically:

```bash frame="terminal"
nf-metro serve map.mmd --open --shutdown-after-complete -- \
  nextflow run my/pipeline -profile docker
```

Without a launch command, point the run at the server yourself:

```bash frame="terminal"
nextflow run ... -with-weblog http://localhost:8080/events
```

## `nf-metro serve-multi`

Run a persistent live server many pipelines can report into. Unlike `serve`, it starts with no map: a pipeline registers its map by POSTing the `.mmd` to `/maps`, then sends weblog events to the run's `/r/<id>/events` endpoint. The index at `http://HOST:PORT/` lists every run with a live status.

```bash frame="terminal"
nf-metro serve-multi [OPTIONS]
```

| Option                                                                                  | Default     | Description                                                                                        |
| --------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------- |
| `--port INTEGER`                                                                        | 8080        | Port to listen on                                                                                  |
| `--host TEXT`                                                                           | `127.0.0.1` | Interface to bind. The default is local only; use `0.0.0.0` to accept connections from other hosts |
| `--theme [nfcore\|nfcore-light\|nfcore-dark\|seqera\|seqera-light\|seqera-dark\|light]` | `nfcore`    | Visual theme, the same choices as `render --theme`                                                 |
| `--overlay [ring\|pulse\|dot\|led]`                                                     | `ring`      | Status-overlay style shown until a viewer picks another in the page                                |
| `--token TEXT`                                                                          | none        | If set, POSTs to `/maps` and `/r/*/events` must supply `?token=...` or an `X-Metro-Token` header   |

The nf-metro Nextflow plugin's `metro.server` mode does the register-and-emit automatically. See [Live progress](/nf-metro/live/#2b-persistent-server-many-runs).

## `nf-metro check-mapping`

Check a map's `%%metro process:` mapping against the pipeline's real processes. It reports processes the map can't show (drift) and station patterns that match nothing (stale), exiting non-zero if any are found, so CI can gate on map fidelity.

```bash frame="terminal"
nf-metro check-mapping [OPTIONS] INPUT_FILE
```

| Option             | Default | Description                                                                                      |
| ------------------ | ------- | ------------------------------------------------------------------------------------------------ |
| `--dag PATH`       | none    | Nextflow `-with-dag` mermaid file; process names are read from its stadium nodes                 |
| `--processes PATH` | none    | Newline-delimited process names (e.g. captured from a run). Authoritative alternative to `--dag` |
| `--ignore TEXT`    | none    | Regex for processes deliberately left unmapped (plumbing). Repeatable                            |

## `nf-metro validate-svg`

Validate a rendered SVG's embedded manifest against the [manifest JSON Schema](/nf-metro/manifest/#manifest-schema).

```bash frame="terminal"
nf-metro validate-svg [OPTIONS] SVG_FILE
```

| Option       | Default | Description                                                                                                                                                                                                                                             |
| ------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--geometry` | off     | Also run the artifact-only render-geometry guards on the drawn ink (label strikes and non-consumer marker crossings), not just the manifest schema. The offset-collapse check needs the engine's assigned offsets and runs only via `render --validate` |

## `nf-metro embed-script`

Print the `attachMetroMap()` embed driver JS to stdout. Load it on a host page alongside an nf-metro SVG to get the documented interactive API.

```bash frame="terminal"
nf-metro embed-script [OPTIONS]
```

| Option                | Default | Description                       |
| --------------------- | ------- | --------------------------------- |
| `-o`, `--output PATH` | stdout  | Write to a file instead of stdout |

See the [embed contract](/nf-metro/embed/#driver-api) for the driver API.
