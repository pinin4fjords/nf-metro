# nf-metro

Generate metro-map-style SVG diagrams from Mermaid graph definitions with `%%metro` directives. Designed for visualizing bioinformatics pipeline workflows (e.g., nf-core pipelines) as transit-style maps where each analysis route is a colored "metro line."

![nf-core/rnaseq metro map](assets/renders/rnaseq_auto.svg)

## Installation

```bash
pip install nf-metro
```

Requires Python 3.10+.

## Quick start

Render a metro map from a `.mmd` file:

```bash
nf-metro render examples/simple_pipeline.mmd -o pipeline.svg
```

Validate your input without rendering:

```bash
nf-metro validate examples/simple_pipeline.mmd
```

Inspect structure (sections, lines, stations):

```bash
nf-metro info examples/simple_pipeline.mmd
```

## CLI reference

### `nf-metro render`

Render a Mermaid metro map definition to SVG.

```
nf-metro render [OPTIONS] INPUT_FILE
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output PATH` | `<input>.svg` | Output SVG file path |
| `--theme [nfcore\|light]` | `nfcore` | Visual theme |
| `--width INTEGER` | auto | SVG width in pixels |
| `--height INTEGER` | auto | SVG height in pixels |
| `--x-spacing FLOAT` | `60` | Horizontal spacing between layers |
| `--y-spacing FLOAT` | `40` | Vertical spacing between tracks |
| `--max-layers-per-row INTEGER` | auto | Max layers before folding to next row |
| `--animate / --no-animate` | off | Add animated balls traveling along lines |
| `--debug / --no-debug` | off | Show debug overlay |
| `--logo PATH` | none | Logo image path (overrides `%%metro logo:` directive) |

### `nf-metro validate`

Check a `.mmd` file for errors without producing output.

```
nf-metro validate INPUT_FILE
```

### `nf-metro info`

Print a summary of the parsed map: sections, lines, stations, and edges.

```
nf-metro info INPUT_FILE
```

## Gallery

See the [Gallery](gallery/index.md) for rendered examples covering simple pipelines, complex multi-line topologies, fan-out/fan-in patterns, fold layouts, and realistic bioinformatics workflows.

## Input format

Input files use a subset of Mermaid `graph LR` syntax extended with `%%metro` directives. See the full [directive reference](https://github.com/pinin4fjords/nf-metro#directive-reference) in the README.

### Directive reference

| Directive | Scope | Description |
|-----------|-------|-------------|
| `%%metro title: <text>` | Global | Map title |
| `%%metro logo: <path>` | Global | Logo image (replaces title text) |
| `%%metro style: <name>` | Global | Theme: `dark`, `light` |
| `%%metro line: <id> \| <name> \| <color>` | Global | Define a metro line |
| `%%metro grid: <section> \| <col>,<row>[,<rowspan>[,<colspan>]]` | Global | Pin section to grid position |
| `%%metro legend: <position>` | Global | Legend position: `tl`, `tr`, `bl`, `br`, `bottom`, `right`, `none` |
| `%%metro file: <station> \| <label>` | Global | Mark a station as a file terminus with a document icon |
| `%%metro entry: <side> \| <lines>` | Section | Entry port hint |
| `%%metro exit: <side> \| <lines>` | Section | Exit port hint |
| `%%metro direction: <dir>` | Section | Flow direction: `LR`, `RL`, `TB` |

## License

[MIT](https://github.com/pinin4fjords/nf-metro/blob/main/LICENSE)
