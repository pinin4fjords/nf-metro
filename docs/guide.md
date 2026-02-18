# Writing metro maps

nf-metro input files use a subset of Mermaid `graph LR` syntax extended with `%%metro` directives. This guide walks through the format from a minimal example up to a full multi-section pipeline.

## Minimal example

The simplest metro map needs just lines, stations, and edges:

```text
%%metro title: Simple Pipeline
%%metro style: dark
%%metro line: main | Main | #4CAF50
%%metro line: qc | Quality Control | #2196F3

graph LR
    input[Input]
    fastqc[FastQC]
    trim[Trimming]
    align[Alignment]
    quant[Quantification]
    multiqc[MultiQC]

    input -->|main| trim
    trim -->|main| align
    align -->|main| quant
    input -->|qc| fastqc
    trim -->|qc| fastqc
    quant -->|qc| multiqc
    fastqc -->|qc| multiqc
```

Save this as `pipeline.mmd` and render it:

```bash
nf-metro render pipeline.mmd -o pipeline.svg
```

The key elements:

- **`%%metro line:`** defines a route with `id | Display Name | #hexcolor`
- **`graph LR`** starts the Mermaid graph (always left-to-right)
- **Stations** use Mermaid node syntax: `node_id[Label]`
- **Edges** carry line IDs: `source -->|line_id| target`
- An edge can carry multiple lines: `a -->|line1,line2| b`

## Adding sections

Sections group related stations into visual boxes. They use Mermaid `subgraph` blocks:

```text
%%metro title: Sectioned Pipeline
%%metro style: dark
%%metro line: main | Main | #4CAF50
%%metro line: qc | Quality Control | #2196F3

graph LR
    subgraph preprocessing [Pre-processing]
        trim[Trimming]
        fastqc[FastQC]
        trim -->|main,qc| fastqc
    end

    subgraph analysis [Analysis]
        align[Alignment]
        quant[Quantification]
        align -->|main| quant
    end

    %% Inter-section edges (outside all subgraph blocks)
    fastqc -->|main| align
    fastqc -->|qc| quant
```

Sections are laid out automatically on a grid based on their dependencies. Edges between stations in different sections must go **outside** all `subgraph`/`end` blocks. nf-metro automatically creates port connections and junction stations at fan-out points.

## Global directives

These go at the top of the file, before `graph LR`:

### Title and theme

```text
%%metro title: nf-core/rnaseq
%%metro style: dark
```

Themes: `dark` (default) or `light`.

### Logo

```text
%%metro logo: path/to/logo.png
```

Replaces the text title with an image. Use the `--logo` CLI flag to override per-render (useful for dark/light variants).

### Lines

```text
%%metro line: star_rsem | Aligner: STAR, Quantification: RSEM | #0570b0
%%metro line: star_salmon | Aligner: STAR, Quantification: Salmon (default) | #2db572
%%metro line: hisat2 | Aligner: HISAT2, Quantification: None | #f5c542
```

Each line needs a unique ID, a display name (shown in the legend), and a hex color.

### Legend

```text
%%metro legend: bl
```

Positions: `tl`, `tr`, `bl`, `br` (corners), `bottom`, `right`, or `none`.

### Grid placement

Sections are placed automatically, but you can pin specific sections:

```text
%%metro grid: postprocessing | 2,0,2
%%metro grid: qc_report | 1,2,1,2
```

Format: `section_id | col,row[,rowspan[,colspan]]`.

### File markers

```text
%%metro file: fastq_in | FASTQ
%%metro file: report_final | HTML
```

Marks a station as a file terminus with a document icon and label.

## Section directives

These go inside `subgraph` blocks.

### Entry and exit hints

```text
subgraph preprocessing [Pre-processing]
    %%metro exit: right | star_salmon, star_rsem, hisat2
    %%metro exit: bottom | pseudo_salmon, pseudo_kallisto
    ...
end
```

Entry/exit hints tell the layout engine which side of the section box lines should enter or leave from. Sides: `left`, `right`, `top`, `bottom`.

Most of the time you can **omit these entirely** and let the auto-layout engine infer them from the graph topology. Explicit hints are useful when:

- You want lines to exit from different sides (e.g., right for some, bottom for others)
- The auto-inferred placement doesn't match your intended layout

### Section direction

```text
subgraph postprocessing [Post-processing]
    %%metro direction: TB
    ...
end
```

Controls the flow direction within a section:

- **`LR`** (default) - left to right
- **`RL`** - right to left, useful for creating serpentine layouts where a section flows back
- **`TB`** - top to bottom, useful for vertical connector sections

## Stations and edges

### Station syntax

Stations use Mermaid node syntax:

```text
node_id[Label]
```

The `node_id` is used in edges and directives. The `Label` is what appears on the map.

### Edge syntax

Edges connect stations and specify which lines travel along them:

```text
%% Single line
trim -->|main| align

%% Multiple lines on the same edge
cat_fastq -->|star_salmon,star_rsem,hisat2| fastqc
```

When multiple lines share an edge, they're drawn as parallel tracks through that connection.

### Forking and joining

Lines diverge when different edges carry different line IDs from the same station:

```text
star -->|star_rsem| rsem
star -->|star_salmon| umi_tools_dedup
hisat2_align -->|hisat2| umi_tools_dedup
```

Lines reconverge when edges from different stations target the same destination. The layout engine handles fork-join patterns automatically.

### Inter-section edges

Edges between stations in different sections go outside all `subgraph`/`end` blocks:

```text
    subgraph preprocessing [Pre-processing]
        ...
        sortmerna[SortMeRNA]
    end

    subgraph alignment [Alignment]
        star[STAR]
        hisat2_align[HISAT2]
        ...
    end

    %% Inter-section edges
    sortmerna -->|star_salmon,star_rsem| star
    sortmerna -->|hisat2| hisat2_align
```

These are automatically rewritten into port-to-port connections with junction stations at fan-out points. You just specify the source and target stations directly.

## Walkthrough: nf-core/rnaseq

The full example is at [`examples/rnaseq_sections.mmd`](https://github.com/pinin4fjords/nf-metro/blob/main/examples/rnaseq_sections.mmd). Here's how the pieces fit together.

The pipeline has five lines representing different analysis routes:

```text
%%metro line: star_rsem | Aligner: STAR, Quantification: RSEM | #0570b0
%%metro line: star_salmon | Aligner: STAR, Quantification: Salmon (default) | #2db572
%%metro line: hisat2 | Aligner: HISAT2, Quantification: None | #f5c542
%%metro line: pseudo_salmon | Pseudo-aligner: Salmon, Quantification: Salmon | #e63946
%%metro line: pseudo_kallisto | Pseudo-aligner: Kallisto, Quantification: Kallisto | #7b2d3b
```

All five share a preprocessing section, then diverge based on aligner choice. The preprocessing section exits right for alignment-based lines and bottom for pseudo-aligners:

```text
subgraph preprocessing [Pre-processing]
    %%metro exit: right | star_salmon, star_rsem, hisat2
    %%metro exit: bottom | pseudo_salmon, pseudo_kallisto
    cat_fastq[cat fastq]
    fastqc_raw[FastQC]
    ...
end
```

Post-processing uses `TB` direction to act as a vertical connector:

```text
subgraph postprocessing [Post-processing]
    %%metro direction: TB
    ...
end
```

The QC section uses `RL` direction to flow backward, creating a serpentine layout:

```text
subgraph qc_report [Quality control & reporting]
    %%metro direction: RL
    ...
end
```

Grid overrides pin sections that the auto-layout can't infer perfectly:

```text
%%metro grid: postprocessing | 2,0,2
%%metro grid: qc_report | 1,2,1,2
```

## Directive reference

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

## Tips

- **Start without sections.** Get your stations and line routing right first, then wrap groups in `subgraph` blocks.
- **Omit entry/exit hints.** The auto-layout engine infers them correctly in most cases. Only add hints when you need multi-side exits or want to override the default.
- **Use `--debug`** to see ports, hidden stations, and edge waypoints: `nf-metro render --debug pipeline.mmd -o debug.svg`
- **Use `nf-metro validate`** to catch errors before rendering.
- **Use `nf-metro info`** to inspect the parsed structure (sections, lines, stations, edges).
