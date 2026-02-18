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

![Minimal example](assets/renders/01_minimal.svg)

The key elements:

- **`%%metro line:`** defines a route with `id | Display Name | #hexcolor`
- **`graph LR`** starts the Mermaid graph (always left-to-right)
- **Stations** use Mermaid node syntax: `node_id[Label]`
- **Edges** carry line IDs: `source -->|line_id| target`
- An edge can carry multiple lines: `a -->|line1,line2| b`

## Adding sections

Sections group related stations into visual boxes using Mermaid `subgraph` blocks. Lines can diverge to different sections, showing how routes split through the pipeline:

```text
%%metro title: Sectioned Pipeline
%%metro style: dark
%%metro line: main | Main | #4CAF50
%%metro line: qc | Quality Control | #2196F3

graph LR
    subgraph preprocessing [Pre-processing]
        input[Input]
        trim[Trimming]
        fastqc[FastQC]
        input -->|main,qc| trim
        trim -->|main,qc| fastqc
    end

    subgraph analysis [Analysis]
        align[Alignment]
        quant[Quantification]
        align -->|main| quant
    end

    subgraph reporting [Reporting]
        multiqc[MultiQC]
        report[Report]
        multiqc -->|qc| report
    end

    fastqc -->|main| align
    fastqc -->|qc| multiqc
```

![Sectioned example](assets/renders/02_sections.svg)

Sections are laid out automatically on a grid based on their dependencies. Edges between stations in different sections must go **outside** all `subgraph`/`end` blocks. nf-metro automatically creates port connections and junction stations at fan-out points.

## Fan-out and fan-in

When lines diverge from a shared section into separate analysis paths and then reconverge, nf-metro stacks the target sections vertically and routes each line to its destination:

```text
%%metro title: Fan-out Pipeline
%%metro style: dark
%%metro line: wgs | Whole Genome | #e63946
%%metro line: wes | Whole Exome | #0570b0
%%metro line: panel | Targeted Panel | #2db572

graph LR
    subgraph preprocessing [Pre-processing]
        fastqc[FastQC]
        trim[Trimming]
        fastqc -->|wgs,wes,panel| trim
    end

    subgraph wgs_analysis [WGS Analysis]
        bwa_wgs[BWA-MEM]
        gatk_wgs[GATK HaplotypeCaller]
        bwa_wgs -->|wgs| gatk_wgs
    end

    subgraph wes_analysis [WES Analysis]
        bwa_wes[BWA-MEM]
        gatk_wes[GATK Mutect2]
        bwa_wes -->|wes| gatk_wes
    end

    subgraph panel_analysis [Panel Analysis]
        minimap[Minimap2]
        freebayes[FreeBayes]
        minimap -->|panel| freebayes
    end

    subgraph annotation [Annotation]
        vep[VEP]
        report[Report]
        vep -->|wgs,wes,panel| report
    end

    trim -->|wgs| bwa_wgs
    trim -->|wes| bwa_wes
    trim -->|panel| minimap
    gatk_wgs -->|wgs| vep
    gatk_wes -->|wes| vep
    freebayes -->|panel| vep
```

![Fan-out example](assets/renders/03_fan_out.svg)

Each line takes a different route through its own analysis section, then all three reconverge at the annotation section. The layout engine handles the junction creation and routing automatically.

## Full example: nf-core/rnaseq

The complete nf-core/rnaseq example at [`examples/rnaseq_sections.mmd`](https://github.com/pinin4fjords/nf-metro/blob/main/examples/rnaseq_sections.mmd) combines all these patterns into a real-world pipeline:

![nf-core/rnaseq](assets/renders/rnaseq_auto.svg)

Five analysis routes share preprocessing, then diverge to different aligners, reconverge at post-processing, and flow through QC. This example also demonstrates **section directions** - while most sections flow left-to-right (`LR`, the default), two other directions create more interesting layouts:

- **`TB`** (top-to-bottom) - the Post-processing section flows vertically, acting as a connector between the aligners above and below
- **`RL`** (right-to-left) - the QC section flows right-to-left, creating a serpentine return path

The layout engine often infers these directions automatically from the graph topology, but you can set them explicitly with `%%metro direction: TB` or `%%metro direction: RL` inside a `subgraph` block. The example also uses:

- **Multiple exit sides** on preprocessing (right for aligners, bottom for pseudo-aligners)
- **Grid overrides** to pin sections the auto-layout can't infer perfectly

See the [Gallery](gallery/index.md) for more rendered examples.

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
