# Importing from Nextflow

nf-metro can import pipeline diagrams directly from Nextflow's built-in DAG output. This lets you visualize any Nextflow pipeline as a metro map without writing the `.mmd` file by hand.

## Generating a Nextflow DAG

Nextflow can export its pipeline DAG in mermaid format:

```bash
nextflow run my_pipeline.nf -preview -with-dag dag.mmd
```

The `-preview` flag skips execution and just generates the DAG. The resulting `dag.mmd` file uses Nextflow's `flowchart TB` mermaid syntax, which is different from nf-metro's `graph LR` format.

## Quick render

The fastest way to get a metro map from a Nextflow DAG is the `--from-nextflow` flag on `nf-metro render`:

```bash
nf-metro render dag.mmd -o pipeline.svg --from-nextflow --title "My Pipeline"
```

This converts and renders in one step. The `--title` option sets the diagram title (otherwise it defaults to the Nextflow DAG's structure).

## Two-step workflow

For more control, convert first and then render separately:

```bash
# Step 1: Convert to nf-metro format
nf-metro convert dag.mmd -o pipeline.mmd --title "My Pipeline"

# Step 2: Render (or hand-tune first)
nf-metro render pipeline.mmd -o pipeline.svg
```

The two-step workflow lets you edit the `.mmd` file before rendering. You can add custom line colors, adjust section names, add entry/exit port hints, or reorganize the layout.

## What the converter does

The converter transforms Nextflow's DAG representation into nf-metro format through several steps:

1. **Node classification** - Nextflow DAGs contain three types of nodes: processes (stadium-shaped), channels/values (square brackets), and operators (circles). The converter keeps processes and drops channel/operator nodes.

2. **Edge reconnection** - After dropping channel and operator nodes, edges are reconnected through them. If process A connects to channel C, which connects to process B, the converter creates a direct A-to-B edge.

3. **Section mapping** - Nextflow subworkflows become nf-metro sections. Processes not inside any subworkflow are grouped into auto-generated sections (typically "Reporting" for terminal processes like MultiQC).

4. **Line detection** - The converter assigns metro lines based on the graph structure:
      - A **main** line follows the longest path through the pipeline
      - **Bypass lines** are created for edges that skip over intermediate sections (e.g., a QC line going from preprocessing directly to reporting)
      - **Spur lines** mark dead-end processes that branch off the main flow (displayed as perpendicular branches)

5. **Label cleanup** - Process names are converted from `SCREAMING_SNAKE_CASE` to `Title Case` and long names are abbreviated to fit the diagram.

## Limitations

- **Bypass routing** - Lines that skip over intermediate sections may visually pass through those sections. This is a known layout limitation. For best results, design pipelines where lines visit a section in every column.

- **Cycle breaking** - Nextflow DAGs can contain cycles (e.g., retry loops). The converter breaks these by removing back-edges detected via DFS.

- **Complex subworkflows** - Very deeply nested subworkflow structures may produce cluttered diagrams. Consider using the two-step workflow and simplifying the converted `.mmd` by hand.

## Example

Starting from a Nextflow pipeline with three subworkflows (Preprocess, Alignment, Variant Calling) plus a standalone MultiQC process:

```bash
nextflow run variant_calling.nf -preview -with-dag vc_dag.mmd
nf-metro convert vc_dag.mmd -o vc.mmd --title "Variant Calling Pipeline"
nf-metro render vc.mmd -o vc.svg
```

The converter maps subworkflows to sections, detects the main analysis flow, creates bypass lines for QC connections, and assigns spur lines for dead-end processes like index files that don't feed downstream.
