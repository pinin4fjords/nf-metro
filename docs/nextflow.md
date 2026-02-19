# Importing from Nextflow

nf-metro can convert Nextflow's built-in DAG output into a metro map. This works well for simple pipelines with a few subworkflows and a clear linear flow. For complex pipelines (like most nf-core pipelines), the automatic conversion is a useful starting point, but you will likely need to hand-tune the resulting `.mmd` file to get a clean diagram.

!!! note "Work in progress"
    Automatic layout of complex, real-world Nextflow pipelines is an active area of development. The converter handles the format translation well, but the layout engine does not yet route bypass lines (lines that skip over intermediate sections) cleanly. We are working on this separately. For now, the recommended workflow for complex pipelines is to convert, then hand-edit.

## Generating a Nextflow DAG

Nextflow can export its pipeline DAG in mermaid format:

```bash
nextflow run my_pipeline.nf -preview -with-dag dag.mmd
```

The `-preview` flag skips execution and just generates the DAG. The resulting file uses Nextflow's `flowchart TB` mermaid syntax, which nf-metro cannot render directly but can convert.

## Converting and rendering

The recommended workflow is to convert first, review and optionally edit the `.mmd`, then render:

```bash
# Convert Nextflow DAG to nf-metro format
nf-metro convert dag.mmd -o pipeline.mmd --title "My Pipeline"

# Review the .mmd file, then render
nf-metro render pipeline.mmd -o pipeline.svg
```

The converted `.mmd` file is plain text that you can edit in any text editor. Common hand-tuning steps:

- Rename lines or change their colors (`%%metro line:` directives)
- Rename sections (the `subgraph` display names)
- Add entry/exit port hints to control line routing at section boundaries
- Remove or merge sections to simplify the layout
- Add `%%metro grid:` directives to override section placement

See the [Guide](guide.md) for the full `.mmd` format reference.

### Quick one-step render

For simple pipelines where hand-tuning is not needed, you can convert and render in one step:

```bash
nf-metro render dag.mmd -o pipeline.svg --from-nextflow --title "My Pipeline"
```

## What works well

The converter handles these patterns cleanly:

- **Linear pipelines** with no subworkflows (all processes in a single section)
- **Pipelines with a few subworkflows** that form a simple chain (preprocessing, alignment, variant calling, reporting)
- **Diamond patterns** where two processes fan out from the same source and reconverge (e.g., GATK and DeepVariant both feeding BCFtools)

## What needs hand-tuning

Real-world Nextflow pipelines often have topologies that the automatic converter and layout engine cannot yet handle cleanly:

- **Many cross-section connections** - QC processes that collect metrics from every stage create bypass lines that visually cross through intermediate sections. The layout engine does not yet route these lines around the sections they skip.
- **Deeply nested subworkflows** - The converter flattens subworkflow nesting to a single level of sections. Complex nesting may lose meaningful groupings.
- **Pipelines with many parallel branches** - Pipelines with more than 3-4 parallel analysis paths can produce cluttered diagrams that benefit from manual section placement via `%%metro grid:` directives.

For these cases, use the two-step workflow: convert first, then edit the `.mmd` to simplify the topology before rendering.

## How the converter works

The converter strips Nextflow's channel and operator nodes (keeping only processes), reconnects edges through the removed nodes, maps subworkflows to sections, and assigns colored metro lines based on the graph structure. Process names are cleaned up from `SCREAMING_SNAKE_CASE` to `Title Case` and long names are abbreviated.
