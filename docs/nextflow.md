# Importing from Nextflow

nf-metro can convert Nextflow's built-in DAG output into a metro map. This works best for simple pipelines with a handful of subworkflows. For complex pipelines like those in nf-core, direct conversion is unlikely to produce a good diagram - you will need to hand-write or heavily edit the `.mmd` file. Improving this is an active area of development.

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

## How the converter works

The converter strips Nextflow's channel and operator nodes (keeping only processes), reconnects edges through the removed nodes, maps subworkflows to sections, and assigns colored metro lines based on the graph structure. Process names are cleaned up from `SCREAMING_SNAKE_CASE` to `Title Case` and long names are abbreviated.
