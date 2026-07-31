# Chronological source census

The census audited repository history, issue bodies, issue comments, and source-bearing pull request commits through 2026-07-31. The repository began in February 2026, so there is no pre-2026 project submission corpus to place in the holdout. Synthetic development fixtures are not population members.

## Included canonical pipelines

The ordered manifest contains 16 canonical slugs: 15 exact sources and one derived source. Exact sources include the early issue-only submissions for nf-core/createtaxdb (#51), nf-core/taxprofiler (#96), nf-core/phaseimpute (#169), nf-core/variantprioritization (#173), sanger-tol/genomeassembly (#174), nf-core/funcprofiler (#248), and sacgf/valor (#484). It also includes first source-bearing commits for nf-core/rnaseq, nf-core/epitopeprediction, nf-core/hlatyping, nf-core/variantbenchmarking, nf-core/differentialabundance, nf-core/sarek, and nf-core/seqinspector, plus the exact nf-core/reportho issue comment.

The sanger-tol/genomeassembly issue source was cross-checked against the pipeline repository. Its first repository version, commit `f9785e5a1208a447b471490968e6525c6938ccac`, is later and adds an organellar-assembly section, so it is not substituted for the earlier issue source.

## Grouped derivatives

- Later rnaseq maps, manual grids, and topology reductions are grouped under nf-core/rnaseq.
- The expanded and auto-layout variantbenchmarking maps are grouped under nf-core/variantbenchmarking.
- `funcprofiler.mmd` and later `funcprofiler_upstream.mmd` copies are grouped under nf-core/funcprofiler.
- `genomeassembly_staggered.mmd` and later issue revisions are grouped under sanger-tol/genomeassembly.
- `longread_variant_calling.mmd` is an anonymised derivative of issue #484 and is grouped under sacgf/valor.
- The seqinspector source was first committed as `single_row_rowspan_neighbor.mmd`, then renamed.

## Derived or unavailable

- nf-core/riboseq: the original external branch or authoring-session first-render source could not be recovered. Issue #1293 has a later full bug reproduction, and issue #1421 has a later tuned gallery candidate. The #1421 source is frozen as `derived` and excluded from the primary denominator. No exact riboseq result is claimed.

## Excluded non-population material

- Generic or synthetic examples such as `simple_pipeline`, `variant_calling`, `genomic_pipeline`, guide maps, topology fixtures, and stress maps do not identify a real canonical pipeline slug.
- `example-org/rnaseq` sources are explicit synthetic reproductions.
- `nf-core/genomeassembler` in issues #1630 and #1631 is a synthetic regression source, not an identifiable real pipeline repository.
- Titles such as `Top-Down Multi-Omics`, `Multi-Omics Integration`, and `Long-read Methylation & Variant Atlas` are synthetic scenario names rather than canonical pipeline slugs.
- Later tuned copies, bug minimisations, and gallery variants are not additional population members after grouping by canonical slug.
