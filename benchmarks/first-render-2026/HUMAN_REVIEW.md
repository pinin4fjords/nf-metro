# First-render baseline human review packet

This packet is ready for human adjudication. It contains no inferred or pre-filled verdicts. Review the exact cases in chronological order. The engine is frozen at `33d8e6ff50bd1a59307c0356878b2393a441f5ef`, with default source directives and no per-case layout tuning.

## Rubric

Choose exactly one verdict for each exact case:

- `accepted_without_correction`: publishable without an engine, grid, direction, port, or route change.
- `minor_polish_only`: a local cosmetic adjustment is desired, with no structural grid, direction, port, or routing correction.
- `major_layout_correction_required`: any structural grid, direction, port, bundle, fan, merge, corridor, or routing correction is required.
- `unusable_or_aborting`: rendering aborts or the intended flow cannot be read without redesign.

One reviewer is sufficient. For every verdict except `accepted_without_correction`, also record:

- `semantic_failure_class`: the kind of failure, such as collision, crossing, excessive spacing, broken bundle, or abort.
- `affected_region`: a compact station, section, or route description.
- `semantic_owner`: the subsystem that should own a correction, such as parser, layout, routing, or render.

Set `issue_history_visible` truthfully. Prefer reviewing a successful SVG before opening its machine record or linked issue history. The machine result is evidence, not a human verdict.

## Exact cases

|   # | Pipeline                      | Review artifact                                                                                                           | Machine evidence                                                                 |
| --: | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
|   1 | nf-core/rnaseq                | [render.svg](baseline/cases/nf-core-rnaseq/seed-0/render.svg)                                                             | [machine.json](baseline/cases/nf-core-rnaseq/seed-0/machine.json)                |
|   2 | nf-core/createtaxdb           | No SVG. Resolution raised `builtins.ValueError`: undeclared metro lines `cat` and `gzip`.                                 | [machine.json](baseline/cases/nf-core-createtaxdb/seed-0/machine.json)           |
|   3 | nf-core/taxprofiler           | [render.svg](baseline/cases/nf-core-taxprofiler/seed-0/render.svg)                                                        | [machine.json](baseline/cases/nf-core-taxprofiler/seed-0/machine.json)           |
|   4 | nf-core/epitopeprediction     | [render.svg](baseline/cases/nf-core-epitopeprediction/seed-0/render.svg)                                                  | [machine.json](baseline/cases/nf-core-epitopeprediction/seed-0/machine.json)     |
|   5 | nf-core/hlatyping             | [render.svg](baseline/cases/nf-core-hlatyping/seed-0/render.svg)                                                          | [machine.json](baseline/cases/nf-core-hlatyping/seed-0/machine.json)             |
|   6 | nf-core/variantbenchmarking   | [render.svg](baseline/cases/nf-core-variantbenchmarking/seed-0/render.svg)                                                | [machine.json](baseline/cases/nf-core-variantbenchmarking/seed-0/machine.json)   |
|   7 | nf-core/phaseimpute           | [render.svg](baseline/cases/nf-core-phaseimpute/seed-0/render.svg)                                                        | [machine.json](baseline/cases/nf-core-phaseimpute/seed-0/machine.json)           |
|   8 | nf-core/variantprioritization | [render.svg](baseline/cases/nf-core-variantprioritization/seed-0/render.svg)                                              | [machine.json](baseline/cases/nf-core-variantprioritization/seed-0/machine.json) |
|   9 | sanger-tol/genomeassembly     | [render.svg](baseline/cases/sanger-tol-genomeassembly/seed-0/render.svg)                                                  | [machine.json](baseline/cases/sanger-tol-genomeassembly/seed-0/machine.json)     |
|  10 | nf-core/funcprofiler          | No SVG. Layout raised `PhaseInvariantError`: a foreign fan diagonal strikes the `FMH FunProfiler` and `HUMAnN v4` labels. | [machine.json](baseline/cases/nf-core-funcprofiler/seed-0/machine.json)          |
|  11 | nf-core/differentialabundance | [render.svg](baseline/cases/nf-core-differentialabundance/seed-0/render.svg)                                              | [machine.json](baseline/cases/nf-core-differentialabundance/seed-0/machine.json) |
|  12 | nf-core/sarek                 | No SVG. Layout raised `PhaseInvariantError`: station `vcf_out` lies outside the `reporting` section after Stage 5.3.      | [machine.json](baseline/cases/nf-core-sarek/seed-0/machine.json)                 |
|  13 | sacgf/valor                   | [render.svg](baseline/cases/sacgf-valor/seed-0/render.svg)                                                                | [machine.json](baseline/cases/sacgf-valor/seed-0/machine.json)                   |
|  14 | nf-core/seqinspector          | [render.svg](baseline/cases/nf-core-seqinspector/seed-0/render.svg)                                                       | [machine.json](baseline/cases/nf-core-seqinspector/seed-0/machine.json)          |
|  15 | nf-core/reportho              | No SVG. Layout raised `PhaseInvariantError`: a foreign fan diagonal strikes the `OrthoInspector online` label.            | [machine.json](baseline/cases/nf-core-reportho/seed-0/machine.json)              |

All four hash seeds produced identical stage status, settled geometry digest, RenderPlan digest, and SVG digest for every case. Identical output artifacts are retained once under `seed-0`; every seed keeps its own machine record.

## Recording a verdict

Create `cases/<case-id>/human.json` only after review. Use this shape and replace every placeholder:

```json
{
  "schema_version": 1,
  "reviewers": [
    {
      "reviewer": "reviewer-id",
      "issue_history_visible": false,
      "verdict": "one-of-the-four-rubric-values",
      "semantic_failure_class": null,
      "affected_region": null,
      "semantic_owner": null
    }
  ],
  "adjudicated_verdict": "one-of-the-four-rubric-values",
  "protocol_deviation": false,
  "protocol_deviation_notes": null
}
```

The three semantic fields may be `null` only for `accepted_without_correction`. Do not calculate acceptance rates until all 15 exact cases have valid human records.

## Provenance exclusion

nf-core/riboseq is not part of the primary denominator. Its original external branch or session source could not be recovered. The later tuned source from issue #1421 is frozen as `derived`, and its machine evidence is available at [machine.json](baseline/cases/nf-core-riboseq/seed-0/machine.json). It aborts during layout on the frozen engine. This machine outcome is not a human verdict and is reported separately.
