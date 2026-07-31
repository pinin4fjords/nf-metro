# First-render robustness baseline

This directory freezes the chronological baseline required by issue #1661. It is an offline holdout, not a gallery. Ordinary `pytest` collection does not load its real pipeline sources.

## Population and census

The population is every identifiable first nf-metro source for a real named pipeline submitted to the nf-metro project from 2026-01-01 through 2026-07-31, inclusive. Sources are grouped by canonical pipeline slug. The earliest valid source is frozen even when it crashes or renders badly.

`manifest.json` is the ordered machine-readable population. Each case has a source bundle and `provenance.json`. [CENSUS.md](CENSUS.md) records inclusion and exclusion decisions. Exact cases alone form the primary denominator. Derived and unavailable sources remain visible but separate.

## No-tuning protocol

- Use the source bytes and source-relative assets in each case directory.
- Use engine commit `33d8e6ff50bd1a59307c0356878b2393a441f5ef` and the source's own directives.
- Do not add, remove, or change layout directives, graph statements, or labels.
- Do not retry with case-specific flags after a failure.
- Run each case in a fresh process with `PYTHONHASHSEED` set to `0`, `1`, `2`, and `43`.
- Preserve stage failures and mark downstream work `not_run`.
- Keep machine findings separate from human verdicts.
- Count only exact, non-deviating cases in the four final rates.

From the repository root, the one offline command is:

```bash
PYTHONPATH=src python scripts/first_render_benchmark.py run benchmarks/first-render-2026 --output benchmarks/first-render-2026/baseline --allow-pending-human
```

Remove `--allow-pending-human` after the review records exist. That final run validates human records and calculates crash-free, strict-invariant-pass, accepted-without-correction, and major-or-unusable rates.

Validate source, asset, provenance, and schema integrity without rendering:

```bash
PYTHONPATH=src python scripts/first_render_benchmark.py verify benchmarks/first-render-2026 --allow-pending-human
```

## Outputs

Every seed has `machine.json`. A successful seed also produces settled geometry, RenderPlan, and SVG artifacts. When all seeds are byte-identical, those larger artifacts are retained under `seed-0` only. If any seed differs, the runner retains every differing seed output without normalization.

The committed `baseline/report.json` aggregates machine status and seed comparisons. Rates remain `null` until human review is complete. Follow [HUMAN_REVIEW.md](HUMAN_REVIEW.md) to record verdicts without inferring them from machine checks.
