# Reproducibility evidence

The baseline was generated twice on 2026-07-31 from the same clean engine source and environment. The first run wrote the committed `baseline/` directory. The second wrote `/tmp/nf-metro-1661-second-run`.

Both runs used:

```bash
PYTHONPATH=src python scripts/first_render_benchmark.py run benchmarks/first-render-2026 --output <output-directory> --allow-pending-human
```

`diff -qr` reported no differences between the two output trees. The corpus verifier, 11 synthetic contract tests, `ruff check`, and `ruff format --check` also passed before the review commit.

The human rubric correction to `minor_polish_only` does not change machine artifacts. No human verdict file existed during either run.
