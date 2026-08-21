# Invariant tests and runtime validators

Steps 4 and 5 in full.

## Step 4: write the invariant test first


### First, check for a pre-existing regression lock

Some issues arrive with their regression infra already built (a fixture, a
gallery row, and a `strict=True` xfail naming the issue). Grep before writing
anything:

```bash
grep -rn "#<N>" tests/ scripts/gallery.yaml examples/topologies/
```

Any hit, or any temptation to add an xfail, means read
[`regression-locks.md`](regression-locks.md) first: an
existing strict-xfail *is* your failing test, and xfail is never a way to defer
an incomplete fix. No hit is the common case - proceed below.

Brief the single writer to do the following before any production code change:

1. Write a test that encodes the invariant the bug violates (e.g. "no two
   stations share a grid cell", "trunk centre is symmetric about the fan
   midpoint"). Place it under `tests/`, ideally extending the layout
   invariants suite.
2. **Parametrise the test over multiple fixtures**, not a single `.mmd`.
   The existing `test_layout_invariants.py` historically over-relies on
   `tests/fixtures/da_pipeline.mmd`; new invariants should be exercised against
   several gallery fixtures so they generalise. This applies to geometry
   invariants. A class (c) structural defect has no fixture corpus to
   parametrise over - one focused test at the named call site is the correct
   lock, and padding it with unrelated fixtures adds cost, not coverage.
3. Run the test and capture that it **fails on `main`**. If it passes, rewrite
   it because it does not encode the bug.
4. Now write the fix.
5. Re-run the test and capture that it passes.

This guarantees the test is meaningful (it caught the bug) and the fix is
meaningful (the test now passes because of the fix, not coincidence).


## Step 5: add a runtime validator


**This step is conditional.** Where the invariant is about layout properties
that could regress silently (overlap, off-grid placement, asymmetry, etc.),
require the writer to add a `_guard_*` function and wire it into
`compute_layout`'s validate block. Where there is no layout property to guard -
a class (c) structural defect, a CLI or docs change - **skip it outright**: the
Step 4 test is the regression lock, and a validator with nothing geometric to
assert is noise. Say in one line that you skipped it and why.

**Know what a validator protects.** `engine.py` sets `_VALIDATE_DEFAULT = False`,
so `compute_layout` skips the validate block in an ordinary render: the guard
fires under tests and explicit `validate=True`, not for a user rendering a map.
It is a test-time invariant, not a runtime abort, and the Step 4 test is what
holds the line.

Validators must **fail loudly** - raise with a clear, contextual error
message. Silent warnings or `print()`s are not acceptable; they get
ignored. The runtime check protects future changes; the unit test pins the
current behaviour.
