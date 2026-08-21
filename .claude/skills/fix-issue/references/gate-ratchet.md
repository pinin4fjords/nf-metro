# Generated-artifact gates

Read this when the change touched `layout/routing/` or added a topology fixture.

Assign gate interpretation to a MID-tier read-only routing specialist. It
classifies each failure and returns the required reconciliation. Only the sole
writer may change fixtures, baselines, sidecars, generated docs, or routing
code.

## The gate-coverage ratchet

Adding, removing, or rewriting an `if`/`while` in a `layout/routing/` module - or
adding a topology fixture that closes a gap - can red one of the three ratchet
tests in `tests/test_routing_gate_coverage.py`. These are **not** flaky; each
names a specific reconciliation you owe in this same PR. Do not silence them by
hand-editing the baseline or the generated matrix doc, and do not delete a
triage entry just to make a test pass.

- `test_no_new_un_exercised_routing_gate_arm` - your change added a gate with an
  un-exercised arm. Either author a fixture that hits both arms, or - if the arm
  is genuinely unreachable - confirm that and regenerate the baseline to
  acknowledge it.
- `test_gate_coverage_baseline_in_sync` - your change closed a gap or removed a
  gate the baseline still lists. Regenerate the baseline.
- `test_triage_sidecar_references_open_gaps` - you edited a gate's condition text
  or removed it, so its entry in `tests/data/routing_gate_triage.json` now names
  a non-gap. Prune (or re-key) that entry.

After the specialist classifies the failure, serialize any fixture, sidecar,
baseline, or doc change to the sole writer. Only that writer runs the mutating
regeneration under the pinned interpreter and commits the result:

```bash
PYTHONPATH=src python scripts/routing_gate_coverage.py --write   # matrix doc + baseline
```

**Gotcha:** the arc model is CPython-version-specific, so these tests **skip**
off the pinned `BASELINE_PYTHON` (3.11). If your fix env is a different Python
you will not see the failure locally - it surfaces only in CI. When in doubt,
regenerate under 3.11. The full methodology (the four verdicts, why these tests
exist, the phantom-arc trap) is in
[`docs/dev/routing_gate_triage.md`](../../../../docs/dev/routing_gate_triage.md);
for a dedicated triage campaign use the `nf-metro-gate-triage` skill.

**`PYTHONPATH=src` is required on all three.** The prescribed env installs no
`nf_metro`, so without it the first two raise `ModuleNotFoundError` and the third
reports 27 subprocess errors that look like gate failures - which is exactly what
provokes the baseline hand-editing this file forbids.

The verifier reruns
`PYTHONPATH=src python -m pytest tests/test_routing_gate_coverage.py -p no:cacheprovider` on the
candidate SHA and confirms the worktree remains unchanged.

## New topology fixture: regenerate the guard-trace golden

Every fixture under `examples/topologies/` carries a committed guard-trace golden
at `tests/data/guard_golden/examples/topologies/<stem>.json` (the ordered list of
which guard fired at which stage). A **new** fixture has no golden yet, so
`tests/test_guard_registry_golden.py` reds with "`<stem>.mmd` absent from the
golden baseline". This is a **full-corpus** gate: targeted fix tests do not cover
it, so it is one of the few cases where a local full-corpus run earns its cost.
The sole writer regenerates and commits the golden:

```bash
NF_METRO_REGEN_GUARD_GOLDEN=1 PYTHONPATH=src python tests/test_guard_registry_golden.py
```

Before committing, the writer checks that only the new fixture's `.json` changed
and reverts unrelated generated changes. Some threshold-sensitive goldens differ
by architecture; regenerate genuine architecture-sensitive traces on Linux
x86_64. The verifier runs the same test without the regeneration environment
variable against the candidate SHA, inspects the committed diff, and confirms no
worktree change.

## Adding or re-tiering a routing check

A new `check_*` in `CHECK_REGISTRY`, or a change to an existing `GuardSpec`'s
`tier`, rewrites guard traces across the whole corpus rather than one fixture's.
Decide the tier before writing the check, regenerate under 3.11, and inspect the
diff's shape: hundreds of changed goldens is expected for a broad tier and a
signal you picked the wrong one for a narrow invariant.

A new topology fixture therefore owes **three** committed artifacts, not one: the
`.mmd`, its `GALLERY_ENTRIES`/`gallery.yaml` row (so the render-diff sees it),
and this guard-trace golden.
