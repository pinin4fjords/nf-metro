---
name: fix-issue
description: Coordinator-led end-to-end workflow for fixing GitHub issues on nf-metro with diagnostic rigor and context-light delegation. Use when the user references a GitHub issue (by number, URL, or description) and wants it fixed. Routes diagnosis, implementation, testing, visual judgment, review, and adjacent fallout to scoped workers while the coordinator retains authority, integration gates, Git/PR mutations, and reporting. Preserves invariant-test-first implementation, runtime validators, evidence-cited verification, /simplify, full-repo lint, CI render review, additive-only history, origin verification, and explicit merge authorisation. Supports autonomous / net-negative requests without inventing merge or issue-closure authority. Trigger on phrases like "fix issue #N", "address #N", "work on issue N", or any request to fix a bug or implement a feature that references an issue. For shepherding a chain of already-existing PRs back to main, see `pr-chain-vet` instead.
---

# Fix Issue

Structured workflow for fixing nf-metro GitHub issues in an isolated worktree.
Emphasises diagnostic-first investigation, invariant tests before code, and
additive-only PR hygiene so a fix never silently regresses the gallery.

**Communication:** keep status updates terse and lead any explanation of a
mechanism or a render with one plain-English sentence before the code or
coordinates. Prefer a narrow table to a wide one. When asked to "explain
simply" or for "less words", cut - don't re-expand.

**Conventions** (substitute if your setup differs):
- Local nf-metro checkout: `~/projects/nf-metro`
- Issues + PRs target the canonical upstream `seqeralabs/nf-metro`. If
  you're working from a fork, resolve the owner with
  `gh repo view --json owner -q .owner.login`.
- micromamba: `/opt/homebrew/bin/micromamba` (macOS Apple Silicon codesign
  workaround). On other platforms, just `micromamba` if it's on PATH.

## Primary invariant: coordinate, delegate, verify independently

Keep the coordinator clean and context-light. It owns user communication and
authority, a compact task/evidence ledger, worker routing, integration gates,
deterministic Git integration and remote mutations, and final reporting. It
does **not** do substantive diagnosis, implementation, domain assessment,
visual judgment, `/simplify`, gate interpretation, or code review. Assign those tasks to workers
with proportionate capability. Choose a model or reasoning level only when the
available worker-launch mechanism exposes that choice and the task benefits from it;
do not hardcode provider-specific names or assume unavailable options.

Maintain a compact ledger containing: issue and authority state; worktree,
branch, base, and writer; current hypothesis and evidence links; worker
assignments and verdicts; changed files and commits; commands and outcomes;
I/N/D visual classifications; fallout; blockers; CI/PR state; next gate. Keep
deep context, long test output, render analysis, and review detail in worker
handoffs or artifacts rather than replaying it into the coordinator.

### Worker brief and handoff contract

Before every assignment, state:

- role and objective; capability/model/reasoning choice only when useful and
  explicitly available;
- read/write authority and exact worktree plus file scope;
- inputs and artifact locations;
- required output/evidence schema and acceptance bar;
- dependencies and stop/escalation conditions.

Allow exactly one writer in each worktree. Give concurrent writers separate
worktrees and non-overlapping write scopes; otherwise serialize them. Keep
diagnostic, verifier, visual-review, and code-review roles read-only and
independent of the writer. Read-only workers never persist tracked, untracked,
or ignored worktree changes; place their logs, caches, and generated evidence
outside the worktree. Readers run concurrently only against a frozen commit SHA or
snapshot, never a live worktree during an active writer assignment. The
coordinator categorically owns pushes, issue and PR edits, merges, retargeting,
and cleanup. User authority determines whether the coordinator acts; it never
transfers that ownership to a worker.

Use one candidate sequence throughout: the sole writer makes local candidate
commit(s), runs mutation-capable hooks or generators, and hands off the exact
SHA. Independent read-only workers verify and review that SHA without changing
it. If fixes are required, serialize them back to the writer and verify the new
SHA. Only the coordinator pushes the accepted SHA and performs remote changes.

Require every worker to return:

1. scope completed;
2. files changed and candidate commit SHA, or an explicit no-change result;
3. exact commands and outcomes;
4. before/after evidence;
5. risks and blockers;
6. acceptance verdict: pass, fail, or blocked with the precise escalation.

A blocked handoff is valid when it satisfies this schema. Re-brief from its
evidence, route a different capability, or escalate to the user when authority
or a material product decision is missing. Never demand an unbounded worker
loop or treat diagnosis as implementation.

### Aggregate-progress review

For long-running work, send the compact ledger plus raw evidence to a fresh,
read-only, high-judgment reviewer after diagnosis, after the first complete
implementation/verification loop, and before the final PR-ready gate. Also do
this after two repeated blocks, material scope growth, conflicting verdicts,
acceptance-bar changes, or multiple active worktrees. For steady work, repeat
after a small cluster of worker handoffs rather than relying on a brittle
time-only timer. Record the review verdict and revise later worker briefs,
scope, or gates from its findings. The coordinator must not substitute its own
substantive review.

## Scope discipline: fix the fallout, don't defer it

Resolve bounded fallout surfaced by diagnosis, implementation, `/simplify`,
lint, review, or CI in the current run. Filing and deferring is the exception.

- A different subsystem, separate worktree, or unfamiliar code is not by itself
  a reason to defer.
- Route each fallout item through the same worker protocol as the primary fix.
  Run independent read-only tasks concurrently. Give any fallout writer its
  own worktree unless the primary writer has finished and the assignment is
  explicitly serialized in the primary worktree.
- Keep a coherent fallout fix in the primary PR; use a sibling PR when that is
  more reviewable. Only the coordinator creates or edits the sibling PR.
- Reroute bounded blocks. Return a structured blocker when completion requires
  missing authority, unavailable capability, external-state change, a material
  user decision, or a genuinely multi-session program. Do not disguise the
  blocker with an xfail or child issue.
- This is not licence for scope creep into features the user didn't ask
  about - it's about not walking away from problems the *current* work
  surfaced.

This applies equally to second findings, gate-coverage gaps, `/simplify`, and
review findings.

## Optional mode: autonomous / net-negative

Activate this mode when the user signals it - "no deferrals", "no giving up",
"drive it to conclusion", "net-negative issues", "work overnight", "I'll leave
you to it", or an explicit "push a complete fix and address all the fallout".
It does not relax the diagnostic or verification rigour of the steps below; it
changes two defaults.

**1. Tighten the deferral bar.** Drive coupled defects and incident fallout to
a complete, reviewable PR in the current run whenever they remain bounded.
Brief each worker with the real acceptance bar: the target render is clean,
tests and validators pass, and every corpus delta receives independent I/N/D
classification. Require byte identity only where no visual delta is intended;
independently accepted I/N deltas are valid evidence. Accept structured blocked
handoffs, then re-route or escalate rather than forcing an unbounded loop.

**2. Track net-negative progress without inventing authority.** Do not open a
child issue merely to defer bounded fallout. Track issues resolved by merged
PRs, reviewable PRs awaiting user authority, and any newly opened issues. Fewer
open issues is the goal, but lack of merge authority or a required user
decision is a legitimate stop. Never claim issue closure before the relevant
PR merges, and never merge only to satisfy the arithmetic.

**Bounded to fallout, not a frontier.** Resolve defects the current work
surfaces. Stop and return a structured blocker for a subsystem rewrite,
sweeping cross-cutting refactor, unavailable capability, external-state block,
or required user decision. Do not recurse into an open-ended program.

**Still in force in this mode - do not over-read "autonomous":**
- Never merge without explicit per-PR authorisation from the user. Autonomy is
  about resolving and pushing *complete* work, not self-authorising merges.
  "Drive to conclusion" means "get each bounded PR green and reviewable, or
  return its structured blocker"; merge only what the user okays, per-PR.
  Pre-authorisation to *work* is not pre-authorisation to *merge*.
- Every step's rigour below still applies to every PR, including the fallout
  ones: diagnostic-first, invariant-test-first, `/simplify`, evidence-cited
  verification, additive-only pushes, origin check, standalone issue bodies.
- Assign an independent verifier to reproduce before/after evidence, test the
  fix, and inspect the actual render. Assign a separate visual reviewer for
  judgment calls. The coordinator may check paths, exit codes, hashes, OIDs,
  and handoff completeness, but must not duplicate substantive verification.
- Report every open-at-end item with its blocker: awaiting merge authority,
  awaiting a user decision, or a genuinely large program. Do not conceal it or
  call a reviewable but unmerged PR "closed".

## Step 1: Understand the Issue

```bash
gh issue view <N> --repo seqeralabs/nf-metro
```

Have a read-only investigator assess the issue against current repository and
remote evidence, then return the problem statement, initial scope, unknowns,
and proposed diagnostic brief. The coordinator records the compact result and
summarizes it to the user. Wait for confirmation before proceeding unless the
user has pre-authorised autonomous work; never infer merge or issue-edit
authority from that permission.

### Issue hygiene

Every issue is run through *this skill* fresh in a later session, so the
**issue body must be standalone and self-contained**. Have the relevant worker
return concise body-ready facts when it learns a cause, repro, or constraint.
Only the coordinator edits the issue body, and only with authority. Do not
scatter facts across comments or retain superseded approaches. Route a
separable defect through "Scope discipline" rather than filing a child issue
and walking away. File only when it is a multi-session undertaking in its own
right, the user has authorised the write, and the new body stands alone.

## Step 2: Worktree + Environment Setup

```bash
# Worktree (always off latest origin/main, never stale local main)
cd ~/projects/nf-metro
git fetch origin main
git worktree add /tmp/nf-metro-fix-<N> -b fix/<N>-<slug> origin/main
```

All repository-changing work for the primary fix happens inside
`/tmp/nf-metro-fix-<N>`. The coordinator performs this deterministic setup,
records the exact base SHA, and assigns one writer. Read-only workers may
inspect only a frozen SHA or snapshot, not the live worktree while the writer
is active. Do not allow a second writer until the first hands off and the
coordinator serializes the next assignment.

### Environment: reuse one persistent env, don't create one per issue

nf-metro is pure Python; the deps (`cairo`, drawsvg, networkx, pillow,
cairosvg, pytest, ruff, mypy, `types-networkx`) change rarely. Creating a
fresh `micromamba` env per issue re-solves and re-downloads all of that
every session for no benefit. Keep **one** long-lived deps env and point it
at the worktree's code per-command:

```bash
# One-time, reused across all issues (skip if it already exists):
ulimit -n 1000000 && export CONDA_OVERRIDE_OSX=15.0 && /opt/homebrew/bin/micromamba create -n nf-metro-dev python=3.11 cairo -y
source ~/.local/bin/mm-activate nf-metro-dev
pip install "drawsvg" "networkx" "pillow" "cairosvg" "pytest" "pytest-xdist" "ruff" "mypy" "types-networkx" "click"
# Refresh this env only when pyproject deps actually change.
```

Then run the worktree's code by prepending its `src/` to `PYTHONPATH` on
each command - **do not** `pip install -e` the worktree into this env:

```bash
source ~/.local/bin/mm-activate nf-metro-dev
cd /tmp/nf-metro-fix-<N>
export PYTHONPATH=/tmp/nf-metro-fix-<N>/src
python -m nf_metro render <file.mmd> -o /tmp/out.svg    # runs THIS worktree
python -m pytest -k <selector>
```

**Why per-command `PYTHONPATH`, not editable install:** an editable install
binds one env's `site-packages` to exactly one worktree path, so it collides
the moment you run two worktrees in parallel. `PYTHONPATH` is set per command
and shadows whatever is installed, so any number of parallel worktree
sessions share the single `nf-metro-dev` env with zero cross-talk. (If you
genuinely want an isolated editable install for one worktree, dedicate a
*separate* env to it - never editable-install a shared env against a
worktree.)

**Commit hooks** need the tools on `PATH` in the same Bash call: the repo
uses `prek` (config `prek.toml`, not `pre-commit`), whose `mypy` hook is
`language: system` and so needs `mypy` on `PATH`. Shell state does not
persist between Bash calls, so run the commit as one call with the env
activated: `source ~/.local/bin/mm-activate nf-metro-dev && cd <worktree> &&
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit ...`.

## Step 3: Diagnostic Before Fix

Assign a read-only diagnostic worker before the writer changes code. **Do not
propose fixes from hypotheses.** Require the worker to reproduce the symptom in
numbers:

1. Render the affected example(s) on the current `main` (the before-state).
2. Inspect the rendered SVG and actual coordinates or element attributes.
3. Restate the bug as "element X has property P=<observed>, expected
   P=<target>" - a concrete numeric or structural claim. Return blocked if the
   worker cannot state it yet; diagnosis must continue before implementation.

Only after the symptom is pinned down to specific numbers may the diagnostic
worker reason about which layout pass or function produced them. Require an
independent diagnostic or aggregate reviewer to challenge any high-impact
domain classification before implementation.

### Check your premise against current `origin/main` first

Diagnose against latest remote, not a stale tree. The coordinator fetches
`origin/main`; the diagnostic worker confirms the bug **still reproduces on
that exact SHA** before reasoning about a cause -
a sibling PR may already have fixed it or changed the very code you're reading.
If the user says something is already addressed, re-fetch and look again before
disagreeing; "I'm looking at outdated code" is a recurring wrong turn. If a
related PR merges mid-session, first require the writer to hand off a clean,
committed candidate. The coordinator may then serialize a base-merge assignment
to that sole writer. Keep conflicts and required edits with the writer. Assign
re-diagnosis on the resulting candidate SHA.

### Classify: authoring mistake or engine bug?

Require the diagnostic worker to decide which of two things it is looking at:

- **(a) An mmd authoring mistake** - the `.mmd` misdescribes the pipeline
  (wrong line on a station, a missing edge, a bad directive). The fix *is* to
  edit the input. `probe_layout.py` labels many of these ("authoring
  mistakes vs engine bugs"); `nf-metro explain` shows the rule each inferred
  decision followed.
- **(b) An engine bug on correct mmd** - the input faithfully describes the
  pipeline and the *engine* lays it out badly. The fix goes in `src/`
  (layout / routing / parser). The reproducing `.mmd` stays untouched.

Record which one it is, in numbers, before briefing the writer.

### Once it's an engine bug, the reproducer is frozen evidence

**Never "fix" an engine bug by editing the input to dodge the bad layout.**
Do not trim labels, remove or reorder stations or lines, split sections, or add
directives to avoid the bad path. This applies to existing and newly authored
fixtures. Legitimate input edits are a faithful new reproducer or correction of
the diagnosed authoring mistake. Treat any additional bad render as fallout.
Step 9 narrowing must gate code on a structural precondition, never reword the
input. If the writer proposes an input workaround during an engine fix, stop
and route the evidence back through diagnosis.

### Diagnostic tooling

The repo bundles two scripts that do exactly this render-and-read-the-numbers
work, usable for **any** layout issue regardless of how it was reported:

```bash
# Validator/crash/guard verdict: parse -> layout -> validate -> route, with
# findings split into authoring mistakes vs engine bugs.
python .claude/skills/nf-metro-stress-render/scripts/probe_layout.py <file.mmd> --json
# Per-section station coordinates, flagging stations off their section trunk,
# off-track in/outputs far from their consumer, and oversized inter-row gaps.
python .claude/skills/nf-metro-stress-render/scripts/inspect_layout.py <file.mmd>
```

Plus `nf-metro explain <file.mmd>` (the rule behind each inferred layout
decision) and `nf-metro info --json` (the structural model). These are
conveniences, not requirements - any way you pin the bug to numbers is fine.

If the issue happens to have been filed by the `nf-metro-stress-render` skill,
it carries a correct-by-construction repro `.mmd` in a `<details>` fold in the
issue body - start from that rather than re-deriving one. Most issues won't have
this; otherwise assign the diagnostic worker to build a faithful reproducer.

## Step 4: Write the Invariant Test FIRST

### First, check for an existing regression lock

Most issues arrive bare, so brief the writer to add the failing test (skip to
the numbered steps below). But some - notably those filed by the
`nf-metro-stress-render` skill - arrive with their regression infra **already
in place**: a fixture in `examples/topologies/`, a `GALLERY_ENTRIES` row in
`scripts/build_gallery.py`, and a `strict=True` xfail test referencing the issue
number. Grep before you write anything:

```bash
grep -rn "#<N>" tests/ scripts/build_gallery.py examples/topologies/
```

- **If a strict-xfail lock exists**, that *is* your failing test - don't write a
  duplicate, and don't re-add the fixture or gallery entry. Confirm it xfails on
  the current tree (it documents the live defect).
- **Completing the fix flips that strict-xfail to XPASS, which reds CI** - that
  is the signal the bug is actually fixed. Finish by **removing the `xfail`
  marker** so the now-passing assertion becomes a permanent positive guard.
  (Deleting the whole test loses the guard; leaving the marker keeps CI red.)
- **If no lock exists** (the common case), proceed with the steps below.

### xfail is a lock on a known bug, not an escape hatch

Do not add an xfail to hide an incomplete fix. Reroute bounded work; if
authority, capability, external state, or a material decision blocks it,
return the structured blocker without muting the test or filing a child issue
as camouflage. Add a new xfail only when the user explicitly accepts a genuine
multi-session deferral and the marker references its standalone issue.

Brief the single writer to do the following before any production code change:

1. Write a test that encodes the invariant the bug violates (e.g. "no two
   stations share a grid cell", "trunk centre is symmetric about the fan
   midpoint"). Place it under `tests/`, ideally extending the layout
   invariants suite.
2. **Parametrise the test over multiple fixtures**, not a single `.mmd`.
   The existing `test_layout_invariants.py` historically over-relies on
   `da_pipeline.mmd`; new invariants should be exercised against several
   gallery fixtures so they generalise.
3. Run the test and capture that it **fails on `main`**. If it passes, rewrite
   it because it does not encode the bug.
4. Now write the fix.
5. Re-run the test and capture that it passes.

This guarantees the test is meaningful (it caught the bug) and the fix is
meaningful (the test now passes because of the fix, not coincidence).

## Step 5: Add a Runtime Validator

Where the invariant is about layout properties that could regress silently
(overlap, off-grid placement, asymmetry, etc.), require the writer to add a
`_guard_*` function and wire it into `compute_layout`'s validate block.

Validators must **fail loudly** - raise with a clear, contextual error
message. Silent warnings or `print()`s are not acceptable; they get
ignored. The runtime check protects future changes; the unit test pins the
current behaviour.

## Step 6: /simplify Pass

After the fix and tests are passing, assign a fresh read-only worker to invoke
the `simplify` skill on the changed code. It returns findings, proposed edits,
and a pass/fail verdict without writing. Re-brief the worktree's sole writer to
apply accepted suggestions and record them in a **separate** local candidate
commit:

```
refactor: tighten <area> after fix for #<N>
```

Keeping `fix:` and `refactor:` commits separate makes the fix itself easy
to review and easy to revert in isolation if regressions surface.
The writer then hands off the exact candidate SHA, and an independent verifier
checks that SHA before it can be accepted.

**Re-running it later:** `/simplify` is expensive, so don't assign it after
every follow-up commit. Only re-run it on the final aggregate diff if later
steps (narrowing a regression, lint/mypy fixes) added a **substantial** chunk
of new production code the first pass never saw. A couple of small,
already-clean follow-up edits don't warrant a second pass.

## Step 7: Lint and Tests

The sole writer runs all mutation-capable formatting, fixing, regeneration,
and hook commands, resolves their changes, then creates the candidate commit.
Never skip hooks with `--no-verify`. `prek` needs the `nf-core` environment and
stub-complete `mypy`:

```bash
micromamba run -n nf-core prek run --all-files
```

Assign an independent read-only verifier the exact candidate SHA. Require a
clean tree before and after and run only against a frozen checkout or snapshot.
Put every cache, temporary file, and log in an external artifact directory;
disable bytecode and in-worktree caches. Return the command, exit status,
concise failure excerpt, and verdict. Use this environment for every verifier
command below:

```bash
export VERIFY_ARTIFACT_DIR=/tmp/nf-metro-verify-<N>-<CANDIDATE_SHA>
mkdir -p "$VERIFY_ARTIFACT_DIR/tmp"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$VERIFY_ARTIFACT_DIR/tmp"
export XDG_CACHE_HOME="$VERIFY_ARTIFACT_DIR/xdg-cache"
test "$(git rev-parse HEAD)" = <CANDIDATE_SHA>
test -z "$(git status --porcelain)"
ruff check --no-cache src/ tests/
ruff format --check --no-cache src/ tests/
mypy --cache-dir="$VERIFY_ARTIFACT_DIR/mypy"
PYTHONPATH=src python -m pytest tests/test_layout_invariants.py -k "<fixture-or-invariant>" -q --no-header -p no:cacheprovider --basetemp="$VERIFY_ARTIFACT_DIR/pytest-tmp"
git diff --exit-code <CANDIDATE_SHA>
test -z "$(git status --porcelain)"
```

Route failures back to the writer, then verify the new SHA. Run the full local
suite only when it earns its cost:

- shared orchestration, parser model, dispatch table, or widely used helper
  changed;
- a targeted pass cannot cover a concrete wider-regression risk;
- explicit admin-merge preparation needs local full-suite confidence.

Otherwise use targeted tests and let CI run the complete matrix.

### Cost discipline (applies throughout)

Layout iteration is where sessions burn tokens and compute. Keep it tight:

- **Reuse the persistent env** (Step 2). Do not `micromamba create` per
  issue - it re-solves the whole dependency set every session for nothing.
- **Default to targeted checks.** Use `--lf`, `-q --no-header -x`, and Python
  3.11 for routing/TB ratchets. Keep full logs outside coordinator context.
- **Have diagnostic workers read coordinates, not rasterize, for non-visual questions.**
  `inspect_layout.py` / `probe_layout.py` print the geometry as cheap text;
  a render -> cairosvg PNG -> open -> image-into-context cycle is far heavier
  and only earns its cost for a genuine *visual* check. "Is station X on the
  trunk?" is a coordinate read, not a screenshot.
- **Poll CI once, in the background.** A single background watch
  (`until gh pr checks <N> ...; done`) pulls you back when checks resolve;
  re-running `gh pr checks` by hand each turn just dumps status into context
  repeatedly.
- **Lean on the CI render-diff for regression review; don't rebuild the
  gallery locally in a loop.** The CI preview (Step 8) is the authoritative
  whole-corpus diff. A local `build_gallery` / render-diff sweep repeated
  many times just duplicates it. Local rendering is for a *single* file's
  quick sanity check.
- **Brief workers to read the big layout files in wide slices and stay oriented.**
  Re-fetching `engine.py` / `fan_bundles.py` / `ordering.py` /
  `routing/*` twenty times over a session is the single largest cache-read
  cost. Read the region once, generously, and keep it in working context.
- **Default `[skip ci]` on work-in-progress pushes** (WIP snapshots, refactor
  passes). Let CI run on the final pre-review push - which this repo needs
  anyway, because the render-diff *is* the visual review. (A commit that
  fixes a known CI failure must re-run CI: no `[skip ci]` on those.)

### If your change touched `layout/routing/`: the gate-coverage ratchet

Assign gate interpretation to a read-only routing specialist. It classifies
each failure and returns the required reconciliation. Only the sole writer may
change fixtures, baselines, sidecars, generated docs, or routing code.

Adding, removing, or rewriting an `if`/`while` in a `layout/routing/`
module - or adding a topology fixture that closes a gap - can red one of
the three ratchet tests in `tests/test_routing_gate_coverage.py`. These
are **not** flaky; each names a specific reconciliation you owe in this
same PR. Do not silence them by hand-editing the baseline or the
generated matrix doc, and do not delete a triage entry just to make a
test pass.

- `test_no_new_un_exercised_routing_gate_arm` - your change added a gate
  with an un-exercised arm. Either author a fixture that hits both arms,
  or - if the arm is genuinely unreachable - confirm that and regenerate
  the baseline to acknowledge it.
- `test_gate_coverage_baseline_in_sync` - your change closed a gap or
  removed a gate the baseline still lists. Regenerate the baseline.
- `test_triage_sidecar_references_open_gaps` - you edited a gate's
  condition text or removed it, so its entry in
  `tests/data/routing_gate_triage.json` now names a non-gap. Prune (or
  re-key) that entry.

After the specialist classifies the failure, serialize any fixture, sidecar,
baseline, or doc change to the sole writer. Only that writer runs the mutating
regeneration under the pinned interpreter and commits the result:

```bash
python scripts/routing_gate_coverage.py --write   # rewrites the matrix doc + baseline
```

**Gotcha:** the arc model is CPython-version-specific, so these tests
**skip** off the pinned `BASELINE_PYTHON` (3.11). If your fix env is a
different Python you will not see the failure locally - it surfaces only
in CI. When in doubt, regenerate under 3.11. The full methodology (the
four verdicts, why these tests exist, the phantom-arc trap) is in
[`docs/dev/routing_gate_triage.md`](../../../docs/dev/routing_gate_triage.md);
for a dedicated triage campaign use the `nf-metro-gate-triage` skill.

The verifier reruns
`python -m pytest tests/test_routing_gate_coverage.py -p no:cacheprovider` on the
candidate SHA and confirms the worktree remains unchanged.

### If you added a topology fixture: regenerate the guard-trace golden

Every fixture under `examples/topologies/` carries a committed guard-trace
golden at `tests/data/guard_golden/examples/topologies/<stem>.json` (the
ordered list of which guard fired at which stage). A **new** fixture has no
golden yet, so `tests/test_guard_registry_golden.py` reds with
"`<stem>.mmd` absent from the golden baseline". This is a **full-corpus**
gate: targeted fix tests do not cover it. The sole writer regenerates and
commits the golden:

```bash
NF_METRO_REGEN_GUARD_GOLDEN=1 python tests/test_guard_registry_golden.py
```

Before committing, the writer checks that only the new fixture's `.json`
changed and reverts unrelated generated changes. Some threshold-sensitive
goldens differ by architecture; regenerate genuine architecture-sensitive
traces on Linux x86_64. The verifier runs the same test without the regeneration
environment variable against the candidate SHA, inspects the committed diff,
and confirms no worktree change.

A new topology fixture therefore owes **three** committed artifacts, not
one: the `.mmd`, its `GALLERY_ENTRIES`/`gallery.yaml` row (Step 8, so the
render-diff sees it), and this guard-trace golden.

## Step 8: Visual Review via Render Preview

### Primary method: CI render preview (authoritative)

The coordinator pushes the branch and creates a draft PR. No worker performs
these remote mutations. The CI workflow
(`.github/workflows/pr-renders.yml`) automatically renders all gallery
examples on both the PR branch and base, generates a before/after visual
diff page, and posts a sticky comment on the PR with the preview link:

```
https://seqeralabs.github.io/nf-metro/_pr/<PR_NUMBER>/
```

### Render-preview verdict gating

Assign a fresh read-only visual reviewer the preview, target before/after
evidence, and acceptance criteria. Require it to inspect every changed example,
classify deltas I/N/D, identify uncertainty, and return an acceptance verdict.
Do not seed it with the writer's preferred interpretation.

The sticky comment ends in a verdict line. Gate the next step on it:

- **"No visual changes detected"** -> a clean result, but **not** a
  licence to merge. Report the verdict and wait for the user to say
  merge. There is no standing auto-merge authorisation.
- **"Ready for review"** (or any wording indicating visual deltas exist)
  -> gate on the independent visual verdict. Re-brief the writer for every D;
  surface accepted I/N deltas and unresolved uncertainty to the user with one
  short evidence-based line per affected gallery example.

Never merge without explicit per-PR user authority. Prior admin merges are not
standing consent. Leave branch deletion to Step 12 so dependent PRs can be
retargeted safely.

### When the user *does* authorise a merge

- **"Merge"** authorises one normal merge-commit attempt:
  `gh pr merge <N> --merge`. Never squash. If review, branch
  protection, or up-to-date policy blocks it, stop and return that blocker.
  Do not escalate to `--admin`.
- **"Admin merge"** explicitly authorises
  `gh pr merge <N> --admin --merge`. If CI is not green, first assign a fresh,
  independent read-only worker to use `pinin4fjords:eco-merge` and determine
  whether the sole unverified delta is CI-irrelevant. The coordinator may run
  the admin merge only with both explicit user admin-merge authority and that
  worker's pass verdict. Otherwise return the blocker. Do not update the branch
  or start fresh CI merely to satisfy up-to-date policy; cancel irrelevant
  in-flight runs only as part of the authorised, accepted eco-merge sequence.

### State the evidence for every "it's fixed" claim

Never assert a fix works without an independent verifier naming what proved it.
Every "resolved" /
"this is fixed" / "renders correctly" claim must cite the **specific render
and the concrete numbers** it was checked against - the file, and the
coordinate or element that moved from the observed value to the target value
you wrote down in Step 3. "I believe it's resolved" with no named render is
not a verdict; it invites the reply "which render did you re-assess on?".

Two traps this closes:

- **"Didn't abort" / "the one invariant passes" is not "renders
  correctly".** Removing an abort can merely expose a poor layout the abort
  was masking. After any layout/routing fix, require the verifier to inspect
  the full render (cropping the region as needed) and run `probe_layout` plus
  `inspect_layout` for the whole-layout picture (crossings, port alignment,
  column gaps), not only the targeted invariant.
- **A clean render-diff verdict only covers the gallery corpus.** It says
  nothing about a NEW fixture that isn't in the gallery yet. Put new
  regression fixtures in `scripts/build_gallery.py` (`GALLERY_ENTRIES`), not
  only `examples/topologies/`, so CI's render-diff makes them visible to a
  human. A topologies-only or tests-only fixture is invisible in the PR
  preview.

Do not present a prototype as an improvement before independent review and the
user's judgment where needed. If the render has problems, revise the writer's
brief; do not defend a weak fix.

### Optional: quick local render of a single file

For a fast sanity check of one specific `.mmd` file before pushing, assign a
read-only verifier to run:

```bash
source ~/.local/bin/mm-activate nf-metro-dev
export PYTHONPATH=/tmp/nf-metro-fix-<N>/src
cd /tmp/nf-metro-fix-<N> && python -m nf_metro render <file.mmd> -o /tmp/<name>.svg
python -c "import cairosvg; cairosvg.svg2png(url='/tmp/<name>.svg', write_to='/tmp/<name>.png', scale=2)"
open /tmp/<name>.png
```

Useful for quick iteration but does not replace the full CI gallery
review.

### Optional: local before/after comparison

For a before/after sweep before pushing, assign a verifier to use the tracked
`.claude/skills/render-topologies` skill.

## Step 9: Narrow Over-Applying Fixes

If the render preview shows the fix changed **more than the targeted
example** unexpectedly, do not ship it as-is. Have the independent visual
reviewer classify each affected example as:

- **I** (improvement) - keep
- **N** (neutral) - keep
- **D** (detrimental) - must be narrowed

The bar is "no **meaningful** visual regression", not pixel-identity. A
subtle spacing or coordinate shift that comes with a cleaner, more elegant
implementation is fine (classify it N or I); do not contort the code to
preserve a byte-identical render. Only a genuine degradation is a D.

For each detrimental delta, assign a diagnostic worker to find the
**precondition** that distinguishes the target case (where the fix helps) from
the regressing case (where it hurts). Re-brief the sole writer to gate the fix
on that precondition (e.g. a topology predicate, a config flag, a layout
property test). Assign fresh re-rendering and re-verification before merging.

A fix with an unaddressed D-delta is not PR-ready. Reroute it or return the
structured blocker that prevents correction or classification.

## Step 10: Accept Candidate, Push, Verify Origin

After the writer hands off candidate commit SHA(s) and independent gates pass,
the coordinator confirms `HEAD` equals the accepted SHA and the tree is clean.
Only the coordinator pushes, creates or edits the PR, and performs later remote
mutations. Open the draft PR:

```bash
cd /tmp/nf-metro-fix-<N>
gh pr create --draft --repo seqeralabs/nf-metro --base main --title "<title>" --body "$(cat <<'EOF'
## Summary
<bullets describing the aggregate diff against main, no narrative>

Fixes #<N>

## Test plan
- [ ] pytest passes (including new invariant test)
- [ ] ruff check + ruff format clean on whole repo
- [ ] Runtime validator added (if applicable)
- [ ] Visual review of [render preview](https://seqeralabs.github.io/nf-metro/_pr/<PR_NUMBER>/)
- [ ] Render-preview verdict: <No visual changes | deltas classified I/N>

Generated with Codex
EOF
)"
```

After every `git push`, **verify origin HEAD matches local**:

```bash
gh pr view <PR_NUMBER> --json headRefOid -q .headRefOid
git rev-parse HEAD
```

The two must match. Prior sessions have lost commits to silent push
failures; do not skip this check.

### Additive only - no force-push, ever

The local pre-push hook blocks force-pushes for a reason. To undo
anything, use `git revert <hash>` and push the revert as a new commit.
Never rewrite shared history (no `--force`, no `--force-with-lease`, no
interactive rebase on a pushed branch). This applies even when "it would
be cleaner" - cleanliness is not worth the risk of silently
dropping work.

An ordinary additive (fast-forward) push is **not** blocked by that hook -
only rewrites are. Don't mistake an unrelated push failure for a force-push
block. The coordinator runs an authorised plain push.

### Narrative belongs in the PR description, not in comments

Do not post explanatory comments on the PR walking through what changed,
what was tried, or what was reverted. Edit the PR description instead:

```bash
gh pr edit <PR_NUMBER> --body-file /tmp/pr-body.md
```

The description should be a standalone summary of the current state of
the diff against main - not a chronology of how the PR got there.

If narrative comments already exist, the coordinator may sweep them via the
GraphQL `deleteIssueComment` mutation only with issue/PR edit authority.
**Keep** the CI sticky render-preview comment.

## Step 11: Drive End-to-End

Before declaring readiness, assign a fresh read-only code reviewer the accepted
candidate SHA, aggregate diff, issue, diagnostic evidence, test artifacts, and
visual verdict.
Require an acceptance verdict covering correctness, scope, invariants, safety,
and unresolved fallout. For a long run, combine this gate with the final
aggregate-progress review and revise any later brief from its findings.
Create the PR as a draft for CI and render evidence. Only after the final code
review and aggregate-progress gate pass may the coordinator run
`gh pr ready <N>`.

A successful fix-issue run is not done when `/simplify` or a test worker
returns. It reaches PR-ready completion when:

1. The fix lands in `src/`, not in a doctored reproducer (Step 3), and the
   "it's fixed" claim cites the render + numbers that prove it (Step 8).
2. Commits are pushed.
3. Origin HEAD verified against local.
4. CI is green on the final commit; any failure interpretation came from an
   assigned verifier or domain specialist.
5. Render-preview verdict is captured and gated on per Step 8.
6. PR description is standalone (per Step 10).
7. Independent verification, visual review when applicable, code review, and
   the final acceptance verdict pass.
8. The coordinator marks the draft PR ready only after those gates pass.

Reroute bounded failures until these gates pass. If missing authority,
unavailable capability, external state, or a material user decision prevents
completion, return the structured blocker with the accepted candidate SHA and
remaining gate. Do not claim PR-ready completion.

## Step 12: Post-Merge Cleanup

Once the PR merges, only the coordinator performs cleanup, and only with user
authority. Before deleting anything, require a clean tree, reconcile local
`HEAD`, pushed remote head, and merged PR head, and confirm no unpushed commits.
Stop on any mismatch. Then use this order:

1. **Retarget any child PRs** based on this branch over to `main` (or
   the next-up base) **first**, via `gh pr edit <child> --base main`.
   Confirm every retarget and stop if any fails; branch deletion can auto-close
   a dependent PR.
2. Delete the **remote** branch: `git push origin --delete fix/<N>-<slug>`
   (or via the GitHub UI's auto-delete on merge).
3. Remove the local worktree: `git worktree remove /tmp/nf-metro-fix-<N>`.
4. Delete the reconciled local branch with `git branch -d fix/<N>-<slug>`.
   Use `-D` only after explicit user authority and proof that `-d` rejects a
   fully reconciled branch for a harmless bookkeeping reason.

Leave the shared `nf-metro-dev` env in place - it is reused across issues
(Step 2), so there is nothing per-issue to remove.

Offer this cleanup to the user; only run it after they agree.

For shepherding a whole stacked chain of PRs back into `main` (rather
than a single issue fix), see `pr-chain-vet`.
