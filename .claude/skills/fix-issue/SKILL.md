---
name: fix-issue
description: Coordinator-led end-to-end workflow for fixing GitHub issues on nf-metro with diagnostic rigor and context-light delegation. Use when the user references a GitHub issue (by number, URL, or description) and wants it fixed. Routes diagnosis, implementation, testing, visual judgment, review, and adjacent fallout to scoped workers at explicit capability tiers, while the coordinator retains authority, integration gates, Git/PR mutations, and reporting. Supports autonomous / net-negative requests without inventing merge or issue-closure authority. Trigger on phrases like "fix issue #N", "address #N", "work on issue N", or any request to fix a bug or implement a feature that references an issue. For shepherding a chain of already-existing PRs back to main, see `pr-chain-vet` instead.
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

## Reference files - load only when the branch is taken

| File | Load when |
| --- | --- |
| [`references/environment.md`](references/environment.md) | setting up the env, running hooks, briefing a verifier |
| [`references/gate-ratchet.md`](references/gate-ratchet.md) | the change touched `layout/routing/` or added a topology fixture |
| [`references/regression-locks.md`](references/regression-locks.md) | the Step 4 grep finds an existing lock, or you are tempted to add an xfail |
| [`references/merge-and-cleanup.md`](references/merge-and-cleanup.md) | pushing, merging, or cleaning up |
| [`references/autonomous-mode.md`](references/autonomous-mode.md) | the user signalled autonomous / net-negative work |

## Primary invariant: coordinate, delegate, verify independently

Keep the coordinator clean and context-light. It owns user communication and
authority, a compact task/evidence ledger, worker routing, integration gates,
deterministic Git integration and remote mutations, and final reporting. It
does **not** do substantive diagnosis, implementation, domain assessment,
visual judgment, `/simplify`, gate interpretation, or code review.

Two separate levers, do not confuse them:

- **Delegate to protect coordinator context.** Anything that produces bulk
  output - issue bodies, test logs, render analysis, file sweeps - goes to a
  worker even when it is mechanically simple, because bytes that land in the
  coordinator are re-read on every subsequent turn while a worker's are
  discarded at handoff. Delegating a cheap task is cheaper than absorbing it.
- **Choose the tier to protect cost.** Delegation decides *where* work runs;
  the tier decides what it costs. A mechanical worker on the top tier is the
  most expensive thing in this workflow.

The coordinator may still run trivial deterministic assertions itself -
`git rev-parse`, an OID or hash comparison, `git status --porcelain`, an exit
code, handoff-schema completeness - because those are a few bytes and spawning
for them buys nothing. It must not substitute its own *substantive* review.

Maintain a compact ledger containing: issue and authority state; worktree,
branch, base, and writer; current hypothesis and evidence links; worker
assignments, tiers, and verdicts; changed files and commits; commands and
outcomes; I/N/D visual classifications; fallout; blockers; CI/PR state; next
gate. Keep deep context, long test output, render analysis, and review detail
in worker handoffs or artifacts rather than replaying it into the coordinator.

### Worker tiers are explicit, never inherited

**Every worker launch must name its capability tier.** Never omit the
model/capability parameter and let the child inherit the session default. An
unset parameter is not "no decision was made", it is "the session's top tier
got picked because nobody decided". Choose the tier *before* spawning and state
it in one line.

If a worker is found running on an inherited default, that is a mistake to
correct - restart it on the intended tier - not a choice to defend. Do not
rationalise keeping the higher tier because the task turned out to suit harder
judgment in hindsight; that call has to happen at spawn time, explicitly, or
not at all.

The tier is the contract, not the model name. Map it to whatever your harness
exposes:

| Tier | Character of the work | Claude Code | Codex |
| --- | --- | --- | --- |
| **LIGHT** | fixed command blocks, exit codes, greps, mechanical reads and reports, no judgment | `haiku` | `luna` |
| **MID** | bounded reasoning against a stated bar: verification, `/simplify`, gate classification, summarising | `sonnet` | `terra` |
| **HIGH** | open-ended judgment where being wrong wastes the run: diagnosis, visual assessment, final review | `opus` | `sol` |

If a harness exposes no such parameter, say so in one line and proceed.
Substitute local equivalents if the names differ; keep the three-tier shape.

Role tiers are fixed by this table and do not drift upward because a task felt
hard:

| Step | Role | Tier |
| --- | --- | --- |
| 1 | issue investigator | LIGHT |
| 3 | diagnostic worker | HIGH |
| 4-7 | sole writer | HIGH for `layout/`, `routing/`, `parser/`; MID elsewhere |
| 6 | `/simplify` worker | MID |
| 7 | lint/test verifier | LIGHT |
| 7 | routing gate specialist | MID |
| 8 | local render / before-after sweep | LIGHT |
| 8 | visual reviewer | HIGH |
| 8 | eco-merge assessor | MID |
| 9 | per-D-delta diagnostic | HIGH |
| 11 | combined code + aggregate reviewer | HIGH |

A LIGHT worker that returns "blocked, this needs judgment" is a correct
outcome, not a failure. Re-route it up a tier rather than pre-emptively
starting high.

### Worker brief template

Fill this in; do not re-improvise the contract in prose each time.

```
ROLE / TIER: <role> at <LIGHT|MID|HIGH>
OBJECTIVE:   <one sentence>
AUTHORITY:   read-only | sole writer in <worktree>
SCOPE:       <worktree path>, files: <paths or "read anywhere, write nothing">
INPUTS:      <SHA, artifact paths, issue number>
ACCEPTANCE:  <the concrete bar>
STOP IF:     <escalation conditions>
RETURN:      the 6-item handoff schema
```

Every worker returns:

1. scope completed;
2. files changed and candidate commit SHA, or an explicit no-change result;
3. exact commands and outcomes;
4. before/after evidence;
5. risks and blockers;
6. acceptance verdict: pass, fail, or blocked with the precise escalation.

A blocked handoff is valid when it satisfies this schema. Re-brief from its
evidence, route a different tier, or escalate to the user when authority or a
material product decision is missing. Never demand an unbounded worker loop or
treat diagnosis as implementation.

### One writer, independent readers

Allow exactly one writer in each worktree. Give concurrent writers separate
worktrees and non-overlapping write scopes; otherwise serialize them. Keep
diagnostic, verifier, visual-review, and code-review roles read-only and
independent of the writer. Read-only workers never persist tracked, untracked,
or ignored worktree changes; place their logs, caches, and generated evidence
outside the worktree. Readers run concurrently only against a frozen commit SHA
or snapshot, never a live worktree during an active writer assignment. The
coordinator categorically owns pushes, issue and PR edits, merges, retargeting,
and cleanup. User authority determines whether the coordinator acts; it never
transfers that ownership to a worker.

Use one candidate sequence throughout: the sole writer makes local candidate
commit(s), runs mutation-capable hooks or generators, and hands off the exact
SHA. Independent read-only workers verify and review that SHA without changing
it. If fixes are required, serialize them back to the writer and verify the new
SHA. Only the coordinator pushes the accepted SHA and performs remote changes.

### Review gates: two mandatory, the rest on trigger

Independent review is load-bearing, but a reviewer that re-reads raw evidence is
the most expensive spawn in the run. Two mandatory gates, both HIGH:

1. **Post-diagnosis gate.** One reviewer that both challenges the domain
   classification (Step 3: authoring mistake or engine bug, and the numeric
   claim behind it) and reviews aggregate progress. A wrong classification
   wastes the whole run, so this gate pays for itself; a second separate
   challenger does not.
2. **Pre-ready gate.** One reviewer combining the Step 11 code review and the
   final aggregate-progress review. These ask nearly the same question against
   the same diff; run them as one brief.

Run an extra mid-loop aggregate review only on a trigger: two repeated blocks,
material scope growth, conflicting worker verdicts, a changed acceptance bar, or
multiple active worktrees. Send it the compact ledger and *links* to evidence,
not the evidence inline. Record every review verdict and revise later briefs,
scope, or gates from its findings.

## Cost discipline (applies throughout)

Layout iteration is where sessions burn tokens and compute. Keep it tight:

- **Name a tier on every spawn** (above). This is the single largest lever.
- **Lean on CI for the full suite.** Locally, run targeted tests: the new
  invariant test, the affected module, `--lf`, `-q --no-header -x`, Python 3.11
  for the routing/TB ratchets. CI runs the complete matrix on push and that is
  the authoritative full-suite signal. Do not run a local full suite per branch
  or per worker. Reserve it for the three cases in Step 7.
- **Reuse the persistent env.** Do not `micromamba create` per issue - it
  re-solves the whole dependency set every session for nothing. See
  [`references/environment.md`](references/environment.md).
- **Read coordinates, not pixels, for non-visual questions.**
  `inspect_layout.py` / `probe_layout.py` print the geometry as cheap text; a
  render -> cairosvg PNG -> open -> image-into-context cycle is far heavier and
  only earns its cost for a genuine *visual* check. "Is station X on the trunk?"
  is a coordinate read, not a screenshot.
- **Poll CI once, in the background.** A single background watch
  (`until gh pr checks <N> ...; done`) pulls you back when checks resolve;
  re-running `gh pr checks` by hand each turn just dumps status into context
  repeatedly.
- **Lean on the CI render-diff for regression review; don't rebuild the gallery
  locally in a loop.** The CI preview (Step 8) is the authoritative whole-corpus
  diff. A local `build_gallery` / render-diff sweep repeated many times just
  duplicates it. Local rendering is for a *single* file's quick sanity check.
- **Brief workers to read the big layout files in wide slices and stay
  oriented.** Re-fetching `engine.py` / `fan_bundles.py` / `ordering.py` /
  `routing/*` twenty times over a session is the single largest cache-read cost.
  Read the region once, generously, and keep it in working context.
- **Default `[skip ci]` on work-in-progress pushes** (WIP snapshots, refactor
  passes). Let CI run on the final pre-review push - which this repo needs
  anyway, because the render-diff *is* the visual review. (A commit that fixes a
  known CI failure must re-run CI: no `[skip ci]` on those.)

## Scope discipline: fix the fallout, don't defer it

Resolve bounded fallout surfaced by diagnosis, implementation, `/simplify`,
lint, review, or CI in the current run. Filing and deferring is the exception.

- A different subsystem, separate worktree, or unfamiliar code is not by itself
  a reason to defer.
- Route each fallout item through the same worker protocol and tier table as the
  primary fix. Run independent read-only tasks concurrently. Give any fallout
  writer its own worktree unless the primary writer has finished and the
  assignment is explicitly serialized in the primary worktree.
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

## Step 1: Understand the Issue

Assign a LIGHT read-only investigator to run

```bash
gh issue view <N> --repo seqeralabs/nf-metro
```

assess the issue against current repository and remote evidence, and return the
problem statement, initial scope, unknowns, and proposed diagnostic brief. The
issue body and any comment thread stay in the worker; only its compact result
reaches the coordinator. The coordinator summarizes that to the user and waits
for confirmation before proceeding unless the user has pre-authorised autonomous
work; never infer merge or issue-edit authority from that permission.

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

Env, `PYTHONPATH`, and commit-hook mechanics:
[`references/environment.md`](references/environment.md). Reuse the one
long-lived `nf-metro-dev` env; never create one per issue.

## Step 3: Diagnostic Before Fix

Assign a HIGH read-only diagnostic worker before the writer changes code. **Do
not propose fixes from hypotheses.** Require the worker to reproduce the symptom
in numbers:

1. Render the affected example(s) on the current `main` (the before-state).
2. Inspect the rendered SVG and actual coordinates or element attributes.
3. Restate the bug as "element X has property P=<observed>, expected
   P=<target>" - a concrete numeric or structural claim. Return blocked if the
   worker cannot state it yet; diagnosis must continue before implementation.

Only after the symptom is pinned down to specific numbers may the diagnostic
worker reason about which layout pass or function produced them. The
post-diagnosis review gate challenges the resulting classification before
implementation starts.

### Check your premise against current `origin/main` first

Diagnose against latest remote, not a stale tree. The coordinator fetches
`origin/main`; the diagnostic worker confirms the bug **still reproduces on
that exact SHA** before reasoning about a cause - a sibling PR may already have
fixed it or changed the very code you're reading. If the user says something is
already addressed, re-fetch and look again before disagreeing; "I'm looking at
outdated code" is a recurring wrong turn. If a related PR merges mid-session,
first require the writer to hand off a clean, committed candidate. The
coordinator may then serialize a base-merge assignment to that sole writer. Keep
conflicts and required edits with the writer. Assign re-diagnosis on the
resulting candidate SHA.

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

### First, check for a pre-existing regression lock

Some issues arrive with their regression infra already built (a fixture, a
gallery row, and a `strict=True` xfail naming the issue). Grep before writing
anything:

```bash
grep -rn "#<N>" tests/ scripts/build_gallery.py examples/topologies/
```

Any hit, or any temptation to add an xfail, means read
[`references/regression-locks.md`](references/regression-locks.md) first: an
existing strict-xfail *is* your failing test, and xfail is never a way to defer
an incomplete fix. No hit is the common case - proceed below.

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

After the fix and tests are passing, assign a fresh MID read-only worker to
invoke the `simplify` skill on the changed code. It returns findings, proposed
edits, and a pass/fail verdict without writing. Re-brief the worktree's sole
writer to apply accepted suggestions and record them in a **separate** local
candidate commit:

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
Never skip hooks with `--no-verify`.

Assign a LIGHT independent read-only verifier the exact candidate SHA. It runs
the fixed command block in
[`references/environment.md`](references/environment.md) against a frozen
checkout, requires a clean tree before and after, keeps every cache and log in
the external artifact directory, and returns the command, exit status, concise
failure excerpt, and verdict. Route failures back to the writer, then verify the
new SHA.

**CI owns the full suite.** Targeted local tests plus the CI matrix is the
default, not a compromise: CI already runs the complete suite across the
supported matrix on every push, and duplicating it locally per branch or per
worker buys no signal. Run a local full suite only when it earns its cost:

- shared orchestration, parser model, dispatch table, or widely used helper
  changed;
- a targeted pass cannot cover a concrete wider-regression risk;
- explicit admin-merge preparation needs local full-suite confidence.

One exception is genuinely full-corpus by construction: a new topology
fixture's guard-trace golden (see below).

### Generated-artifact gates

If the change touched `layout/routing/` or added a topology fixture, three
ratchet tests and a guard-trace golden may red, each naming a specific
reconciliation you owe in this same PR. Do not hand-edit baselines to silence
them. Procedure, verdicts, and the Python 3.11 pinning gotcha:
[`references/gate-ratchet.md`](references/gate-ratchet.md).

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

Assign a fresh HIGH read-only visual reviewer the preview, target before/after
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

Merge authority, push hygiene, and cleanup:
[`references/merge-and-cleanup.md`](references/merge-and-cleanup.md). Leave
branch deletion to Step 12 so dependent PRs can be retargeted safely.

### State the evidence for every "it's fixed" claim

Never assert a fix works without an independent verifier naming what proved it.
Every "resolved" / "this is fixed" / "renders correctly" claim must cite the
**specific render and the concrete numbers** it was checked against - the file,
and the coordinate or element that moved from the observed value to the target
value you wrote down in Step 3. "I believe it's resolved" with no named render
is not a verdict; it invites the reply "which render did you re-assess on?".

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

### Local renders

A single-file sanity render, or a local before/after sweep, belongs in a LIGHT
read-only worker that returns the verdict and artifact path rather than the
imagery. Commands:
[`references/environment.md`](references/environment.md). Neither replaces the
CI gallery review.

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

For each detrimental delta, assign a HIGH diagnostic worker to find the
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
- [ ] Targeted tests pass locally (including new invariant test); CI matrix runs the full suite
- [ ] ruff check + ruff format clean on whole repo
- [ ] Runtime validator added (if applicable)
- [ ] Visual review of [render preview](https://seqeralabs.github.io/nf-metro/_pr/<PR_NUMBER>/)
- [ ] Render-preview verdict: <No visual changes | deltas classified I/N>
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

No force-push, ever, and no narrative comments on the PR - both in
[`references/merge-and-cleanup.md`](references/merge-and-cleanup.md).

## Step 11: Drive End-to-End

Before declaring readiness, run the **pre-ready gate**: one fresh HIGH read-only
reviewer given the accepted candidate SHA, aggregate diff, issue, diagnostic
evidence, test artifacts, and visual verdict. It covers correctness, scope,
invariants, safety, unresolved fallout, *and* aggregate progress in a single
brief. Revise any later brief from its findings. Create the PR as a draft for CI
and render evidence. Only after that gate passes may the coordinator run
`gh pr ready <N>`.

A successful fix-issue run is not done when `/simplify` or a test worker
returns. It reaches PR-ready completion when:

1. The fix lands in `src/`, not in a doctored reproducer (Step 3), and the
   "it's fixed" claim cites the render + numbers that prove it (Step 8).
2. Commits are pushed.
3. Origin HEAD verified against local.
4. CI is green on the final commit - including the full test matrix, which is
   CI's job, not a local run's; any failure interpretation came from an assigned
   verifier or domain specialist.
5. Render-preview verdict is captured and gated on per Step 8.
6. PR description is standalone.
7. Independent verification, visual review when applicable, and the pre-ready
   gate pass.
8. The coordinator marks the draft PR ready only after those gates pass.

Reroute bounded failures until these gates pass. If missing authority,
unavailable capability, external state, or a material user decision prevents
completion, return the structured blocker with the accepted candidate SHA and
remaining gate. Do not claim PR-ready completion.

## Step 12: Post-Merge Cleanup

Retarget child PRs first, then delete remote branch, worktree, local branch, in
that order, and only with user authority. Full procedure and the reconciliation
checks that gate it: [`references/merge-and-cleanup.md`](references/merge-and-cleanup.md).

For shepherding a whole stacked chain of PRs back into `main` (rather
than a single issue fix), see `pr-chain-vet`.
