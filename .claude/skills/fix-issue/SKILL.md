---
name: fix-issue
description: Coordinator-led workflow for fixing GitHub issues on nf-metro: diagnostic-first, invariant-test-first, delegated to tiered workers. Use when the user references a GitHub issue (by number, URL, or description) and wants it fixed. Handles autonomous / net-negative requests. Trigger on "fix issue #N", "address #N", "work on issue N", or any request to fix a bug or implement a feature that references an issue. For shepherding a chain of existing PRs back to main, see `pr-chain-vet` instead.
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

## Reference files - load only the one you need

| File | Load when |
| --- | --- |
| [`worker-contract.md`](references/worker-contract.md) | every brief points workers here; read it yourself once |
| [`procedure.md`](references/procedure.md) | Steps 1, 2, 6, 7, 10, 11, 12 in full |
| [`diagnosis.md`](references/diagnosis.md) | Step 3: pinning the defect, classification, tooling |
| [`tests-and-validators.md`](references/tests-and-validators.md) | Steps 4 and 5: the failing test, then the guard |
| [`visual-review.md`](references/visual-review.md) | Steps 8 and 9: verdict gating, evidence rules, narrowing |
| [`environment.md`](references/environment.md) | env, hooks, the verifier command block, local renders |
| [`gate-ratchet.md`](references/gate-ratchet.md) | the diff touched `layout/routing/`, or added a topology fixture |
| [`regression-locks.md`](references/regression-locks.md) | the Step 4 grep found a lock, or you want to add an xfail |
| [`merge-and-cleanup.md`](references/merge-and-cleanup.md) | the PR body, pushing, merging, cleanup |
| [`autonomous-mode.md`](references/autonomous-mode.md) | the user signalled autonomous / net-negative work |

## Primary invariant: coordinate, delegate, verify independently

The coordinator owns user communication and authority, a compact ledger, worker
routing, integration gates, Git integration and remote mutations, and final
reporting. It does **not** do substantive diagnosis, implementation, domain
assessment, visual judgment, `/simplify`, gate interpretation, or code review.

Two separate levers, do not confuse them:

- **Delegate to protect coordinator context.** Anything producing bulk output -
  issue bodies, test logs, render analysis, file sweeps - goes to a worker even
  when it is mechanically simple, because bytes that land in the coordinator are
  re-read on every subsequent turn while a worker's are discarded at handoff.
  Delegating a cheap task is cheaper than absorbing it.
- **Choose the tier to protect cost.** Delegation decides *where* work runs; the
  tier decides what it costs. A mechanical worker on the top tier is the most
  expensive thing in this workflow.

The coordinator does not read `src/` at all. `engine.py`, `ordering.py`,
`fan_bundles.py`, and `routing/*` should only ever occupy a worker's context.
Reading them "just to orient" is the largest avoidable context cost in the run.

It may still run trivial deterministic assertions itself - `git rev-parse`, a
hash or OID comparison, `git status --porcelain`, an exit code, handoff-schema
completeness - because those are a few bytes and spawning for them buys nothing.
It must not substitute its own *substantive* review.

The ledger tracks: issue and authority state; worktree, branch, base, writer;
current hypothesis and evidence links; worker assignments, tiers and verdicts;
changed files and commits; commands and outcomes; I/N/D classifications;
fallout; blockers; CI/PR state; next gate. Hold only the **live slice** in
context - current gate, open blockers, accepted SHA, active assignments. Append settled rows to a ledger file outside the
worktree and cite it. Keep deep context, long test output, render analysis, and
review detail in worker handoffs or artifacts. Without this the ledger is the
one item that grows every turn and is re-read on all of them.

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
hard. A role **re-briefed** later in the run (the writer applying `/simplify`
findings, or narrowing a delta) keeps the tier it was spawned at.

| Step | Role | Tier |
| --- | --- | --- |
| 1 | issue investigator | LIGHT |
| 3 | diagnostic worker | HIGH |
| 4-7 | sole writer | HIGH when the diff changes geometry-affecting logic in `layout/`, `routing/`, or `parser/`; MID for a class (c) structural change in those dirs that alters no geometry, or for anything outside them. Highest tier wins on a mixed diff |
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

### Prefer the named agent types

`.claude/agents/` defines one type per role with its tier and tool set already
set: `fix-issue-investigator`, `fix-issue-diagnostician`, `fix-issue-writer`,
`fix-issue-simplifier`, `fix-issue-verifier`, `fix-issue-gate-specialist`,
`fix-issue-visual-reviewer`, `fix-issue-reviewer`.

Spawn by role name **and** pass the tier's model explicitly. Belt and braces,
deliberately: the explicit model is the guarantee, since a spawn-time model is
verified to take precedence over the definition. The definition's model is only
a safety net - documentation says omitting the parameter falls back to it rather
than to the session's model, but that direction is **not verified here**, so
never rely on it alone. The read-only roles carry no `Edit`/`Write` tool, which
makes read-only structural rather than advisory (`Bash` can still write, so the
instruction still matters).

Two roles need no repo architecture context: the investigator reading an issue,
and the verifier running a fixed command block. Those may use the built-in
`Explore` type with an explicit model instead - the only type that loads
**neither** CLAUDE.md, saving roughly 5k tokens per spawn (verified: `Explore`
sees no project and no user CLAUDE.md, and holds only `Bash`, `Read`, `Skill`,
`ToolSearch`). Never route the diagnostician, writer, visual reviewer, or
reviewer through it: they need the architecture map and the station-as-elbow
constraint.

### Worker brief template

Fill this in. Do not restate the authority rules, the return schema, or the
verifier command block in the brief: they live in
[`references/worker-contract.md`](references/worker-contract.md) and the worker
reads them there. Restating them per spawn spends coordinator output that then
gets re-read on every later turn.

```
ROLE / TIER: <role> at <LIGHT|MID|HIGH>
OBJECTIVE:   <one sentence>
AUTHORITY:   read-only | sole writer in <worktree>
SCOPE:       <paths this worker may read or write>
INPUTS:      <frozen SHA, artifact dir, issue number>
ACCEPTANCE:  <the concrete bar for this task>
STOP IF:     <escalation conditions specific to this task>
DECIDE:      <options this worker must surface rather than pick, if any>
CONTRACT:    follow .claude/skills/fix-issue/references/worker-contract.md
```

Check every handoff against the six items: scope, files+SHA, commands and
outcomes, evidence, risks/blockers, verdict. Evidence arrives **as a path plus
the one figure that carries the verdict** - never pasted coordinate dumps,
render analysis, or test logs. A blocked handoff satisfying the schema is a
valid outcome: re-brief from its evidence, route a different tier, or escalate
to the user when authority or a material product decision is missing. Never
demand an unbounded worker loop or treat diagnosis as implementation. The
contract file carries the full wording, including what a worker does when the
work exceeds its briefed tier.

### Review gates: two mandatory, the rest on trigger

Independent review is load-bearing, but a reviewer that re-reads raw evidence is
the most expensive spawn in the run. Two mandatory gates, both HIGH:

1. **Post-diagnosis gate.** One reviewer that both challenges the domain
   classification (Step 3: authoring mistake, engine bug, or input-independent
   structural defect, and the numeric or structural claim behind it) and reviews aggregate progress. A wrong classification
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
- **One CI-triggering push per round.** A push runs the full test matrix *and*
  renders the whole gallery twice, on the PR branch and the base. It is the most
  expensive action in this workflow. Batch accepted fixes into one push instead
  of pushing each one as it lands.

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

## The twelve steps

Each step's detail is in the reference named beside it. Do not skip a step
because its detail is not inline.

1. **Understand the issue.** A LIGHT investigator reads it and returns problem
   statement, scope, unknowns, and a proposed diagnostic brief. The issue body
   stays in the worker. Wait for user confirmation unless autonomous work is
   pre-authorised. [`procedure.md`](references/procedure.md)
2. **Worktree and environment.** Coordinator creates the worktree off latest
   `origin/main`, records the base SHA, assigns one writer.
   [`procedure.md`](references/procedure.md),
   [`environment.md`](references/environment.md)
3. **Diagnose before fixing.** No fix proposed from a hypothesis. The defect
   becomes a numeric claim (geometry) or named call sites (structural), or the
   verdict is "not a bug" held to the same standard. Then the mandatory
   post-diagnosis gate. [`diagnosis.md`](references/diagnosis.md)
4. **Write the failing invariant test first**, and confirm it fails on `main`
   before any production change.
   [`tests-and-validators.md`](references/tests-and-validators.md),
   [`regression-locks.md`](references/regression-locks.md)
5. **Add a runtime validator** where a layout property could regress silently.
   Conditional: skip it outright for a class (c) structural defect, and say so.
   [`tests-and-validators.md`](references/tests-and-validators.md)
6. **`/simplify` pass**, run by default; skipping needs all four triviality
   conditions and must be declared.
   [`procedure.md`](references/procedure.md)
7. **Lint and tests.** Writer runs the mutating commands and commits; a LIGHT
   verifier re-runs the fixed block against the frozen SHA. CI owns the full
   suite. [`procedure.md`](references/procedure.md),
   [`environment.md`](references/environment.md),
   [`gate-ratchet.md`](references/gate-ratchet.md)
8. **Visual review via the CI render preview.** The coordinator's single push
   and draft-PR creation happens here. Any delta at all gets HIGH eyes on every
   changed render. [`visual-review.md`](references/visual-review.md)
9. **Narrow an over-applying fix.** Every D-delta gets a precondition that
   separates the helped case from the hurt one.
   [`visual-review.md`](references/visual-review.md)
10. **Accept the candidate, verify origin.** `HEAD` equals the accepted SHA, the
    tree is clean, and the pushed ref matches local. Query the ref, not the PR
    API, which lags. [`procedure.md`](references/procedure.md),
    [`merge-and-cleanup.md`](references/merge-and-cleanup.md)
11. **The pre-ready gate**, then `gh pr ready`. One HIGH reviewer covering code
    review and aggregate progress together.
    [`procedure.md`](references/procedure.md)
12. **Post-merge cleanup**, only with authority, children retargeted first.
    [`merge-and-cleanup.md`](references/merge-and-cleanup.md)

For shepherding a whole stacked chain of PRs back into `main` rather than a
single issue fix, see `pr-chain-vet`.
