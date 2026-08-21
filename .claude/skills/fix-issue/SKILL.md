---
name: fix-issue
description: Coordinator-led workflow for fixing GitHub issues on nf-metro: diagnostic-first, invariant-test-first, delegated to tiered workers. Use when the user references a GitHub issue (by number, URL, or description) and wants it fixed. Handles autonomous / net-negative requests. Trigger on "fix issue #N", "address #N", "work on issue N", or any request to fix a bug or implement a feature that references an issue. Use this whenever the work starts from a filed issue, including layout and routing bugs. For a bad render you are looking at with no issue filed, see `nf-metro-layout-fix`; for shepherding a chain of existing PRs back to main, see `pr-chain-vet`.
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

## Reference files: load only what you own

**The coordinator reads only the files marked `coord`.** The rest are
worker-facing: name the file in the brief and let the worker read it in its own
context, which is discarded at handoff. Reading a worker's reference "to check
its work" is what the gates are for.

| File | Owner | Load when |
| --- | --- | --- |
| [`coordinator.md`](references/coordinator.md) | **coord** | always: briefing the issue, worktree setup, the push, origin check, pre-ready gate, cleanup |
| [`agent-types.md`](references/agent-types.md) | **coord** | choosing an agent type, or checking the model resolution order |
| [`scope-discipline.md`](references/scope-discipline.md) | **coord** | fallout appears and you are tempted to defer it |
| [`merge-and-cleanup.md`](references/merge-and-cleanup.md) | **coord** | the PR body, pushing, merging, cleanup |
| [`autonomous-mode.md`](references/autonomous-mode.md) | coord | the user signalled autonomous / net-negative work |
| [`worker-contract.md`](references/worker-contract.md) | worker | every brief names it |
| [`diagnosis.md`](references/diagnosis.md) | diagnostician | its brief names it |
| [`tests-and-validators.md`](references/tests-and-validators.md) | writer | its brief names it |
| [`writer-steps.md`](references/writer-steps.md) | writer | its brief names it |
| [`regression-locks.md`](references/regression-locks.md) | writer | the Step 4 grep found a lock, or an xfail is proposed |
| [`gate-ratchet.md`](references/gate-ratchet.md) | gate specialist | `layout/routing/` changed, or a fixture was added |
| [`environment.md`](references/environment.md) | any worker running commands | its brief names it |
| [`visual-review.md`](references/visual-review.md) | visual reviewer | its brief names it |

**After an auto-compaction, re-invoke this skill.** Claude Code re-attaches only
the first 5,000 tokens of each skill, sharing a 25,000-token budget across all of
them, so a long run can lose the back half of this file or drop it entirely once
other skills are invoked. The ordering below puts the procedure and the tier
contract first for that reason; do not move the rationale above them.

## The twelve steps

Each step's detail is in the reference named beside it. Do not skip a step
because its detail is not inline.

1. **Understand the issue.** A LIGHT investigator reads it and returns problem
   statement, scope, unknowns, and a proposed diagnostic brief. The issue body
   stays in the worker. Wait for user confirmation unless autonomous work is
   pre-authorised. [`coordinator.md`](references/coordinator.md)
2. **Worktree and environment.** Coordinator creates the worktree off latest
   `origin/main`, records the base SHA, assigns one writer.
   [`coordinator.md`](references/coordinator.md)
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
   [`writer-steps.md`](references/writer-steps.md)
7. **Lint and tests.** Writer runs the mutating commands and commits; a LIGHT
   verifier re-runs the fixed block against the frozen SHA. CI owns the full
   suite. [`writer-steps.md`](references/writer-steps.md),
   [`gate-ratchet.md`](references/gate-ratchet.md)
8. **Visual review via the CI render preview.** The coordinator's single push
   and draft-PR creation happens here. Any delta at all gets HIGH eyes on every
   changed render. [`visual-review.md`](references/visual-review.md)
9. **Narrow an over-applying fix.** Every D-delta gets a precondition that
   separates the helped case from the hurt one.
   [`visual-review.md`](references/visual-review.md)
10. **Accept the candidate, verify origin.** `HEAD` equals the accepted SHA, the
    tree is clean, and the pushed ref matches local. Query the ref, not the PR
    API, which lags. [`coordinator.md`](references/coordinator.md),
    [`merge-and-cleanup.md`](references/merge-and-cleanup.md)
11. **The pre-ready gate**, then `gh pr ready`. One HIGH reviewer covering code
    review and aggregate progress together.
    [`coordinator.md`](references/coordinator.md)
12. **Post-merge cleanup**, only with authority, children retargeted first.
    [`merge-and-cleanup.md`](references/merge-and-cleanup.md)

For shepherding a whole stacked chain of PRs back into `main` rather than a
single issue fix, see `pr-chain-vet`.

## Worker tiers, briefs, and gates

### Worker tiers are explicit, never inherited

**Every worker launch names its tier explicitly.** Decide it before spawning and
say so in one line. Omitting the parameter is not a decision, it is a default
nobody chose - and where it lands depends on the resolution order in
[`agent-types.md`](references/agent-types.md), including one environment variable
that overrides every tier in this table. Never leave it to that.

The tier is the contract, not the model name: `haiku`/`sonnet`/`opus` on Claude
Code, `luna`/`terra`/`sol` on Codex, three tiers whatever the harness calls them.

Tiers come from the table below and do not drift upward because a task felt hard.
A re-briefed role keeps the tier it was spawned at. If you find a worker running
on an inherited default, restart it on the intended tier rather than justifying
the one it got: that call belongs at spawn time or not at all.

| Step | Role | Agent type | Tier |
| --- | --- | --- | --- |
| 1 | issue investigator | `fix-issue-investigator` | LIGHT |
| 3 | diagnosis | `fix-issue-diagnostician` | HIGH; MID when the issue already names its own single-site cause and the brief is confirm-or-refute |
| 3, 11 | the two review gates | `fix-issue-reviewer` | HIGH |
| 4-7 | sole writer | `fix-issue-writer` | HIGH when the diff changes geometry-affecting logic in `src/nf_metro/layout/` (including its `routing/` package) or `src/nf_metro/parser/`; MID for a class (c) structural change in those dirs that alters no geometry, or for anything outside them. Highest tier wins on a mixed diff |
| 6 | `/simplify` review | `fix-issue-simplifier` | MID |
| 7 | lint/test verification | `fix-issue-verifier` | LIGHT |
| 7 | routing gate classification | `fix-issue-gate-specialist` | MID |
| 8 | local render / before-after sweep | `fix-issue-renderer` | LIGHT |
| 8 | admin-merge gate | `fix-issue-merge-assessor` | HIGH - it gates shipping code CI has not verified |
| 8, 9 | visual judgment and D-delta narrowing | `fix-issue-visual-reviewer` | HIGH |

A LIGHT worker that returns "blocked, this needs judgment" is a correct
outcome, not a failure. Re-route it up a tier rather than pre-emptively
starting high.

`effort` is a second dial but fixed per definition, with no per-invocation
form: see [`agent-types.md`](references/agent-types.md).

Resolution order when tiers collide, including the `CLAUDE_CODE_SUBAGENT_MODEL`
override that beats everything in this skill:
[`agent-types.md`](references/agent-types.md).

### Prefer the named agent types

`.claude/agents/` defines one type per role, tier and tool set already set:
`fix-issue-investigator`, `fix-issue-diagnostician`, `fix-issue-writer`,
`fix-issue-simplifier`, `fix-issue-verifier`, `fix-issue-gate-specialist`,
`fix-issue-visual-reviewer`, `fix-issue-reviewer`, `fix-issue-renderer`,
`fix-issue-merge-assessor`.

Spawn by role name **and** pass the model. Both, deliberately: the per-invocation
model is verified to beat the definition, and the definition catches a spawn
where it was forgotten. Every definition carries an explicit `tools` allowlist,
which matters because a subagent with no allowlist inherits every tool the main
conversation has - including all MCP servers, and including `Agent`, which would
let a worker spawn untiered children of its own.

The read-only roles carry no `Edit` or `Write`, but they all hold `Bash`, so
treat that as a backstop and not a guarantee; the instruction is what enforces
read-only. The structural lever, and why `permissionMode` is not it, is in
[`agent-types.md`](references/agent-types.md).

Substituting `Explore`/`Plan` skips both CLAUDE.md files but discards the role
definition and blocks re-briefing; measured at ~$0.24 a run, so it is a curiosity
rather than a lever. See [`agent-types.md`](references/agent-types.md).

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

### Gates and writer discipline

Two mandatory review gates, one writer per worktree, readers always fresh and
independent: [`coordinator.md`](references/coordinator.md).

## Primary invariant: coordinate, delegate, verify independently

The coordinator owns user communication and authority, the ledger, worker
routing, integration gates, and final reporting. It **alone** pushes, creates or
edits issues and PRs, merges, retargets, and cleans up. It does **not** do
substantive diagnosis, implementation, domain assessment, visual judgment,
`/simplify`, gate interpretation, or code review.

Two separate levers, do not confuse them:

- **Delegate to protect coordinator context.** Bulk output - issue bodies, test
  logs, render analysis, file sweeps - goes to a worker even when the task is
  mechanically trivial: coordinator bytes are re-read every turn, a worker's are
  discarded at handoff. Delegating a cheap task is cheaper than absorbing it.
- **Choose the tier to protect cost.** Delegation decides *where* work runs; the
  tier decides what it costs. A mechanical worker on the top tier is the most
  expensive thing in this workflow.

The coordinator does not read `src/` at all: `routing/inter_section_handlers.py`,
`routing/invariants.py`, `phases/guards.py` and `engine.py` - the largest files
in the tree - belong only in a worker's context. This is
hygiene, not a headline saving - measured, it is worth under 1% of a run.

It may run trivial deterministic assertions itself - `git rev-parse`, a hash or
OID comparison, `git status --porcelain`, an exit code, handoff-schema
completeness - since spawning for a few bytes buys nothing. It must not
substitute its own *substantive* review.

The ledger tracks: issue and authority state; worktree, branch, base, writer;
current hypothesis and evidence links; worker assignments, tiers and verdicts;
changed files and commits; commands and outcomes; I/N/D classifications;
fallout; blockers; CI/PR state; next gate. Hold only the **live slice** in
context - current gate, open blockers, accepted SHA, active assignments. Append settled rows to a ledger file outside the
worktree and cite it. Keep deep context, long test output, render analysis, and
review detail in worker handoffs or artifacts. Without this the ledger is the
one item that grows every turn and is re-read on all of them.

## Cost discipline (applies throughout)

Layout iteration is where sessions burn tokens and compute. Keep it tight:

- **Name a tier on every spawn** (above). Measured at 5.7% of spend in that same sample - real, but note that most of the per-spawn gap between tiers is *turns*
  (216 vs 65), not price per token, so the tier table gets some credit that
  belongs to task selection. The writer's turn count is the bigger lever.
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
- **Do not hold a large context idle across a CI run.** Waiting is where the
  second-largest cost lives: in the same one-off measurement, 468 turns (1.8%
  of all turns) were 9.8% of spend, each a full re-encode averaging 358k tokens
  at the 1-hour cache premium, caused by idle gaps busting the cache. End the turn after the push and pick the run up fresh, or collapse the
  ledger to its live slice before waiting. When you do watch, watch once in the
  background (`until gh pr checks <N> ...; done`) rather than re-running
  `gh pr checks` each turn, which dumps status into context repeatedly.
- **Lean on the CI render-diff for regression review; don't rebuild the gallery
  locally in a loop.** The CI preview (Step 8) is the authoritative whole-corpus
  diff. A local `build_gallery` / render-diff sweep repeated many times just
  duplicates it. Local rendering is for a *single* file's quick sanity check.
- **Brief workers to read the big layout files in wide slices and stay
  oriented.** Re-fetching `routing/inter_section_handlers.py` (7.4k lines),
  `routing/invariants.py` (6.7k) or `phases/guards.py` (6.6k) twenty times over a session is the single largest cache-read cost.
  Read the region once, generously, and keep it in working context.
- **Push policy is governance, not economy**, and it lives with the push:
  [`merge-and-cleanup.md`](references/merge-and-cleanup.md).

## Scope discipline

Resolve bounded fallout surfaced by diagnosis, implementation, `/simplify`, lint,
review, or CI **in the current run**. Filing and deferring is the exception, and
a different subsystem is not by itself a reason to defer. Full rules, including
when a sibling PR is the right home and what a legitimate blocker looks like:
[`scope-discipline.md`](references/scope-discipline.md).
