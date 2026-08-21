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

## Reference files

Load the one you are on. Be honest about what this saves: `worker-contract`,
`procedure`, `diagnosis`, `tests-and-validators`, `visual-review`, `environment`
and `merge-and-cleanup` are all needed on a normal run, so deferral is the
saving, not elimination. Only `gate-ratchet`,
`regression-locks`, `agent-types` and `autonomous-mode` are genuinely
conditional.

| File | Load when |
| --- | --- |
| [`worker-contract.md`](references/worker-contract.md) | every brief points workers here; read it yourself once |
| [`procedure.md`](references/procedure.md) | Steps 1, 2, 6, 7, 10, 11, 12 in full |
| [`diagnosis.md`](references/diagnosis.md) | Step 3: pinning the defect, classification, tooling |
| [`tests-and-validators.md`](references/tests-and-validators.md) | Steps 4 and 5: the failing test, then the guard |
| [`visual-review.md`](references/visual-review.md) | Steps 8 and 9: fetching renders, verdict gating, narrowing |
| [`environment.md`](references/environment.md) | env, hooks, the verifier command block, local renders |
| [`gate-ratchet.md`](references/gate-ratchet.md) | the diff touched `layout/routing/`, or added a topology fixture |
| [`regression-locks.md`](references/regression-locks.md) | the Step 4 grep found a lock, or you want to add an xfail |
| [`merge-and-cleanup.md`](references/merge-and-cleanup.md) | the PR body, pushing, merging, cleanup |
| [`agent-types.md`](references/agent-types.md) | you are considering `Explore` in place of a role type |
| [`autonomous-mode.md`](references/autonomous-mode.md) | the user signalled autonomous / net-negative work |

**After an auto-compaction, re-invoke this skill.** Claude Code re-attaches only
the first 5,000 tokens of each skill, sharing a 25,000-token budget across all of
them, so a long run can silently lose the back half of this file or drop it
entirely once other skills are invoked. The ordering below puts the procedure and
the tier contract first for that reason; do not move the rationale above them.

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

## Worker tiers, briefs, and gates

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

The tier is the contract, not the model name. On Claude Code that is
`haiku`/`sonnet`/`opus`; on Codex, `luna`/`terra`/`sol`. Substitute local
equivalents elsewhere and keep the three-tier shape; if a harness exposes no such
parameter, say so in one line and proceed.

Role tiers are fixed by this table and do not drift upward because a task felt
hard. A role **re-briefed** later in the run (the writer applying `/simplify`
findings, or narrowing a delta) keeps the tier it was spawned at.

| Step | Role | Tier |
| --- | --- | --- |
| 1 | issue investigator | LIGHT |
| 3 | diagnostic worker | HIGH; MID when the issue already names its own single-site cause and the brief is confirm-or-refute |
| 4-7 | sole writer | HIGH when the diff changes geometry-affecting logic in `layout/`, `routing/`, or `parser/`; MID for a class (c) structural change in those dirs that alters no geometry, or for anything outside them. Highest tier wins on a mixed diff |
| 6 | `/simplify` worker | MID |
| 7 | lint/test verifier | LIGHT |
| 7 | routing gate specialist | MID |
| 8 | local render / before-after sweep | LIGHT |
| 8 | eco-merge assessor | HIGH - it gates shipping code CI has not verified |
| 8 | visual reviewer | HIGH |
| 9 | per-D-delta diagnostic | HIGH |
| 11 | combined code + aggregate reviewer | HIGH |

A LIGHT worker that returns "blocked, this needs judgment" is a correct
outcome, not a failure. Re-route it up a tier rather than pre-emptively
starting high.

**`effort` is a second dial, but a fixed one.** Definitions take `effort`
(`low`/`medium`/`high`/`xhigh`/`max`) and it is set for the life of the
definition: there is no per-invocation `effort`, only a per-invocation `model`.
So the LIGHT roles run `low` and the judgment roles run `high`, and the
persistent writer cannot be dialled down for its mechanical re-briefs. Accept
that: the retained context is worth more than the thinking tokens, which is all
`effort` moves.

### The model resolution order

Highest wins:

1. the `CLAUDE_CODE_SUBAGENT_MODEL` environment variable;
2. the per-invocation `model` parameter;
3. the agent definition's `model` frontmatter;
4. the main conversation's model.

If `CLAUDE_CODE_SUBAGENT_MODEL` is set it overrides **every** tier decision in
this skill, so check it once at session start. An organisation `availableModels`
allowlist can also substitute a model at any of these levels.

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

The read-only roles carry no `Edit` or `Write`. Treat that as a backstop, not a
guarantee: they all hold `Bash`, so a worker that ignores its brief can still
write, and `permissionMode` will not save you - under the parent's auto mode a
subagent's `permissionMode` is ignored. The instruction is what enforces
read-only. If you want it enforced structurally, the lever is the per-subagent
`hooks` field: a `PreToolUse` matcher on `Bash` rejecting `git push`,
`gh pr merge` and `gh pr edit`. That matters most for
`fix-issue-merge-assessor`, whose `Skill` grant points at a procedure ending in
`gh pr merge --admin`.

`Explore`/`Plan` skip both CLAUDE.md files (~5k tokens a spawn) but discard the
role definition and cannot be re-briefed: see
[`agent-types.md`](references/agent-types.md) before substituting one.

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

1. **Post-diagnosis gate.** One reviewer that challenges the domain
   classification (Step 3: authoring mistake, engine bug, input-independent
   structural defect, or no defect at all, and the claim behind whichever came
   back). A wrong classification wastes the whole run, so this gate pays for
   itself; a second separate challenger does not.
2. **Pre-ready gate.** One reviewer combining the Step 11 code review and the
   final aggregate-progress review. These ask nearly the same question against
   the same diff; run them as one brief.

Run an extra mid-loop aggregate review only on a trigger: two repeated blocks,
material scope growth, conflicting worker verdicts, a changed acceptance bar, or
multiple active worktrees. Send it the compact ledger and *links* to evidence,
not the evidence inline. Record every review verdict and revise later briefs,
scope, or gates from its findings.

### One writer, independent readers

Allow exactly one writer in each worktree. Give concurrent writers separate
worktrees and non-overlapping write scopes; otherwise serialize them. Keep
diagnostic, verifier, visual-review, and code-review roles read-only and
independent of the writer. Read-only workers never persist tracked, untracked,
or ignored worktree changes; place their logs, caches, and generated evidence
outside the worktree. Readers run concurrently only against a frozen commit SHA
or snapshot, never a live worktree during an active writer assignment. User
authority determines whether the coordinator acts; it never transfers that
ownership to a worker.

The writer is **one continuing worker**, not a fresh spawn per step. Continue it
with `SendMessage` addressed to its agent ID; a fresh `Agent` call creates a new
instance and re-pays the largest read cost in the run. A per-invocation `model`
still applies on resume. Steps 6, 7
and 9 re-brief the same agent so it keeps the large layout modules it already
read in working context; re-spawning it would re-pay the largest read cost in
the run. Readers, by contrast, are fresh each time so their judgment stays
independent.

Use one candidate sequence throughout: the sole writer makes local candidate
commit(s), runs mutation-capable hooks or generators, and hands off the exact
SHA. Independent read-only workers verify and review that SHA without changing
it. If fixes are required, serialize them back to the writer and verify the new
SHA.

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

The coordinator does not read `src/` at all: `engine.py`, `ordering.py`,
`fan_bundles.py` and `routing/*` belong only in a worker's context. Reading them
"just to orient" is the largest avoidable cost in the run.

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

## Scope discipline

Resolve bounded fallout surfaced by diagnosis, implementation, `/simplify`, lint,
review, or CI **in the current run**. Filing and deferring is the exception, and
a different subsystem is not by itself a reason to defer. Full rules, including
when a sibling PR is the right home and what a legitimate blocker looks like:
[`scope-discipline.md`](references/scope-discipline.md).
