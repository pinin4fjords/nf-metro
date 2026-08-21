# Procedure detail

The steps whose detail does not live in a more specific reference. Read the one
you are on; the spine in SKILL.md says when each applies.

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
[`environment.md`](environment.md). Reuse the one
long-lived `nf-metro-dev` env; never create one per issue.

## Step 6: /simplify Pass

**Run this by default.** `/simplify` is worthwhile on anything beyond the
trivial, and "the diff is small" is not by itself a reason to skip it. Skip only
when **every** one of these holds, and say in one line that you did:

- the production diff adds no new function, class, branch, or code path;
- it does not touch a shared helper, a dispatch table, or anything with more
  than one caller;
- it is under roughly 20 lines of non-test production code;
- no later step (narrowing, lint, review) has added production code since.

A five-line change that introduces a branch, or threads an argument through
three call sites, is not trivial. If you are weighing whether it qualifies, it
does not: run the pass.

After the fix and tests are passing, assign a fresh MID read-only worker to
invoke the `pinin4fjords:simplify` skill on the changed code. It returns findings, proposed
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
[`environment.md`](environment.md) against a frozen
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
fixture's guard-trace golden, under "Generated-artifact gates" below.

### Generated-artifact gates

Two distinct triggers, and the path is literal. The gate-coverage ratchet
scans `src/nf_metro/layout/routing/` **only** (`ROUTING_DIR` in
`scripts/routing_gate_coverage.py`, with branch coverage scoped to it), so it
trips on an `if`/`while` change inside that nested package, or on a new fixture
that closes a gap. A change elsewhere under `layout/` - `engine.py`, a phase
module, `ordering.py` - cannot trip it, so do not spend a worker checking
defensively. The guard-trace golden is separate and trips on any **new** fixture
under `examples/topologies/`. When either fires, each failure names a specific
reconciliation you owe in this same PR. Do not hand-edit baselines to silence
them. Procedure, verdicts, and the Python 3.11 pinning gotcha:
[`gate-ratchet.md`](gate-ratchet.md).


## Step 10: accept candidate, verify origin

After the writer hands off candidate commit SHA(s) and independent gates pass,
the coordinator confirms `HEAD` equals the accepted SHA and the tree is clean.
Only the coordinator pushes, creates or edits the PR, and performs later remote
mutations. Step 8 has normally already pushed and opened the draft PR: **do not
re-push or re-create it**, edit the existing body (`gh pr edit`) and go straight
to the origin check. The body template is in
[merge-and-cleanup.md](merge-and-cleanup.md).

## Step 11: Drive End-to-End

Before declaring readiness, run the **pre-ready gate**: one fresh HIGH read-only
reviewer given the accepted candidate SHA, aggregate diff, issue, diagnostic
evidence, test artifacts, and visual verdict. It covers correctness, scope,
invariants, safety, unresolved fallout, *and* aggregate progress in a single
brief. Revise any later brief from its findings. The draft PR already exists
from Step 8; this step adds no push and no new PR. Only after the gate passes
may the coordinator run `gh pr ready <N>`.

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
checks that gate it: [`merge-and-cleanup.md`](merge-and-cleanup.md).

For shepherding a whole stacked chain of PRs back into `main` (rather
than a single issue fix), see `pr-chain-vet`.
