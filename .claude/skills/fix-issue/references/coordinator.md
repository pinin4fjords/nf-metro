# Coordinator actions

The steps the coordinator executes itself. Everything else it briefs a worker to
do; do not read the worker references to "check their work" - that is what the
independent gates are for, and reading them puts worker-facing bytes in the
context that is re-read every turn.

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
git worktree add /tmp/nf-metro-fix-<N> -b fix/<N>-<slug> --no-track origin/main
```

`--no-track` matters: without it the branch takes `main` as upstream, and a bare
`git push` then fails with "the upstream branch does not match the name of your
current branch" and helpfully suggests `git push origin HEAD:main`. Push
explicitly instead:

```bash
git push -u origin fix/<N>-<slug>
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
