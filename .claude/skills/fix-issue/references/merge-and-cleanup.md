# Push hygiene, merge authority, cleanup

Read this at the push/merge/cleanup end of a run.

## Additive only - no force-push, ever

The local pre-push hook blocks force-pushes for a reason. To undo anything, use
`git revert <hash>` and push the revert as a new commit. Never rewrite shared
history (no `--force`, no `--force-with-lease`, no interactive rebase on a pushed
branch). This applies even when "it would be cleaner" - cleanliness is not worth
the risk of silently dropping work.

An ordinary additive (fast-forward) push is **not** blocked by that hook - only
rewrites are. Don't mistake an unrelated push failure for a force-push block.

## Narrative belongs in the PR description, not in comments

Do not post explanatory comments on the PR walking through what changed, what was
tried, or what was reverted. Edit the PR description instead:

```bash
gh pr edit <PR_NUMBER> --body-file /tmp/pr-body.md
```

The description should be a standalone summary of the current state of the diff
against main - not a chronology of how the PR got there.

If narrative comments already exist, the coordinator may sweep them via the
GraphQL `deleteIssueComment` mutation only with issue/PR edit authority. **Keep**
the CI sticky render-preview comment.

## When the user *does* authorise a merge

Never merge without explicit per-PR user authority. Prior admin merges are not
standing consent. A clean render-diff verdict is not consent either.

- **"Merge"** authorises one normal merge-commit attempt:
  `gh pr merge <N> --merge`. Never squash. If review, branch protection, or
  up-to-date policy blocks it, stop and return that blocker. Do not escalate to
  `--admin`.
- **"Admin merge"** explicitly authorises `gh pr merge <N> --admin --merge`. If
  CI is not green, first assign a fresh `fix-issue-merge-assessor` (HIGH, read-only) to
  use `pinin4fjords:eco-merge` and determine whether the sole unverified delta is
  CI-irrelevant. The coordinator may run the admin merge only with both explicit
  user admin-merge authority and that worker's pass verdict. Otherwise return the
  blocker. Do not update the branch or start fresh CI merely to satisfy
  up-to-date policy; cancel irrelevant in-flight runs only as part of the
  authorised, accepted eco-merge sequence.

## Post-merge cleanup

Once the PR merges, only the coordinator performs cleanup, and only with user
authority. Before deleting anything, require a clean tree, reconcile local
`HEAD`, pushed remote head, and merged PR head, and confirm no unpushed commits.
Stop on any mismatch. Then use this order:

1. **Retarget any child PRs** based on this branch over to `main` (or the next-up
   base) **first**, via `gh pr edit <child> --base main`. Confirm every retarget
   and stop if any fails; branch deletion can auto-close a dependent PR.
2. Delete the **remote** branch: `git push origin --delete fix/<N>-<slug>` (or via
   the GitHub UI's auto-delete on merge).
3. Remove the local worktree: `git worktree remove /tmp/nf-metro-fix-<N>`.
4. Delete the reconciled local branch with `git branch -d fix/<N>-<slug>`. Use
   `-D` only after explicit user authority and proof that `-d` rejects a fully
   reconciled branch for a harmless bookkeeping reason.

Leave the shared `nf-metro-dev` env in place - it is reused across issues, so
there is nothing per-issue to remove.

Offer this cleanup to the user; only run it after they agree.

## Draft PR body template

Used once, at Step 8/10. The preview link needs the PR number, which does not
exist until the PR does: create the body with the placeholder, then fill it in
with `gh pr edit` at Step 10.

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
git ls-remote origin <branch> | cut -f1
git rev-parse HEAD
```

The two must match. Prior sessions have lost commits to silent push failures;
do not skip this check. Query the ref, not `gh pr view --json headRefOid`: the
PR API lags a push by seconds and will report the previous SHA, which reads as
a lost commit when nothing is wrong. If they genuinely differ, re-query the ref
once before treating it as a failure.
`merge-and-cleanup.md`.
