# Scope discipline: fix the fallout, don't defer it

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
