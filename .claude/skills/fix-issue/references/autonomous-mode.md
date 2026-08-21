# Optional mode: autonomous / net-negative

Read this only when the user signals the mode: "no deferrals", "no giving up",
"drive it to conclusion", "net-negative issues", "work overnight", "I'll leave
you to it", or an explicit "push a complete fix and address all the fallout".

It does not relax the diagnostic or verification rigour of the main steps; it
changes two defaults.

## 1. Tighten the deferral bar

Drive coupled defects and incident fallout to a complete, reviewable PR in the
current run whenever they remain bounded. Brief each worker with the real
acceptance bar: the target render is clean, tests and validators pass, and every
corpus delta receives independent I/N/D classification. Require byte identity
only where no visual delta is intended; independently accepted I/N deltas are
valid evidence. Accept structured blocked handoffs, then re-route or escalate
rather than forcing an unbounded loop.

## 2. Track net-negative progress without inventing authority

Do not open a child issue merely to defer bounded fallout. Track issues resolved
by merged PRs, reviewable PRs awaiting user authority, and any newly opened
issues. Fewer open issues is the goal, but lack of merge authority or a required
user decision is a legitimate stop. Never claim issue closure before the
relevant PR merges, and never merge only to satisfy the arithmetic.

## Bounded to fallout, not a frontier

Resolve defects the current work surfaces. Stop and return a structured blocker
for a subsystem rewrite, sweeping cross-cutting refactor, unavailable
capability, external-state block, or required user decision. Do not recurse into
an open-ended program.

## Still in force in this mode - do not over-read "autonomous"

- Never merge without explicit per-PR authorisation from the user. Autonomy is
  about resolving and pushing *complete* work, not self-authorising merges.
  "Drive to conclusion" means "get each bounded PR green and reviewable, or
  return its structured blocker"; merge only what the user okays, per-PR.
  Pre-authorisation to *work* is not pre-authorisation to *merge*.
- Every step's rigour still applies to every PR, including the fallout ones:
  diagnostic-first, invariant-test-first, `/simplify`, evidence-cited
  verification, additive-only pushes, origin check, standalone issue bodies.
- The worker tier table still binds. Autonomy is not a reason to promote every
  role to HIGH, and a long unattended run is exactly where an unset tier
  silently bills the top model for hours.
- Reproducing before/after evidence and testing the fix is the Step 7 verifier
  at LIGHT, per the role table; judgment calls go to a separate HIGH visual
  reviewer. Autonomy does not re-tier a role. Name the tier at spawn time like
  any other assignment.
- Report every open-at-end item with its blocker: awaiting merge authority,
  awaiting a user decision, or a genuinely large program. Do not conceal it or
  call a reviewable but unmerged PR "closed".
