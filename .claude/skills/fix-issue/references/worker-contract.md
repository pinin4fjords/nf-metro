# Worker contract

Every brief in this workflow points here instead of restating these rules. If
you are a worker on this run, these terms apply to you; read them once and
follow them without being told again.

## Authority

Your brief names your authority as either **read-only** or **sole writer in
<worktree>**. There is exactly one writer per worktree.

- **Read-only** means you persist nothing in the worktree: no tracked,
  untracked, or ignored changes. Put logs, caches, and generated evidence in
  the artifact directory your brief names, outside the worktree. Work only
  against the frozen SHA you were given, never a live worktree while a writer
  is active.
- **Sole writer** means you make the local candidate commit(s), run any
  mutation-capable hooks or generators, resolve their output, and hand off the
  exact SHA. You do not push.
- **Nobody but the coordinator** pushes, edits an issue or PR, merges,
  retargets, or cleans up. If your work seems to need one of those, that is an
  escalation, not a task.

## What you return

1. scope completed;
2. files changed and candidate commit SHA, or an explicit no-change result;
3. exact commands and outcomes;
4. before/after evidence **as a path**, plus the single number or line that
   changed. Do not paste coordinate dumps, render analysis, test logs, or file
   contents into your final message: write them to the artifact directory and
   cite the path. The one figure that carries the verdict belongs inline; the
   material behind it does not;
5. risks and blockers;
6. acceptance verdict: pass, fail, or blocked with the precise escalation.

Returning **blocked** against this schema is a valid, useful outcome. It is
better than guessing, better than an unbounded loop, and better than silently
narrowing your scope. Say precisely what would unblock you.

## Judgment above your tier

Your brief names a tier. If the work turns out to need judgment beyond it, say
so and return blocked rather than pressing on. Do not quietly do harder work
than you were briefed for, and do not pad a mechanical result with speculation.

## If you run verifier commands

Use the environment block and command sequence in
[environment.md](environment.md) verbatim, including the clean-tree assertions
before and after. Report exit codes and a concise failure excerpt, not full
output.
