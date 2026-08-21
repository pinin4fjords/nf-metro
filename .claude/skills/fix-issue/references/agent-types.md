# Agent types and the Explore trade-off

Read before substituting `Explore` for a named role type.

## When to use `Explore` instead

`Explore` and `Plan` are the only agent types that skip both CLAUDE.md files and
the git-status snapshot, worth about 5k tokens per spawn here (project 10.9KB +
user 9.7KB). There is no frontmatter field that exempts a custom type, so this is
the only lever. `Explore` is read-only - `Write` and `Edit` are denied - and it
is one-shot: it returns no agent ID and cannot be resumed or re-briefed.

Two consequences before you reach for it:

- It runs its own system prompt, **not** your role definition, so everything in
  `fix-issue-verifier.md` or `fix-issue-investigator.md` is discarded. The entire
  brief has to be in the task message.
- It inherits the main conversation's model, so omitting the `model` parameter
  runs it at the session tier - the exact failure this skill exists to prevent.

Worth it for the investigator reading an issue and the verifier running a fixed
command block, where the repo architecture is irrelevant. Never for the
diagnostician, writer, visual reviewer, or reviewer: they need the architecture
map and the station-as-elbow constraint.
