# Agent types and the Explore trade-off

Read before substituting `Explore` for a named role type.

## When to use `Explore` instead

`Explore` and `Plan` are the only agent types that skip both CLAUDE.md files and
the git-status snapshot. Measured with tiktoken on the actual files, that is
**3,849 tokens per spawn** here (project CLAUDE.md 2,246 + user 1,603). Measured over the historical corpus that is worth about $1.65 a run once the
resident re-reads are counted, not just the one-off cache write. There is no frontmatter field that exempts a custom type, so this is
the only lever. `Explore` is read-only - `Write` and `Edit` are denied - and it
is one-shot: it returns no agent ID and cannot be resumed or re-briefed.

Two consequences before you reach for it:

- It runs its own system prompt, **not** your role definition, so everything in
  `fix-issue-verifier.md` or `fix-issue-investigator.md` is discarded. The entire
  brief has to be in the task message.
- It inherits the main conversation's model, so omitting the `model` parameter
  runs it at the session tier - the exact failure this skill exists to prevent.

Eligible only for the investigator and the verifier, where the repo architecture
is irrelevant - but measurement says the saving is negligible and the lost
re-brief is not, so prefer the named role types. Never for the
diagnostician, writer, visual reviewer, or reviewer: they need the architecture
map and the station-as-elbow constraint.

## The model resolution order

Highest wins:

1. the `CLAUDE_CODE_SUBAGENT_MODEL` environment variable;
2. the per-invocation `model` parameter;
3. the agent definition's `model` frontmatter;
4. the main conversation's model.

If `CLAUDE_CODE_SUBAGENT_MODEL` is set it overrides **every** tier decision in
this skill, so check it once at session start. An organisation `availableModels`
allowlist can also substitute a model at any of these levels.

## Effort, and enforcing read-only

**`effort` is a second dial, but a fixed one.** Definitions take `effort`
(`low`/`medium`/`high`/`xhigh`/`max`) and it is set for the life of the
definition: there is no per-invocation `effort`, only a per-invocation `model`.
So the LIGHT roles run `low` and the judgment roles run `high`, and the
persistent writer cannot be dialled down for its mechanical re-briefs. Accept
that: the retained context is worth more than the thinking tokens, which is all
`effort` moves.

The read-only roles carry no `Edit` or `Write`. Treat that as a backstop, not a
guarantee: they all hold `Bash`, so a worker that ignores its brief can still
write, and `permissionMode` will not save you - under the parent's auto mode a
subagent's `permissionMode` is ignored. The instruction is what enforces
read-only. If you want it enforced structurally, the lever is the per-subagent
`hooks` field: a `PreToolUse` matcher on `Bash` rejecting `git push`,
`gh pr merge` and `gh pr edit`. The roles holding `Skill` are the simplifier and the
renderer; `fix-issue-merge-assessor` deliberately holds none, so it cannot reach
a procedure that performs the merge it is reviewing.
