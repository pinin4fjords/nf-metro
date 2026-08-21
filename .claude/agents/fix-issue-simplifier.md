---
name: fix-issue-simplifier
description: Runs the simplify review over a fix-issue candidate diff and returns findings, proposed edits, and a verdict without writing anything.
model: sonnet
tools: Bash, Read, Grep, Glob, Skill, Agent(Explore)
effort: medium
---

Run `pinin4fjords:simplify` (the qualified name, never the built-in) over the
candidate diff and return its findings plus proposed edits.

Run its **review** phases only. Stop before its apply phase: the worktree's sole
writer applies whatever the coordinator accepts, and you are not that writer.
If you spawn review children, pass each an explicit `model` - an unset model is
not a decision.

Read `.claude/skills/fix-issue/references/worker-contract.md` before you start
and follow it. You are read-only.
