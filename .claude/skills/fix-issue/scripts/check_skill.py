#!/usr/bin/env python3
"""Self-checks for the fix-issue skill. Run from the repo root.

Exits non-zero on any failure so this can gate a commit or a CI job.
Covers what a reader cannot verify by eye: link resolution, tier naming,
owner-split accounting, agent-definition/tier-table agreement, shell syntax,
and the token budget against the post-compaction re-attachment cap.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL = Path(".claude/skills/fix-issue")
AGENTS = Path(".claude/agents")
REATTACH_CAP = 5000
# Coordinator-owned references. ALWAYS_COORD is the subset a normal run reads,
# which is what the resident-token figure is about; autonomous-mode is
# coordinator-owned but only read when the user signals that mode.
ALWAYS_COORD = {"coordinator.md", "agent-types.md", "scope-discipline.md", "merge-and-cleanup.md"}
COORD_REFS = ALWAYS_COORD | {"autonomous-mode.md"}

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def md_files() -> list[Path]:
    return [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]


def check_links() -> None:
    for f in md_files() + sorted(AGENTS.glob("*.md")):
        for _, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", f.read_text()):
            if target.startswith(("http", "#")):
                continue
            if not (f.parent / target.split("#")[0]).resolve().exists():
                fail(f"broken link in {f.name}: {target}")


def check_steps() -> None:
    body = (SKILL / "SKILL.md").read_text()
    spine = set(re.findall(r"^(\d{1,2})\. \*\*", body, re.M))
    for f in md_files():
        for n in set(re.findall(r"\bStep (\d+)\b", f.read_text())):
            if n not in spine:
                fail(f"{f.name} references Step {n}, absent from the spine")


def check_tiers_named() -> None:
    """Every spawn instruction must name a tier; the table is the contract."""
    tiers = ("LIGHT", "MID", "HIGH")
    for f in md_files():
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if not re.search(r"\b[Aa]ssign\b.*\b(worker|reviewer|verifier|specialist|investigator|assessor)\b", line):
                continue
            ctx = " ".join(lines[max(0, i - 2) : i + 2])
            if not any(t in ctx for t in tiers) and "sole writer" not in ctx:
                fail(f"{f.name}:{i} assigns a worker without naming a tier")


def check_agent_definitions() -> None:
    """Definitions must parse, and their model must agree with the tier table."""
    tier_model = {"LIGHT": "haiku", "MID": "sonnet", "HIGH": "opus"}
    table = (SKILL / "SKILL.md").read_text()
    for a in sorted(AGENTS.glob("fix-issue-*.md")):
        head = a.read_text().split("---")[1]
        fields = dict(re.findall(r"^(\w+):\s*(.+)$", head, re.M))
        for required in ("name", "description", "model", "tools", "effort"):
            if required not in fields:
                fail(f"{a.name} missing frontmatter field '{required}'")
        if fields.get("name") != a.stem:
            fail(f"{a.name} name '{fields.get('name')}' != filename")
        if fields.get("model") not in tier_model.values():
            fail(f"{a.name} model '{fields.get('model')}' is not a tier model")
        if "Agent" in fields.get("tools", "") and "Agent(" not in fields.get("tools", ""):
            fail(f"{a.name} grants unrestricted Agent; use Agent(<type>)")
        if a.stem not in table:
            fail(f"{a.name} is not named in SKILL.md")


def check_owner_split() -> None:
    """The coordinator must not be told to read worker-facing references."""
    body = (SKILL / "SKILL.md").read_text()
    rows = re.findall(r"^\| \[`([^`]+)`\][^|]*\| ([^|]+)\|", body, re.M)
    if not rows:
        fail("reference table has no owner column")
    for name, owner in rows:
        declared_coord = "coord" in owner
        if declared_coord != (name in COORD_REFS):
            fail(f"{name}: owner column says '{owner.strip()}', COORD_REFS says {name in COORD_REFS}")


def check_shell_blocks() -> None:
    for f in md_files():
        for m in re.finditer(r"```bash\n(.*?)```", f.read_text(), re.S):
            code = re.sub(r"<[^<>\n]{1,40}>", "PLACEHOLDER", m.group(1))
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
                fh.write(code)
                path = fh.name
            for shell in ("bash", "zsh"):
                r = subprocess.run([shell, "-n", path], capture_output=True, text=True)
                if r.returncode:
                    fail(f"{shell} syntax error in {f.name}: {r.stderr.strip()[:120]}")


def check_guards_self_fail() -> None:
    """`set -e` is inert here: the Bash tool runs zsh and evals the block, so
    ERREXIT never fires. Any bare `test ...` line is a decorative guard."""
    for f in md_files():
        for m in re.finditer(r"```bash\n(.*?)```", f.read_text(), re.S):
            block = m.group(1)
            if re.search(r"^\s*set -[a-z]*e", block, re.M):
                fail(f"{f.name}: shell block relies on `set -e`, which does not fire in this harness")
            for line in block.splitlines():
                stripped = line.strip()
                if stripped.startswith("test ") and "||" not in stripped:
                    fail(f"{f.name}: bare guard `{stripped[:60]}` cannot fail the block; add `|| die ...`")


def check_token_budget() -> None:
    try:
        import tiktoken
    except ImportError:
        print("  note: tiktoken absent, token budget unchecked")
        return
    enc = tiktoken.get_encoding("o200k_base")
    count = lambda p: len(enc.encode(p.read_text()))
    body = count(SKILL / "SKILL.md")
    if body > REATTACH_CAP:
        fail(f"SKILL.md is {body} tokens, over the {REATTACH_CAP} re-attachment cap")
    coord = body + sum(count(SKILL / "references" / n) for n in sorted(ALWAYS_COORD))
    print(f"  SKILL.md {body} tokens ({REATTACH_CAP - body} headroom); coordinator-resident {coord}")


def main() -> int:
    for check in (
        check_links,
        check_steps,
        check_tiers_named,
        check_agent_definitions,
        check_owner_split,
        check_shell_blocks,
        check_guards_self_fail,
        check_token_budget,
    ):
        check()
    if failures:
        print(f"FAIL: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK: fix-issue skill self-checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
