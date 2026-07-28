"""Geometry for one revision: a private worktree, one checkout, one extract.

Each revision is measured with **its own** engine and **its own** fixture files,
but a single fixed set of feature definitions, so a feature's meaning cannot
drift underneath the dataset. That is what a detached worktree buys: the
extractor is invoked from today's checkout while ``sys.path`` points at the
revision's ``src/``.

Records are cached per SHA, so a revision measured once is free thereafter. The
cache matters most for a sweep: consecutive merges to ``main`` share a commit
(one merge's ``mergeCommit`` is the next one's ``mergeCommit^1``), so capturing
*k* PRs costs about *k + 1* extractions rather than *2k*.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
REPO_ROOT = S.parents[2]
EXTRACTOR = S / "extract_features.py"
CACHE = S / "geometry"

ENGINE_PATHS = ("src/nf_metro/layout", "src/nf_metro/render", "src/nf_metro/parser")
"""Paths whose change can move geometry."""


class GeometryError(RuntimeError):
    """A revision's geometry could not be produced (bad checkout or extractor)."""


class MissingRevision(GeometryError):
    """The commit is not reachable from anywhere, so it can never be measured.

    Distinct from a failed measurement: the 2026-07-27 history rewrite left
    pre-rewrite merge commits unreachable, and no amount of retrying will bring
    them back. A caller sweeping many revisions can record these as settled and
    keep retrying the merely-failed ones.
    """


def git(
    *args: str, cwd: Path | None = None, timeout: int = 300
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd or REPO_ROOT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def rev_parse(spec: str) -> str | None:
    return git("rev-parse", spec).stdout.strip() or None


def touches_engine(before: str, after: str) -> bool:
    """Whether a revision range's own diff reaches code that can move geometry."""
    files = git("diff", "--name-only", f"{before}..{after}").stdout
    return any(p in files for p in ENGINE_PATHS)


def ensure_local(sha: str) -> None:
    """Fetch ``sha`` if it is not already in the object store."""
    if git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0:
        return
    git("fetch", "origin", timeout=900)
    if git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        raise MissingRevision(
            f"commit {sha[:9]} is not in this clone and origin lacks it"
        )


def ensure_worktree(path: Path) -> Path:
    """A detached worktree of this repo at ``path``, created if absent."""
    if not (path / ".git").exists():
        r = git("worktree", "add", str(path), "--detach", "HEAD")
        if r.returncode:
            raise GeometryError(f"worktree add {path} failed: {r.stderr.strip()[:300]}")
    return path


def remove_worktree(path: Path) -> None:
    git("worktree", "remove", "--force", str(path))


def geometry_at(sha: str, *, worktree: Path, cache: Path = CACHE) -> dict:
    """The extractor's per-fixture record for ``sha``, via ``cache``."""
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{sha}.json"
    if dest.exists():
        rec = json.loads(dest.read_text())
        if "fixtures" in rec:
            return rec
        raise GeometryError(f"{sha[:9]} previously failed: {rec.get('error')}")

    ensure_local(sha)
    co = git("checkout", "--force", "--detach", sha, cwd=worktree, timeout=180)
    if co.returncode:
        raise GeometryError(f"checkout of {sha[:9]} failed: {co.stderr.strip()[:300]}")
    ex = subprocess.run(
        [
            sys.executable,
            str(EXTRACTOR),
            "--worktree",
            str(worktree),
            "--sha",
            sha,
            "--out",
            str(dest),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if ex.returncode or not dest.exists():
        raise GeometryError(f"extract of {sha[:9]} failed: {ex.stderr.strip()[-600:]}")
    rec = json.loads(dest.read_text())
    if "fixtures" not in rec:
        raise GeometryError(f"{sha[:9]} yielded no fixtures: {rec.get('error')}")
    return rec
