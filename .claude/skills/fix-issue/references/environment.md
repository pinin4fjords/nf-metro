# Environment and commit hooks

## Reuse one persistent env, don't create one per issue

nf-metro is pure Python; the deps (`cairo`, drawsvg, networkx, pillow, cairosvg,
pytest, ruff, mypy, `types-networkx`) change rarely. Creating a fresh
`micromamba` env per issue re-solves and re-downloads all of that every session
for no benefit. Keep **one** long-lived deps env and point it at the worktree's
code per-command:

```bash
# One-time, reused across all issues (skip if it already exists):
ulimit -n 1000000 && export CONDA_OVERRIDE_OSX=15.0 && /opt/homebrew/bin/micromamba create -n nf-metro-dev python=3.11 cairo -y
source ~/.local/bin/mm-activate nf-metro-dev
pip install "drawsvg" "networkx" "pillow" "cairosvg" "pytest" "pytest-xdist" "ruff" "mypy" "types-networkx" "click"
# Refresh this env only when pyproject deps actually change.
```

Then run the worktree's code by prepending its `src/` to `PYTHONPATH` on each
command - **do not** `pip install -e` the worktree into this env:

```bash
source ~/.local/bin/mm-activate nf-metro-dev
cd /tmp/nf-metro-fix-<N>
export PYTHONPATH=/tmp/nf-metro-fix-<N>/src
python -m nf_metro render <file.mmd> -o /tmp/out.svg    # runs THIS worktree
python -m pytest -k <selector>
```

**Why per-command `PYTHONPATH`, not editable install:** an editable install binds
one env's `site-packages` to exactly one worktree path, so it collides the moment
you run two worktrees in parallel. `PYTHONPATH` is set per command and shadows
whatever is installed, so any number of parallel worktree sessions share the
single `nf-metro-dev` env with zero cross-talk. (If you genuinely want an
isolated editable install for one worktree, dedicate a *separate* env to it -
never editable-install a shared env against a worktree.)

Python 3.11 is the pinned interpreter for the routing/TB ratchets; those tests
skip silently off it (see [gate-ratchet.md](gate-ratchet.md)).

## Commit hooks

Hooks need the tools on `PATH` in the same Bash call: the repo uses `prek`
(config `prek.toml`, not `pre-commit`), whose `mypy` hook is `language: system`
and so needs `mypy` on `PATH`. Shell state does not persist between Bash calls,
so run the commit as one call with the env activated:

```bash
source ~/.local/bin/mm-activate nf-metro-dev && cd <worktree> && \
  PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit ...
```

Never skip hooks with `--no-verify`. Run hooks on the changed files, which is
what the commit itself does anyway:

```bash
micromamba run -n nf-core prek run --files $(git diff --cached --name-only)
```

`prek run --all-files` sweeps the entire repo, including a cold `mypy` over all
of `src/`. Reserve it for a change to the hook config itself or to a
repo-wide generated artifact; it is not the default.

## Verifier environment

Give every read-only verifier this block so its caches and logs land outside the
worktree and it can prove the tree is unchanged:

```bash
export VERIFY_ARTIFACT_DIR=/tmp/nf-metro-verify-<N>-<CANDIDATE_SHA>
mkdir -p "$VERIFY_ARTIFACT_DIR/tmp"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$VERIFY_ARTIFACT_DIR/tmp"
export XDG_CACHE_HOME="$VERIFY_ARTIFACT_DIR/xdg-cache"
# mypy's cache is deliberately NOT per-SHA: a fresh cache per candidate makes
# every verifier run cold. Keep it outside the worktree and reuse it.
test "$(git rev-parse HEAD)" = <CANDIDATE_SHA>
test -z "$(git status --porcelain)"
ruff check --no-cache src/ tests/
ruff format --check --no-cache src/ tests/
mypy --cache-dir=/tmp/nf-metro-verify-mypy-cache   # persistent, outside the worktree
PYTHONPATH=src python -m pytest tests/test_layout_invariants.py -k "<fixture-or-invariant>" -q --no-header -p no:cacheprovider --basetemp="$VERIFY_ARTIFACT_DIR/pytest-tmp"
git diff --exit-code <CANDIDATE_SHA>
test -z "$(git status --porcelain)"
```

## Local renders

Both of these belong in a LIGHT read-only worker that returns the verdict and
the artifact path, not the imagery, unless a genuine visual question needs the
picture in front of a HIGH reviewer. Neither replaces the CI gallery review.

Fast sanity check of one specific `.mmd` before pushing:

```bash
source ~/.local/bin/mm-activate nf-metro-dev
export PYTHONPATH=/tmp/nf-metro-fix-<N>/src
cd /tmp/nf-metro-fix-<N> && python -m nf_metro render <file.mmd> -o /tmp/<name>.svg
python -c "import cairosvg; cairosvg.svg2png(url='/tmp/<name>.svg', write_to='/tmp/<name>.png', scale=2)"
open /tmp/<name>.png
```

For a before/after sweep before pushing, brief the worker to use the tracked
`.claude/skills/render-topologies` skill.
