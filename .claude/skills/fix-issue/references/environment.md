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
cd ~/projects/nf-metro && pip install ".[dev,docs]"   # docs extra carries cairosvg
# Install from pyproject, never a hand-listed set: `lark` is a hard runtime
# dependency and `coverage` is needed by the gate ratchet, and a hand-written
# list drifts silently. Non-editable on purpose - PYTHONPATH shadows it below.
# Refresh this env only when pyproject deps actually change.
```

Then run the worktree's code by prepending its `src/` to `PYTHONPATH` on each
command - **do not** `pip install -e` the worktree into this env:

```bash
source ~/.local/bin/mm-activate nf-metro-dev
cd /tmp/nf-metro-fix-<N>
export PYTHONPATH=/tmp/nf-metro-fix-<N>/src
python -m nf_metro render <file.mmd> -o /tmp/out.svg --no-chrome-css
python -m pytest -k <selector>
```

CLAUDE.md now recommends `PYTHONPATH` over an editable install, which agrees
with this file. fix-issue work uses `nf-metro-dev`. Note that the
`render-topologies` skill brings its own per-issue editable env, so a renderer
briefed to use it is a deliberate exception to the rule below.

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
  git commit ...
```

Never skip hooks with `--no-verify`. Run hooks on the changed files, which is
what the commit itself does anyway:

```bash
source ~/.local/bin/mm-activate nf-metro-dev && cd <worktree> && \
  git diff --cached --name-only -z | xargs -0 -r prek run --files
```

Activate rather than `micromamba run`: the `micromamba-env` skill explains why
`run` must be avoided on this machine. `prek`'s `mypy` hook is
`language: system`, so it needs the env that carries stub-complete `mypy`.

`prek run --all-files` sweeps the entire repo, including a cold `mypy` over all
of `src/`. Reserve it for a change to the hook config itself or to a
repo-wide generated artifact; it is not the default.

## Verifier environment

Give every read-only verifier this block so its caches and logs land outside the
worktree and it can prove the tree is unchanged:

```bash
# NEVER rely on `set -e` here. The Bash tool runs zsh and evals the block, where
# ERREXIT does not fire: `(set -e; false; echo SURVIVED)` prints SURVIVED. Every
# guard below therefore fails itself explicitly.
die() { echo "VERIFY FAILED: $*" >&2; exit 1; }
source ~/.local/bin/mm-activate nf-metro-dev || die "env not activated"
export VERIFY_ARTIFACT_DIR=/tmp/nf-metro-verify-<N>-<CANDIDATE_SHA>-$$   # $$ so two readers on one SHA do not collide
mkdir -p "$VERIFY_ARTIFACT_DIR/tmp" || die "artifact dir"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$VERIFY_ARTIFACT_DIR/tmp"
export XDG_CACHE_HOME="$VERIFY_ARTIFACT_DIR/xdg-cache"
# The frozen checkout. Nothing else creates it, and reading the writer's live
# worktree is forbidden. Prune first: a previous run that died mid-block leaves
# the worktree behind and `add` then fails with "already exists".
export VERIFY_ROOT="$VERIFY_ARTIFACT_DIR/src"
git -C ~/projects/nf-metro worktree prune
test ! -e "$VERIFY_ROOT" || die "stale verify worktree at $VERIFY_ROOT; remove it first"
git -C ~/projects/nf-metro worktree add --detach "$VERIFY_ROOT" <CANDIDATE_SHA> || die "worktree add"
cd "$VERIFY_ROOT" || die "cd"

test "$(git rev-parse HEAD)" = "$(git rev-parse <CANDIDATE_SHA>)" || die "HEAD is not the candidate SHA"
st=$(git status --porcelain) || die "git status failed"   # empty stdout from a failed git reads as clean
test -z "$st" || die "tree dirty before checks"
ruff check --no-cache src/ tests/ || die "ruff check"
ruff format --check --no-cache src/ tests/ || die "ruff format"
mypy --cache-dir=/tmp/nf-metro-verify-mypy-cache || die "mypy"
PYTHONPATH=src python -m pytest tests/test_layout_invariants.py -k "<fixture-or-invariant>" \
  -q --no-header -p no:cacheprovider -p no:warnings --basetemp="$VERIFY_ARTIFACT_DIR/pytest-tmp" || die "pytest"
git diff --exit-code <CANDIDATE_SHA> || die "checks modified tracked files"
st=$(git status --porcelain) || die "git status failed"
test -z "$st" || die "tree dirty after checks"
cd ~ && git -C ~/projects/nf-metro worktree remove --force "$VERIFY_ROOT"
echo "VERIFY OK <CANDIDATE_SHA>"
```

Report the final line. **An absent `VERIFY OK` is a failure**, whatever the exit
status looked like: `die` is what makes a guard bite, and a block that merely
"ran" proves nothing.


`mypy` needs no target: `pyproject.toml` sets `files = ["src"]`. Run it exactly
as written rather than adding a path. The cache is deliberately outside the
per-SHA directory so verification is not cold every time.

## Local renders

Both of these belong in a LIGHT read-only worker that returns the verdict and
the artifact path, not the imagery, unless a genuine visual question needs the
picture in front of a HIGH reviewer. Neither replaces the CI gallery review.

`--no-chrome-css` is required on anything you intend to rasterise: without it
cairosvg aborts on the `var()` chrome custom properties.

Fast sanity check of one specific `.mmd` before pushing:

```bash
source ~/.local/bin/mm-activate nf-metro-dev
export PYTHONPATH=/tmp/nf-metro-fix-<N>/src
cd /tmp/nf-metro-fix-<N> && python -m nf_metro render <file.mmd> -o /tmp/<name>.svg --no-chrome-css
python -c "import cairosvg; cairosvg.svg2png(url='/tmp/<name>.svg', write_to='/tmp/<name>.png', scale=2)"
open /tmp/<name>.png
```

For a before/after sweep before pushing, brief the worker to use the tracked
`.claude/skills/render-topologies` skill.
