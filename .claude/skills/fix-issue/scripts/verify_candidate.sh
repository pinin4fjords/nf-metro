#!/usr/bin/env bash
# Verify a candidate SHA against a frozen checkout. Prints VERIFY OK on success;
# an absent VERIFY OK is a failure whatever the exit status looked like.
#
# A script, not a documented block: `set -e` does fire here, so every guard bites
# without needing to be written defensively, and the frozen worktree is created
# and removed in the same process that asserts against it.
set -euo pipefail

die() { echo "VERIFY FAILED: $*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
usage: verify_candidate.sh --candidate SHA [--selector K] [--repo DIR] [--art DIR]
       verify_candidate.sh --self-test

--selector is passed to pytest -k; omit it to run the layout invariant suite.
USAGE
}

CANDIDATE=""
SELECTOR=""
REPO="$HOME/projects/nf-metro"
ART=""
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --candidate) CANDIDATE=$2; shift 2 ;;
    --selector) SELECTOR=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --art) ART=$2; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
done

if [ "$SELF_TEST" = 1 ]; then
  rc=0
  # The point of the script is that a failed guard stops the run. Prove it.
  want=a; got=b
  ( set -euo pipefail; [ "$got" = "$want" ] || die "mismatch"; echo "PAST GUARD" ) >/dev/null 2>&1 \
    && { echo "  FAIL  a failed guard did not stop the run"; rc=1; } \
    || echo "  ok    a failed guard stops the run"
  # An empty `git status` from a failed git must not read as a clean tree.
  if (cd /tmp && git status --porcelain >/dev/null 2>&1); then st=ok; else st=failed; fi
  [ "$st" = failed ] && echo "  ok    git failure is distinguishable from clean" \
    || { echo "  FAIL  git failure looks like a clean tree"; rc=1; }
  [ $rc -eq 0 ] && echo "verify_candidate.sh self-test OK" || echo "verify_candidate.sh self-test FAILED"
  exit $rc
fi

[ -n "$CANDIDATE" ] || { usage; die "missing --candidate"; }
: "${ART:=/tmp/nf-metro-verify-$CANDIDATE-$$}"
mkdir -p "$ART/tmp" || die "cannot create $ART"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$ART/tmp" XDG_CACHE_HOME="$ART/xdg-cache"

root="$ART/src"
git -C "$REPO" worktree prune
[ ! -e "$root" ] || die "stale verify worktree at $root; remove it first"
git -C "$REPO" worktree add --detach "$root" "$CANDIDATE" >/dev/null || die "worktree add"
cleanup() { cd "$HOME"; git -C "$REPO" worktree remove --force "$root" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cd "$root"

[ "$(git rev-parse HEAD)" = "$(git rev-parse "$CANDIDATE")" ] || die "HEAD is not the candidate"
st=$(git status --porcelain) || die "git status failed"
[ -z "$st" ] || die "tree dirty before checks"

ruff check --no-cache src/ tests/ || die "ruff check"
ruff format --check --no-cache src/ tests/ || die "ruff format"
# mypy needs no target: pyproject sets files = ["src"]. Cache lives outside the
# per-SHA dir so verification is not cold every time.
mypy --cache-dir=/tmp/nf-metro-verify-mypy-cache || die "mypy"

if [ -n "$SELECTOR" ]; then
  PYTHONPATH=src python3 -m pytest tests/test_layout_invariants.py -k "$SELECTOR" \
    -q --no-header -p no:cacheprovider -p no:warnings --basetemp="$ART/pytest-tmp" || die "pytest"
else
  PYTHONPATH=src python3 -m pytest tests/test_layout_invariants.py \
    -q --no-header -p no:cacheprovider -p no:warnings --basetemp="$ART/pytest-tmp" || die "pytest"
fi

git diff --exit-code "$CANDIDATE" || die "checks modified tracked files"
st=$(git status --porcelain) || die "git status failed"
[ -z "$st" ] || die "tree dirty after checks"
echo "VERIFY OK $CANDIDATE"
