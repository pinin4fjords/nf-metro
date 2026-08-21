#!/usr/bin/env bash
# Render each changed stem at the base and candidate SHAs and rasterise both, so
# a reviewer compares like with like. Reports every stem it could not render
# rather than skipping it: a stem that renders on base and fails on the candidate
# is the regression.
set -euo pipefail

die() { echo "RENDER FAILED: $*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
usage: render_pairs.sh --base SHA --candidate SHA [--art DIR] [--stems FILE] [--repo DIR]
       render_pairs.sh --self-test

Reads stems from $ART/stems.txt (or --stems), or from scripts/gallery.yaml when
that file is absent, which is the fallback for a preview that cannot be trusted.
Writes $ART/{base,cand}-<stem>.png and prints a reconciliation summary.
USAGE
}

BASE=""
CANDIDATE=""
ART=""
STEMS=""
REPO="$HOME/projects/nf-metro"
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=$2; shift 2 ;;
    --candidate) CANDIDATE=$2; shift 2 ;;
    --art) ART=$2; shift 2 ;;
    --stems) STEMS=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
done

# render_only mixes lists of plain strings with lists of dicts; handling only one
# shape raises TypeError, which is how the documented fallback used to fail.
enumerate_corpus() {
  python3 - "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
ids = [e["id"] for e in cfg.get("gallery", [])]
ids += [e["id"] for e in cfg.get("pipelines", [])]
ids += [e if isinstance(e, str) else e["id"]
        for group in cfg.get("render_only", {}).values() for e in (group or [])]
print("\n".join(sorted(set(ids))))
PY
}

# A gallery id may carry a pipeline_ prefix its source file does not, and the
# corpus spans examples/ and tests/fixtures/, so resolve across the whole tree.
resolve_stem() {
  local stem=$1 list=$2 hit
  for candidate in "$stem" "${stem#pipeline_}"; do
    hit=$(grep -E "(^|/)${candidate}\.mmd$" "$list" | head -1) || true
    [ -n "$hit" ] && { printf '%s' "$hit"; return 0; }
  done
  return 1
}

self_test() {
  local tmp; tmp=$(mktemp -d); local rc=0
  check() { [ "$2" = "$3" ] && echo "  ok    $1" || { echo "  FAIL  $1: got $2 want $3"; rc=1; }; }

  cat > "$tmp/gallery.yaml" <<'YML'
gallery:
  - id: alpha
pipelines:
  - id: beta
render_only:
  guide_examples:
    - gamma
  nextflow_conversions:
    - id: delta
YML
  check "corpus enumeration handles both shapes" \
        "$(enumerate_corpus "$tmp/gallery.yaml" | tr '\n' ' ')" "alpha beta delta gamma "

  printf 'examples/alpha.mmd\ntests/fixtures/beta.mmd\n' > "$tmp/list"
  check "resolve plain stem" "$(resolve_stem beta "$tmp/list")" "tests/fixtures/beta.mmd"
  check "resolve pipeline_ prefix" "$(resolve_stem pipeline_alpha "$tmp/list")" "examples/alpha.mmd"
  resolve_stem nothing "$tmp/list" >/dev/null 2>&1 && { echo "  FAIL  unresolvable stem should fail"; rc=1; } \
    || echo "  ok    unresolvable stem reports failure"
  rm -rf "$tmp"
  [ $rc -eq 0 ] && echo "render_pairs.sh self-test OK" || echo "render_pairs.sh self-test FAILED"
  return $rc
}

[ "$SELF_TEST" = 1 ] && { self_test; exit $?; }
[ -n "$BASE" ] && [ -n "$CANDIDATE" ] || { usage; die "missing argument"; }
: "${ART:=/tmp/nf-metro-visual-$CANDIDATE}"
: "${STEMS:=$ART/stems.txt}"
mkdir -p "$ART" || die "cannot create $ART"
command -v cairosvg >/dev/null 2>&1 || python3 -c "import cairosvg" 2>/dev/null \
  || die "cairosvg unavailable; activate the nf-metro-dev env first"

if [ ! -s "$STEMS" ]; then
  echo "no stem list; enumerating the whole corpus from gallery.yaml" >&2
  enumerate_corpus "$REPO/scripts/gallery.yaml" > "$STEMS" || die "corpus enumeration failed"
fi

unresolved=0 failed=0 rendered=0
for side in base cand; do
  case $side in base) sha=$BASE ;; cand) sha=$CANDIDATE ;; esac
  wt="$ART/wt-$sha"
  git -C "$REPO" worktree add --detach "$wt" "$sha" >/dev/null || die "worktree add $side"
  git -C "$wt" ls-files '*.mmd' > "$ART/all-mmd.txt"
  while read -r stem; do
    [ -n "$stem" ] || continue
    if ! rel=$(resolve_stem "$stem" "$ART/all-mmd.txt"); then
      echo "UNRESOLVED: $stem" >&2; unresolved=$((unresolved + 1)); continue
    fi
    # --no-chrome-css bakes concrete colours; without it cairosvg aborts on var().
    if ! PYTHONPATH="$wt/src" python3 -m nf_metro render "$wt/$rel" \
           -o "$ART/$side-$stem.svg" --no-chrome-css 2>"$ART/$side-$stem.err"; then
      echo "RENDER-FAILED($side): $stem" >&2; failed=$((failed + 1)); continue
    fi
    python3 -c 'import cairosvg,sys;cairosvg.svg2png(url=sys.argv[1],write_to=sys.argv[1][:-4]+".png",scale=2)' \
      "$ART/$side-$stem.svg"
    rendered=$((rendered + 1))
  done < "$STEMS"
  git -C "$REPO" worktree remove --force "$wt" >/dev/null || true
done

echo "stems=$(wc -l < "$STEMS" | tr -d ' ') rendered=$rendered unresolved=$unresolved render_failed=$failed"
echo "Read the base-<stem>.png / cand-<stem>.png pairs in $ART."
echo "A stem that renders on base and fails on cand IS the regression; failures on"
echo "both sides pre-date this PR - diff the .err files, a new reason is a finding."
