#!/usr/bin/env bash
# Establish what the render preview says, and which examples changed.
#
# This is a script rather than a documented block because the logic is a chain
# that has to share state: as separate Bash calls, variables do not survive
# between steps and `set -e` does not fire under the harness's eval. Here both
# work. Run --self-test to exercise the parsing without touching the network.
set -euo pipefail

die() { echo "VISUAL FAILED: $*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
usage: visual_preview.sh --pr N --branch REF --candidate SHA [--art DIR]
       visual_preview.sh --self-test

Exit 0 and print "VERDICT: no visual changes" when the run was clean; there is
nothing to review and no page is published in that case.
Exit 0 and print "STEMS: <n>" with $ART/stems.txt populated when deltas exist.
Any other outcome exits non-zero with a reason.
USAGE
}

PR=""
BRANCH=""
CANDIDATE=""
ART=""
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --pr) PR=$2; shift 2 ;;
    --branch) BRANCH=$2; shift 2 ;;
    --candidate) CANDIDATE=$2; shift 2 ;;
    --art) ART=$2; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
done

# The sticky comment carries five wordings. Only the first two are verdicts.
classify_sticky() {
  local body_file=$1
  grep -q "no visual changes detected" "$body_file" && { echo clean; return; }
  grep -q "was not generated because" "$body_file" && { echo ci-failed; return; }
  grep -qE "rendering in progress|still publishing" "$body_file" && { echo pending; return; }
  echo deltas
}

self_test() {
  local tmp; tmp=$(mktemp -d); local rc=0
  check() { [ "$2" = "$3" ] && echo "  ok    $1" || { echo "  FAIL  $1: got $2 want $3"; rc=1; }; }

  # A real sticky body is multi-line with the HTML marker last, which is why a
  # line-wise tail of it silently matches nothing.
  printf '**Render preview**: no visual changes detected. All renders match `main`.\n<!-- Sticky Pull Request Commentrender-preview -->\n' > "$tmp/clean"
  check "clean verdict" "$(classify_sticky "$tmp/clean")" clean
  printf '**Render preview** was not generated because a prerequisite check or the render job failed.\n' > "$tmp/failed"
  check "ci-failed verdict" "$(classify_sticky "$tmp/failed")" ci-failed
  printf '**Render preview**: rendered and pushed; GitHub Pages is still publishing it.\n' > "$tmp/pending"
  check "pending verdict" "$(classify_sticky "$tmp/pending")" pending
  printf '**Render preview**: ready for review: https://example/_pr/1/\n' > "$tmp/deltas"
  check "deltas verdict" "$(classify_sticky "$tmp/deltas")" deltas

  # Stem extraction and the run marker both come off the published page.
  printf '<meta name="nf-metro-render-run" content="12345">\n<div class="diff-entry" id="alpha"><div class="diff-entry" id="beta">\n' > "$tmp/page"
  check "marker extraction" "$(extract_marker "$tmp/page")" 12345
  check "stem extraction" "$(extract_stems "$tmp/page" | tr '\n' ' ')" "alpha beta "
  rm -rf "$tmp"
  [ $rc -eq 0 ] && echo "visual_preview.sh self-test OK" || echo "visual_preview.sh self-test FAILED"
  return $rc
}

extract_marker() {
  grep -o 'nf-metro-render-run" content="[0-9]*"' "$1" | grep -o '[0-9]*' | head -1
}

extract_stems() {
  grep -oE '<div class="diff-entry" id="[^"]+"' "$1" | sed -E 's/.*id="([^"]+)".*/\1/' | sort -u
}

[ "$SELF_TEST" = 1 ] && { self_test; exit $?; }
[ -n "$PR" ] && [ -n "$BRANCH" ] && [ -n "$CANDIDATE" ] || { usage; die "missing argument"; }
: "${ART:=/tmp/nf-metro-visual-$CANDIDATE}"
mkdir -p "$ART" || die "cannot create $ART"

run=$(gh run list --workflow pr-renders.yml --branch "$BRANCH" --limit 1 \
        --json databaseId,headSha) || die "gh run list failed"
built=$(printf '%s' "$run" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d[0]["headSha"] if d else "")')
runid=$(printf '%s' "$run" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d[0]["databaseId"] if d else "")')
[ -n "$built" ] && [ -n "$runid" ] || die "no pr-renders run for $BRANCH yet"
[ "$(git rev-parse "$built")" = "$(git rev-parse "$CANDIDATE")" ] \
  || die "renders were built from $built, not the candidate $CANDIDATE"

# Anchor on the sticky marker: a human comment mentioning "Render preview" would
# otherwise be selected as the latest match. jq prints a literal null on no match.
gh pr view "$PR" --json comments \
  -q '[.comments[] | select(.body | contains("Sticky Pull Request Commentrender-preview"))] | last | .body' \
  > "$ART/sticky.txt" || die "gh pr view failed"
[ -s "$ART/sticky.txt" ] || die "no sticky render-preview comment yet"
grep -qx "null" "$ART/sticky.txt" && die "no sticky render-preview comment yet"

case "$(classify_sticky "$ART/sticky.txt")" in
  clean)     echo "VERDICT: no visual changes"; exit 0 ;;
  ci-failed) die "the render job itself failed; fix CI, waiting will not help" ;;
  pending)   die "renders not published yet; wait and re-run" ;;
esac

# The preview publishes one inlined index.html; there are no fetchable .svg files
# and the page is multi-megabyte, so never read it into context.
curl -fsS "https://seqeralabs.github.io/nf-metro/_pr/$PR/" -o "$ART/index.html" \
  || die "page absent though the comment reported deltas"
marker=$(extract_marker "$ART/index.html")
[ -n "$marker" ] || die "no run marker in page"
# keep_files: true leaves an earlier push's page in place, so headSha is not enough.
[ "$marker" = "$runid" ] || die "page is from run $marker, not $runid: stale preview"

extract_stems "$ART/index.html" > "$ART/stems.txt"
[ -s "$ART/stems.txt" ] || die "no stems parsed from a page that reported deltas"
echo "STEMS: $(wc -l < "$ART/stems.txt" | tr -d ' ')"
