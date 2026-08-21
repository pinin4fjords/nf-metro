# Visual review and narrowing

Step 8's detail and Step 9 in full.

### Primary method: CI render preview (authoritative)

**This is the one and only push and draft-PR creation in the run.** The
coordinator does it here, once; no worker performs these remote mutations.
Steps 10 and 11 do not repeat it - Step 10 verifies the push landed and edits
the PR body, Step 11 only flips the draft to ready. Every push costs a full CI
matrix plus a gallery render of both branches, so batch accepted fixes into one
push rather than pushing per fix.

The one exception is a failure that **cannot** be seen locally. The routing
ratchets skip off Python 3.11, so they surface only in CI; if the diff touches
`src/nf_metro/layout/routing/`, require the verifier to run under 3.11 so the
failure lands before the push rather than after it. When CI still reveals a
gate failure the local run could not, fixing it earns a second push - that is
not a violation of the rule, it is the rule's stated exception. The CI workflow
(`.github/workflows/pr-renders.yml`) automatically renders all gallery
examples on both the PR branch and base, generates a before/after visual
diff page, and posts a sticky comment on the PR with the preview link:

```
https://seqeralabs.github.io/nf-metro/_pr/<PR_NUMBER>/
```

### Getting the renders in front of the reviewer

The preview publishes **one** file, `index.html`, with every render inlined by
`_inline_svg` in `scripts/build_render_diff.py`. There are no fetchable `.svg`
files, the page is multi-megabyte, and the inlined markup is full of `var()` and
`light-dark()` that cairosvg cannot parse. So: never `Read` the preview page, and
do not try to download renders from it.

**Read the verdict before reaching for the page.** When a PR has no visual
changes the preview is never published at all: `build_render_diff.py` returns
early, `pr-renders.yml` sets `has_changes=false`, and every deploy step in
`pr-render-publish.yml` is gated on `has_changes == 'true'`. So a 404 is the
*expected* result of a clean run, not a reason to wait.

```bash
die() { echo "VISUAL FAILED: $*" >&2; exit 1; }
source ~/.local/bin/mm-activate nf-metro-dev || die "env"   # `python` is not on PATH otherwise
export ART=/tmp/nf-metro-visual-<CANDIDATE_SHA>; mkdir -p "$ART" || die "artifact dir"
run=$(gh run list --workflow pr-renders.yml --branch <HEAD_BRANCH> --limit 1 \
        --json databaseId,headSha,conclusion) || die "gh run list"
built=$(echo "$run" | python -c "import json,sys;d=json.load(sys.stdin);print(d[0]['headSha'] if d else '')")
test -n "$built" || die "no pr-renders run for this branch yet"
runid=$(echo "$run" | python -c "import json,sys;d=json.load(sys.stdin);print(d[0]['databaseId'] if d else '')")
test -n "$runid" || die "no pr-renders run for this branch yet"
# Normalise both sides: gh returns 40 chars, a hand-copied candidate may be short.
test "$(git rev-parse "$built")" = "$(git rev-parse <CANDIDATE_SHA>)" \
  || die "renders were built from $built, not the candidate"
# `last | .body`, not `| tail -1`: the body is multi-line and its last LINE is
# the sticky HTML marker, so a line-wise tail makes every grep below miss.
# Anchor on the sticky marker, not the prose: a human comment saying "the Render
# preview looks fine" would otherwise be selected as the latest match.
gh pr view <PR_NUMBER> --json comments \
  -q '[.comments[] | select(.body | contains("Sticky Pull Request Commentrender-preview"))] | last | .body' \
  > "$ART/sticky.txt" || die "gh pr view failed"
# jq prints the literal "null" and exits 0 when nothing matches.
# zsh rejects `a && ! b`, and this harness is zsh, so keep the guards separate.
test -s "$ART/sticky.txt" || die "no sticky render-preview comment yet"
grep -qx "null" "$ART/sticky.txt" && die "no sticky render-preview comment yet"
grep -q "no visual changes detected" "$ART/sticky.txt" && { echo "VERDICT: no visual changes"; exit 0; }
grep -q "was not generated because" "$ART/sticky.txt" \
  && die "the render job itself failed; fix CI, waiting will not help"
grep -qE "rendering in progress|still publishing" "$ART/sticky.txt" && die "not ready yet, wait"
```

Exit 0 with "no visual changes" **is** the verdict for a clean run: report it and
stop. Only continue when the comment says deltas exist.

**Then prove the published page is this run's.** `pr-render-publish.yml` deploys
with `keep_files: true` and only cleans up when the PR closes, so a previous
push's page survives. Comparing the workflow's `headSha` is not enough: push A
with deltas publishes a page, push B without deltas publishes nothing, and the
headSha check passes while the page you fetch is A's. The page carries the
discriminator - `build_render_diff.py` writes
`<meta name="nf-metro-render-run" content="{run id}">`.

```bash
die() { echo "VISUAL FAILED: $*" >&2; exit 1; }   # each block is self-contained
export ART=/tmp/nf-metro-visual-<CANDIDATE_SHA>; mkdir -p "$ART" || die "artifact dir"
# Re-derive runid here: block variables do not survive between Bash calls.
runid=$(gh run list --workflow pr-renders.yml --branch <HEAD_BRANCH> --limit 1 \
          --json databaseId -q '.[0].databaseId') || die "gh run list"
test -n "$runid" || die "no pr-renders run for this branch yet"
curl -fsS "https://seqeralabs.github.io/nf-metro/_pr/<PR_NUMBER>/" -o "$ART/index.html" \
  || die "page absent though the comment reported deltas; Pages may still be publishing"
marker=$(grep -o 'nf-metro-render-run" content="[0-9]*"' "$ART/index.html" \
           | grep -o '[0-9]*') || die "no run marker in page"
test "$marker" = "$runid" || die "page is from run $marker, not $runid: stale preview"
grep -oE '<div class="diff-entry" id="[^"]+"' "$ART/index.html" \
  | sed -E 's/.*id="([^"]+)".*/\1/' | sort -u > "$ART/stems.txt"
test -s "$ART/stems.txt" || die "no stems parsed from a page that reported deltas"
wc -l < "$ART/stems.txt"
```

The sticky comment has five wordings, not two: no visual changes, ready for review,
rendering in progress, Pages still publishing, and "was not generated because a
prerequisite check or the render job failed". Only the first two are verdicts.
The last is not a wait: CI failed, and no amount of waiting fixes it.

**If provenance fails, do not fall back to the stale list.** Enumerate the corpus
from `scripts/gallery.yaml` and compare both sides yourself: the delta you care
about is exactly the part an old preview cannot show.

```bash
die() { echo "VISUAL FAILED: $*" >&2; exit 1; }
source ~/.local/bin/mm-activate nf-metro-dev || die "env"
export ART=/tmp/nf-metro-visual-<CANDIDATE_SHA>; mkdir -p "$ART" || die "artifact dir"
# render_only mixes list-of-str (guide_examples, test_fixtures) with list-of-dict
# (nextflow_conversions), so handle both or this raises TypeError.
python -c "import yaml;c=yaml.safe_load(open('scripts/gallery.yaml'));\
ids=[e['id'] for e in c.get('gallery',[])]+[e['id'] for e in c.get('pipelines',[])];\
ids+=[e if isinstance(e,str) else e['id'] for g in c.get('render_only',{}).values() for e in (g or [])];\
print('\n'.join(sorted(set(ids))))" > "$ART/stems.txt" || die "could not enumerate gallery.yaml"
wc -l < "$ART/stems.txt"   # the whole corpus, not just the changed subset
```

Then run the both-sides sweep below over that list and diff the PNG pairs
yourself; `render_only` groups can map an id to a different output name, so
expect some `UNRESOLVED` and report them. In a
measured quarter-sample of the 248 ids missing from one stale preview's list, four
rendered differently and five aborted on the candidate but not the base.

### Rendering both sides

Then render each stem at the base SHA and the candidate SHA and rasterise both.
This block is **self-contained**: shell state does not survive between Bash
calls, so it re-exports `ART` rather than relying on the block above.

```bash
die() { echo "VISUAL FAILED: $*" >&2; exit 1; }
export ART=/tmp/nf-metro-visual-<CANDIDATE_SHA>; mkdir -p "$ART" || die "artifact dir"
test -s "$ART/stems.txt" || die "run the provenance block first"
source ~/.local/bin/mm-activate nf-metro-dev || die "env"
for side in base cand; do
  case $side in base) SHA=<BASE_SHA>;; cand) SHA=<CANDIDATE_SHA>;; esac
  W="$ART/wt-$SHA"
  git -C ~/projects/nf-metro worktree add --detach "$W" "$SHA" || die "worktree add $side"
  git -C "$W" ls-files '*.mmd' > "$ART/all-mmd.txt"
  while read -r stem; do
    # pipeline_<stem> is the same .mmd with the same options, so one verdict covers both.
    for cand in "$stem" "${stem#pipeline_}"; do
      f=$(grep -E "(^|/)${cand}\.mmd$" "$ART/all-mmd.txt" | head -1) && [ -n "$f" ] && break
    done
    if [ -z "${f:-}" ]; then echo "UNRESOLVED: $stem" >&2; continue; fi
    if ! PYTHONPATH="$W/src" python -m nf_metro render "$W/$f" \
         -o "$ART/$side-$stem.svg" --no-chrome-css 2>"$ART/$side-$stem.err"; then
      echo "RENDER-FAILED($side): $stem" >&2
      continue
    fi
    python -c "import cairosvg,sys; cairosvg.svg2png(url=sys.argv[1], write_to=sys.argv[1][:-4]+'.png', scale=2)" "$ART/$side-$stem.svg"
  done < "$ART/stems.txt"
  git -C ~/projects/nf-metro worktree remove --force "$W"
done
```

Three failure shapes, and only one is benign:

- `RENDER-FAILED(cand)` where base rendered: **this is the regression**, a
  D-delta of the worst kind. Never skip it.
- `RENDER-FAILED` on **both** sides: pre-existing at head, so not this PR's
  regression - but still report it by name, and `diff` the two `.err` files: the
  same fixture aborting for a *different reason* is a finding. There is no
  declared allow-list of aborting fixtures in this repo, so never treat an abort
  as expected without comparing both sides yourself.
- `UNRESOLVED`: a gallery id with no matching `.mmd`, e.g. one rendered from a
  Nextflow DAG or with non-default options. Name it; do not let it vanish.

Reconcile the count before judging anything: the number of `cand-*.png` files
must equal `wc -l < "$ART/stems.txt"` minus the `UNRESOLVED` lines. Report every
unresolved stem by name; a gallery entry whose output id differs from its source
stem lands here, and skipping it silently is how a regression ships.

`Read` the `base-<stem>.png` / `cand-<stem>.png` pairs. Every stem in
`stems.txt` owes an I/N/D verdict; if you could not produce an image for one,
say which and why rather than passing over it silently.

### Render-preview verdict gating

Read the sticky comment's verdict line first, then size the review to it. A
LIGHT worker may report a literal "No visual changes detected" verdict, since
there is nothing to judge. **The moment any delta exists, a fresh HIGH read-only
visual reviewer inspects every changed example**, classifies deltas I/N/D,
identifies uncertainty, and returns an acceptance verdict. Do not seed it with
the writer's preferred interpretation, and do not downgrade this because the
issue predicted no visual change - a delta nobody expected is the most important
kind. Every changed render gets HIGH eyes.

The sticky comment ends in a verdict line. Gate the next step on it:

- **"no visual changes detected"** (lowercase, as emitted) -> a clean result, but **not** a
  licence to merge. Report the verdict and wait for the user to say
  merge. There is no standing auto-merge authorisation.
- **"Ready for review"** (or any wording indicating visual deltas exist)
  -> gate on the independent visual verdict. Re-brief the writer for every D;
  surface accepted I/N deltas and unresolved uncertainty to the user with one
  short evidence-based line per affected gallery example.

Merge authority, push hygiene, and cleanup:
[`merge-and-cleanup.md`](merge-and-cleanup.md). Leave
branch deletion to Step 12 so dependent PRs can be retargeted safely.

### State the evidence for every "it's fixed" claim

Never assert a fix works without an independent verifier naming what proved it.
Every "resolved" / "this is fixed" / "renders correctly" claim must cite the
**specific render and the concrete numbers** it was checked against - the file,
and the coordinate or element that moved from the observed value to the target
value you wrote down in Step 3. "I believe it's resolved" with no named render
is not a verdict; it invites the reply "which render did you re-assess on?".

Two traps this closes:

- **"Didn't abort" / "the one invariant passes" is not "renders
  correctly".** Removing an abort can merely expose a poor layout the abort
  was masking. After any layout/routing fix, the LIGHT verifier runs `probe_layout` and
  `inspect_layout` on the *candidate* SHA and hands over the numbers; the HIGH
  visual reviewer inspects the full render (cropping as needed) and makes the
  judgment. Do not ask a LIGHT worker to assess a render - the after-state counterpart to
  Step 3's before-state reading, not a repeat of it - for the whole-layout picture (crossings, port alignment,
  column gaps), not only the targeted invariant.
- **A clean render-diff verdict only covers the gallery corpus.** It says
  nothing about a NEW fixture that isn't in the gallery yet. Add new regression
  fixtures to `scripts/gallery.yaml` (`GALLERY_ENTRIES` in
  `scripts/build_gallery.py` is derived from it), not only `examples/topologies/`, so CI's render-diff makes them visible to a
  human. A topologies-only or tests-only fixture is invisible in the PR
  preview.

Do not present a prototype as an improvement before independent review and the
user's judgment where needed. If the render has problems, revise the writer's
brief; do not defend a weak fix.

### Local renders

This is about **review**, not about the writer looking at its own draft. A
writer iterating on an in-progress change renders as often as it needs to inside
its own loop; it is not independent of its own work, so a spawn there buys
nothing. For review, a single-file sanity render or a local before/after sweep
belongs in a LIGHT read-only worker that returns the verdict and artifact path
rather than the imagery. Commands:
[`environment.md`](environment.md).

**Pick one, never both for the same SHA.** The local sweep computes the same
before/after corpus diff that CI already computes, so running it alongside the
CI preview is paying twice for one answer. Use it only when you need that answer
*before* spending a CI cycle; once the preview exists, the preview is the
review.

## Step 9: narrow over-applying fixes

If the render preview shows the fix changed **more than the targeted
example** unexpectedly, do not ship it as-is. Have the independent visual
reviewer classify each affected example as:

- **I** (improvement) - keep
- **N** (neutral) - keep
- **D** (detrimental) - must be narrowed

The bar is "no **meaningful** visual regression", not pixel-identity. A
subtle spacing or coordinate shift that comes with a cleaner, more elegant
implementation is fine (classify it N or I); do not contort the code to
preserve a byte-identical render. Only a genuine degradation is a D.

For each detrimental delta, assign a HIGH diagnostic worker to find the
**precondition** that distinguishes the target case (where the fix helps) from
the regressing case (where it hurts). Re-brief the sole writer to gate the fix
on that precondition (e.g. a topology predicate, a config flag, a layout
property test). Assign fresh re-rendering and re-verification before merging.

Each D-delta round costs a HIGH diagnostician, a writer re-brief, a fresh render
and another CI cycle - roughly $25-35. After two rounds stop and report rather
than looping: a fix needing a third narrowing is usually the wrong shape, and
whether to keep paying is the user's call.

A fix with an unaddressed D-delta is not PR-ready. Reroute it or return the
structured blocker that prevents correction or classification.
