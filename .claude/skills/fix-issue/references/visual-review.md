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

Use the page only to learn *which* examples changed, then render those locally.

```bash
set -euo pipefail
export ART=/tmp/nf-metro-visual-<PR_NUMBER>; mkdir -p "$ART"
curl -sL "https://seqeralabs.github.io/nf-metro/_pr/<PR_NUMBER>/" -o "$ART/index.html"
# Each changed example is a diff-entry div keyed by fixture stem.
grep -oE '<div class="diff-entry" id="[^"]+"' "$ART/index.html"   | sed -E 's/.*id="([^"]+)".*/\1/' | sort -u > "$ART/stems.txt"
wc -l < "$ART/stems.txt"     # how many examples the reviewer owes a verdict on
```

Then render each stem at the base SHA and the candidate SHA and rasterise both.
This block is **self-contained**: shell state does not survive between Bash
calls, so it re-exports `ART` rather than relying on the block above.

```bash
set -euo pipefail
export ART=/tmp/nf-metro-visual-<PR_NUMBER>
: "${ART:?}"                        # abort loudly rather than writing to /
source ~/.local/bin/mm-activate nf-metro-dev
for side in base cand; do
  case $side in base) SHA=<BASE_SHA>;; cand) SHA=<CANDIDATE_SHA>;; esac
  W="$ART/wt-$side"
  git -C ~/projects/nf-metro worktree add --detach "$W" "$SHA"
  # The corpus spans examples/, tests/fixtures/ and tests/fixtures/hash_seed_determinism/,
  # so resolve against the whole tree, not one root.
  git -C "$W" ls-files '*.mmd' > "$ART/all-mmd.txt"
  while read -r stem; do
    for cand in "$stem" "${stem#pipeline_}"; do
      f=$(grep -E "(^|/)${cand}\.mmd$" "$ART/all-mmd.txt" | head -1) && [ -n "$f" ] && break
    done
    if [ -z "${f:-}" ]; then echo "UNRESOLVED: $stem" >&2; continue; fi
    # Some corpus fixtures abort by design at head. Never let one kill the sweep:
    # under `set -e` an unguarded failure here ends the whole review.
    if ! PYTHONPATH="$W/src" python -m nf_metro render "$W/$f" \
         -o "$ART/$side-$stem.svg" --no-chrome-css 2>"$ART/$side-$stem.err"; then
      echo "RENDER-FAILED($side): $stem" >&2; continue
    fi
    python -c "import cairosvg,sys; cairosvg.svg2png(url=sys.argv[1], write_to=sys.argv[1][:-4]+'.png', scale=2)" "$ART/$side-$stem.svg"
  done < "$ART/stems.txt"
  git -C ~/projects/nf-metro worktree remove --force "$W"
done
```

A stem that renders on `base` but is `RENDER-FAILED` on `cand` **is the
regression** - that is a D-delta of the worst kind, not a stem to skip. One that
fails on both was already broken at head; say so and move on. The `.err` files
hold the abort message.

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

- **"No visual changes detected"** -> a clean result, but **not** a
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
  was masking. After any layout/routing fix, require the verifier to inspect
  the full render (cropping the region as needed) and run `probe_layout` plus
  `inspect_layout` on the *candidate* SHA - the after-state counterpart to
  Step 3's before-state reading, not a repeat of it - for the whole-layout picture (crossings, port alignment,
  column gaps), not only the targeted invariant.
- **A clean render-diff verdict only covers the gallery corpus.** It says
  nothing about a NEW fixture that isn't in the gallery yet. Put new
  regression fixtures in `scripts/build_gallery.py` (`GALLERY_ENTRIES`), not
  only `examples/topologies/`, so CI's render-diff makes them visible to a
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

A fix with an unaddressed D-delta is not PR-ready. Reroute it or return the
structured blocker that prevents correction or classification.
