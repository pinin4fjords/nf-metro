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

`Read` takes filesystem paths and the preview embeds SVG, which `Read` cannot
display, so the reviewer must fetch and rasterise before it can judge anything.
`Bash` covers all of this; do not reach for `WebFetch`, which returns markdown
and is useless for a render.

```bash
export ART=/tmp/nf-metro-visual-<PR_NUMBER>; mkdir -p "$ART"
BASE=https://seqeralabs.github.io/nf-metro/_pr/<PR_NUMBER>
curl -sL "$BASE/" -o "$ART/index.html"
grep -oE '[A-Za-z0-9_./-]+\.svg' "$ART/index.html" | sort -u > "$ART/changed.txt"
while read -r f; do curl -sL "$BASE/$f" -o "$ART/$(basename "$f")"; done < "$ART/changed.txt"
source ~/.local/bin/mm-activate nf-metro-dev
for s in "$ART"/*.svg; do
  python -c "import cairosvg,sys; cairosvg.svg2png(url=sys.argv[1], write_to=sys.argv[1][:-4]+'.png', scale=2)" "$s"
done
```

Then `Read` each `.png`. If a render was produced without `--no-chrome-css` the
rasterise step aborts on `var()` chrome properties; re-render that one locally
with the flag rather than skipping the example.

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
