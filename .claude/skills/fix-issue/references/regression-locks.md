# Pre-existing regression locks and the xfail rule

Read this when `grep -rn "#<N>" tests/ scripts/build_gallery.py examples/topologies/`
finds anything, or when you are tempted to add an xfail.

## An issue that arrives with its infra already in place

Most issues arrive bare. But some - notably those filed by the
`nf-metro-stress-render` skill - arrive with their regression infra **already
built**: a fixture in `examples/topologies/`, a `GALLERY_ENTRIES` row in
`scripts/build_gallery.py`, and a `strict=True` xfail test referencing the issue
number.

- **If a strict-xfail lock exists**, that *is* your failing test. Don't write a
  duplicate, and don't re-add the fixture or gallery entry. Confirm it xfails on
  the current tree; that is what documents the live defect.
- **Completing the fix flips that strict-xfail to XPASS, which reds CI.** That
  is the signal the bug is actually fixed. Finish by **removing the `xfail`
  marker** so the now-passing assertion becomes a permanent positive guard.
  Deleting the whole test loses the guard; leaving the marker keeps CI red.
- **If no lock exists** (the common case), the writer adds the failing test per
  Step 4.

## xfail is a lock on a known bug, not an escape hatch

Do not add an xfail to hide an incomplete fix. Reroute bounded work. If
authority, capability, external state, or a material decision blocks it, return
the structured blocker without muting the test or filing a child issue as
camouflage.

Add a **new** xfail only when all three hold:

1. the user explicitly accepts a genuine multi-session deferral;
2. the marker references a standalone issue whose body stands alone;
3. it is `strict=True`, so completing the fix reds CI rather than passing
   silently.

A new topology fixture owes three committed artifacts, not one - see
[gate-ratchet.md](gate-ratchet.md).
