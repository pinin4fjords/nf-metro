---
title: "Envelope settlement design record"
description: Dated corpus measurements and rejected alternatives that informed envelope settlement.
---

# Envelope settlement design record

This page records corpus measurements and rejected alternatives that informed
the envelope-settlement design. These observations describe particular commits.
They are not layout invariants and must not be read as current corpus totals.
The enforceable specification remains in
[`src/nf_metro/layout/CONTRACT.md`](../../src/nf_metro/layout/CONTRACT.md).

Every figure on this page is historical evidence. A named test identifies a
related live invariant, not an assertion of the surrounding snapshot totals.
The referenced commits preserve the code and corpus used for each measurement.

## Boundary measurement experiments

The settlement work merged in `36089574` measured several alternatives while
defining which boxes bound a corridor.

Landing boxes were removed from a reservation's blockers only when every claim
in that reservation landed in the box. The corpus published the same claims on
557 reservations with or without that reduction. Six reservations in four
fixtures had claims that disagreed about a landing box. A union would have
removed a blocker needed by other claims in the same reservation. The
`reportho.metro` column 4/5 corridor is retained as the concrete regression case
in
`test_a_box_only_one_claims_run_ends_inside_bounds_the_whole_reservation`.

The settlement contract originally quoted 557 realised gap reservations and
1007 claims as if those totals were pinned. Re-measuring `bd0b805a` for issue
#1699 produced 588 reservations and 1064 claims. The actual invariant and the
two tolerated identities remained green in
`tests/test_reserved_claim_consumption.py`; only the unasserted aggregate totals
had drifted. The live contract therefore states the invariant and the pinned
exceptions without a corpus-size snapshot.

The launch-anchor experiment found two planned bottom-exit fan fixtures whose
junction stood 10px inside the gap below its box. Treating that anchor as the
blocker grew each row gap by 10px. The other 367 renders in that snapshot were
byte-identical. No reservation in that corpus had claims that disagreed about a
launch anchor, so the intersection reduction had no observable counterexample.

Charging unclaimed strokes as peer width found 10 fixtures whose boundary was
too narrow for the peer it drew. Seven renders changed, each growing only in
height by 11px to 16px. The motivating case was
`merge_around_below_leftmost.mmd`, where a merge trunk return leg sat 14px below
a box edge without carrying its own claim.

Reading containment from emitted polylines instead of the ledger's published
occupied interval changed the strict-path result from 37 refused fixtures to 11
in the settlement snapshot. The ledger interval records the first routing pass;
the emitted polyline records where the settled reroute is drawn.

## Origin-independence experiment

The origin experiment translated settling fixtures by 0.1, 0.3, 1/3, 7.7,
1000.1, -0.1, and -7.7 pixels. Twenty-five fixtures both settled and translated
rigidly at every tested offset. Ten allocated a different width at some origin
when the deficit was a bare binary64 subtraction. None did when the deficit was
normalised by `measured_distance`.

Only `examples/differentialabundance_default.mmd` and
`tests/fixtures/da_pipeline.mmd` rendered differently between the two
arithmetics in that snapshot. Their 14px deficit was observed as
`14.000000000000057`, so a bare ceiling allocated 15px. The resulting maps were
802px tall instead of 801px. The other 367 renders were byte-identical.

Four fixtures could not be used for the origin comparison because a uniform
translation changed route shape instead of moving it rigidly:
`convergence_stacked_sink`, `same_line_fan_distinct_descent`, and `seed_15` at
1/3, plus `seed_77` at 7.7. The current property test is
`test_the_allocation_is_a_function_of_the_deficit_not_the_canvas_origin`.

## Canvas corridor measurement

The pre-content-clearance state at `36089574` was re-measured for issue #1699.
The corpus at that commit produces 149 canvas-corridor observations.
Twenty-nine published claim intervals are short on the content side, but they
represent 28 distinct `RouteReservationId` values. The duplicated identity is
`route-reservation:1af1d2d15818ce454433bad8`, produced by the mirror fixtures
`tb_lr_exit_left.mmd` and `tb_lr_exit_right.mmd`. Measuring the emitted
polylines produces 28 short observations. The extra published-interval result
is `bypass_leftward_far_side_entry.mmd`.

The original 144 / 28 / 29 figures were therefore a mixture of two dimensions:
28 distinct reservation identities and 29 fixture observations. They also came
from an earlier corpus population. Commit `aff0b101` replaced that model with
final content-boundary measurement and enforcement. On `bd0b805a`, the same
measurement finds 149 canvas observations and zero content-side shortfalls.

Before the content-clearance fix, 18 of the 28 drawn shortfalls used an
over-top band inside the header reservation. No measured case put route ink
inside the box of the header actually drawn there. Tightening the claim to the
drawn header alone was insufficient: 10 of those 18 still kept only 22px of the
26px edge clearance, while the other 10 of the 28 shortfalls missed a box edge
by 2px to 6px. This was the evidence for moving the corridors away from the box
edges rather than merely narrowing the header claim.

## Reroute ledger experiment

In the settlement snapshot, comparing the frozen ledger with the reroute ledger
found no corridor present in only one ledger. Twenty-one corridors in 11
fixtures requested less width after rerouting. None requested more, so the
frozen ledger did not under-size a boundary drawn by that corpus. The current
single-case regression test is
`test_a_corridor_the_reroute_resizes_is_named_rather_than_invisible`.

## Convergence planning experiments

Planning the three convergences in
`cross_column_perp_entry_overflow.mmd` changed the snapshot population from 30
compatibility convergences and 22 planned convergences to 27 compatibility and
25 planned. The render dropped two overshoot stubs, remained pixel-identical,
and retained a 1325x1781 canvas. The planned trunks were at x = 554, 558, and
562 and landed at y = 1617.4.

At the time of the settlement merge, 12 live compatibility systems occupied 12
fixtures and nine distinct system-id strings. `COMPATIBILITY_CORPUS` contained
14 rows because two planned fixtures were retained as controls. That snapshot
ran 168 grants and observed zero divergent grants.
`tests/test_capacity_probe.py` checks each listed row's verdict and each probe's
grant structure; it does not assert these aggregate snapshot totals.

The longitudinal-blocker experiment covered 14 out-of-band claims. Thirteen
had every blocker on the violated side overlap or abut the drawn leg. The
remaining case, `fan_bypass_shared_band`, had distant blockers but an abutting
section set its violated edge. Filtering blockers only by longitudinal overlap
dropped boxes beside corridor elbows, changed eight renders, reversed one
vertical leg, and raised the out-of-band total from 21 to 32. This rejected the
filter as a settlement rule.

Shared-trunk laning asked 89px for `merge_around_below_leftmost`, compared with
90px while the planner declined the system and treated the second trunk as a
peer. The planned result shortened the settled map by 1px. The live contract
states the formula rather than this snapshot value;
`test_a_boundary_is_charged_for_the_unfiled_leg_drawn_in_it` checks the peer
width, minimum-width sum, and available-capacity inequality.

The capacity probe also established why dependent coordinates must be
re-derived after a grant. Leaving junctions behind made five systems appear to
reach allocation: `merge_bottom_row_bypass` and
`merge_feeder_shared_channel_gap` from 19.5px,
`ambiguous_exit_continuation` from 256px, `merge_right_entry` from 576px, and
`merge_trunk_out_of_range_section` at 656px. Applying equivalent widening
through settlement kept the conflicts at 0px separation. Those results came
from stranded junctions rather than usable capacity.

A conditional demand was rejected for the two `OPPOSING_OPENING_CHANNEL`
systems. Both arms leave one junction coordinate and every settlement-owned
translation carries that junction with the sections it joins, so the arms
remain 0px apart at every tested capacity. The missing object is a shared-channel
planning decision, not a distance settlement can allocate.

## Port and caption experiments

The post-layout label experiment found 38 fixtures that grew a port-bearing
edge at render time. Four did so during a pass with no later re-observation.
Repeated re-observation was not a general fixpoint: `top_entry_left_neighbour`
moved its producer box 6px per round, while `bypass_fan_in_outer_slot`
contracted by half per round. This supported one re-observation followed by
holding anchored edges.

The caption-band snapshot contained 1224 captions. Every caption fit the band
stated by its chosen side. A fixed band above `bbox_y` would have rejected 39:
36 captions below their box and three rotated captions. Twenty captions had a
neighbour's caption ink or reserved band in their claimed strip, all caused by
an above-caption title overhanging into the next column.

Ranking the leftmost clear slot ahead of every off-band position affected 20
captions. Each was only 4px from a descending stroke while the declined edge
was 22px to 60px clear. Clearance ranking sent 18 to a bottom edge with 42.0px
to 60.4px clearance and two to a roomier in-band slot with 19.3px and 50.0px
clearance. Nineteen fixtures rendered differently and
`cross_column_perp_entry_overflow` grew by 8px, or 0.45 percent, to contain its
bottom-row caption.

Moving Tier-A render guards from the first route observation to settled
geometry exposed 27 failures among 356 rendering fixtures: 23 in the own-section
interior guard and 11 in the port-boundary guard, with overlap between the two
sets. Twenty-six were ports whose boxes had grown away from them. The remaining
fixture, `cross_column_perp_entry_overflow`, also failed on the first pass. No
fixture failed only on the first pass.

## Layout-pass ownership survey

The settlement design surveyed eight local layout passes across 329 fixtures by
snapshotting every section box and station coordinate around each engine call.
No surveyed pass moved another section's origin or applied one delta to an owned
set. Stage 4.7 was the boundary case: it slid a whole section body in 28 of 427
calls, but derived the amount independently for each section.

Only `push_lower_rows_after_bbox_grow` matched settlement ownership. Its local
measurement became `BoundaryClearanceDemand`, and settlement replaced its row
translation. Paying both a reservation demand and a clearance demand in
sequence over-allocated 0.6px in `diagonal_labels` and 0.2px in
`longread_variant_calling`, so settlement takes their maximum and translates
once.

The render-time push fired for one of 369 fixtures in that snapshot, while five
row boundaries were short without a correction. Settlement closed three:
`manual_rl_row_nonconsumer_bypass`, `packed_cell_cellmate_bypass`, and
`packed_cell_cellmate_bypass_adjacent`, each by 9px. The two rail cases were
excluded because widening their row boundary turned flat routes into staircases:
seven of 91 routes in `sarek_metro` and four of 11 in
`rail_pitch_vs_labels`.

`_tighten_lower_rows_after_shrink` was rejected because it decreases separation,
which settlement forbids. `_reserve_row_gap_for_top_padding` was rejected because
its translation must precede a local box resize, while settlement freezes box
size. `_shift_graph_into_canvas` and `_snap_canvas_y_to_grid` were rejected
because they translate the whole graph uniformly and therefore change no
pairwise separation.

Suppressing the two canvas translations left every pairwise facing-box
separation identical in all 357 fixtures that reached a settled render graph.
With bare binary64 deficit subtraction, `differentialabundance_default.mmd` and
`da_pipeline.mmd` were the two exceptions: their functional/plots gap measured
91px with the canvas translation and 90px without it. Both satisfied the 90px
reservation, but the result depended on canvas origin. `measured_distance`
removed that coupling.

## Late vertical port alignment

The Stage 6.16 scope experiment found that applying full entry-port alignment
would move nine horizontal-flow ports, including longread `small_variants` by
86px. Removing the vertical-only pass restored two S-kinks: longread `phasing`
by 16.8px and `tb_file_termini` reporting by -14px. Repositioning junctions after
settlement moved 17 junctions across the snapshot corpus, some by hundreds of
pixels. This evidence supports the stage's vertical-only alignment followed by
axis-generic junction positioning.

## Other provenance removed from the specification

The axis policy grew out of repeated direction-specific TB mirrors. The
`AxisFrame` and `lanes_run_along_y` vocabulary replaced mixed checks such as
`direction == "TB"` and `direction not in ("LR", "RL")` across row alignment,
grid snapping, trunk selection, and section placement. TB-only helpers with no
LR mirror remain direct because wrapping their coordinates in an axis frame
would add indirection without sharing an implementation.

The flat Stage.N naming rule replaced an organic suffix tree containing names
such as `13d2` and `13k2`. PR #342 included a `Phase 13k` to `Phase 13k2`
rename. The contract retains only the prospective rule: allocate the next
sequential Stage.N number and give the stage a tested postcondition.
