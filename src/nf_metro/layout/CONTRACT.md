# Layout Stage Contract

Per-stage pre/postconditions for `_compute_section_layout` in
`src/nf_metro/layout/engine.py`. The pipeline is a long chain of mutating
passes over a shared `MetroGraph`; this doc records what each pass assumes
and what it guarantees, so that adding or reordering a stage doesn't
silently violate a downstream pass's expectations.

## How to read this doc

- **Stage tag** matches the `# Stage X.Y:` comments inside
  `_compute_section_layout`. The first digit is the stage number (1-6,
  see "Stage overview" below); the second is sequential within the
  stage.
- **Lines** point at the entry comment of the stage in `engine.py` at the
  current HEAD. Re-grep `# Stage ` if the file shifts.
- **Precondition** = what the helper assumes. Pass-A stages assume
  global coordinates and ports on bbox edges; Pass-C stages assume
  finalised station Ys.
- **Postcondition** = the property the stage establishes (and that
  later stages may depend on).
- **Invariants preserved** = state the stage does NOT touch. Useful when
  asking "can I move this stage earlier?"
- **Related tests** = invariants in `tests/test_layout_invariants.py`
  that exercise the postcondition. Many tests are full-pipeline
  end-to-end checks (no single stage owns them outright); the mapping is
  "this stage is the one that establishes the property the test
  asserts," not "this test fails iff this stage regresses."
- **Lifecycle** classifies the stage by one objective question: does the
  property it establishes still hold at the *final* layout boundary?
  - **invariant** - it does. The one-line final-boundary property is
    given. (Some invariants are re-asserted by later re-runs of the same
    helper; re-assertion *maintains* the invariant, it does not negate
    it.)
  - **transient** - a later stage deliberately overrides it, so the stage
    has no final-boundary property to declare. The superseding stage is
    named.

  The distinction is about the *property*, not the coordinates. A stage
  stays **invariant** when a later stage recomputes the exact coordinates
  but the abstract property it established (no-kink flow, horizontal port
  connection, filled top band, grid-snapped Y) still holds at the end -
  that is maintenance. It is **transient** only when a later stage
  discards the decision itself, replacing the property with a different
  layout (flush row tops giving way to content-hugging tops; an early fan
  re-fanned around the final trunk Y). The test in doubt: does the
  property survive to the end, or is the decision overwritten?

  Lifecycle answers "what does this phase guarantee at the end" and is
  pinned by `tests/test_contract_lifecycle.py`. It is **orthogonal** to
  the question #365 explored: "is this invariant safe to *lift* into a
  declarative run-anytime `maintain()` registry?" Liftability requires
  the invariant *plus* idempotency, order-independence, and no (or a
  gated) precondition - properties the lifting work (#463, #464)
  establishes empirically. So an inline `liftable:` qualifier appears
  **only** where liftability is non-trivially anything other than "yes";
  its absence is not a promise of liftability.

A stage whose purpose isn't crisp here is a structural-debt signal -
those rows are flagged "UNCLEAR" in the Notes column. Don't paper over
them; investigate before adding another stage next to them.

## Coordinate-system convention

Stages split into three regimes:

1. **Pre-Stage-2.1**: stations have section-local coordinates. Bboxes are
   in local coordinates.
2. **Post-Stage-2.1**: stations and bboxes are in global canvas
   coordinates. Ports do not yet exist on bbox edges.
3. **Post-Stage-3.1**: ports sit on bbox edges (validated by
   `_guard_ports_on_boundaries`).

## Axis vocabulary (TB policy)

TB sections run the identical LR machinery and swap axes only at coordinate
assignment (`single_section.py`). Every heuristic written against the LR
*interpretation* of `x`/`y` (horizontal trunks, layers spread along X, lines
stacked along Y) is wrong-by-default for TB, and the historical fix was to
hand-write a one-off `if direction == "TB"` mirror per heuristic. That count
only grew.

The sanctioned alternative is the `AxisFrame` primitive in `geometry.py`:
`AxisFrame.for_direction(direction, x_spacing, y_spacing)` returns the
**primary** axis (the layer/flow axis: X for LR/RL, Y for TB) and the
**secondary** axis (the track axis: Y for LR/RL, X for TB), each carrying its
`step` and `get`/`set` accessors, plus `primary_sign` (`-1` for RL, which runs
the LR primary axis reversed). A heuristic expressed against primary/secondary
instead of raw `x`/`y` has a TB path that is *the same code* as its LR path, so
it needs no branch.

**Lane sign (`secondary_sign`).** A transpose is a reflection: it flips
chirality, so a TB path written as an axis swap diverges from LR in behaviour,
not just orientation. The cure is a true 90-degree rotation, which a transpose
is not. `AxisFrame.secondary_sign` carries the lane fan direction: a 90-degree-CW
rotation maps LR's screen-down lane (`+Y`) to screen-left (`-X`), so TB is `-1`;
LR/RL are `+1` (RL reverses only the primary) and BT is `+1`, the flow-axis
reflection of TB (`primary_sign = -1`, `secondary_sign = +1`) so an upward flow
fans its lanes to the `+X` side. The **sanctioned offset->coordinate path** applies
this sign at the *draw accessor*, never to a stored offset, which stays positive:

- `geometry.station_lane_coord(frame, station, offset)` -> `station.y + offset`
  (LR), `station.x - offset` (TB): the screen coordinate of a positive lane
  offset from a station.
- `geometry.lane_delta(frame, offset)` -> `secondary_sign * offset`: the signed
  secondary-axis displacement for a positive offset, station-free.
- `geometry.lane_delta_to_normal_offset(lane_delta, travel)` bridges a lane delta
  to the bundle builder's right-normal offset (`routing.bundle._right_normal`),
  the sole point where the lane-sign and builder-normal conventions meet. The
  builder itself fans purely geometrically along `_right_normal` of travel and is
  not per-axis; rotation lives *above* it, in this offset->coordinate mapping.

`secondary_sign` governs the offsets of lines inside a station or bundle. Fan
station tracks use a separate plan-owned appearance sign: tracks progress along
the positive secondary axis for LR, RL, TB, and BT, then mirror when a feeder
arrives from the positive end of that axis. This keeps the hub on the nearest
track without changing bundle chirality. Symmetric fans remain centred; the sign
only determines their branch order on screen.

**Policy:** no new one-off TB branches. A heuristic that needs TB awareness is
the trigger to convert it to the axis vocabulary, not to add another branch.
This is machine-enforced by `tests/test_tb_branch_ratchet.py`, which counts
`"TB"` literals / `.TB` attribute accesses across the layout package and fails
CI if the total rises above its baseline (mirroring the corner-radius and
gate-coverage ratchets). Migrating a heuristic onto `AxisFrame` removes its
branch and lowers the count; lower the baseline in the same change to lock it in.

**Row / lane membership** is the inter-section corollary. The row-level passes
align the **Y (lane) axis**: row grouping, row trunk-Y alignment, the shared
row Y-grid, top-aligning row-mates. A horizontal-flow (LR/RL) section stacks
its lines along Y, so it is a first-class member of that machinery; a
vertical-flow (TB/BT) section stacks lines along X and shares no row Y-grid, so
those passes leave its Y alone. The predicate for this is
`geometry.lanes_run_along_y(direction)` (built on `AxisFrame.axes_for_direction`,
which names a section's axes without needing spacings). It replaced the
historical mix of `direction == "TB"` and `direction not in ("LR", "RL")`
exclusions in `row_align.py`, `grid_snap.py`, `_common._section_trunk_y`, and
`section_placement.py`, and underlies `_common._is_fold_section`
(`grid_row_span > 1 or not lanes_run_along_y(...)`), the row-fold predicate that
routes a section's exit ports through the fold path rather than the row passes.

**Deliberately left direct (not contortion-migrated).** Per the same judgement
as the in-section migration, a *single-branch* TB-only heuristic with no LR
mirror gains no polymorphism from `AxisFrame` - expressing its reads as
`frame.primary`/`frame.secondary` would just rename `.x`/`.y` inside code that
only ever runs for one direction. These stay direct in `phases/ports.py`:
`_align_tb_entry_port` (its TB-trunk branch; the function also serves the LR/RL
perpendicular case), `_clamp_tb_entry_port`, `_resolve_tb_exit_y`,
`_align_tb_section_bbox_bottoms`, and `_tb_trunk_x` (the secondary-axis trunk
coordinate is a *median* for a vertical section but the bundle-connected topmost
for a horizontal one - `_section_trunk_y` - so the two are not the same code and
should not be forced behind one name). The `_apply_tb_fold_spans` selection is a
domain grouping, not an axis swap, and likewise stays.

## Validate-mode guards

`compute_layout(validate=True)` runs these guards at fixed checkpoints:

| Checkpoint | Guards |
|---|---|
| after Stage 1.1 | `_guard_section_bboxes_positive` |
| after Stage 2.1 | finite coords, stations-in-sections, bboxes-positive |
| after Stage 3.1 | ports-on-boundaries |
| after exit-port align + row re-flush and the X-axis perp-port inset (Stages 3.4 to 3.5) | ports-on-boundaries |
| after each Pass C sub-stage (bisection) | finite coords, bboxes-positive, ports-on-boundaries, station-x-column-drift, plus three phase-gated guards (see below) |
| after final | bisection set (all unconditional) + off-track-above-anchor, row-trunk-cy-consistent, inter-section-routes-in-row-band |

Bisection checkpoints fire after every Pass C sub-stage (see the
`# Stage 5.2:` through `# Stage 6.16:` comments in
`_compute_section_layout`). Three guards
hold continuously only from a specific checkpoint onward, and the
bisection runner skips them earlier; see `_BISECTION_FIRST_VALID` in
`engine.py` for the threshold table:

| Guard | First valid checkpoint | Transient because |
|---|---|---|
| `_guard_stations_in_sections` | after Stage 5.3 | Stage 5.2's off-track lift moves stations above the section bbox; Stage 5.3 grows the bbox to enclose them. |
| `_guard_no_station_overlap` | after Stage 6.4 | Pre-snap fan placement can sit a fraction of a pitch off the row grid; Stage 6.4's snap pulls every station onto the grid while keeping same-column stations on distinct slots, after which markers must be collision-free. |
| `_guard_no_line_crosses_non_consumer` | after Stage 6.14 | A sparse loop-side station sits on the trunk Y until Stage 6.14 shifts it to a half-grid offset; before that, sibling line bundles pass through its marker bbox. |

Three further guards are excluded from the bisection set entirely
(meaningful only at the final boundary); the `_run_pass_c_guards`
docstring in `engine.py` is the authoritative list.

Guard bodies live in `phases/guards.py` and are imported into `engine.py`;
the bisection runner is `_run_pass_c_guards`.

## Anchor invariant

The **anchors** of a section are its port stations: synthetic points on the
section boundary where the inter-section line bundle crosses. A port anchors
the trunk on whichever axis its side dictates - LEFT/RIGHT (LR/RL) ports fix
the Y at which the bundle runs horizontally, TOP/BOTTOM (TB/BT) ports fix the
X at which it runs vertically - and a port's cross-axis (an LR port's X, a TB
port's Y) is likewise pinned to the section boundary by port positioning.
Anchors are set only by structural phases - port positioning along the section
DAG (align/snap entry/exit ports, inter-section port-pair snap), the row trunk
alignment (4.8), grid snapping, the inter-row cascade (6.13/6.14 phase 2) and
uniform canvas/row translation.

The **content-placement** phases - fan-out / full-bundle redistribution (4.9,
4.10), band-fill (6.1, 6.2), the 2-branch symfan half-grid (6.3), full-bundle
recenter (6.7), balance-around-trunk (6.11) and loop-side recenter (6.12) -
position content *around* the resolved anchors and must never move one. Each
runs through the `_run_placement` wrapper in `_compute_section_layout`, which
under `validate=True` calls `_guard_anchors_frozen_during_placement` to assert
that no port's `(x, y)` changed across the phase. The snapshot
(`_port_anchor_snapshot`) covers **every port on every side, on both axes** -
not just the LR/RL-Y subset - so the guard catches any anchor movement
regardless of port side or axis (a phase that nudged a TOP/BOTTOM port, or an
LR port's X, would be caught too). This separation (structural anchors vs.
dependent placement) is what makes the layout forward-resolvable: content is a
function of the frozen anchors, not the reverse.

### Content-placement purity

`_guard_anchors_frozen_during_placement` only forbids a content phase from
*moving* an anchor. A stronger property holds and is machine-checked separately:
every content-placement phase is a **pure function of (frozen anchors +
structure)**. The Y it assigns to the stations it governs depends only on the
frozen port anchors and the section structure (tracks, edges, columns), never on
the mutable intermediate state earlier phases happen to have left behind
(current station Y, section `bbox` geometry). This is strictly stronger than the
idempotence locked by `test_content_placement_idempotent` (#488): purity means
re-running, re-ordering, *or perturbing the non-anchor state* cannot change a
phase's output. `tests/test_content_placement_pure.py` (#491) is the guard - it
perturbs the non-anchor state before each phase and asserts the governed
stations land identically, the test-time counterpart to the anchor-frozen guard.

The phases that genuinely need an intermediate quantity - the empty-band slack
in 6.1 / 6.2, the balance arrangement in 6.11 - read it from a frozen *placement
reference* (`_snapshot_placement_refs` populates `graph._placement_ref_y` /
`_placement_ref_bbox_top`; phases read it via `_ref_y` / `_ref_bbox_top`)
captured once right before the consumer, rather than from live geometry. The
reference equals the live geometry at capture time. Planned fan materialisation
uses the same boundary pattern without a graph-state channel:
`_snapshot_planned_fan_centrelines` captures a read-only centreline mapping
after structural settlement and passes it into `_apply_planned_fan_geometry`.
These frozen inputs preserve the established render while keeping placement
independent of mutable station and bbox geometry.

## Inter-phase state protocol

Some stages hand intermediate results to later stages through private
`graph._*` fields rather than through station coordinates. These channels are
declared as data in [`phase_state.py`](phase_state.py) (`PHASE_FIELD_REGISTRY`),
which records each field's writer stage, its reader stages, and why it exists;
`tests/test_phase_state_registry.py` keeps that registry in sync with the
dataclass fields, the engine stage list, and this document.

Fields whose reader genuinely depends on the writer having run call
`require_phase_field` just before the read, which raises `PhaseInvariantError`
under `validate=True` when the writer stage has not completed in the current
pass:

- `graph._row_y_grid_info` - written by Stage 1.2 (`_align_row_y_grids`); read
  by the grid-group port snap (Stage 4.2-4.4), fan re-centre (6.3/6.7), and
  grid snap (6.4).
- `graph.half_grid_station_ids` - written by Stage 6.3 (`center_ports` only),
  Stage 6.4's own midpoint restore and Stage 6.17
  (`diamond_style='symmetric'`); read by the Stage 6.4 grid snap, which must
  skip these half-pitch stations. Stage 6.18 both reads the set and clears the
  marking off any station it seats back on a full row, so the post-layout
  readers (the straddle guard, the co-fanned drop-clearance rule in
  `routing/intra_handlers.py`) see only stations still at half pitch.
- `graph.symfan_trunk_station_ids` - written by Stage 6.3 (`center_ports` only);
  read by the Stage 6.4 grid snap, which must skip these source/trunk stations
  so they stay on the symfan's local frame instead of snapping to a rowspan
  neighbour's fractional row-grid origin.
- `graph._consumers_grid_snapped` - set right after the Stage 6.4 snap; the
  Stage 6.6 off-track reanchor carries its own always-on guard on it.

The remaining channels tolerate an unwritten value by design (their read sites
fall back to live geometry or a `None`/empty default), so they are documented in
the registry but carry no runtime check: `graph._struct_height_below_top`
(snapshotted after 6.15a, read by the 6.13 cascade), `graph._placement_ref_y` /
`graph._placement_ref_bbox_top` (frozen before 6.1/6.11, read via `_ref_y` /
`_ref_bbox_top`), `graph._base_y_spacing` (recorded before the spread loop
when `y_spacing` is auto-resolved), and `graph._resolved_x_spacing` (the
resolved column pitch recorded before layout, read as the cross-axis off-track
step for vertical-flow sections).

A further group crosses a subsystem boundary rather than two numbered stages,
so their `PhaseFieldSpec` names a lifecycle phase (`pre-layout`, `post-layout`,
`station-offset-layout`, `rail-layout`) in place of a stage id. They carry no
runtime check either:

- `graph._cross_column_perp_bridges` - sections whose perpendicular drop was
  bridged across grid columns, accumulated by the Stage 3.2 / 3.4 port
  alignment; routing's render-curve invariant reads it to relax its abort to a
  warning for those bundles.
- `graph._fold_compressed_sections` - recorded at parse time for sections a
  lowered fold threshold relocated; read by the fold-exit-side guard and the
  render fold-abort chokepoint. A resolve-time flow reversal is recorded as a
  `FLOW_REORIENTED_DIRECTION` decision in `graph.layout_provenance`; routing's
  exit-port offset reads that typed reason instead of a second section set.
- `graph._linear_entry_pill_lines_cache` - accepted linear-entry cohorts
  projected by each station-offset computation. Marker bbox, label, and render
  consumers use the cohort with the offset map produced by that computation;
  the empty default means no entry frame owns marker geometry.
- `graph._rail_y` - the per-section `{line_id: rail_y}` map produced by the
  opt-in rail-mode layout; read by the rail router, label placement, and rail
  guards, empty when rail mode is off.
- `graph._defer_final_guards` / `graph._after_final_deferred` - pass-control
  flags `compute_layout` uses so the final-geometry guards defer while the
  pre-bypass passes run, then validate the settled post-bypass geometry once.

## Stage overview

The pipeline groups into six stages aligned with the coord-regime
transitions and the Pass A / Pass B / Pass C divisions used throughout
this doc.  See [`docs/dev/layout_pipeline.mdx`](../../../docs/dev/layout_pipeline.mdx)
for a prose walkthrough of each stage; the matching
`# ---- Stage N - ... ----` comment dividers in `_compute_section_layout`
mark each stage's start in the source.  Stage-table entries below appear
in pipeline order.

## Stage table

### Stage 1.1: internal section layout
- **Purpose**: Lay out each section's real stations in section-local
  coordinates via layer/track assignment.
- **Helper**: `_layout_single_section` (`phases/single_section.py`).
- **Precondition**: Parser has populated `graph.sections`, `graph.stations`,
  `graph.edges`. Section directions and grid positions inferred by
  `auto_layout`. Ports exist in the graph but are not yet positioned.
- **Postcondition**: For every section with real stations, the section
  subgraph (returned via `section_subgraphs[sec_id]`) has every real
  station assigned a local `(x, y)`, a `layer`, and a `track`. Section
  `bbox_x/y/w/h` reflect the local content extent.
- **Invariants preserved**: Ports (`is_port=True`) are not positioned.
  Inter-section edges in `graph.edges` are untouched. Junctions are not
  positioned.
- **Related tests**: `test_section_bbox_contains_all_content`,
  `test_loop_column_stations_share_x`.
- **Lifecycle:** invariant - each station's layer/track and
  section-local relative layout persist to the end; Stage 2.1 only
  translates them into global coordinates, it does not re-lay them out.

### Stage 1.2: align row Y grids
- **Purpose**: Snap station Ys to a shared row-wide grid so same-row
  same-direction sections agree on grid pitch and slot count.
- **Helper**: `_align_row_y_grids` (`phases/row_align.py`).
- **Precondition**: Stage 1.1 complete; sections still in local
  coordinates; section subgraphs available.
- **Postcondition**: Within each `(grid_row, direction)` group, all
  multi-station layers share one Y grid. Bbox `w/h` unchanged from
  Stage 1.1 (only station Ys shift). `graph._row_y_grid_info` stores
  grid metadata for the debug overlay.
- **Invariants preserved**: Isolated stations (sole layer occupants
  with off-grid Y) keep original Y - hub centering survives. Section
  bbox dimensions unchanged.
- **Related tests**: `test_row_trunk_marker_cy_consistent`,
  `test_all_stations_snap_to_grid`.
- **Lifecycle:** invariant - the shared per-row Y grid holds at the
  final boundary (re-asserted by Stage 6.4's grid snap).

### Stage 1.3: section placement
- **Purpose**: Place sections on the canvas grid via topological
  layering of the section DAG.
- **Helper**: `place_sections` in `section_placement.py`.
- **Precondition**: Sections have bboxes from Stage 1.1 and grid
  positions from `auto_layout`. Still all local-coord.
- **Postcondition**: Every section has `offset_x`, `offset_y` set such
  that `(local + offset)` lands sections on a non-overlapping grid.
- **Column seating**: A column is seated on the edge its members' box
  extents grow away from (`box_growth_sign`, `layout/geometry.py`): the left
  one by default, the right one when a member's extent grows leftward (its
  flow runs that way, or its lanes fan that way). A right-seated member's
  slack is its column's width minus its own `_effective_section_width`, the
  same measure the column width was reserved from, so the column's boxes
  land on one X.
- **Disconnected graphs**: When the section meta-graph has 2+
  weakly-connected components and the author pinned no explicit
  `%%metro grid:` positions, each component is placed in its own local
  column grid (so a wide component never inflates another's columns)
  and the components are stacked vertically in a deterministic order
  (ascending min original row, then descending size, then smallest
  section id), left-aligned and separated by `section_y_gap`. Any
  explicit grid override falls back to the shared single-grid path.
- **Invariants preserved**: Station local coords unchanged. Bboxes
  still local-coord.
- **Runtime guard**: `_guard_independent_components_disjoint` (under
  `validate=True`) asserts stacked components occupy disjoint vertical
  bands.
- **Lifecycle:** invariant - the section grid (column/row placement,
  non-overlap) holds at the final boundary.

### Stage 1.4: renumber sections
- **Purpose**: Renumber sections by connected route continuity, using visual
  lanes to choose between alternative continuations.
- **Helper**: `_renumber_sections_by_route` (`phases/canvas.py`).
- **Precondition**: Section grid positions and directions finalised.
- **Postcondition**: Each disconnected flow is numbered completely before the
  next. The nearest connected section on the current lane is preferred;
  parallel branch starts remain together; joins wait for aligned or independent
  predecessor routes. A secondary cross-row route may rejoin a section already
  numbered on a dominant row. Authored numbers are preserved, and automatic
  sections take the lowest unused positive numbers.
- **Invariants preserved**: Section IDs, station coords, bboxes,
  edges. Pure metadata pass.
- **Related tests**: `tests/test_section_numbering.py`.
- **Lifecycle:** invariant - `number` metadata is final
  (cosmetic, never recomputed).

### Stage 1.5: offset overshoot correction
- **Purpose**: Grow `x_offset`/`y_offset` when section local extents
  reach left/above the canvas origin, so global coords stay positive
  after Stage 2.1.
- **Helper**: inline.
- **Precondition**: Section `offset_x/y` and local `bbox_x/y` set.
- **Postcondition**: For every laid-out section, `offset_{x,y} +
  bbox_{x,y} + {x,y}_offset >= section_{x,y}_padding`.
- **Invariants preserved**: Section bboxes (local), station local
  coords, grid layout.
- **Lifecycle:** invariant - positive in-canvas coordinates hold at the
  end (the canvas top margin is maintained by Stage 6.15 /
  `_shift_graph_into_canvas`).

### Stage 2.1: local-to-global translation
- **Purpose**: Translate every real station and section bbox into
  global canvas coordinates.
- **Helper**: inline.
- **Precondition**: Stage 1.3 / 3b complete; `section.offset_{x,y}`,
  `x_offset`, `y_offset` final.
- **Postcondition**: Every real station's `x, y` and every section's
  `bbox_x, bbox_y` are global. `bbox_w, bbox_h` unchanged. Section
  subgraphs (local-coord) still exist but are not used downstream.
- **Invariants preserved**: Ports remain unpositioned. Junctions
  unpositioned.
- **Validate guards run after**: finite coords, stations-in-sections,
  bboxes-positive.
- **Related tests**: `test_section_bbox_contains_all_content` (the
  containment invariant first holds here).
- **Lifecycle:** invariant - the global-coordinate regime is permanent;
  every later stage works in global coordinates.

### Stage 3.1: position ports on section boundaries
- **Purpose**: Place every port on its section's bbox edge at the
  section's nominal centre line for its side.
- **Helper**: `position_ports` in `section_placement.py`.
- **Precondition**: Section bboxes in global coords (Stage 2.1).
- **Postcondition**: Every port station's `(x, y)` lies on the bbox
  edge corresponding to its side, within `GUARD_TOLERANCE`. Ports
  start at the bbox-edge midpoint for their side.
- **Invariants preserved**: Real station coords, section bboxes,
  junctions.
- **Validate guard after**: `_guard_ports_on_boundaries`.
- **Lifecycle:** invariant - ports sit on their bbox edges at the final
  boundary (guarded continuously by `_guard_ports_on_boundaries`).

### Stage 3.2: align LR entry ports
- **Purpose**: For LEFT/RIGHT entry ports, set Y to the incoming
  source's Y so the inter-section horizontal run is straight; for
  TOP/BOTTOM entry ports, set X / Y accordingly.
- **Helper**: `_align_entry_ports` (`phases/ports.py`), dispatching to
  `_align_lr_entry_port` and `_align_tb_entry_port`.
- **Precondition**: Stage 3.1 placed ports on bbox edges. Junction
  positions are unknown - the helper uses `_resolve_source_xy` to
  derive junction coords on-the-fly.
- **Postcondition**: Each entry port's coordinate on the axis along
  its bbox edge matches its source's coordinate on that axis (within
  the section's bbox extent).
- **Invariants preserved**: Real station coords (Pass-A is port- and
  bbox-only). Exit ports. Junctions still unpositioned.
- **Related tests**: `test_no_kink_at_section_boundary` (the
  straight-run property this phase establishes).
- **Lifecycle:** invariant - the entry-port straight-run (no-kink) Y
  holds at the end (re-asserted by Stages 5.5 / 6.16).

### Stage 3.3: shift LR/RL perp-entry internal stations
- **Purpose**: When an LR/RL section has a TOP or BOTTOM (perpendicular)
  entry port, shift internal stations' X so the entry port has
  in-section runway before stations begin.
- **Helper**: `_shift_lr_perp_entry_stations` (`phases/single_section.py`).
- **Precondition**: Stage 3.2 finalised LR/RL entry-port X for perp
  entries.
- **Postcondition**: Internal stations in such sections sit at least
  `ENTRY_SHIFT_LR * x_spacing` away from the perp entry port X, and the
  section's own bbox still contains the shifted run: a drop inside the
  run's span shifts the run further than `_adjust_lr_entry_inset`
  reserved, so the trailing edge grows by the uncovered remainder.
- **Invariants preserved**: Station Y, ports (the flow-axis exit ports
  re-pin to a moved edge), bboxes (X shift is bbox-bounded).
- **Related tests**: `test_terminus_not_directly_after_diagonal`,
  `test_no_kink_at_section_boundary` (entry-side geometry),
  `test_lr_perp_port_pair_1539.py::test_run_stays_inside_its_own_box`.
- **Lifecycle:** invariant - the perpendicular-entry runway
  (internal-station X clearance) holds at the final boundary.

### Stage 3.4: align fold-section exit ports
- **Purpose**: For row-spanning (fold) and TB-direction sections,
  shift LEFT/RIGHT exit ports to the target section's entry Y. May
  push the target section down via `_resolve_tb_exit_y`; the move then
  re-flushes the tops of the rows it pushed so it cleans up after
  itself rather than leaving the correction to a separate stage.
  Also seats every single-row LR/RL section's TOP/BOTTOM exit past its
  trailing station (`_align_perpendicular_exit_port`), whether or not
  the section also carries a flow-aligned port: an exit left on its
  feeder's own X collapses the turn to a zero-length corner that the
  lane fan then splays the wrong way round.
- **Helper**: `_align_exit_ports` (`phases/ports.py`), dispatching to
  `_align_lr_exit_port` and finishing with a `_top_align_row_sections`
  (`phases/row_align.py`) call scoped to the pushed rows.
- **Precondition**: Entry ports aligned (Stage 3.2); target sections
  positioned (Stage 1.3/4).
- **Postcondition**: Exit ports on fold/TB sections sit at the same Y
  as their target section's entry port (within section bbox extent);
  same-row contiguous-column sections whose top the exit move disturbed
  share `bbox_y` again (station/port Ys shift by the same delta,
  preserving Stage 3.2 alignment). The row re-flush is a transient
  intermediate property, not a final guarantee: Stage 6.15a later grows
  a fanned section's bbox top above the flush line, so finished same-row
  tops are not guaranteed flush (measured ~40px non-flush on
  `terminal_symmetric_fan` / `trunk_through_fan`; see Stage 4.7, which
  re-flushes and carries the same transient tag).
- **Invariants preserved**: Real station coords. Entry-port Ys.
- **Validate guard after**: `_guard_ports_on_boundaries` (the row
  re-flush preserves port-on-edge by shifting ports with stations).
- **Related tests**: `test_no_kink_at_section_boundary`,
  `test_inter_section_route_y_stays_within_row_band`,
  `test_exit_port_row_reflush`.
- **Lifecycle:** invariant - the fold/TB exit-port no-kink Y holds at
  the end (re-asserted by Stage 5.5).

### Stage 3.5: reserve the perpendicular-port edge inset on X
- **Purpose**: Grow a horizontal-flow (LR/RL) section's left and right
  bbox edges so each TOP/BOTTOM port keeps `PERP_PORT_EDGE_INSET` from
  them, the X-axis rotation of the inset the Y sizing keeps for a
  vertical flow's LEFT/RIGHT ports. X sizing measures real stations
  only, so a port seated past the trailing station (Stage 3.4) or
  dragged onto a drop column lands inside the padding band with nothing
  to push the edge out. Each such port owes its facing edges the inset on
  its own; the two edges are not levelled against each other, because an
  edge already held further out by content or a routing band is not the
  port's doing.
- **Helper**: `_reserve_perp_port_edge_inset` (`phases/bbox.py`),
  followed by `reenforce_column_gaps` (`section_placement.py`) when any
  box grew.
- **Precondition**: Perpendicular port X settled (Stages 3.2 to 3.4 are
  the last to move one relative to its own box).
- **Postcondition**: No LR/RL TOP/BOTTOM port's outermost drawn lane sits
  within `PERP_PORT_EDGE_INSET` of its section's left or right edge
  (`port_bundle_edge_reach`, as on the Y axis); adjacent columns still keep
  `MIN_INTER_SECTION_GAP`.
- **Invariants preserved**: Station coords; every port's own edge
  anchoring (LEFT/RIGHT ports move with the edge they are pinned to).
- **Validate guard after**: `_guard_ports_on_boundaries`.
- **Related tests**:
  `test_perp_port_edge_clearance_1494.py::test_horizontal_perp_ports_keep_the_designed_inset`.
- **Lifecycle:** invariant - the inset holds at the final boundary.

### Stage 3.6: level a grid column's shared-runway X edges
- **Purpose**: Give column mates that start their content at one X a
  common bbox edge on that side, so the runway between that edge and the
  shared content column is the same width in each. The X half of the same
  levelling primitive the row top-align uses (Stages 5.3 / 6.9), narrowed:
  a grid row's sections share a trunk Y, so their tops are always
  comparable, whereas a grid column's sections share no trunk X, and
  levelling boxes whose content starts at different X moves an edge
  without moving anything a viewer reads. Both X edges are levelled,
  because unlike the box top - which carries the header badge, and so is
  privileged by text a rotation does not carry with it - neither X edge is
  the one a column must agree on.
- **Helper**: `_level_column_anchor_edges` (`phases/bbox.py`), grouping
  via `_column_contiguous_row_groups` (`phases/_common.py`) then
  `_shared_anchor_runway_runs` (`phases/bbox.py`), levelling each run
  through `level_group_anchor_edges` (`phases/bbox.py`, shared with the
  row top-align).
- **Precondition**: Every X-axis box mover has run - Stage 1.1 sizing,
  the Stage 1.3 column seating, the Stage 3.3 perp-entry runway grow, the
  Stage 3.5 perp inset - so the levelled edge is not re-broken by a later
  widen.
- **Postcondition**: For each X side, within each maximal run of adjacent
  grid rows in one column whose sections' content stations nearest that
  side share an X, every section shares the run's outermost edge on that
  side, except one held short by a neighbour overlapping its own row band
  (which keeps `MIN_INTER_SECTION_GAP` of inter-column corridor). Members
  of a packed cell are out of scope: they sit side-by-side along X inside
  one cell, so no common vertical edge exists. Two kinds of section break a
  run: a rail-flagged one, because `_retrofit_section_rails_phase`
  re-derives its interior from its bbox, so growing its edge would slide
  its stations rather than widen a runway in front of them; and one whose
  exit port rides the edge under test, because that port's coordinate is
  where the inter-section route leaves and the clearances downstream of it
  are measured from there.
- **Invariants preserved**: Station coords (only `bbox_x` / `bbox_w`
  move); the opposite edge of the pair being levelled; every port's own
  edge anchoring (LEFT/RIGHT ports move with the edge they are pinned
  to). Because a run's members share a content X on the side being
  levelled, growing each to the run's outermost edge on that side can only
  raise a narrower runway to the widest already present in the run, so the
  spread of runway widths within a grid column never grows.
- **Validate guard after**: `_guard_ports_on_boundaries`.
- **Related tests**: `test_grid_column_anchor_edge.py`.
- **Lifecycle:** invariant - the levelled edge holds at the final
  boundary for the sections the stage moved
  (`_guard_column_run_shares_its_anchored_edge`).

### Stage 4.1: align ports to downstream
- **Purpose**: For non-fold LR/RL sections, pull exit-entry port
  pairs toward the downstream section's internal stations so lines
  flow without detour.
- **Helper**: `_align_ports_to_downstream` (`phases/ports.py`).
- **Precondition**: Section geometry final (Pass A complete).
- **Postcondition**: Each non-fold LR/RL exit-entry pair Y sits near
  the downstream section's connected station Y.
- **Invariants preserved**: Section bboxes (movement is bbox-bounded,
  Stage 4.6/c recompute bboxes where needed). Real stations.
- **Related tests**: `test_no_kink_at_section_boundary`.
- **Lifecycle:** invariant - exit/entry pairs flow to the downstream
  section (no-kink) at the final boundary (refined, not undone, by Stage
  5.5).

### Stage 4.2: snap sole-layer stations to ports
- **Purpose**: When a port-connected station is the only occupant of
  its layer, snap it to the port Y so the connection is horizontal.
- **Helper**: `_snap_sole_layer_stations_to_ports` (`phases/ports.py`).
- **Precondition**: Stage 4.1 settled port Ys.
- **Postcondition**: Sole-layer port-connected stations share Y with
  their port. Multi-station layers are skipped (would risk collision).
- **Invariants preserved**: Multi-station layer Ys. Shared row-Y grid
  is not respected here (Stage 6.4 re-snaps).
- **Related tests**: `test_section_entry_hub_on_grid` (downstream).
- **Lifecycle:** invariant - the horizontal sole-layer-station-to-port
  connection holds at the end (re-snapped onto the grid by Stage 6.4).

### Stage 4.3: snap grid-group entry ports
- **Purpose**: For grid-group sections (skipped by Stage 4.2), snap entry
  ports to the connected first-internal-station Y - straight
  port-to-station connection.
- **Helper**: `_snap_grid_group_entry_ports` (`phases/ports.py`).
- **Precondition**: Stage 4.2 complete.
- **Postcondition**: Grid-group entry ports share Y with their first
  connected internal station.
- **Invariants preserved**: Internal station Y. Exit ports.
- **Lifecycle:** invariant - grid-group entry ports share Y with their
  first connected station at the final boundary.

### Stage 4.4: snap grid-group exit ports
- **Purpose**: Mirror of Stage 4.3 for exit ports - snap to the downstream
  entry port's Y (which Stage 4.3 just snapped to a grid station).
- **Helper**: `_snap_grid_group_exit_ports` (`phases/ports.py`).
- **Precondition**: Stage 4.3 complete (downstream entry ports snapped).
- **Postcondition**: Grid-group exit ports share Y with their
  downstream entry port (i.e. with the downstream's connected
  station).
- **Invariants preserved**: Internal stations.
- **Lifecycle:** invariant - grid-group exit ports share Y with their
  downstream entry port at the final boundary.

### Stage 4.5: space ports from termini
- **Purpose**: Push ports away from terminus stations so a routed
  line clears any file-icon caption / label by at least `y_spacing`.
- **Helper**: `_space_ports_from_termini` (`phases/ports.py`).
- **Precondition**: Port Ys settled by Stages 4.1 to 4.4.
- **Postcondition**: For every (port, terminus) pair in the same
  section, `|port.y - terminus.y| >= y_spacing` (modulo bbox bounds).
  Bboxes may expand via `_expand_bbox_for_y` to keep ports on edges.
- **Invariants preserved**: Real non-terminus station Y. Other
  sections.
- **Lifecycle:** invariant - the port-to-terminus clearance holds at the
  final boundary.

### Stage 4.6: recompute grid-group bboxes
- **Purpose**: Reset grid-group bboxes to symmetric `max_y_pad`
  padding around final non-port station Y range, then expand for any
  ports outside.
- **Helper**: `_recompute_grid_group_bboxes` (`phases/row_align.py`).
- **Precondition**: Port Ys final (Stage 4.5).
- **Postcondition**: Each grid-group section bbox snugly bounds its
  content with consistent top/bottom padding.
- **Invariants preserved**: Station and port Ys.
- **Lifecycle:** transient - the snug grid-group bbox is superseded by
  the final bbox sizing in Stage 6.13 (bottom) and Stage 6.15a (top).

### Stage 4.7: re-run top-align
- **Purpose**: Re-flush row tops after Stage 4.5 expanded bboxes via
  `_expand_bbox_for_y` (the same row-top alignment Stage 3.4 applies to
  the rows it pushes, here run over every row).
- **Helper**: `_top_align_row_sections` (`phases/row_align.py`).
- **Precondition**: Stages 4.5 / 4.6 complete.
- **Postcondition**: Same-row contiguous-column sections share
  `bbox_y` (station/port Ys shift by the same delta).
- **Invariants preserved**: Relative station-to-section position inside
  each shifted section. Bbox heights.
- **Lifecycle:** transient - superseded by Stage 6.15a, which grows a
  fanned section's bbox top above the flush line.

### Stage 4.8: align row trunk Ys
- **Purpose**: Within each row, shift content downward in shallower
  sections so the inter-section trunk bundle passes through at a
  single Y, then seat each eligible flow exit on its internal carrier
  row so the level change occurs in the inter-section corridor.
- **Helpers**: `_align_row_trunk_ys` (`phases/row_align.py`), then
  `_reconcile_flow_exit_carrier_anchors` (`phases/ports.py`).
- **Precondition**: Stage 4.7 done.
- **Postcondition**: For sections in a row's contiguous column run,
  the trunk Y is the row's deepest pre-pass trunk Y. A non-fold LR/RL
  exit selected by `flow_exit_carrier_anchor` shares its carrier Y;
  its downstream entry remains on the consumer row. Row-spanning
  sections are skipped.
- **Invariants preserved**: Bbox tops, downstream entry coordinates,
  perpendicular exits, and row-spanning sections.
- **Lifecycle:** invariant - the per-row trunk Y is consistent at the
  final boundary (`test_row_trunk_marker_cy_consistent`).

### Stage 4.9: redistribute fan-out siblings
- **Purpose**: For each fan-out column with a unique trunk junction
  (one station carrying the full bundle plus >=2 side branches),
  redistribute side stations symmetrically around the trunk Y. No-op
  unless `graph.center_ports` (guard inside the helper, not at the call
  site).
- **Helpers**: `_snapshot_planned_fan_centrelines` and
  `_apply_planned_fan_geometry` (`phases/planned_fans.py`) materialise complete
  semantic plans first; `_redistribute_fanout_siblings`
  (`phases/fan_bundles.py`) handles unsupported fans.
- **Precondition**: Trunk Ys aligned (Stage 4.8).
- **Postcondition**: In qualifying columns, fan-out siblings sit
  symmetrically around the section's LR/RL port trunk anchor (the trunk
  station's own Y only when the section has no such port). Linear chains,
  fan-in structures, and file inputs are left in place.
- **Invariants preserved**: Trunk station Y. Off-track stations.
- **Purity**: semantic plans read a centreline frozen immediately after
  structural settlement; legacy fans centre on the frozen port anchor. Neither
  path depends on a governed station's live Y (#491).
- **Lifecycle:** transient - superseded by Stage 6.7 / 6.11, which
  re-fan the siblings against the final trunk Y (this fan uses the early
  trunk Y).

### Stage 4.10: redistribute full-bundle columns (engine.py)
- **Purpose**: When a column has no unique trunk (every station
  carries the full bundle - e.g. Reporting's Shiny + Quarto),
  symmetrically fan stations around the local LR port Y. No-op unless
  `center_ports` (guard inside the helper, not at the call site).
- **Helper**: `_redistribute_full_bundle_columns` (`phases/fan_bundles.py`).
- **Precondition**: Stage 4.9 ran.
- **Postcondition**: Full-bundle columns sit symmetric around the
  LR port Y.
- **Why both this and Stage 6.7**: Stage 6.7
  (``_recenter_full_bundle_columns``) re-fans the same columns
  using the final trunk Y, which can have drifted from Stage 4.10's
  port-Y anchor.  Stage 4.10's output is *not* redundant: the
  intermediate symmetric layout is read by Pass C's bbox-growth
  and compaction passes (an empty trunk row in fanned columns lets
  Stages 5.4 / 6.13 shrink the section bbox to the compact extent).
  Skipping Stage 4.10 changes intermediate bbox sizes and is not
  empty-render-diff -- the two passes are load-bearing in combination.
- **Invariants preserved**: Other columns.
- **Lifecycle:** transient - superseded by Stage 6.7, which re-fans the
  full-bundle columns against the final trunk Y (this fan uses the local
  port Y).

### Stage 5.1: position junctions
- **Purpose**: Place each junction station in the inter-section gap
  at the exit port's Y (fan-out) or near the entry port (merge).
- **Helper**: `_position_junctions` (`phases/junctions.py`).
- **Precondition**: All port Ys final (Pass B complete).
- **Postcondition**: Every junction has finite `(x, y)`. Fan-out
  junctions sit at `exit_port.y` plus a `JUNCTION_MARGIN` X offset
  toward the targets; merge junctions sit at
  `max(pred.x) + JUNCTION_MARGIN, entry_port.y`.
- **Invariants preserved**: Real stations, ports.
- **Lifecycle:** invariant - junctions track their ports at the final
  boundary (`junction.xy == _compute_junction_xy(ports)`, re-established
  after every later port move).

### Stage 5.2: lift off-track stations (engine.py)
- **Purpose**: Offset off-track file artefacts one step clear of their
  anchor along the section's cross axis (Y for an LR/RL trunk, X for a
  TB/BT one; `section_cross_axis`), stacking when several share one
  anchor. An input's anchor is its consumer; a producer-fed sink's anchor
  is its producer (see `_off_track_anchor_of`). Grow bbox along the cross
  axis to fit the band and along the flow axis to fit the icon extent;
  nudge same-side ports back to the new edges.
- **Helper**: `_lift_off_track_stations`.
- **Precondition**: Stage 5.1 complete; all on-track Ys final.
- **Postcondition**: Each off-track station sits at
  `anchor_cross +/- n*step` (n = stack rank) on the cross axis, keeping
  its own flow-axis (layer) coordinate. The `step` is the cross pitch:
  `y_spacing` for a horizontal section (base content pitch
  `graph._base_y_spacing` on a single-trunk section, so the diagonal-label
  widening doesn't strand the icon, issue #580), or the resolved column
  pitch for a vertical section (`_off_track_lift_step`). Section bbox
  extends to fit.  May leave the topmost section above the canvas margin --
  ``_shift_graph_into_canvas`` runs immediately afterwards to restore the
  margin (called explicitly by the caller, not by the helper).
- **Invariants preserved**: On-track station Y. Other sections' Ys
  (only the canvas Y-offset may shift the world uniformly).
- **Related tests**: `test_off_track_inputs_above_consumer`,
  `test_off_track_outputs_above_and_adjacent_to_producer`,
  `test_off_track_icons_ordered_by_consumer_y`.
- **Lifecycle:** invariant - off-track stations sit a step clear of their
  anchor on the cross axis at the final boundary. *liftable:* only behind
  a "consumers final" precondition - the anchor uses the consumer/producer's
  final Y and is re-applied by Stages 6.6 / 6.8 (#463).

### Stage 5.3: re-align row bbox tops only
- **Purpose**: After Stage 5.2 grew some bboxes upward, grow other
  same-row bboxes upward to match. Station Ys in unlifted sections
  preserved.
- **Helper**: `_top_align_row_bboxes_only` (`phases/row_align.py`).
- **Precondition**: Stage 5.2 may have lifted some bboxes.
- **Postcondition**: Within each row's contiguous column group, all
  bboxes share `bbox_y` (heights extended upward as needed).
- **Invariants preserved**: All station / port Ys.
- **Lifecycle:** transient - superseded by Stage 6.15a (flush row tops,
  as Stage 4.7).

### Stage 5.4: compact row content to bbox top
- **Purpose**: Shift each row's column-group up by the smallest
  above-content slack, then shrink bbox heights to remove the empty
  band. Preserves trunk alignment.
- **Helper**: `_compact_row_content_to_bbox_top` (`phases/row_align.py`).
- **Precondition**: Bbox tops aligned (Stage 5.3).
- **Postcondition**: Each row's contiguous column group's bbox top
  sits at `min(content_top) - section_y_padding`, except where
  `_perp_port_lead_edge_reserve` caps the shift so a perpendicular port
  keeps `PERP_PORT_EDGE_INSET` inside the edge -- there the top stays
  higher and the group's content keeps more than the padding above it.
  The reserve is measured from the port station, which is also its topmost
  drawn lane: a port's bundle staggers below it, never above
  (`port_bundle_edge_reach`).
  Stations shift up by the same delta as their bbox.
- **Invariants preserved**: Inter-station relative positions inside
  each section. Trunk Y stays aligned across the row.
- **Related tests**: `test_section_bbox_has_bottom_padding`.
- **Lifecycle:** transient - superseded by Stage 6.1 (fans content back
  into the band) and Stage 6.13 (re-sizes the bbox bottom).

### Stage 5.5: snap inter-section port pairs + reposition junctions
- **Purpose**: Snap exit/entry port pairs in the same row to a shared
  Y (the entry's), then re-run Stage 5.1 to put junctions back on the
  exit port.
- **Helper**: `_snap_inter_section_port_pairs` (`phases/balancing.py`) then
  `_position_junctions`.
- **Precondition**: Row compaction done; port pair Ys may have drifted.
- **Postcondition**: Within each row, every LEFT/RIGHT exit port and
  its connected LEFT/RIGHT entry port share a Y. Junctions back at
  exit-port Y.
- **Invariants preserved**: Internal station Y in each section.
- **Related tests**: `test_no_kink_at_section_boundary`,
  `test_inter_section_route_y_stays_within_row_band`.
- **Lifecycle:** invariant - LR/RL exit-entry port pairs share a Y
  (no-kink) and junctions track their ports at the final boundary.

### Stage 6.1: fan free content upward
- **Purpose**: When the row's compaction leaves visible empty top
  band but the section has trunk-candidate sibling stations,
  fan those upward into the empty band.
- **Helper**: `_fan_free_content_upward` (`phases/balancing.py`).
- **Precondition**: Trunk Y aligned (Stage 4.8). Compaction done
  (Stage 5.4).
- **Postcondition**: Eligible sections fan stations upward by at most
  one `y_spacing` slot, balancing content above/below trunk.
- **Invariants preserved**: Trunk station Y. Off-track stations
  (sections with off-track band are skipped).
- **Purity**: top slack and anchor are read from the frozen placement
  reference (see Content-placement purity), not live geometry (#491).
- **Related tests**: `test_section_top_band_filled`,
  `test_section1_input_above_trunk`.
- **Lifecycle:** invariant - the filled top band / content balanced
  around the trunk holds at the final boundary
  (`test_section_top_band_filled`). Stage 6.11 can fill the same band on
  the same section, but moves a *disjoint* station set (strict-subset,
  non-trunk siblings; this stage moves only full-bundle trunk
  candidates), so it does not override this placement.

### Stage 6.2: fan source inputs upward
- **Purpose**: Companion to Stage 6.1 for source-stack sections (single
  full-bundle trunk + subset-bundle file inputs at the entry column).
  Lift trunk-nearest source inputs into the empty top band.
- **Helper**: `_fan_source_inputs_upward` (`phases/balancing.py`).
- **Precondition**: Stage 6.1 done.
- **Postcondition**: Section is top- and bottom-weighted around the
  trunk row instead of stacked below it.
- **Invariants preserved**: Trunk station Y.
- **Purity**: trunk anchor is the frozen LR/RL port Y and the lift count
  reads the frozen placement-reference bbox top, not live geometry (#491).
- **Lifecycle:** invariant - source-stack sections stay
  top-and-bottom-weighted around the trunk at the final boundary.

### Stage 6.3: 2-branch symfan half-grid compaction (engine.py)
- **Purpose**: Sections containing exactly a 2-branch symmetric fan
  (no off-track / constraining content) collapse onto half-pitch
  offsets so the section is 1 grid-unit tall instead of 2. The two
  branches may be fed from upstream (entry port or a terminus source
  icon) or from a single in-section non-terminus source whose two
  consumers are equal siblings (identical line sets); that source is
  the fan hub and is excluded from the branch count. Records the placed
  branches on the public `MetroGraph.half_grid_station_ids` field so
  Stage 6.4 leaves them alone -- this is the only cross-phase channel
  for half-grid placement. The fan's remaining on-track stations (its
  source/trunk) are recorded on `MetroGraph.symfan_trunk_station_ids`
  so Stage 6.4 keeps them on the same local frame; a single in-section
  equal-sibling source hub is additionally moved to the trunk Y so the
  fork is a balanced Y-split rather than collinear with one branch.
  Gated on `center_ports`.
- **Helper**: `_apply_half_grid_2branch_symfan`
  (classification via `_symfan_branches_hub` /
  `_section_symfan_uses_half_grid`).
- **Precondition**: Stages 6.1 / 6.2 done; symfan classification stable
  (`_section_symfan_uses_half_grid`).
- **Postcondition**: Eligible symfan pairs share half-pitch offsets
  from the trunk Y; an in-section equal-sibling source hub sits on the
  trunk Y, centred between them. `graph.half_grid_station_ids` contains
  the branch IDs; `graph.symfan_trunk_station_ids` contains the fan's
  source/trunk IDs.
- **Invariants preserved**: Trunk station Y. Other sections.
- **Related tests**: `test_symfan_pairs_share_y`.
- **Lifecycle:** invariant - 2-branch symfan pairs keep their half-pitch
  offsets at the final boundary (Stage 6.4 skips
  `graph.half_grid_station_ids`); only Stage 6.18 may seat one on a full
  row, and only once its straddling partner has moved away.

### Stage 6.4: snap all Y to grid (engine.py)
- **Purpose**: Final pass snapping every station and port Y to the
  nearest row-wide grid slot, removing fractional Ys left by earlier
  shifts. Stations listed in `graph.half_grid_station_ids` (populated
  by Stage 6.3) are skipped so they keep their intentional half-pitch
  Y. The stage then restores each fan-in target to the midpoint of its
  sources. For a symmetric diamond, it also restores the fork hub and
  its unbranched trunk to the branch midpoint. This keeps the complete
  centreline straight. A restored station joins
  `graph.half_grid_station_ids` if it sits half a pitch from the branch
  grid.
- **Helper**: `_snap_all_y_to_grid`, with
  `_restore_convergence_midpoints` / `_restore_divergence_midpoints`
  and `_centreline_trunk_followers` (`phases/fan_bundles.py`) for the
  restores.
- **Precondition**: All semantic Y shifts done. If Stage 6.3 ran,
  `graph.half_grid_station_ids` is populated.
- **Postcondition**: Every station and port Y is a grid slot of the
  per-section / per-row pitch (except marked half-grid stations). A
  symmetric diamond's fork hub, join and trunk run share one Y.
- **Invariants preserved**: X coordinates (tested by
  `test_grid_snap_does_not_mutate_x`). Half-grid station Ys.
- **Related tests**: `test_all_stations_snap_to_grid`,
  `test_grid_snap_does_not_mutate_x`,
  `test_fork_and_join_hub_share_centreline`,
  `test_ported_fan_centreline_reaches_ports_and_trunk`.
- **Lifecycle:** invariant - every (non-half-grid) station/port Y is a
  grid slot at the final boundary (re-asserted canvas-wide by Stage
  6.15).

### Stage 6.5: align TB-section bbox bottoms
- **Purpose**: Extend TB-section bbox bottom to match the downstream
  LR/RL section's *settled content* bottom so the line doesn't look
  pinned to the TB bbox edge, and the straight inter-section run clears
  both section bottoms by the same distance. The target's settled
  content bottom (`_predict_section_content_bottom`) is used rather than
  its live `bbox_h`, which the later bbox-shrink phase may collapse.
- **Helper**: `_align_tb_section_bbox_bottoms` (`phases/ports.py`).
- **Precondition**: All station/port Ys final (post-snap).
- **Postcondition**: For each TB section feeding an LR/RL target,
  `tb.bbox_y + tb.bbox_h >= target settled content bottom`. After the
  bbox-shrink phase the two edges are level for a straight run (guarded
  by `_guard_fold_lr_exit_sections_share_bbox_bottom`, #1162).
- **Invariants preserved**: All station and port Ys. Other bboxes.
- **Lifecycle:** invariant - TB-section bbox bottoms align with their
  downstream LR/RL target at the final boundary.

### Stage 6.6: reanchor off-track to consumer (engine.py)
- **Purpose**: Re-pin each off-track station `n*step` clear of its anchor
  on the cross axis using the anchor's final snapped coordinate (Stage 5.2
  used pre-snap ones); the anchor is the consumer for an input, the
  producer for a sink. Recompute the lift-side bbox edge to fit the band
  (grow **or** shrink); grow the opposite and flow edges as needed.
- **Helper**: `_reanchor_off_track_to_consumer`.
- **Precondition**: Stage 6.4 snapped consumers to grid. Enforced
  explicitly via `graph._consumers_grid_snapped` (set right after the
  Stage 6.4 snap); the helper raises `PhaseInvariantError` if it runs
  while unset, so the dependence on snapped consumers is no longer
  implicit in call position (#463).
- **Postcondition**: Off-track stations sit `n * step` clear of their
  anchor's final cross coordinate. The lift-side bbox edge hugs the band
  (recompute-to-fit, so re-running is order-independent). May leave the
  topmost section above the canvas margin -- ``_shift_graph_into_canvas``
  runs immediately afterwards (called explicitly by the caller, not by the
  helper).
- **Invariants preserved**: On-track station Y.
- **Related tests**: `test_off_track_inputs_above_consumer`,
  `test_off_track_outputs_above_and_adjacent_to_producer`,
  `test_reanchor_off_track_requires_snapped_consumers`,
  `test_reanchor_off_track_bbox_fit_is_reversible`.
- **Lifecycle:** invariant - off-track stations sit a step clear of their
  anchor's final cross coordinate. *liftable:* as a **precondition-gated** invariant
  (#463): the bbox fit is now reversible, but the helper *raises* when
  `_consumers_grid_snapped` is unset, so a run-anytime `maintain()` pass
  must check that flag and skip while consumers are pre-snap rather than
  call-and-catch. Registry integration deferred to #459.

### Stage 6.7: re-center full-bundle columns (engine.py)
- **Purpose**: Re-fan full-bundle columns around the row's final trunk
  Y (Stage 4.10 used the local port Y which may now be stale).
  Gated on `center_ports`.
- **Helper**: `_recenter_full_bundle_columns`, then the port-seating pair
  `_center_lr_entry_ports_on_fork` / `_center_lr_exit_ports_on_join`, which
  seat a flow-aligned port on the centreline of the two-way fork it feeds or
  the two-way join that feeds it. A port already level with one of those
  branches is left there: that is a dead-end fan's legitimate seat, where the
  branch's track is the trunk the inter-section run continues along.
- **Precondition**: Final inter-section trunk Y known (post-snap).
- **Postcondition**: Full-bundle columns are symmetric around the
  row's final trunk Y; a flow-aligned port bounding a two-way fork or join
  sits on one of its branches' tracks or on their midpoint.
- **Invariants preserved**: Off-track Y anchoring (re-established by
  Stage 6.8) and bbox-top alignment (re-established by Stage 6.9)
  are temporarily broken; both are restored before leaving the
  `if center_ports:` block.
- **Lifecycle:** invariant - full-bundle columns are symmetric around
  the row's final trunk Y at the boundary; no later stage re-fans them,
  though Stage 6.18 seats a half-pitch member on a full row once its
  straddling partner has moved away.
  *liftable:* no - one-shot, order-dependent (computes against the final
  trunk Y, so a premature run is wrong).

### Stage 6.8: re-anchor off-track after recenter (engine.py)
- **Purpose**: The Stage 6.7 recenter moves consumers to the final
  trunk-anchored Y, leaving off-track icons stranded at the old
  consumer Y (and overlapping the consumer station). Re-pin each
  off-track at `consumer.y - n*y_spacing` on the post-recenter grid.
  Followed by ``_shift_graph_into_canvas`` to handle bbox grow that
  pushed the topmost section above the canvas margin.  Gated on
  `center_ports`.
- **Helper**: `_reanchor_off_track_to_consumer` (same helper as
  Stage 6.6; called again here on the post-recenter Ys).
- **Precondition**: Stage 6.7 has re-centred full-bundle columns.
- **Postcondition**: Off-track inputs sit one or more pitches above
  their post-recenter consumer. Section tops are recomputed to fit the
  off-track band (grow or shrink), so re-running is order-independent.
- **Invariants preserved**: Row top-alignment may be broken when a
  bbox top moved; Stage 6.9 restores it.
- **Lifecycle:** invariant - off-track inputs sit a pitch above their
  post-recenter consumer at the final boundary. *liftable:* as a
  **precondition-gated** invariant (#463): reversible bbox fit, but the
  helper raises while `_consumers_grid_snapped` is unset, so a run-anytime
  `maintain()` pass must check that flag and skip until consumers are
  snapped rather than call-and-catch. Registry integration deferred to
  #459.

### Stage 6.9: re-run row top-align (engine.py)
- **Purpose**: A Stage 6.8 bbox grow can leave the grown section's
  bbox top above its row mates'. Pull row mates' bbox tops up to
  match so the section row stays flush along its top edge. Gated on
  `center_ports`.
- **Helper**: `_top_align_row_bboxes_only` (same helper as Stage 5.3).
- **Precondition**: Stage 6.8 has re-anchored off-track inputs.
- **Postcondition**: Row bboxes flush at the top across all row mates.
- **Invariants preserved**: Station Ys (only bbox tops move).
- **Lifecycle:** transient - superseded by Stage 6.15a (flush row tops,
  as Stage 4.7).

### Stage 6.10: align terminus to upstream
- **Purpose**: After Stage 6.7 re-pitched fanned columns, a single-station
  downstream column (e.g. a `file` terminus) may have stayed at its
  pre-fan Y. Pin it back onto its sole upstream's Y.
- **Helper**: `_align_terminus_to_upstream` (`phases/single_section.py`).
- **Precondition**: Stage 6.7 re-centered fans.
- **Postcondition**: Single-station downstream columns share Y with
  their unique upstream.
- **Invariants preserved**: Multi-station columns.
- **Related tests**: `test_terminus_not_directly_after_diagonal`.
- **Lifecycle:** invariant - single-station downstream columns share Y
  with their unique upstream at the final boundary.

### Stage 6.11: balance section content around trunk
- **Purpose**: Auto-balance pass. For sections whose final layout
  still has an empty band above the trunk while more siblings sit
  below than above, lift bottommost movable siblings into the empty
  top band. U-turn-safe and bbox-bounded.
- **Gating**: Early-returns unless `graph.layout_provenance` contains at least
  one author-owned grid decision and `graph.center_ports` is set (scoped to
  explicit-`%%metro grid:` + centre-ports pipelines), so it is a no-op on
  auto-laid graphs.
- **Helper**: `_balance_section_content_around_trunk` (`phases/balancing.py`).
- **Precondition**: All earlier 13-phase reshuffles done.
- **Postcondition**: Sibling count above trunk >= sibling count below
  trunk (where movable), inside bbox.
- **Invariants preserved**: Trunk station Y. Sections that already
  balance are left alone.
- **Purity**: an in-scope reset restores every station to its frozen
  placement-reference Y before the lift/swap loop, and the band gates /
  feeder check read the reference, so the balance decision does not depend
  on live geometry (#491).
- **Related tests**: `test_section_top_band_filled`.
- **Lifecycle:** invariant - section content is balanced around the
  trunk (siblings above >= below, where movable) at the final boundary.

### Stage 6.12: recenter loop side stations
- **Purpose**: Recompute the X of fan-out side stations (one trunk
  predecessor, one trunk successor - "loop side" stations like propd,
  dream, DESeq2 around limma) to the midpoint of their actual diagonal
  corner Xs from the routing geometry.
- **Helper**: `_recenter_loop_side_stations` (`phases/balancing.py`).
- **Precondition**: All Y phases done; routing geometry derivable.
- **Postcondition**: Loop side stations sit at the visual centre of
  their horizontal loop run.
- **Invariants preserved**: Station Y. Pure-side-branch classification
  is strict (see `test_loop_recenter_only_for_pure_side_branches`).
- **Related tests**: `test_fan_station_centered_on_loop`,
  `test_loop_recenter_only_for_pure_side_branches`,
  `test_loop_column_stations_share_x`.
- **Lifecycle:** invariant - loop-side stations sit at the visual centre
  of their loop run at the final boundary.

### Stage 6.13: shrink and tighten rows
- **Purpose**: Shrink each section's bbox bottom to
  `max_content_y + section_y_padding` (phase 1), then pull lower-row
  sections up to close any vertical slack the shrink revealed
  (phase 2).  Phase 1 handles bbox bottoms that drifted after earlier
  passes lifted content; phase 2 handles the pre-shrink row-height
  overestimate when a rowspan section collapses to less than its
  row claim.  Phase 2 must run as a second pass over the graph so
  every section's shrink is finalised before row-gap deficits are
  measured.  Phase 2 reads `bbox_y + bbox_h` from Phase 1's content-hugging
  bbox as the row-ending extent.  If `graph._struct_height_below_top`
  is populated, its per-section height is used instead (reconstructed
  on the current bbox top); that dict is populated after Stage 6.15a
  so it records the fully settled extent for structural-extent fidelity
  checks, not as a cascade input.
- **Helper**: `_shrink_and_tighten_rows` (orchestrates
  `_shrink_bboxes_to_content_bottom` then
  `_tighten_lower_rows_after_shrink`).
- **Precondition**: All content Ys final.
- **Postcondition**: Section bbox bottoms sit `section_y_padding`
  below the deepest content (trunk alignment unaffected -- only
  bottom shrinks), and clear the lowest drawn lane of every port the box
  holds: `PERP_PORT_EDGE_INSET` for a perpendicular port, otherwise
  `PERP_PORT_EDGE_CLEARANCE`, both measured from the port's outermost lane
  rather than the port station (`port_bundle_edge_reach`).  For each row pair,
  the row gap is `section_y_gap` (no more, no less, except where rowspan
  sections filled their full row claim).  A row pair claimed by
  `_merge_trunk_row_minimums` keeps that wider minimum between the two row
  *envelopes*: the trunk's channel crosses the whole boundary, so no
  column-overlapping section pair bounds it and none records the two rows as
  related (its connectors are rewritten through fan and merge nodes).
- **Invariants preserved**: Bbox tops. Within-row trunk Ys. Bbox
  heights of upper rows.
- **Related tests**: `test_section_bbox_has_bottom_padding`,
  `test_section_bbox_matches_content_extent`.
- **Lifecycle:** invariant - content-hugging bbox bottoms and correct
  inter-row gaps hold at the final boundary (maintained by Stage 6.14,
  which restores the gap via `push_lower_rows_after_bbox_grow` whenever
  it grows a bbox downward). *liftable:* no - one-shot, order-dependent
  (computes against the final content extent).

### Stage 6.14: shift and propagate loop stations
- **Purpose**: Shift sparse loop-side stations (one inbound, one
  outbound, single-line consumer) onto a half-pitch Y when sharing
  the full-row Y with a busier sibling whose inbound bundle would
  otherwise breeze-past the sparse station's marker.  When a shift
  grows a section's bbox downward, push lower-row sections down
  internally to restore `section_y_gap`.
- **Helper**: `_shift_and_propagate_loop_stations`
  (calls `push_lower_rows_after_bbox_grow` when any bbox grew).
- **Precondition**: Bundle Ys final.
- **Postcondition**: Sparse single-line loop stations whose row Y
  conflicts with a busier sibling's bundle move to a half-pitch
  offset (may grow bbox downward).  Row gaps preserved across any
  bbox grow.
- **Invariants preserved**: Busy sibling Y. Bundle Y. Within-row Ys
  of unaffected sections.
- **Related tests**: `test_lines_dont_cross_non_consumer_markers`,
  `test_no_icon_overlaps_line_path`,
  `test_row_gap_accommodates_bypass`.
- **Lifecycle:** invariant - sparse loop-side stations keep their
  half-pitch offset at the final boundary; row gaps preserved across any
  bbox grow.

### Stage 6.15a: fit bbox tops to content (grow and shrink)
- **Purpose**: Size each bbox top to `section_y_padding` above its highest
  marker, bounded by the row above. Grows when fan re-distribution (Stages
  4.9 / 4.10 / 6.7 / 6.11) lifted a branch above the line the bbox was sized
  for, crowding the topmost marker (issue #406). Shrinks when the transient
  row-top flush left an empty band above content with nothing in it (no port
  or bypass helper); a band holding a port or bypass helper is left intact.
  The upward grow can breach the canvas top margin, so
  `_shift_graph_into_canvas` runs immediately after. That shift keeps every
  section `section_y_padding` below the canvas top and, on a titled map, keeps
  every *drawn* section `TITLE_BAND_CLEARANCE` below it so the header badge
  clears the title band (issue #1273).
- **Helper**: `_fit_bboxes_to_content_top` (`phases/bbox.py`), then
  `_shift_graph_into_canvas`.
- **Precondition**: All content Ys final (post-6.14).
- **Postcondition**: Each bbox top sits `section_y_padding` above its
  highest marker, or `PERP_PORT_EDGE_INSET` above the topmost drawn lane of a
  perpendicular port that reaches higher, whichever is further out. For a
  section with an empty band (no port / bypass above content) the padding term
  is an equality, not just a floor: the excess band is reclaimed. Both port
  terms are measured from the port's outermost lane rather than the port
  station (`port_bundle_edge_reach`), and a port the inset does not cover still
  owes `PERP_PORT_EDGE_CLEARANCE` past that lane.
- **Invariants preserved**: Station Ys (only bbox tops move). Resolves #406.
- **Related tests**: `test_section_bbox_has_top_padding`,
  `test_section_bbox_top_hugs_content`.
- **Lifecycle:** invariant - each bbox top hugs its highest marker at the
  final boundary (a full `section_y_padding`, an equality for empty-band
  sections), the final top-sizing pass. Row-top flush alignment is not a
  maintained property; it is transient scaffolding superseded here.

### Stage 6.15b: distribute stacked rows across a rowspan band
- **Purpose**: When a column holds single-row sections stacked one per grid
  row beside an adjacent `grid_row_span > 1` section spanning those rows,
  distribute them across that section's vertical band so the topmost's bbox
  top meets the band top and the bottommost's bbox bottom meets the band
  bottom. Otherwise a `center_ports` fan in the top section spreads above the
  band into the title space, and the bottom section floats high with slack
  beneath it.
- **Helper**: `_distribute_stacked_rows_in_rowspan_band` (`phases/row_align.py`),
  after the Stage 6.15a fit and before `_shift_graph_into_canvas`.
- **Precondition**: Bbox tops content-fitted (post-fit), bboxes final-sized.
- **Postcondition**: For a qualifying stack (one section per band row, with
  band slack), the topmost top equals the band top and the bottommost bottom
  equals the band bottom; sections shift without resizing.
- **Invariants preserved**: Bbox heights; intra-section station geometry
  (each section's stations and ports shift together).
- **Related tests**: `test_stacked_rows_fill_rowspan_band`; runtime guard
  `_guard_stacked_rows_fill_rowspan_band`. Resolves #1207, #1209.
- **Lifecycle:** invariant - a qualifying stack fills its rowspan band at the
  final boundary.

### Stage 6.15: snap canvas to the y-grid
- **Purpose**: After all settling, restore canvas-wide grid alignment.
  Stage 6.4 snaps to a per-row grid, but later helpers (notably
  `_shift_graph_into_canvas` shifting by a non-grid amount) can leave a
  uniform residue; shift the whole canvas back onto integer `y_spacing`
  multiples.
- **Helper**: `_snap_canvas_y_to_grid`.
- **Precondition**: All other Y phases done.
- **Postcondition**: Real stations sharing a single non-zero residue are
  shifted onto integer `y_spacing` multiples; mixed-residue (multi-row)
  layouts and half-grid / convergence stations are left untouched. A
  candidate grid shift is rejected if it would pull the top above the
  canvas margin or (on a titled map) a drawn section into the title band.
- **Invariants preserved**: Relative station/section/port Ys (the whole
  canvas moves by one delta).
- **Related tests**: `test_auto_y_spacing_fits_content`.
- **Lifecycle:** invariant - canvas-wide grid alignment holds at the
  final boundary (the last Y pass; only ports/junctions move after, via
  Stage 6.16).

### Stage 6.16: re-align vertical-flow entry ports + re-anchor junctions
- **Purpose**: A vertical-flow (TB/BT) section's perpendicular entry port is
  pinned a fixed offset above its first internal station, so the late vertical
  settling (Stages 6.13-6.15) that shifts the section's content drags the entry
  port off the upstream feeder Y it was snapped to in Stage 3.2, re-introducing
  an inter-section S-kink. Re-run the port alignment for vertical-flow sections
  to re-snap them, then re-anchor every junction (any direction) to the settled
  exit/entry port Ys, since junctions live in inter-section space and the
  settling phases leave them stale.
- **Helper**: `_align_entry_ports(graph, vertical_only=True)`
  (`phases/ports.py`), then `_position_junctions`.
- **Precondition**: All vertical settling done (post-6.15).
- **Postcondition**: Vertical-flow entry ports share their upstream feeder's
  Y; all junctions re-anchored to the settled ports.
- **Invariants preserved**: Horizontal-flow (LR/RL) entry/exit geometry, which
  `vertical_only` leaves on the positions the settling phases deliberately gave
  it.
- **Validate guard after**: bisection set ("after Stage 6.16").
- **Lifecycle:** invariant - vertical-flow entry ports share their upstream
  feeder Y (no-kink) and junctions track them at the final boundary.
- **Why this pass stays (axis-generic, not removed)**: the port re-align is
  scoped (`vertical_only`), not TB-special-cased, but it is load-bearing and
  irreducible. Re-running the *full* alignment here would drag horizontal-flow
  ports (9 across the corpus, e.g. longread `small_variants` by +86px) off
  their settled positions, so the scope cannot be dropped; and removing the
  pass re-introduces the S-kink on the vertical-flow ports it corrects (2
  across the corpus: longread `phasing` +16.8px, `tb_file_termini` `reporting`
  -14px). The companion `_position_junctions` is not TB-specific at all - it
  re-anchors stale junctions (any direction) after the settling phases (17
  across the corpus, some by hundreds of px).

### Stage 6.17: semantic fan settlement and symmetric compaction (engine.py)
- **Purpose**: Re-materialise every planned semantic fan against its settled
  centreline. Under `diamond_style='symmetric'`, a planned two-way fan keeps
  mirrored half-pitch lanes around that centreline even when topology identifies
  one branch as the unique continuation. For unsupported legacy fans, compact
  each clean 2-way fork-join diamond (`_iter_symmetric_diamonds`) onto half-pitch
  offsets `trunk_y +/- 0.5 * y_spacing`, so the diamond reads as a tight
  one-grid-unit bubble rather than straddling the trunk at full pitch
  (as tall as a 3-way fan with an empty trunk row between its branches).
  Per-diamond, so a diamond compacts even when it shares a section with a
  wider fan (which keeps its full-pitch slots) and regardless of
  `center_ports`. Records the branches on
  `MetroGraph.half_grid_station_ids`. Runs after every trunk-settling
  pass, so the branches straddle the section trunk's final Y exactly; the
  compaction only moves them inward toward the trunk, so it never breaks bbox
  containment.
- **Helpers**: `_snapshot_planned_fan_centrelines` captures the settled frame,
  `_apply_planned_fan_geometry` materialises it, then
  `_apply_half_grid_symmetric_diamonds` for symmetric legacy geometry.
- **Precondition**: Trunk Ys and section bboxes settled (post-6.16).
- **Postcondition**: Each planned station realises its immutable relative frame.
  Each symmetric two-way fan straddles one centreline at half pitch; legacy
  diamond branch IDs are recorded in `graph.half_grid_station_ids`.
- **Invariants preserved**: Trunk station Y, ports, section bboxes, unrelated
  row-mate bbox tops, and wider fan full-pitch slots.
- **Related tests**: `test_symmetric_diamond_compacts_to_half_pitch`,
  `test_symmetric_diamond_both_branches_deviate`,
  `test_symmetric_style_keeps_planned_two_way_fan_on_shared_centreline`,
  `test_planned_fan_does_not_level_unrelated_row_bbox_tops`,
  `_guard_symmetric_diamond_branches_straddle_trunk`, and
  `_guard_planned_fan_frame_realised`.
- **Lifecycle:** invariant - symmetric diamond branches keep their
  half-pitch offsets at the final boundary; only Stage 6.18 may move one,
  and only when its straddling partner is gone.

### Stage 6.18: orphaned half-pitch expansion (engine.py)
- **Purpose**: A half-pitch offset encodes one side of a symmetric pair
  straddling the section trunk, so the pair reads as one compact grid
  unit. Stage 6.10's `_align_terminus_to_upstream` may pull a terminus
  member onto its producer's trunk Y, leaving the other member holding an
  offset that straddles nothing and rendering as a branch stranded
  between two grid rows. `_straddles_nothing` mirrors each marked
  station's offset about the section's LR/RL port anchor; with no station
  at the mirrored slot, the branch is seated one full row from the anchor
  on the side it already sits, its half-grid marking cleared, and the
  section bbox grown over the moved branch alone. Stations marked
  half-grid whose settled Y is already a whole number of rows from the
  anchor are left alone. A Stage 6.4 centreline has no mirror member, so
  the fork hub and its midpoint trunk are exempt.
- **Helper**: `_expand_orphaned_half_grid_stations`
  (`phases/fan_bundles.py`), sharing `_half_grid_frame` /
  `_straddles_nothing` with the invariant test.
- **Precondition**: Every pass that places or dissolves a half-pitch pair
  has run (post-6.17), so the half-grid marks are final.
- **Postcondition**: No station in `graph.half_grid_station_ids` sits half
  a pitch off its section's LR/RL port anchor with the mirrored slot
  empty.
- **Invariants preserved**: Trunk station Y, ports, bbox containment.
  Deliberately not preserved: the half-grid marker set (the seated
  station's id is discarded, so the post-layout readers see only stations
  still at half pitch) and the section bbox extent, which grows over the
  seated branch. No runtime `_guard_*` arms this postcondition:
  `test_half_grid_stations_straddle_in_pairs` covers it across the corpus
  without the abort risk a live guard would add to novel input.
- **Related tests**: `test_half_grid_stations_straddle_in_pairs`.
- **Lifecycle:** invariant - the expanded branch keeps its full-row Y at
  the final boundary (no later Y mutation). The cleared marker reaches the
  next `_layout_once` pass, which re-derives the marks from scratch.

### Stage 6.18a: refit planned fan bbox tops (engine.py)
- **Purpose**: Stage 6.17 can move planned fan content after the general bbox
  fit in Stage 6.15a. Remove top slack left by that final placement without
  forcing the section to share a top edge with its row mates.
- **Helper**: `refit_empty_section_tops_to_content` (`phases/bbox.py`), scoped
  by `planned_fan_layout_section_ids` (`phases/planned_fans.py`).
- **Precondition**: Planned fan geometry and half-pitch expansion are settled
  (post-6.18).
- **Postcondition**: A planned fan section with an unused top band has exactly
  `section_y_padding` above its highest visible content.
- **Invariants preserved**: Station and route geometry, unrelated section
  bboxes, and top bands used by ports or bypass helpers.
- **Related tests**: `test_section_bbox_top_hugs_content` and
  `_guard_section_top_padding`.
- **Lifecycle:** invariant - no geometry or bbox phase follows this refit.

## Post-layout routing boundary: exit-turn planning

- **Purpose**: Decide source-lane order and turn axes for every complete
  inter-section exit group before route emission.
- **Helpers**: `compute_station_offsets` produces the base offset map.
  `_route_edges` calls `build_exit_turn_execution` once, immediately before
  dispatch. That call plans complete groups and commits their owned compact
  offsets to the routing context.
- **Precondition**: Layout coordinates and topology resolution are settled and
  remain immutable. The mutable per-line offset map has completed all local,
  port, junction, and rail-boundary phases.
- **Postcondition**: Each supported exit group has compact active lanes, one
  assignment per outbound member, and any needed ordered turn axes, lane
  transitions, references, and runway demands. Any unsupported member places
  the whole group on the legacy path.
- **Invariants preserved**: Station, port, junction, and section coordinates.
  The planner may commit per-line station offsets at its owned seam. Downstream
  passes may change unowned route geometry but cannot move, remove, or replace
  a planner-owned source-turn segment or lane transition. Re-seating a planned
  axis derives the opening corner from the source-lane displacement; the corner
  at the other end of that axis belongs to its destination or transition family
  and keeps that family's radius.
- **Related tests**: `tests/test_exit_turn_planner.py`,
  `tests/test_route_plan.py`, and the topology fixtures
  `leftward_up_exit_turn_order.mmd` and
  `terminated_exit_lane_compaction.mmd`.
- **Lifecycle:** invariant - every planned lane, lane transition, route family,
  and turn axis matches the final routed paths, and every assignment is
  consumed exactly once at the render boundary.

## Post-layout routing boundary: convergence planning

- **Purpose**: Give each complete semantic convergence one immutable target-side
  decision before route emission.
- **Helpers**: `_route_edges` calls `build_convergence_plan_execution` after
  exit-turn planning and before dispatch. Canonical inter-section templates
  provide the planned trunk, approaches, joins, and continuation geometry.
- **Precondition**: The semantic route scaffold, exit-turn decisions, station
  offsets, layout coordinates, topology resolution, and compatibility merge
  classification are settled.
- **Postcondition**: Every supported convergence records complete authored and
  resolved membership, its merge and entry bundle, primary trunk and structural
  reason, axis, extent, flanks and terminal caps, stable feeder and lane order,
  opening-turn coordinate, exact joins, handedness, runway, continuation,
  resource conflicts, and endpoint ownership. Unsupported geometry places
  every convergence in the route system on the legacy path. Incomplete
  semantic membership and programming errors fail the invariant.
- **Invariants preserved**: Planning does not move stations, ports, junctions,
  section boxes, or unrelated offsets. Templates consume plan-owned joins and
  covered continuations during dispatch. Coincidence and normalization passes
  may inspect but cannot move or replace plan-owned convergence geometry.
- **Related tests**: `tests/test_convergence_planner.py`,
  `tests/test_merge_branch_trunk_invariant.py`, `tests/test_route_plan.py`, and
  the frozen hash-seed fixtures.
- **Lifecycle:** invariant - every planned feeder retains its exact join, every
  trunk retains its planned axis, flanks and terminal caps, every emitted
  continuation ends at its owned endpoint, and every covered continuation names
  its carrier.

## Post-layout render boundary: envelope settlement

- **Purpose**: Give every grid boundary the width it owes, by translating whole
  grid rows and whole grid columns and nothing else. Two demands say what a
  boundary owes and both are settled by one translation apiece: the width a
  reserved corridor's `RouteReservation` requires, and the clearance a
  `BoundaryClearanceDemand` measures between the boxes facing across it, which a
  render-time box resize can eat with no run involved. A boundary carrying both
  is widened once, by the larger. Being the single owner of the translation is
  the point: no separate row push runs before or behind this stage to make up a
  shortfall it could have paid.
- **Helpers**: `settle_route_envelopes` (`layout/envelope_settlement.py`),
  driven from `_settle_render_geometry` in `render/svg.py`. Each pass
  re-measures live geometry through `realise_reservation`, and re-measures the
  clearance demands the same way and for the same reason -- a figure taken
  before its own earlier translations would be stale.
  `measure_row_gap_clearance` (`layout/phases/bbox.py`) states the row-axis
  clearance demands; the demand vocabulary itself is
  `layout/settlement_demand.py`, held apart from settlement so a layout phase
  can state a demand without importing the routing stack the ledger is built on.
  Rail layouts raise no clearance demand: their row pitch comes from the
  interchange idiom rather than the declared section gap, and widening one of
  their boundaries to that gap turns a flat inter-row run into a staircase --
  a decision change, which `_assert_settlement_decisions_frozen` refuses.
- **Precondition**: `compute_layout` has finished, routing has published the
  reservation ledger, render-time label wrapping has taken its bbox growth, and
  the header-collision reconcile has run. Local station geometry, section bbox
  sizes, port anchors, plan frames, lane orders, and author pins are frozen.
- **Allowed writes**: `Section.bbox_x` / `Section.bbox_y` and the `x` / `y` of
  the stations and ports those sections own, all by one shared per-boundary
  amount. Junctions live in inter-section space and are reproduced by routing.
- **Translation ownership**: A section belongs to the band holding its grid
  start, so a boundary owns every section starting at or beyond it. A section
  straddling the boundary starts above it and stays: carrying it would take its
  upper portion into the gap above and narrow that separation. Both sets are
  recorded on the `SettlementTranslation`, and holding a straddling section is
  sound exactly when it bounds none of the corridors the translation settled --
  if it did, the widening never reached them. That is asserted on the settled
  geometry, together with the monotone claim, by re-measuring every facing pair
  of boxes and every straddling section's corridors.
- **A corridor is not bounded by a box its own runs end inside**: a boundary is
  measured from the section edges facing it, and a section spans it -- occupying
  it rather than bounding it -- when its box crosses the boundary. A run whose
  last leg stops at a station of some box has entered that box just as surely, so
  `RouteReservation.landing_section_ids` names it and
  `_row_region_measurement` / `_column_region_measurement` drop it from both
  sides. Without that, an entry lead-in is charged `INTER_ROW_HEADER_CLEARANCE`
  off the header of the very box it is arriving at, which nothing can satisfy:
  the leg's own endpoint is inside the box, so no widening of the boundary brings
  it into band, and settlement spends real height chasing a demand that cannot
  close. The set is the *intersection* over a reservation's claims, because one
  reservation states one measurement: a box only stops bounding the boundary when
  every run sharing the corridor ends inside it, and a box one of them merely
  passes bounds it for all of them. Region *selection* is unaffected -- it
  asks which boundary a run occupies, not what that boundary has room for -- so
  the corpus raises exactly the same claims on the same 557 reservations.
- **A corridor is bounded by the station its own runs launch from**: a
  pre-routing plan that emits its runs out of a station standing inside the gap
  fixes the length of the opening leg and refuses emitted geometry that shortens
  it, so no widening of the far side brings the run any nearer to that station.
  `RouteReservation.launch_anchors` names it with the runway it owes, and
  `_launch_anchored_measurement` folds it into the region edge on the side of
  the run it stands on: the band the reservation states is then the band the
  plan is free to occupy, and the width the boundary is asked for is the width
  that band needs. Without it the measurement reads the departed box's edge, a
  proxy that sits behind the launch station, and states a band the plan cannot
  reach -- the mirror of the landing case above, and the reason
  `_route_planned_bottom_exit_right_landings` can seat its traverse in the band
  its own reservation realises instead of at a floor the ledger disagrees with.
  The set is the intersection over the reservation's claims, for the same reason
  `landing_section_ids` is. Settlement is unaffected by this blocker: an
  anchor stands on the side a translation holds still, so the ownership lemma
  below still gives the corridor the full widening it asks for. Measured on the
  corpus, two fixtures raise an anchored corridor -- both planned bottom-exit
  fans, whose junction stands 10px into the gap below its box -- and each grows
  its row gap by that 10px; the other 367 renders are byte-identical.
- **The width a boundary is asked for holds every corridor confined with each
  one**: a reservation's `minimum_width` is
  `negative_side_clearance + bundle_width + peer_width + positive_side_clearance`,
  and `peer_width` (`_peer_widths` in `layout/route_reservations.py`) is what the
  corridors sharing the boundary take beside this one. Two corridors crossing one
  boundary compete only when both hold: their runs overlap along it
  (`spans_share_corridor`), and neither one's own measured band can hold them the
  distance apart they need -- a pair whose bands already reach that far is settled
  however the boundary grows, and asks nothing of it. That reach is measured in
  the order the pair is drawn in, since that is the only order any seating may
  produce: the router moves a corridor up to its neighbour's lane and never past
  it, so crediting the pair with the better of the two orderings would report a
  boundary settled that in fact has no seating at all. Where they do compete, the
  demand is the stack in drawn order: each neighbouring pair contributes
  `cotravelling_lane_clearance` (`layout/geometry.py`), which states in one place
  what `_required_channel_clearance` asks of counter-running channels and
  `_overlays_distinct_line` of co-travelling ones -- nothing between two tracks of
  one line running together, `OFFSET_STEP` between distinct co-travelling lines, a
  turn radius between a line and its own return leg, `BUNDLE_TO_BUNDLE_CLEARANCE`
  between counter-running distinct lines -- and never less than the pair is
  already drawn at, so a widening cannot be answered by bringing the pair
  together. Each competing reservation states the same stack, so settlement's
  per-boundary maximum widens the boundary once for all of them and the
  single-sweep argument below is untouched: a larger `minimum_width` is a larger
  capacity deficit and nothing else.
  A claim is not what makes a stroke take room, so the stack holds every leg
  drawn in the boundary and not only the filed ones. The region search asks which
  boundary a leg *crosses*, and a leg that dips into a gap and returns to the row
  it left crosses none -- it is drawn in that gap all the same. Such a leg is
  charged against a boundary whose measured gap its coordinate falls inside and
  whose own claims travel a stretch of corridor it shares, reading that
  boundary's band because it holds no reservation of its own. It is charged as a
  peer rather than filed as a claim because a reservation is a corridor a run may
  be *seated in*, and a leg no boundary crosses has no such corridor to be held
  inside: filing one would state a band a frozen route shape cannot reach, and
  gate its containment against a corridor it never enters. Charging only the
  filed lanes states a boundary wide enough for one stroke where two are drawn,
  and the second is left wherever the narrow gap forced it -- in
  `examples/topologies/merge_around_below_leftmost.mmd`, a merge trunk's return
  leg 14px below a box edge that asks `INTER_ROW_EDGE_CLEARANCE` of it, ungated
  because it carries no claim. Measured on the corpus, 10 fixtures state a peer
  width their boundary lacks and 7 of them render differently for it; each of the
  7 grows in height only, by 11 to 16px.
- **Postcondition**: No boundary still owes what it was measured for. For a
  clearance demand that is one count, re-measured on the settled geometry by
  `_assert_clearance_demands_are_met`: it follows arithmetically from the
  ownership lemma below, and is checked anyway because the lemma is a property of
  two predicates staying in step. For a reservation, every row-gap and
  column-gap reservation *contains* the run
  drawn in it. Containment is three counts, all of which
  `assert_reservations_are_settled` refuses on the strict path: non-negative
  capacity slack (the region is wide enough at all), and non-negative slack on
  each side (the run is drawn inside it, not seated off centre with one side
  absorbing the whole surplus and the other overrun). The two counts read
  different evidence, and must. Capacity is a property of the reservation and the
  settled envelopes, so it is measured by re-realising the reservation against the
  ledger settlement was handed. Where in the region the run *sits* is only
  knowable from the emitted polylines: the published ledger records the demand --
  frozen claims projected through the translations -- so its occupied interval
  states where the first pass observed the run, not where the settled re-route
  drew it, and a boundary widened so that the re-route can move into the new room
  leaves that interval untouched. The side slacks are therefore measured by
  `drawn_corridor_containment` on the polylines the renderer is about to draw,
  through each claim's own `(path_rank, segment_rank .. segment_end_rank + 1)`
  point range; `_settle_render_geometry` builds them once and hands the same list
  to the guard and to the renderer. That the frozen plan's ranks still name the
  re-routed geometry's points is what `_assert_settlement_decisions_frozen`
  already guarantees: it compares one signature entry per point pair, in route
  order, so equal fingerprints mean equal route order and equal point counts, and
  `apply_route_offsets` maps points one for one. Measured on the corpus, reading
  the drawn coordinate is the difference between 37 fixtures refused on the strict
  path and 11.
  Capacity holds unconditionally, for every arrangement an author can express, by
  the **ownership lemma**: `_row_region_measurement` splits the sections beside
  boundary `b` into an upper set `{row_end(s) <= b-1}` and a lower set
  `{grid_row(s) >= b}`, and `_translation_ownership(b)` moves exactly
  `{grid_row(s) >= b}` and holds everything else. Those are the same inequality,
  so a translation raises the corridor's `end` by its full amount and leaves
  `start` fixed: the corridor widens by exactly what was asked. Columns are the
  same statement on `grid_col`. The premises are that `amount = ceil(deficit /
  SETTLEMENT_QUANTUM) * SETTLEMENT_QUANTUM >= deficit` (`quantised_allocation`)
  with translations unbounded above; that no directive pins a canvas coordinate
  or a maximum separation (`grid:` fixes grid indices,
  `section_x_gap`/`section_y_gap` are floors, `width`/`height` size the viewport,
  and `legend:` is not a corridor
  blocker); that row and column offsets are cumulative sums over ascending
  index, so "A above B" implies `A.grid_row <= B.grid_row`; and that section
  sizes are frozen between settlement and the guard, `shift_section` writing only
  origins. The same inequality covers a `BoundaryClearanceDemand`: every box its
  shortfall is measured *from* ends at row `b-1` or above and every box it is
  measured *to* starts at `b` or beyond, including the bypass-span and
  row-envelope variants, whose deeper edge belongs to a section in the upper row.
  Consequently the strict deficit path is a backstop against ledger or
  ownership drift rather than an authoring outcome: an "infeasible pinned
  arrangement" is not a state this model admits. The guard stays because the
  lemma is a property of two predicates staying in step, which a future edit
  could break. `tests/test_envelope_settlement.py` measures the lemma directly
  over one fixture per pin class -- explicit grid, row span, column span,
  inferred span, fold-driven rows, and all four flow directions -- asserting that
  a boundary's negative blockers are disjoint from the sections its translation
  moves, its positive blockers are contained in them, and no blocker straddles
  the boundary. A boundary that every relevant section spans
  across has no side to measure, so it is never selected as a corridor's region
  in the first place -- the measurement bounds a boundary by the sections lying
  wholly on each side of it, and raises otherwise. Every convergence system left
  on the compatibility path carries a `CompatibilityOwnership` record measured by
  `attribute_compatibility_systems` on the plan the map draws: the tightest
  capacity slack across the corridors that system reserved, the
  `ConvergenceConflict` its planner recorded (kind, axis, both run coordinates,
  and the distance between them), and the `SettlementReach` verdict deciding
  whether any offset this stage owns changes that distance. Two runs one
  translated band carries together keep their distance whatever settlement
  does; runs in different bands only ever get further apart, which is the wrong
  direction for a conflict whose relief is one shared channel. The owner comes
  from `ConvergenceConflictKind`, so it follows from the check that fired rather
  than from re-reading its wording.
- **Origin-independence**: The width a boundary is widened by is a function of
  its deficit and nothing else, so one arrangement described at two canvas
  origins allocates identically. This is the quantisation lemma's other half,
  and neither half is sufficient alone. `amount >= deficit` on its own permits
  an allocation that follows the coordinates a gap happens to be measured
  between: a gap is a difference of two box edges, binary64 subtraction of two
  coordinates carrying decimal fractions leaves an error set by the magnitude of
  the operands rather than by the distance between them, and `ceil` amplifies
  whatever it is handed into a whole `SETTLEMENT_QUANTUM`. The two halves hold
  together because the resolution belongs to the measurement rather than to the
  ceiling: `measured_distance` (`layout/route_reservations.py`) states every
  ledger width and every containment slack at `COORD_GROUP_DIGITS_FINE`, two
  orders of magnitude finer than `COORD_TOLERANCE_FINE`, so the ceiling
  allocates no less than the deficit it is handed and the deficit it is handed
  is the one the geometry states. A `RealisedRouteReservation`'s own two side
  slacks are raw subtractions, because every consumer reads them against
  `COORD_TOLERANCE`, a band 1e13 times that error: the resolution is owed where
  a reader is finer than the tolerance, which is the ceiling here and the sign
  test in `drawn_corridor_containment`. Measured on the corpus, 25 fixtures both
  settle and are translated rigidly by every offset tried (0.1, 0.3, 1/3, 7.7,
  1000.1, -0.1, -7.7 -- none a whole pixel, none representable in binary64). Ten
  of those 25 allocate a different width at some origin when the deficit is a
  bare subtraction, and none does when it is a `measured_distance`. Two of the
  ten are `examples/differentialabundance_default.mmd` and
  `tests/fixtures/da_pipeline.mmd`, whose own origin is the one that reads long:
  their `functional`/`plots` deficit of 14.0 measures as `14.000000000000057`,
  which the ceiling answers with 15px, and each map is 801px tall against the
  802px a bare subtraction gives. Those are the only two renders in the corpus
  the two arithmetics disagree on; the other 367 are byte-identical, and both
  deltas are a rigid 1px translation of the sections below the boundary.
  `test_the_allocation_is_a_function_of_the_deficit_not_the_canvas_origin` holds
  the property over both axes. Four fixtures are outside its reach because a
  uniform translation makes their router draw a different shape rather than the
  same one moved (`convergence_stacked_sink`, `same_line_fan_distinct_descent`
  and `seed_15` at 1/3, `seed_77` at 7.7); the test establishes rigidity before
  it compares, so it measures the quantiser and not that.
- **Invariants preserved**: No row or column separation decreases. Section
  sizes, a station's position within its section, plan-owned frames, lane
  order, port sides, and author-pinned grid relationships are unchanged.
- **Out of scope**: Canvas-side corridors, whose far boundary is the canvas
  edge rather than a grid neighbour; closing one grows a margin, which no row
  or column offset owns. They are gated separately, by
  `assert_canvas_corridors_hold_their_claims`, which runs once the render has
  sized its canvas -- the first point at which the number a canvas claim is
  measured against exists, and the reason the settlement guard could never
  measure one. A run is filed against a canvas side only when it lies beyond the
  extreme of every placed section, so the margin it is measured within is the one
  it occupies, and its clearance on that side is `CANVAS_EDGE_CLEARANCE`: the
  stroke's half-width plus a direction chevron, which is what is drawn there,
  scaled through `canvas_edge_clearance()` because `stroke_scale` multiplies the
  stroke but not the chevron's arms. A turn radius is not, because an arc beside
  the canvas is inscribed inboard of the centreline. The guard gates on that
  margin -- `canvas_edge_slack`, the room between the ink and the edge -- and on
  total capacity, which are different claims: a corridor can hold every pixel it
  reserved and bank all of it on the side facing content, leaving its stroke and
  chevron drawn through the margin and clipped by the viewport. Across the
  `examples` and `tests/fixtures` corpus, 144 canvas corridors are realised and
  none is short of its canvas margin or of total capacity, measured either from
  the published claim interval or from the drawn polylines. A canvas corridor's
  *content*-facing side is not gated here, because no growth of the canvas moves
  a section box edge or header badge. 28 of those 144 keep less than the
  clearance they claim from that content when their drawn ink is measured (29 from
  the published claim interval), 18 of them an over-top band within the band
  `INTER_ROW_HEADER_CLEARANCE` reserves for a header badge. That clearance is a
  longitudinally blind envelope, and it is charged at routing time, before any
  caption has a position: it assumes the badge protrudes above `bbox_y` as
  `section_header_top` states, and where a section draws its caption below or
  beside its box, or at an x the band does not reach, the reserved band holds no
  badge. The reservation cannot be narrowed to the caption drawn, because the
  caption is chosen from the routed polylines and the routes are placed against
  this clearance (see *A caption's reserved band is the one on the side it took*);
  reserving the band unconditionally is what keeps the caption's default position
  always available, and its cost is slack rather than a defect. Measured
  over all 28, the count of fixtures in which route ink enters a drawn badge's own
  box is zero, so gating this side today would refuse renders for clearance from
  something not drawn there. Each is instead published as an attributed
  `reservation-deficit` record on the plan for the box-edge and header guards to
  own, and tightening the claim to the badge actually drawn is #1693. That issue
  is not closed by localising the claim alone: `INTER_ROW_HEADER_CLEARANCE` is
  `SECTION_HEADER_PROTRUSION + INTER_ROW_EDGE_CLEARANCE`, and charging only the
  edge clearance where no badge is drawn under the run still leaves 10 of the 18
  short, because they keep 22px of the 26px edge clearance rather than the full
  26 that 8 of them keep. The remaining 10 deficits are 2 to 6px short of
  `EDGE_TO_BUNDLE_CLEARANCE` or `INTER_ROW_EDGE_CLEARANCE` against a box edge with
  no badge involved. Gating the content side therefore needs those corridors moved
  off the box edges they hug, not a narrower claim.
- **Transactional**: The pre-settlement coordinates are restored before any
  exception propagates, so a failure leaves the graph as settlement found it.
  The reservation ledger is read-only here.
- **Idempotence**: A second pass over settled geometry finds no positive
  deficit of either kind and writes nothing, so running settlement twice is an
  exact geometry no-op.
- **Termination**: Settlement runs once, against one ledger. That pass visits
  each adjacent-index boundary once in ascending order; translating everything
  from boundary `b` onward widens `b` by exactly that amount, leaves earlier
  boundaries' blockers stationary, and moves later boundaries' blockers
  together, so boundaries do not interfere and the pass is finite in the number
  of boundaries. It deliberately does not iterate: re-routing the settled
  geometry publishes a different ledger (corridors appear, vanish, and change
  their required width), so settling against successive ledgers would be a
  fixpoint search over a moving constraint set with no convergence argument.
  The plan the closing guard measures is therefore the frozen ledger projected
  through the translations, not the re-routed one. A demand only the re-routed
  geometry reveals is consequently not chased, and `attach_reroute_ledger_delta`
  records it as a non-blocking plan diagnostic so it is named rather than
  invisible. It compares each corridor's description together with the width it
  asks for, since a boundary whose corridor survives at a different
  `minimum_width` is one the translations were sized wrongly for. Measured on
  the corpus, its gap demand names no corridor either ledger lacks, and 21
  corridors across 11 fixtures whose required width the re-route states lower
  than settlement was sized for. None is stated higher, so the frozen ledger
  never under-sizes a boundary the render draws.
- **Consumed by**: the re-route. `_settle_render_geometry` hands the
  pre-settlement ledger back to `observe_route_edges_centred` whenever it holds
  any reservation, which builds `ReservedCorridors`
  (`layout/routing/reserved_bands.py`) by re-measuring each row-gap and
  column-gap reservation on the settled geometry. One axis-neutral measurement
  serves both, keyed by the higher grid index the boundary separates (the lower
  row, the right column). `_center_inter_row_channel` and
  `centre_inter_column_channel` place a claimed channel inside that band rather
  than deriving one from the row or column edges, and a published band always
  holds a channel, so a claimed corridor cannot take the narrow-gap fallback.
  Where a handler or normalisation pass sizes a channel from the boxes it has to
  hand -- `bypass_bottom_y`'s trunk depth, the L-shape and wrap clearance
  floors, `_clamp_inter_row_band_top`'s stack limit -- that proxy is applied
  through the band (`held_in_reserved_band`) so the reservation's answer wins
  where the two disagree. A boundary whose claims intersect to nothing, and
  every gap the ledger never reached, keep the row- or column-edge derivation.
- **A band bounds a corridor, it does not assign it a lane**: every claim
  crossing one boundary realises the same band, so corridors placed in it
  independently cannot see each other and two can settle less than one
  `OFFSET_STEP` apart, which draws two distinct lines as a single two-tone
  stripe. `_separate_fused_cotravelling_runs` closes the pass chain by
  restoring the step across every corridor at once, moving a whole track (each
  run of one line on one lane through one corridor) so a fused fan-out cannot
  be split, and never moving a track a plan owns.
  `check_no_fused_cotravelling_lines` is its postcondition on the render
  chokepoint.
- **Containment is closed on the drawn geometry, not in the handlers**:
  `ReservedCorridors` answers "what is clear at this boundary", which is the
  intersection of every claim crossing it. That cannot separate two corridors
  crossing one boundary in opposite directions -- their intersection is narrower
  than either, sometimes a single coordinate -- so a pass allocating several
  corridors across one boundary at once (`_materialize_gap_slots`) keeps the raw
  gap instead. Eight post-passes therefore position channels without reading the
  ledger: `_separate_opposing_inter_row_trunks`, `_materialize_trunk_slots`,
  `_spread_diagonal_bundles`, `_bundle_divergent_distinct_traverses`,
  `_coincide_fanout_opening_descents`, `_stagger_convergent_distinct_lines`,
  `_coincide_same_line_tracks`, `_materialize_gap_slots`.
  `_hold_runs_in_corridor_clearance` closes the difference last instead, on the
  routed geometry. **A leg the ledger claims is held inside its own claim's
  realised band**, read through `ReservedCorridors.for_segment` by the claim's
  `(source, target, line_id, segment_rank)` identity, which is the same band the
  closing guard scores it against. Consuming the reservation rather than
  re-deriving one is the whole point of having allocated it: settlement widened
  that boundary for this corridor over the corridor's own declared span, and a
  band read back off live geometry can only ever confirm wherever the leg already
  sits. A leg no claim names has no reservation to consume and keeps the gap
  measurement (`gap_corridor_clearance_band`, which states the reservation's
  arithmetic against live geometry), which is what the first routing pass -- the
  one that publishes the ledger and has none to read -- runs on.
  Bundles move rigidly and only into the space their gap-mates leave them, so no
  move fuses two lines onto one stroke. How much room a pair needs is
  `cotravelling_lane_clearance`, the same rule the ledger sizes boundaries by, so
  a corridor is never denied a coordinate the ledger allocated it on a separation
  the ledger did not charge for. A bundle every shift is denied retries with the
  peers denying it as one rigid group: two corridors owed one boundary between
  them are seated by the same widening and neither can reach it alone, and a
  rigid move leaves every separation inside the group exactly as drawn.
  Measured on the corpus, every one of the 1007 claims carried by 557 realised
  gap reservations is drawn inside the band its own reservation realises, and
  `tests/test_reserved_claim_consumption.py` holds the whole corpus to that with
  no exceptions. All but two are exact: the pair are the `hic_reads` lane turning up
  into `scaffolding` in `examples/genomeassembly.mmd` and in its organellar twin,
  each drawn 1.00px past its inter-column channel's positive edge, which is
  inside the `COORD_TOLERANCE` the bound allows. Their channel's lowest lane is a
  planned exit turn's descent, standing 4px above the band floor, and the stack
  seated from it takes 15px of the band's 18. That shortfall is a position rather
  than a width -- the reservation's own `minimum_width` is met with 14px to spare
  -- and settlement cannot pay it in any case: `SETTLEMENT_QUANTUM` is
  `COORD_TOLERANCE`, `_settle_axis` acts only above it, and
  `ReservationCoordinateTranslation` refuses an amount that small, so the least
  translation this stage can express is 2px and a 1px deficit is below the
  resolution the ledger works at.
  The last claims to come into band were the merge trunks of
  `tests/fixtures/regressions/cross_column_perp_entry_overflow.mmd`, and the
  planner owns all three of them. A merge feeding a TOP or BOTTOM entry port is
  seated on the vertical lead-in that port receives (`_position_merge_junction`),
  which puts its six feeders in the row corridor they claim and puts the merged
  trunk in the column the port's own crossing gives that line. Three things have
  to hold together for a plan to state that column.
  **The drop lands where its siblings land.** The junction-to-port hop is seated
  on `_perp_entry_landing_x` -- the port-crossing X the intra-section departure
  leaves from and every bundled feeder lands on -- and ends on the port's own
  edge. Carrying the lane offset along the axis the hop travels instead runs it 4
  and 8px past the boundary for `tumor_only` and `somatic`, on a column no
  sibling stands in; `_shared_terminal_axis` then finds no feeder terminating
  where the hop does, and the plan falls back to `OUTGOING_CONTINUATION` with its
  trunk disagreeing with its own landings.
  **A corridor shared with an unowned member is not contested.** The unowned
  member is `annotation__exit_right_3 -> reporting__entry_top_7`: the same line,
  landing on the plan's own entry port, which is the pair
  `_convergent_port_groups` groups and `_coincide_same_line_tracks` fuses onto
  one column. `UNOWNED_MEMBER_CORRIDOR` measures the planned trunk against a
  `_trial_route` taken before that fusion, so `_fuses_onto_trunk` exempts exactly
  the run the fusion seats on the trunk -- the route's own final approach into
  the plan's port, within `EDGE_TO_BUNDLE_CLEARANCE` of it, landing where it
  lands -- and nothing wider. A conflict anywhere else along the same route is a
  second corridor the fusion never touches, which is why the check still fires
  for `examples/genomic_pipeline.mmd`,
  `tests/fixtures/regressions/stacked_collector_fanin.mmd` and
  `examples/topologies/merge_right_entry.mmd`.
  **A plan claims the segments its axis describes, and no more.** A trunk axis
  collapses its flanks onto its own coordinate when the trunk turns straight into
  the port, and `_trunk_segment_ranks` matched those zero-length flanks as runs,
  which claims every leg passing through the corner they state -- here the
  horizontal runway -- and through `convergence_owns_segment_boundary` the
  feeder's opening descent before it. That took the descent out of
  `_divergent_source_groups`, the pass that fuses each line's descents at one
  source onto the column its bundle occupies there, and the feeder stood one lane
  off its own colour: three doubled strokes over 40-60px, each overlapping a
  neighbouring line's lane. The corner itself stays owned by the boundary rule
  around the trunk's own run, so only coordinates the axis never stated are
  handed back. For the same reason the landing states **no**
  opening turn where a bundle outside the convergence seats its column
  (`_bundled_sibling_owns_opening_column`): `_divergent_source_groups` draws its
  reference from the bundled members, and a lone feeder's own handler column is
  not the plan's to freeze.
  Together those make all three convergences `PLANNED` with
  `SHARED_TERMINAL_APPROACH` trunks on x = 554, 558 and 562, each landing on the
  port at y = 1617.4, and take the corpus from 30 compatibility convergences to
  27 and from 22 planned to 25. One render moves, this fixture's, and it moves by
  dropping the two overshoot stubs and nothing else: the SVG differs, the raster
  is pixel-identical, and the canvas is unchanged at 1325x1781.
  #1660 admits a compatibility disposition only against evidence, and that
  evidence is measured on the settled map: a conflict's two runs are the same
  column, 0.00px apart in one translated column band
  (`SettlementReach.SEPARATION_FIXED`), so no offset this stage owns separates
  them, and the corpus publishes no `WITHIN_REACH` compatibility system at all.
  That last figure is weaker than it reads and should not be quoted without this
  qualification: `_settlement_reach` returns `WITHIN_REACH` only for a conflict
  whose relief is not `ConflictRelief.SHARED_CHANNEL`, and 9 of the corpus's 12
  compatibility systems carry `SHARED_CHANNEL` relief, so for those the verdict
  follows from the conflict kind rather than from geometry. What the measurement
  does establish, on the 3 that could answer either way, is that a row or column
  offset does not change the distance between the two drawn coordinates. It does
  **not** establish that a wider boundary would still leave the planner unable to
  allocate both members, which is the question #1657's exit criteria asked.
  `capacity_probe.probe_settlement_capacity` answers that one directly, and its
  answer is that **#1657's exit criteria are met for all 12**: every one of them
  stays on the compatibility path under every capacity the probe grants, up to
  sixteen times what one competing pair of runs costs, which is the evidence the
  criteria ask for. The probe copies the settled graph, translates whole rows and
  columns to widen the boundaries a system is measured at, re-derives the
  coordinates that follow from where those sections then sit, re-runs convergence
  planning on the copy, and reads the disposition.
  **12 is the live population**, carried by 12 fixtures and collapsing to 9
  distinct system-id strings because sibling fixtures share connector names.
  `COMPATIBILITY_CORPUS` in `tests/test_capacity_probe.py` carries **14 rows**,
  because a fixture whose systems the planner came to own is retained as a
  control rather than dropped: its row asserts that it publishes no compatibility
  system at all, which fails the moment one reappears. The counts must not be
  conflated -- 14 is a set of probed identities, 12 is how many of them are on the
  compatibility path. `cross_column_perp_entry_overflow` is one of the two
  controls; the other is the measurement the shared-channel decision below came
  from.
  `merge_around_below_leftmost` was planned at 22.5px of extra claimed-boundary
  capacity and at every larger capacity granted, which established that what held
  it was an allocation -- and equally that the allocation was buying a decision,
  since the separation the planner needed grew at half the widening rate.
  The probe is not on the render path: it plans the map fourteen more times
  per compatibility system, so it is diagnostic machinery that
  `tests/test_capacity_probe.py` runs and no render pays for. Its result is only
  meaningful against a reproduced baseline, so each system is first re-planned
  untouched and one whose control does not come back on the compatibility path is
  refused rather than measured; and its positive answer is reachable by
  construction, which
  `test_a_starved_system_is_handed_back_the_capacity_that_starved_it` shows by
  taking 10px out of `fan_in_merge`'s reserved boundaries until the planner drops
  it onto compatibility and watching the probe return 10.75px.
  The re-derivation is what makes a grant a counterfactual about capacity rather
  than about the probe. A junction's coordinates are a function of the ports it
  joins, and `_settle_render_geometry` derives them from the sections before
  every re-route, so a grant that translates sections and leaves the junctions
  where they were hands the planner a map the pipeline cannot draw. On that map
  the two arms of one fan stop meeting at a shared turn coordinate -- not because
  the boundary got wider but because the junction that sets the coordinate was
  left behind -- and the planner falls silent about a pair it no longer
  recognises as sharing a channel. A grant that skips it reads as five further
  systems reaching allocation (`merge_bottom_row_bypass` and
  `merge_feeder_shared_channel_gap` from 19.5px, `ambiguous_exit_continuation`
  from 256px, `merge_right_entry` from 576px, and
  `merge_trunk_out_of_range_section` planned at 656px but not above it), none of
  which the settled pipeline reproduces: handing those systems' own claimed
  boundaries 39, 78 and 156px through `settle_route_envelopes` itself leaves the
  re-route's planner on compatibility with its conflict measured at the same
  0.00px separation every time.
  That is also why a **conditional demand** -- the planner publishing what it
  would have needed when it declines for space, so settlement allocates against
  it in the one pass it makes -- closes nothing for the two
  `OPPOSING_OPENING_CHANNEL` systems. Their conflict is two arms of one fan
  turning on a single coordinate and opening opposite ways, and the coordinate
  belongs to the junction both arms leave. Every offset this stage owns carries
  that junction with the sections it joins, so both arms move together and the
  0.00px between them is invariant under the whole space of allocations this
  stage can make. There is no amount to publish:
  `test_capacity_carries_a_shared_opening_turn_instead_of_opening_it` reads the
  separation back at every capacity on the ladder and finds it unchanged. The
  demand would have to be for a *decision* -- one shared channel, per
  `ConflictRelief.SHARED_CHANNEL` -- which is the emission owner's
  (`plan-driven opposing-opening emission (#1658)`), not a distance settlement
  could allocate.
  Where that decision belongs to the convergence planner it is made rather than
  declined, and this stage's part in it is to charge for the result and nothing
  more. `_settle_shared_trunk_channels` lanes the runs of one route system's
  trunks. Each convergence plan reads its trunk geometry off a trial route taken
  with no knowledge of its siblings, so two plans of one system whose trunks take
  the same channel derive the same coordinate and each believes the channel is its
  own; the system assigns the lanes, by `cotravelling_lane_clearance` -- a full
  turn radius between a line and its own return leg, and nothing between two runs
  going the same way, which stay one fused stroke. Both channels a trunk shares
  are laned by that rule: the one its central run travels, and the one its flanks
  turn out into.
  Three properties of that decision matter here.
  It publishes **no demand of its own**. A lane is a drawn stroke, so the boundary
  carrying it is charged for it exactly as every other stroke is: the second lane
  is a run in the row gap, `_peer_widths` reads it, and `minimum_width` states the
  pair. On `merge_around_below_leftmost` that is 26 negative + 0 bundle + 11 peer
  + 52 positive = 89px against a 78px gap, realised at required 89.00 / available
  89.00 with the trunks on 196 and 207. `BoundaryClearanceDemand` is for a
  boundary owed clearance by something that is *not* a drawn run, so it is the
  wrong vocabulary for a lane and is not used.
  It is taken on the **first** routing pass, so this stage realises a demand
  against a plan that already exists. `_assert_settlement_decisions_frozen` is
  therefore unmodified and holds: disposition, membership, lane order and frame
  are identical either side of the sweep. Lanes are measured from the trunk that
  arrived first rather than from a boundary edge, so widening the boundary moves
  neither of them, which is what makes the separation invariant under allocation
  instead of growing at half the widening rate.
  It is **cheaper than the allocation it replaces**. Charging the pair as a peer
  while the planner still declined asked 90px of that boundary; laning it asks
  89px and plans the system, so the settled map is 1px shorter than the one the
  compatibility path drew.
  Two corridors confined at one boundary are not a source of
  residue either: `peer_width`
  states the room they take together, so settlement widens the boundary for both.
  Nor is it longitudinal blindness in the band's blockers, which was measured
  and ruled out: of the 14 out-of-band claims that measurement covered, 13 have
  every blocking section on their violated side overlapping or abutting the
  drawn leg, and the one that does not
  (`fan_bypass_shared_band`, whose two 148px-away blockers bound the side it
  holds) has its violated edge set by a section abutting the run. Selecting
  blockers by longitudinal overlap alone was measured over the corpus and is a
  regression, not a fix: it drops the box a corridor's own elbow turns beside
  (16 to 26px past the run's end in 4 of those 14), whose removal widens the band
  enough that the re-route re-centres the run in it, changing 8 renders and
  flipping one vertical leg's direction, which
  `_assert_settlement_decisions_frozen` refuses outright; out-of-band claims rise
  from 21 to 32 as region selection reclassifies corridors onto other boundaries.
- **Related tests**: `tests/test_envelope_settlement.py`,
  `tests/test_reserved_corridor_placement.py`, and
  `assert_reservations_are_settled` in `layout/phases/guards.py`.
- **Lifecycle:** invariant - the settled geometry satisfies every reservation
  settlement owns, and re-running it changes nothing.

### A port travels with the box edge it is anchored to

Seating a label grows its section box outward (`_clamp_label_to_section`,
`_place_tb_label`, `_grow_section_for_box`), and that growth is render-time: the
wrapped text and the marker positions routing centres are not known until the
render path has both. A port's side names the edge it is pinned to, so an edge
that moves without it leaves the port inside its own box, its inbound run
crossing the drawn border and traversing the interior to reach it.
`carry_ports_with_section_edges` (`phases/ports.py`) therefore moves every port
already on a moved edge by that edge's displacement, at the step that moves it,
and `_settle_render_geometry` re-observes the routes so each still terminates on
its port.

The re-observation is one step, not a fixpoint. Routing centres a station marker
on its flat run, so lengthening that run by moving the port moves the marker,
hence its label, hence the edge, hence the port again. That relation is a
contraction on some topologies and has unit gain on others
(`top_entry_left_neighbour` walks its `producer` box right by 6px per round
without settling, `bypass_fan_in_outer_slot` halves), so iterating it is not
bounded. A label pass with no re-observation behind it instead gives its growth
back on the anchored edges (`hold_port_anchored_edges`), leaving the port where
the drawn runs land and the label seated within its bbox margin. That is also
what keeps growth out of a settled corridor: settlement measured its boundaries
against these edges, so a post-settlement pass that pushed one further out would
spend clearance already promised -- `bypass_fan_in_outer_slot`'s
inter-column-channel has 40.5px of the 40px it reserved, and 5.6px of unheld
growth takes it below.

Across `examples/` and `tests/fixtures/`, 38 fixtures grow a port-bearing edge
at render time and 4 do so on a pass with nothing behind it.

### A caption's reserved band is the one on the side it took

`SECTION_HEADER_PROTRUSION` above a box top is the band the layout reserves for
that box's caption, and `section_header_top` states it. Routing is charged against
it (`section_header_safe_cap`, `INTER_ROW_HEADER_CLEARANCE`), and that charge is
unconditional because it is made before any caption has a position: the caption is
picked from the routed polylines it has to avoid, so a reservation that followed
the caption would be a route-place-route fixpoint. What the reservation buys is
that the caption's default top-left position is always available, and what it
costs where the caption goes elsewhere is slack.

A fixed band above `bbox_y` is therefore the wrong thing to hold a *drawn* caption
to, in both directions. It is too small: a wrapped title grows away from the box
until it reaches the map title or the box above (`_max_lines_upward`), and even a
single line reaches `SECTION_NUM_CIRCLE_R_LARGE + SECTION_NUM_Y_OFFSET +
SECTION_LABEL_HALF_HEIGHT_RATIO * font` past the edge, which passes the 26px
reservation once the section label font passes 13.75px - `sarek_metro` at
`font_scale` 1.3 draws 31.64px and `near_edge_exit_corner` 27.80px. And it is in
the wrong place: a caption below or beside its box is not in that band at all,
while the gap it does occupy is the one that has to hold it.

`header_band_room` (`render/section_header.py`) therefore states the band from the
placement, on whichever side the caption hangs off: down to whatever stands above
the box and never less than the default position's own reach; up from the box
bottom to the next row's top less the `SECTION_HEADER_PROTRUSION` that box
reserves for its own badge; or out to the section beside. `header_band_protrusion`
states how far the ink reaches into it, the resolver only offers a side whose room
holds the caption, and `check_section_headers_hold_the_reserved_band` re-reads both
off the drawn placements and refuses the render with `SectionHeaderBandError`
otherwise. Across the corpus every one of the 1224 drawn captions is inside the
band its own side states, against 39 outside a fixed band above `bbox_y` (36 below
the box, 3 rotated). What that statement does not buy is an *empty* band: 20 of the
1224 have a neighbour's caption ink or reserved band inside their own claimed
strip, every one of them an `above` caption whose title is too wide for its box and
overhangs into the next column - the `height_capped` case
`check_section_headers_fit_box_width` exempts, and a box-width problem rather than
a band one.

Stating the band per side is what lets a caption take the clear side. An
uncontested default position wins outright; once a route crosses it, the band slot
and the bottom edge are ranked by the room each keeps from route ink, with a
rotated side header a lower tier below both (see the module docstring).

Ranking any clear slot along the band ahead of every position leaving it was
measured over the corpus and is the wrong rule: the leftmost clear shift keeps
exactly `SECTION_HEADER_ROUTE_PAD` from the stroke it stepped past by
construction, so it held all 20 of the captions it applied to within 4.00px of a
descending stroke while the edge those captions declined stood 22 to 60px clear.
Under the clearance ranking 18 of the 20 take a bottom edge at 42.0 to 60.4px and
2 take a roomier slot in the band at 19.3px and 50.0px, which is the band winning
where it is genuinely open. Those 20 captions are the whole of the ranking's
effect on the corpus: 19 fixtures render differently and one of them grows,
`cross_column_perp_entry_overflow` by 8px of height (+0.45%) for the canvas to
hold a bottom-row caption. The canvas-corridor figures above are unmoved at
144 / 29 / 18, since no route or box edge moves.

### Tier-A layout guards read the settled geometry

`assert_render_layout_invariants` runs once per render, next to
`assert_render_header_clearance`, on the routes and offsets the renderer is
handed. Measured on the first routing pass instead, it certifies geometry that
label placement, the header reconcile and settlement then move: 27 of the 356
rendering fixtures failed it on their settled geometry while passing it there,
23 on `_guard_inter_section_route_clears_own_section_interior` and 11 on
`_guard_ports_on_boundaries`. 26 of the 27 were a port its box had grown away
from rather than the wrapped bundle those guards were written for; the 27th,
`cross_column_perp_entry_overflow`, already refuses on the first pass. No
fixture fails on the first pass and passes on the settled routes, so the single
settled evaluation loses no coverage and the earlier call was redundant rather
than complementary.

### Row, bbox-top and canvas responsibilities, and which of them settlement owns

Settlement translates whole rows and whole columns. Several older passes also
move something, so each was examined for whether *its* move is the one
settlement makes. Bypassing a whole pass cannot answer that: it deletes two
different responsibilities at once and only shows that the pair matters. So
each pass is split first, and only the second half is a candidate:

- a **local** responsibility -- measuring a shortfall and resizing or
  repositioning one box against its own content;
- a **global** responsibility -- translating sections the pass does not own, to
  absorb what the local half revealed.

Settlement's translation is exactly `_apply_translation` over
`_translation_ownership(b)`, which moves `{grid_row(s) >= b}` and holds the
rest. A pass is a migration candidate when its global half applies one amount
to that same set, in the direction that widens the boundary.

| Pass | Local half | Global half | Candidate |
| --- | --- | --- | --- |
| `_shrink_bboxes_to_content_bottom` | `bbox.py:693`, `:704` own `bbox_h`; `:694`, `:705` own BOTTOM ports. Reads row-mate bottoms as a floor (`_row_mate_bottoms`) without writing them | none | no |
| `_fit_bboxes_to_content_top` | `bbox.py:1168`, `:1176` `move_section_bbox_min_edge` on the section being fitted. `_section_fit_top` *reads* the row above as a ceiling and writes nothing | none | no |
| Stage 4.7 `_top_align_row_sections` | `row_align.py:535-540`: a per-section `delta = section.bbox_y - min_top`, applied by `shift_section` to that section's own body | none -- no set-wide amount exists | no |
| Stage 5.3 / 6.9 `_top_align_row_bboxes_only` | `row_align.py:573` per-member `grow_section_bbox_to_anchor`, own `bbox_y` / `bbox_h` and own TOP ports | none | no |
| Stage 6.6 / 6.8 `_reanchor_off_track_to_consumer` | `off_track.py:1508` own off-track station cross-axis; `:1839`, `:1840`, `:1867`, `:1868` grow the passed section's own edges | none | no |
| Stage 6.16 `_align_entry_ports` + `_position_junctions` | `_set_port_y` / `_set_port_x` own entry port; `ports.py:396`, `:501`, `:1812` `_expand_bbox_for_y` on its own box; `:494-498` a rigid slide of its own section in `_mirror_entry_section_to_seam`; `:1744`, `:1755` the *feeding* section's exit-port Y when `_clamp_tb_entry_port` bites. `junctions.py:173`-`:226` writes junctions and nothing else | none | no |
| `_tighten_lower_rows_after_shrink` | `bbox.py:1389` the slack each lower row may rise by | `bbox.py:1401` `_shift_rows_from(graph, r, -slack)` | no -- wrong direction |
| `_reserve_row_gap_for_top_padding` | `bbox.py:1123` the padding band a blocked section is short of, via `_section_fit_top` / `_section_content_hug_top` | `bbox.py:1126` `_shift_rows_from(graph, r, deficit)` | no -- ordered before a resize |
| `push_lower_rows_after_bbox_grow` | `measure_row_gap_clearance` (`bbox.py:179`) | `bbox.py:329` `_shift_rows_from(graph, boundary, deficit)` | **yes -- migrated** |
| `_shift_graph_into_canvas` (`phases/canvas.py:311`) | `_canvas_top_shortfall` | `_translate_graph_y`: every section, uniformly | no -- not compensation |
| `_snap_canvas_y_to_grid` (`phases/grid_snap.py:271`) | the dominant `station.y % y_spacing` residue | `_translate_graph_y`: every section, uniformly | no -- not compensation |

Measured over all 329 fixtures under `examples/` and `tests/fixtures/`, each of
the eight local passes was wrapped at its engine call site and every section box
and station coordinate snapshotted across the call. None writes another
section's origin, and no call applies one delta to a set: where several boxes
did move by an equal amount, they had entered the call sharing the coordinate
the amount was derived from, or the move was an edge grow with a compensating
`bbox_h`. Stage 4.7 is the boundary case -- it is the one local pass that slides
a whole section body, 28 calls of 427 -- and it stays local because the amount
is derived per section from that section's own top, confined to one contiguous
column group within one grid row.

#### The one that migrated

`push_lower_rows_after_bbox_grow` restores the clearance a row boundary is owed
after something grew a box into it. Its demand is now
`BoundaryClearanceDemand`, settlement's second demand alongside the reservation
ledger, and the render path hands settlement the measurement instead of pushing
rows itself. What made this migratable, where the sentence this section used to
carry said it was not:

- The demand is not a corridor, and a `RouteReservation` cannot express it. That
  type requires authored `connector_ids`, non-empty `claimant_member_ids` and
  `claims`, a `RouteReservationClaim` per claim naming a real polyline
  point-pair range with a positive travel interval, `lanes` partitioning those
  claims, a `route_family_ids`, and a `direction` along the boundary -- every one
  of which is a property of a *drawn run*. A boundary owed padding has no run.
  So the vocabulary the demand needed was a second demand type, not a synthetic
  reservation.
- Nothing else about it was out of reach. Its ownership predicate is already
  settlement's: every box the shortfall is measured *from* ends at row `b-1` or
  above and every box it is measured *to* starts at `b` or beyond, which are the
  two halves of `_translation_ownership(b)`. Its per-section write, via
  `_shift_rows_from`, is `shift_section` -- the same write `_apply_translation`
  makes. Junction re-derivation is the render path's `reanchor_junctions`
  either way.
- A boundary carrying both demands is now widened **once**, by the larger. It
  was previously widened twice in succession, and the sum was larger than either
  needed: on `diagonal_labels` and `longread_variant_calling` the two owners
  together left 0.6px and 0.2px more than the single translation does.

Measured on the corpus, the render-time push fired on exactly **1** fixture of
369, and 5 row boundaries were left short of the clearance they owe with nothing
to correct them -- because the call site was gated on the header reconcile
having fired rather than on a shortfall existing. Settling the demand closes 3
of those 5 (`manual_rl_row_nonconsumer_bypass`,
`packed_cell_cellmate_bypass`, `packed_cell_cellmate_bypass_adjacent`, each
recovering the 9px its inter-row bundle was declared) and creates none.

The other 2 are rail layouts (`sarek_metro`, `rail_pitch_vs_labels`), which the
demand is deliberately not raised for: rail mode pitches adjacent rows so a line
runs between them without turning, and widening one of those boundaries to the
declared gap turns those flat runs into staircases -- 7 routes of 91 and 4 of 11
respectively, which `_assert_settlement_decisions_frozen` refuses as a decision
change rather than a translation. `tests/test_envelope_settlement.py` measures
that consequence rather than asserting the exclusion.

#### The two that cannot migrate, and why it is not a render count

`_tighten_lower_rows_after_shrink` pulls rows **up** by the slack a bottom
shrink revealed, closing a gap that is wider than it needs to be. Settlement's
invariant is that no row or column separation decreases, asserted by
`_assert_no_separation_decreased` on every settled layout. The two are opposite
operations, so there is no amount to publish: a demand for this move would be a
demand settlement is defined to refuse.

`_reserve_row_gap_for_top_padding` widens a boundary so that
`_fit_bboxes_to_content_top`, which runs immediately after it, can then grow a
box top into the room. The fact its demand would have to carry is not a
distance -- the distance is `fit_top - hug` and is perfectly expressible -- but
an *order*: the widening is only useful before a resize, and settlement may not
resize a box (`bbox_h` and `bbox_w` are frozen across it, which
`test_settlement_preserves_frozen_local_geometry` holds). By the time settlement
runs, the grow it was making room for has already been refused its ceiling and
the box is the size it will be drawn at. Settling the same distance then buys a
wider gap and no padding, which is not what the pass achieves.

Both of these also run inside `compute_layout`, before routing has published any
ledger at all, and the row positions they produce are inputs to the routing that
creates the ledger. That is a second, independent reason, but the direction and
ordering arguments above stand without it.

#### The two canvas passes were never candidates

`_shift_graph_into_canvas` and `_snap_canvas_y_to_grid` both measure something
and then call `_translate_graph_y`, which moves every section by one amount. A
uniform translation of everything changes no distance between any two boxes, so
there is no separation for settlement to own. They are canvas placement, not
compensation.

That was verified rather than assumed: each pass's `_translate_graph_y` call was
suppressed with its measurement left running, and the full set of pairwise
facing box separations (`_axis_gaps`, both axes) compared per fixture on the
settled render graph. Of the 357 fixtures that reach one, every fixture shows
every separation identical on both axes, which is what a uniform translation has
to show, so the argument above needs no exception. That the argument holds
without one rests on **Origin-independence** above: a canvas origin can reach a
row separation only through the binary64 arithmetic of a settlement deficit, and
`measured_distance` closes that route. Taking the deficit as a bare subtraction
instead, `examples/differentialabundance_default.mmd` and
`tests/fixtures/da_pipeline.mmd` are the two fixtures that break it -- their
`functional`/`plots` gap reads 91.0px with the canvas translation and 90.0px
without it, because a 14.0px deficit measures as `14.000000000000057` at that
origin and is answered with 15px. Both widths satisfy the 90px the corridor
reserves, so neither render is wrong; the coupling is.


## Cross-stage contract: semantic fan planning

- **Purpose**: Give one immutable owner to a complete authored fan or diamond,
  including its branches, opening and landing order, relative lanes, runway
  demands, exact offset slots, centreline members, and dedicated route
  emissions.
- **Helpers**: `build_fan_plan_execution` runs before Stage 1.
  `_apply_planned_fan_port_geometry` seats owned boundary anchors.
  `_snapshot_planned_fan_centrelines` freezes each settled structural
  centreline before `_apply_planned_fan_geometry` materialises the relative
  frame at Stages 4.9 and 6.17. Routing applies `FanOffsetCarrier` assignments
  before dispatch.
- **Precondition**: Authored connector identity and resolver lineage are
  complete. Effective grid decisions are available even though section canvas
  coordinates are not.
- **Postcondition**: A fan is wholly `PLANNED` or wholly `LEGACY`. A planned
  fan has exact structural ownership and complete relative geometry. A
  symmetric two-way fan uses mirrored lanes around one centreline; structural
  continuation identity does not convert that appearance into a trunk-plus-peel
  frame. Its absolute centreline source is fixed by the planner, so later grid,
  port, or topology mutations cannot select another anchor. A legacy fan claims
  no layout geometry, offsets, anchor, or route emissions and records one
  deterministic reason.
- **Invariants preserved**: Planned materialisation reads frozen anchors and
  cannot move an unowned port or station. Structural membership is independent
  of route-emission ownership. Each claimed route emission is produced exactly
  once and carries its plan and emitter identity.
- **Related tests**: `tests/test_fan_plans.py` and the fan-plan topology
  fixtures listed in `examples/topologies/README.md`.
- **Lifecycle:** invariant - the same fan decision is consumed by layout,
  offset assignment, routing, validation, and diagnostics for one layout pass.

## Unclear / structural-debt signals

No open signals at this time. Add new entries here when phase
pre/postconditions reveal a candidate for cleanup.

## Adding a new stage: checklist

When adding a new stage to `_compute_section_layout`, document the
following before merging:

1. **Stage tag**: pick the next sequential number within the
   appropriate stage (e.g. a new Stage 6.x sub-step gets the next
   integer after Stage 6.16).  Historical note: the organic phase
   suffix tree (`13d2`, `13k2`, the `Phase 13k` -> `Phase 13k2`
   rename in PR #342) is what the flat Stage.N scheme is designed to
   prevent.
2. **Helper location**: top-level function in `engine.py` (or a new
   module if it's substantial). Stage comments in the function body
   must reference the helper.
3. **Precondition**: what state on the graph the helper assumes.
   Mention coordinate-system regime (local vs global), whether ports
   are positioned, whether junctions are positioned, and whether
   trunks/grids are final.
4. **Postcondition**: the property the stage guarantees. Be concrete -
   "Y values are snapped to the row grid" not "Y values look nice".
5. **Invariants preserved**: what the stage does NOT change. Crucial
   for reasoning about reorder safety. Bboxes? Other sections?
   Off-track stations? Half-grid marker set?
6. **Related tests**: which invariants in `tests/test_layout_invariants.py`
   defend the postcondition. If none, add one - stages without test
   coverage are how the Phase-13-suffix sprawl happened in the first
   place.
7. **Validate-mode coverage**: if the stage introduces a new property
   that should hold permanently, add a `_guard_*` helper and call it
   from `validate=True` mode.
8. **Update this doc**: extend the per-stage table above and call out
   any cross-stage coupling in the structural-debt section.
