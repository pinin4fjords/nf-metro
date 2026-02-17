# nf-metro Refactor Plan

**Goal**: Clean up AI-iteratively-built code while keeping outputs pixel-identical.
**Safety net**: All 15 topology fixtures + rnaseq regression tests must pass unchanged after each step. Render all topologies before and after to diff PNGs.
**Baseline**: 213 tests, 6210 source lines, 2258 test lines.

---

## Phase 0: Establish baseline (do first, before any code changes)

1. **Generate golden renders** - Run `scripts/render_topologies.py` plus rnaseq examples to PNG. Store checksums as the regression baseline.
2. **Run full test suite** - Confirm all 213 tests pass. This count must not drop.
3. **Run ruff** - Fix any existing lint/format issues in a separate commit so refactor diffs are clean.
4. **Add pytest-cov** - Get a baseline coverage number. Add to dev deps in pyproject.toml.

---

## Phase 1: Extract constants (low risk, high readability win)

**Problem**: Magic numbers scattered across 6+ files.

Known constants to extract (non-exhaustive):
- engine.py: `x_spacing=60.0`, `y_spacing=40.0`, `x_offset=80.0`, `y_offset=120.0`, `char_width=7.0`, `label_pad=6.0`
- svg.py: `gap=30.0`, `inset=10.0`, `curve_radius=10.0`
- routing.py: `offset_step`, port margins, diagonal run lengths
- labels.py: padding values, font metrics
- section_placement.py: grid spacing defaults

**Action**: Create `src/nf_metro/layout/constants.py` for layout/routing constants and `src/nf_metro/render/constants.py` for render-side values. Replace all inline literals with imports.

**Files touched**: routing.py, engine.py, labels.py, svg.py, animate.py, section_placement.py, ordering.py

---

## Phase 2: Standardize naming conventions - SKIPPED

**Original problem**: Inconsistent abbreviations throughout.

**Audit result**: The codebase already follows a consistent convention:
- Short forms (`sid`, `sec_id`, `lid`, `pid`) for loop variables and local identifiers
- Long forms (`station_id`, `section_id`, `line_id`, `port_id`) for dataclass fields and public parameters
- `node_id` in `mermaid.py` is domain-correct (parsing Mermaid "nodes")

Forcing a single form per concept would make either loops verbose or fields cryptic. No action needed.

---

## Phase 3: Split routing.py (highest impact, moderate risk)

**Problem**: routing.py is 1299 lines. `route_edges()` alone is ~657 lines (28-684) with intra-section, inter-section, and TB-section routing all interleaved.

**Action**: Split into a `routing/` subpackage:
- `routing/intra_section.py` - Regular LR/RL section routing, station offsets
- `routing/inter_section.py` - L-shaped inter-section routing, bundle ordering, `_inter_column_channel_x()`
- `routing/tb_section.py` - TB section special cases (BOTTOM exit, LEFT/RIGHT exit, entry)
- `routing/reversal.py` - `_detect_reversed_sections()` and its nested helpers (lines 1108-1299, 192 lines)
- `routing/common.py` - `RoutedPath`, `_compute_bundle_info()`, `compute_station_offsets()`, offset helpers
- `routing/__init__.py` - Re-export `route_edges()` as the public API

Keep `route_edges()` as a thin dispatcher that calls into the submodules.

**Files touched**: routing.py (split), engine.py (import path), svg.py (import path), test_routing.py (import path)

---

## Phase 4: Break up engine.py god functions (high impact, moderate risk)

**Problem**: `_compute_section_layout()` is ~275 lines (86-360) with 7 inline phases. `_align_entry_ports()` is ~199 lines (409-607) with deep nesting.

**Action**:
- Extract each phase of `_compute_section_layout()` into a named function: `_phase_internal_layout()`, `_phase_section_placement()`, `_phase_global_coords()`, `_phase_port_positioning()`, `_phase_junction_positioning()`, `_phase_entry_alignment()`, `_phase_exit_alignment()`. The orchestrator becomes a clear sequence of calls.
- Extract `_align_entry_ports()` TB clamping logic into `_clamp_tb_entry_port()`.
- Extract repeated `ys = [s.y for s in ...]` patterns into a `_station_y_range()` helper.
- Extract `_compute_fork_join_gaps()` (lines 753-834) - already a separate function, verify it's well-bounded.

**Files touched**: engine.py

---

## Phase 5: Clean up mermaid.py parser (moderate impact, low risk)

**Problem**: `_resolve_sections()` is 174 lines (294-467) doing 3 distinct things. Junction creation logic is deeply nested.

**Action**: Split into:
- `_build_entry_side_mapping()` - Build per-line entry side lookup
- `_classify_edges()` - Separate internal vs inter-section edges
- `_create_ports_and_junctions()` - Port creation, junction creation, edge rewriting

**Files touched**: mermaid.py

---

## Phase 6: Simplify auto_layout.py (moderate impact, moderate risk)

**Problem**: `_assign_grid_positions()` is 158 lines (121-278) with implicit col_step toggling for serpentine folds. `_infer_port_sides()` is 124 lines (540-663) of nested voting logic.

**Action**:
- Extract fold/serpentine logic from `_assign_grid_positions()` into `_apply_fold_wrapping()`.
- Extract port-side voting from `_infer_port_sides()` into `_compute_side_votes()`.
- Consider a `FoldDirection` enum instead of toggling col_step between -1/+1.
- `_transitive_successors()` is used by `_optimize_rowspans()` - not dead code, leave it.

**Files touched**: auto_layout.py

---

## Phase 7: Test infrastructure consolidation (low risk)

**Problem**: Duplicated inline graph strings across test files. No conftest.py. Helper functions scattered.

Current test inventory:
| File | Lines | Tests |
|------|-------|-------|
| test_auto_layout.py | 462 | ~18 |
| test_topology_validation.py | 461 | ~95 (parametrized) |
| test_layout.py | 364 | ~11 |
| test_parser.py | 275 | ~19 |
| test_render.py | 157 | ~12 |
| test_routing.py | 116 | ~6 |
| layout_validator.py | 423 | (test infra) |

**Action**:
- Create `tests/conftest.py` with shared fixtures (`simple_linear_graph`, `simple_diamond_graph`, `two_section_graph`).
- Create `tests/helpers.py` with `GraphBuilder` class and `render_pipeline()` helper.
- Deduplicate repeated graph-string definitions across test files.
- Move repeated validator call patterns into shared fixtures.

**Files touched**: All test files, new conftest.py, new helpers.py

---

## Phase 8: Module boundary cleanup (low risk)

**Problem**: Parser imports from layout (`infer_section_layout` imported inside `parse_metro_mermaid`). Root `__init__.py` has no `__all__`.

Current `__all__` coverage:
- `layout/__init__.py`: `["compute_layout"]`
- `parser/__init__.py`: `["parse_metro_mermaid"]`
- `render/__init__.py`: `["render_svg"]`
- `themes/__init__.py`: `["THEMES", "NFCORE_THEME", "LIGHT_THEME"]`
- Root `__init__.py`: missing

**Action**:
- Add `__all__` to root `__init__.py`.
- Consider whether `infer_section_layout` should be called by the parser or by the caller. If by the caller, simplify the parser/layout coupling.
- Document the module dependency DAG: parser -> layout -> routing -> render.

**Files touched**: All `__init__.py` files, possibly mermaid.py

---

## Phase 9: Remove dead code and vestigial artifacts (low risk)

Confirmed dead code:
- `grammar.lark` - unused (parser uses regex). Remove or add a clear comment.
- `route_inter_section_edges()` in routing.py (lines 858-939) - only called from `test_routing.py`, never from production code. Either mark as test utility or remove and inline in test.

**Action**:
- Remove `grammar.lark` (or move to `docs/` with explanation).
- Audit `route_inter_section_edges()` - if test-only, move to test helpers.
- Grep for TODO/FIXME/HACK markers and address or document.
- Check for commented-out code blocks.

**Files touched**: Various.

---

## Execution Strategy

- **One phase per commit** on this branch. Each commit should pass all tests independently.
- **Phases 0-1** are safe warm-ups with near-zero regression risk.
- **Phase 3** (routing split) is the highest-value refactor but needs careful import management.
- **Phase 4** (engine extraction) is the second highest value.
- **Phases 5-6** are moderate wins.
- **Phases 7-9** are cleanup/polish.
- After each phase: run full test suite + render all topologies + diff PNGs against golden baseline.

## File Size Reference (actual, from main as of 2026-02-17)

| File | Lines | Priority |
|------|-------|----------|
| routing.py | 1299 | Critical - split target (Phase 3) |
| engine.py | 834 | Critical - function extraction (Phase 4) |
| svg.py | 725 | Low - well-structured |
| auto_layout.py | 700 | Moderate - fold/voting logic (Phase 6) |
| section_placement.py | 593 | Low - stable |
| mermaid.py | 467 | Moderate - parser split (Phase 5) |
| animate.py | 307 | Low - stable |
| test_topology_validation.py | 461 | Phase 7 target |
| test_auto_layout.py | 462 | Phase 7 target |
| layout_validator.py | 423 | OK (test infra) |
| test_layout.py | 364 | Phase 7 target |
| test_parser.py | 275 | Phase 7 target |
| ordering.py | 243 | Low - stable |
| labels.py | 222 | Low - stable |
| model.py | 206 | Clean |
| cli.py | 180 | Clean |
| test_render.py | 157 | Phase 7 target |
| legend.py | 149 | Clean |
| test_routing.py | 116 | Phase 7 target |
| icons.py | 111 | Clean |
| style.py | 45 | Clean |
| layers.py | 43 | Clean |
