"""Parametrized topology stress tests for the auto-layout engine.

Loads diverse .mmd fixtures, runs layout, and validates programmatically
for layout defects. Also includes topology-specific assertions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from layout_validator import (
    Severity,
    check_coordinate_sanity,
    check_edge_waypoints,
    check_port_boundary,
    check_section_overlap,
    check_station_containment,
    validate_layout,
)

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid

TOPOLOGIES_DIR = Path(__file__).parent / "fixtures" / "topologies"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Collect all topology fixtures
TOPOLOGY_FILES = sorted(TOPOLOGIES_DIR.glob("*.mmd"))
TOPOLOGY_IDS = [f.stem for f in TOPOLOGY_FILES]

# Include rnaseq as regression guard
RNASEQ_FILE = EXAMPLES_DIR / "rnaseq_sections.mmd"


def _load_and_layout(path: Path):
    """Parse a .mmd file and run layout."""
    text = path.read_text()
    graph = parse_metro_mermaid(text)
    compute_layout(graph)
    return graph


# --- Parametrized validation across all topologies ---


@pytest.fixture(params=TOPOLOGY_FILES, ids=TOPOLOGY_IDS)
def topology_graph(request):
    """Load and lay out each topology fixture."""
    return _load_and_layout(request.param)


class TestTopologyValidation:
    """Run all validator checks against every topology."""

    def test_no_section_overlap(self, topology_graph):
        violations = check_section_overlap(topology_graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_station_containment(self, topology_graph):
        violations = check_station_containment(topology_graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_port_boundary(self, topology_graph):
        violations = check_port_boundary(topology_graph)
        # Port boundary is a warning, but we still flag issues
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_coordinate_sanity(self, topology_graph):
        violations = check_coordinate_sanity(topology_graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_edge_waypoints(self, topology_graph):
        violations = check_edge_waypoints(topology_graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_all_stations_have_coordinates(self, topology_graph):
        """Every real station should have been assigned non-default coords."""
        for sid, station in topology_graph.stations.items():
            if station.is_port or sid in topology_graph.junctions:
                continue
            if station.section_id is None:
                continue
            # At least one coordinate should be non-zero (offset is >= 80)
            assert station.x != 0 or station.y != 0, (
                f"Station '{sid}' still at origin (0,0)"
            )


# --- Regression guard: rnaseq example ---


class TestRnaseqRegression:
    """Ensure the rnaseq example passes all layout checks."""

    @pytest.fixture
    def rnaseq_graph(self):
        return _load_and_layout(RNASEQ_FILE)

    def test_no_section_overlap(self, rnaseq_graph):
        violations = check_section_overlap(rnaseq_graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_station_containment(self, rnaseq_graph):
        violations = check_station_containment(rnaseq_graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_coordinate_sanity(self, rnaseq_graph):
        violations = check_coordinate_sanity(rnaseq_graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_edge_waypoints(self, rnaseq_graph):
        violations = check_edge_waypoints(rnaseq_graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_all_sections_placed(self, rnaseq_graph):
        """All 5 rnaseq sections should have valid bounding boxes."""
        assert len(rnaseq_graph.sections) == 5
        for sid, section in rnaseq_graph.sections.items():
            assert section.bbox_w > 0, f"Section '{sid}' has zero width"
            assert section.bbox_h > 0, f"Section '{sid}' has zero height"


# --- Topology-specific assertions ---


class TestTopologySpecific:
    """Targeted assertions for individual topologies."""

    def test_fan_out_creates_junction(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "wide_fan_out.mmd")
        # With 4 targets from one source, we expect junction(s)
        assert len(graph.junctions) > 0, "Fan-out should create junction stations"

    def test_fan_out_has_5_sections(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "wide_fan_out.mmd")
        assert len(graph.sections) == 5

    def test_fan_in_has_5_sections(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "wide_fan_in.mmd")
        assert len(graph.sections) == 5

    def test_deep_linear_has_7_sections(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "deep_linear.mmd")
        assert len(graph.sections) == 7
        # Sections should progress left to right (or with fold)
        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_parallel_independent_separate_rows(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "parallel_independent.mmd")
        # DNA and RNA chains should not overlap
        violations = check_section_overlap(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)
        # Should have 4 sections
        assert len(graph.sections) == 4

    def test_diamond_grid_structure(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "section_diamond.mmd")
        # 4 sections: start, branch_left, branch_right, finish
        assert len(graph.sections) == 4
        # Start should be in col 0, branches in col 1, finish in col 2
        start = graph.sections["start"]
        bl = graph.sections["branch_left"]
        br = graph.sections["branch_right"]
        finish = graph.sections["finish"]
        assert start.grid_col < bl.grid_col
        assert start.grid_col < br.grid_col
        assert bl.grid_col < finish.grid_col
        assert br.grid_col < finish.grid_col

    def test_diamond_branches_different_rows(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "section_diamond.mmd")
        bl = graph.sections["branch_left"]
        br = graph.sections["branch_right"]
        # Branches should be stacked vertically (different rows)
        assert bl.grid_row != br.grid_row

    def test_single_section_no_ports(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "single_section.mmd")
        assert len(graph.sections) == 1
        # Single section with no inter-section edges should have no ports
        assert len(graph.ports) == 0
        assert len(graph.junctions) == 0

    def test_single_section_valid(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "single_section.mmd")
        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_asymmetric_tree_sections(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "asymmetric_tree.mmd")
        # 7 sections total
        assert len(graph.sections) == 7
        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_mixed_port_sides_structure(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "mixed_port_sides.mmd")
        assert len(graph.sections) == 3
        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_multi_line_bundle_all_6_lines(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "multi_line_bundle.mmd")
        assert len(graph.lines) == 6
        assert len(graph.sections) == 3
        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_complex_multipath_structure(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "complex_multipath.mmd")
        assert len(graph.sections) == 6
        assert len(graph.lines) == 4
        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_rnaseq_lite_structure(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "rnaseq_lite.mmd")
        assert len(graph.sections) == 5
        assert len(graph.lines) == 3
        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_variant_calling_structure(self):
        graph = _load_and_layout(TOPOLOGIES_DIR / "variant_calling.mmd")
        assert len(graph.sections) == 6
        assert len(graph.lines) == 4
        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    # --- Fold topology tests ---

    def test_fold_fan_across_structure(self):
        """Fan-out/fan-in across a fold boundary."""
        graph = _load_and_layout(TOPOLOGIES_DIR / "fold_fan_across.mmd")
        assert len(graph.sections) == 7
        assert len(graph.lines) == 3

        # normalize is the fold section (TB direction, rowspan=3)
        normalize = graph.sections["normalize"]
        assert normalize.direction == "TB"
        assert normalize.grid_row_span == 3

        # Three quant sections stacked at the same column
        tmt = graph.sections["tmt_quant"]
        lfq = graph.sections["lfq_quant"]
        dia = graph.sections["dia_quant"]
        assert tmt.grid_col == lfq.grid_col == dia.grid_col
        assert len({tmt.grid_row, lfq.grid_row, dia.grid_row}) == 3

        # stat_analysis is RL (post-fold return row)
        stat = graph.sections["stat_analysis"]
        assert stat.direction == "RL"

        # All grid_cols are non-negative
        for sid, sec in graph.sections.items():
            assert sec.grid_col >= 0, f"{sid} has negative grid_col={sec.grid_col}"

        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_fold_double_structure(self):
        """Double fold producing a serpentine (zigzag) layout."""
        graph = _load_and_layout(TOPOLOGIES_DIR / "fold_double.mmd")
        assert len(graph.sections) == 10
        assert len(graph.lines) == 2

        # Two fold sections (TB direction)
        calling = graph.sections["calling"]
        integration = graph.sections["integration"]
        assert calling.direction == "TB"
        assert integration.direction == "TB"

        # Serpentine: row 0 (LR), row 1 (RL), row 2 (LR)
        row0_secs = [s for s in graph.sections.values() if s.grid_row == 0]
        row1_secs = [s for s in graph.sections.values() if s.grid_row == 1]
        row2_secs = [s for s in graph.sections.values() if s.grid_row == 2]
        assert len(row0_secs) == 4  # input_qc, alignment, base_recal, calling
        assert len(row1_secs) == 4  # hard_filter .. integration
        assert len(row2_secs) == 2  # reporting, archival

        # Row 1 post-fold sections flow RL
        hard_filter = graph.sections["hard_filter"]
        annotation = graph.sections["annotation"]
        interpretation = graph.sections["interpretation"]
        assert hard_filter.direction == "RL"
        assert annotation.direction == "RL"
        assert interpretation.direction == "RL"

        # Row 2 post-second-fold sections flow LR
        reporting = graph.sections["reporting"]
        archival = graph.sections["archival"]
        assert reporting.direction == "LR"
        assert archival.direction == "LR"

        # All grid_cols are non-negative (the double-fold bug fix)
        for sid, sec in graph.sections.items():
            assert sec.grid_col >= 0, f"{sid} has negative grid_col={sec.grid_col}"

        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)

    def test_fold_stacked_branch_structure(self):
        """Stacked sections near fold + post-fold branching."""
        graph = _load_and_layout(TOPOLOGIES_DIR / "fold_stacked_branch.mmd")
        assert len(graph.sections) == 8
        assert len(graph.lines) == 3

        # integration is fold section (TB, rowspan=3)
        integration = graph.sections["integration"]
        assert integration.direction == "TB"
        assert integration.grid_row_span == 3

        # Three analysis sections stacked at same column
        rna = graph.sections["rna_analysis"]
        atac = graph.sections["atac_analysis"]
        prot = graph.sections["protein_analysis"]
        assert rna.grid_col == atac.grid_col == prot.grid_col
        assert len({rna.grid_row, atac.grid_row, prot.grid_row}) == 3

        # bio_interp and tech_qc are post-fold, stacked at same column
        bio = graph.sections["bio_interp"]
        tech = graph.sections["tech_qc"]
        assert bio.grid_col == tech.grid_col
        assert bio.grid_row != tech.grid_row

        # bio_interp is RL (post-fold, successor to left)
        assert bio.direction == "RL"

        # All grid_cols are non-negative
        for sid, sec in graph.sections.items():
            assert sec.grid_col >= 0, f"{sid} has negative grid_col={sec.grid_col}"

        violations = validate_layout(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        assert not errors, "\n".join(v.message for v in errors)
