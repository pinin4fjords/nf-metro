"""Tests for label placement helpers."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.geometry import segment_intersects_bbox as _segment_intersects_bbox
from nf_metro.layout.labels import (
    LabelPlacement,
    _avoid_diagonal_routes,
    _compute_port_label_preference,
    _wrap_text_to_chars,
)
from nf_metro.layout.routing.common import OffsetRegime
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import (
    Edge,
    LayoutGeometryWarning,
    MetroGraph,
    Port,
    PortSide,
    Station,
)
from nf_metro.render.svg import build_render_plan
from nf_metro.themes import THEMES

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_graph(stations, edges, ports):
    """Build a minimal MetroGraph for label tests."""
    g = MetroGraph()
    for s in stations:
        g.stations[s.id] = s
    g.edges = list(edges)
    for p in ports:
        g.ports[p.id] = p
    return g


class TestComputePortLabelPreference:
    """Tests for _compute_port_label_preference."""

    def test_exit_port_below_prefers_label_above(self):
        """Station with exit port below should prefer label above."""
        g = _make_graph(
            stations=[
                Station(id="a", label="A", x=100, y=100),
                Station(id="p", label="", x=120, y=200, is_port=True),
            ],
            edges=[Edge(source="a", target="p", line_id="L1")],
            ports=[Port(id="p", section_id="s", side=PortSide.BOTTOM, is_entry=False)],
        )
        pref = _compute_port_label_preference(g)
        assert pref["a"] is True  # above

    def test_exit_port_above_prefers_label_below(self):
        """Station with exit port above should prefer label below."""
        g = _make_graph(
            stations=[
                Station(id="a", label="A", x=100, y=200),
                Station(id="p", label="", x=120, y=100, is_port=True),
            ],
            edges=[Edge(source="a", target="p", line_id="L1")],
            ports=[Port(id="p", section_id="s", side=PortSide.TOP, is_entry=False)],
        )
        pref = _compute_port_label_preference(g)
        assert pref["a"] is False  # below

    def test_entry_port_ignored(self):
        """Entry ports should not produce a label preference."""
        g = _make_graph(
            stations=[
                Station(id="p", label="", x=50, y=200, is_port=True),
                Station(id="a", label="A", x=100, y=100),
            ],
            edges=[Edge(source="p", target="a", line_id="L1")],
            ports=[Port(id="p", section_id="s", side=PortSide.LEFT, is_entry=True)],
        )
        pref = _compute_port_label_preference(g)
        assert "a" not in pref

    def test_same_y_ignored(self):
        """Ports at the same Y as the station should not produce a preference."""
        g = _make_graph(
            stations=[
                Station(id="a", label="A", x=100, y=100),
                Station(id="p", label="", x=200, y=100, is_port=True),
            ],
            edges=[Edge(source="a", target="p", line_id="L1")],
            ports=[Port(id="p", section_id="s", side=PortSide.RIGHT, is_entry=False)],
        )
        pref = _compute_port_label_preference(g)
        assert "a" not in pref

    def test_max_dx_filters_distant_ports(self):
        """Ports beyond max_dx should not override label side."""
        g = _make_graph(
            stations=[
                Station(id="a", label="A", x=100, y=100),
                Station(id="p", label="", x=300, y=200, is_port=True),
            ],
            edges=[Edge(source="a", target="p", line_id="L1")],
            ports=[Port(id="p", section_id="s", side=PortSide.BOTTOM, is_entry=False)],
        )
        # dx=200 exceeds max_dx=120
        pref = _compute_port_label_preference(g, max_dx=120)
        assert "a" not in pref

        # Without limit, preference is present
        pref_no_limit = _compute_port_label_preference(g, max_dx=0)
        assert pref_no_limit["a"] is True

    def test_conflicting_ports_cancel(self):
        """Ports on both sides should cancel the preference."""
        g = _make_graph(
            stations=[
                Station(id="a", label="A", x=100, y=150),
                Station(id="p1", label="", x=120, y=100, is_port=True),
                Station(id="p2", label="", x=120, y=200, is_port=True),
            ],
            edges=[
                Edge(source="a", target="p1", line_id="L1"),
                Edge(source="a", target="p2", line_id="L2"),
            ],
            ports=[
                Port(id="p1", section_id="s", side=PortSide.TOP, is_entry=False),
                Port(id="p2", section_id="s", side=PortSide.BOTTOM, is_entry=False),
            ],
        )
        pref = _compute_port_label_preference(g)
        assert "a" not in pref

    def test_multiple_consistent_ports_keep_preference(self):
        """Multiple exit ports on the same side should reinforce the preference."""
        g = _make_graph(
            stations=[
                Station(id="a", label="A", x=100, y=100),
                Station(id="p1", label="", x=110, y=200, is_port=True),
                Station(id="p2", label="", x=120, y=250, is_port=True),
            ],
            edges=[
                Edge(source="a", target="p1", line_id="L1"),
                Edge(source="a", target="p2", line_id="L2"),
            ],
            ports=[
                Port(id="p1", section_id="s", side=PortSide.BOTTOM, is_entry=False),
                Port(id="p2", section_id="s", side=PortSide.BOTTOM, is_entry=False),
            ],
        )
        pref = _compute_port_label_preference(g)
        assert pref["a"] is True  # both below -> prefer above


@dataclass
class _FakeEdge:
    source: str = ""
    target: str = ""


@dataclass
class _FakeRoute:
    edge: _FakeEdge = field(default_factory=_FakeEdge)
    line_id: str = "L1"
    points: list = field(default_factory=list)
    offset_regime: OffsetRegime = OffsetRegime.BAKED


class TestSegmentIntersectsBbox:
    """Tests for _segment_intersects_bbox."""

    def test_segment_inside_bbox(self):
        assert _segment_intersects_bbox(5, 5, 10, 10, (0, 0, 20, 20))

    def test_segment_crosses_bbox(self):
        assert _segment_intersects_bbox(-5, 10, 25, 10, (0, 0, 20, 20))

    def test_segment_outside_bbox(self):
        assert not _segment_intersects_bbox(100, 100, 200, 200, (0, 0, 20, 20))

    def test_diagonal_clips_corner(self):
        assert _segment_intersects_bbox(0, 30, 30, 0, (10, 10, 20, 20))

    def test_diagonal_misses_bbox(self):
        # Diagonal passes well clear of the bbox.
        assert not _segment_intersects_bbox(0, 0, 5, 5, (50, 50, 60, 60))


class TestAvoidDiagonalRoutes:
    """Tests for _avoid_diagonal_routes."""

    def test_label_flipped_off_diagonal(self):
        g = MetroGraph()
        g.stations["a"] = Station(id="a", label="A", x=100, y=200)
        # Label placed above the station (y_max = 195) right where a
        # diagonal route segment crosses.
        placement = LabelPlacement(station_id="a", text="A", x=100, y=195, above=True)
        # Diagonal segment passes through the label area above.
        route = _FakeRoute(points=[(50, 250), (150, 150)])
        _avoid_diagonal_routes([placement], g, [route], None)
        # Should have flipped to below.
        assert placement.above is False
        assert placement.y > 200

    def test_horizontal_segment_ignored(self):
        g = MetroGraph()
        g.stations["a"] = Station(id="a", label="A", x=100, y=200)
        placement = LabelPlacement(station_id="a", text="A", x=100, y=195, above=True)
        # Pure horizontal segment crossing the label area.
        route = _FakeRoute(points=[(0, 195), (200, 195)])
        _avoid_diagonal_routes([placement], g, [route], None)
        # Should not flip - horizontal trunk routes aren't treated as
        # label obstacles.
        assert placement.above is True
        assert placement.y == 195

    def test_no_route_collision_no_flip(self):
        g = MetroGraph()
        g.stations["a"] = Station(id="a", label="A", x=100, y=200)
        placement = LabelPlacement(station_id="a", text="A", x=100, y=195, above=True)
        # Diagonal far away from the label.
        route = _FakeRoute(points=[(500, 500), (600, 600)])
        _avoid_diagonal_routes([placement], g, [route], None)
        assert placement.above is True
        assert placement.y == 195


# The issue-1768 reporter's map: a wide-label section beside a narrow one, so
# the narrow section's labels are the ones forced to wrap.
_CROWDED_LABELS_MMD = """%%metro line: main | Main | #e6007e
%%metro grid: wide, narrow | 0,0

graph LR
    subgraph wide [Wide neighbour]
        w1[BEDTools genomecov]
        w2[UMI-tools deduplicate]
        w3[Infer strandedness]

        w1 -->|main| w2
        w2 -->|main| w3
    end

    subgraph narrow [Narrow]
        a[plastid P-site]
        b[plastid wiggle]
        c[Quantify ORF P-sites]

        a -->|main| b
        b -->|main| c
    end

    w3 -->|main| a
"""


def _drawn_label_texts(graph: MetroGraph) -> dict[str, str]:
    """Station id -> the name-label text the renderer will draw, wraps included."""
    plan = build_render_plan(graph, THEMES["nfcore"])
    return {p.station_id: p.text for p in plan.labels if p.station_id}


def _assert_tokens_intact(graph: MetroGraph, context: str) -> None:
    """Every drawn label must be its node label with only whitespace re-flowed."""
    mangled = {
        sid: text
        for sid, text in _drawn_label_texts(graph).items()
        if (station := graph.stations.get(sid)) is not None
        and text.split() != station.label.split()
    }
    assert not mangled, (
        f"{context}: wrapping mutated label tokens (mid-word hyphenation): "
        + ", ".join(f"{sid}={text!r}" for sid, text in sorted(mangled.items()))
    )


class TestWrapTextToChars:
    """Wrapping re-flows whitespace; it never breaks inside a token."""

    @pytest.mark.parametrize(
        "label",
        [
            "plastid wiggle",
            "gffcompare",
            "RNA-SeQC",
            "sambamba markdup",
            "Quantify ORF P-sites",
            "UMI-tools deduplicate",
            "Infer strandedness",
        ],
    )
    @pytest.mark.parametrize("budget", [1, 2, 4, 6, 8, 12, 40])
    def test_tokens_survive_every_budget(self, label: str, budget: int) -> None:
        assert _wrap_text_to_chars(label, budget).split() == label.split()

    def test_breaks_on_whitespace_before_widening(self) -> None:
        assert _wrap_text_to_chars("plastid wiggle", 7) == "plastid\nwiggle"

    def test_single_token_overflows_rather_than_splitting(self) -> None:
        assert _wrap_text_to_chars("gffcompare", 4) == "gffcompare"

    def test_oversized_token_does_not_strand_its_neighbours(self) -> None:
        assert (
            _wrap_text_to_chars("Quantify ORF P-sites", 4) == "Quantify\nORF\nP-sites"
        )


class TestDrawnLabelsKeepTokensWhole:
    """No station label the renderer draws is hyphenated mid-word (#1768)."""

    @pytest.mark.parametrize("x_spacing", [None, 40.0, 60.0, 70.0])
    def test_crowded_labels_never_hyphenate(self, x_spacing: float | None) -> None:
        graph = parse_metro_mermaid(_CROWDED_LABELS_MMD)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_layout(graph, x_spacing=x_spacing)
        _assert_tokens_intact(graph, f"x_spacing={x_spacing}")

    @pytest.mark.parametrize(
        "fixture",
        [
            "examples/topologies/render_labelwrap_row_gap.mmd",
            "examples/topologies/paired_input_fan_branch_tree.mmd",
            "examples/topologies/packed_cell_right_exit_left_entry_wrap.mmd",
            "examples/variantbenchmarking.mmd",
            "tests/fixtures/multiline_labels.mmd",
        ],
    )
    def test_wrapping_corpus_fixture_never_hyphenates(self, fixture: str) -> None:
        graph = parse_metro_mermaid((REPO_ROOT / fixture).read_text())
        compute_layout(graph)
        _assert_tokens_intact(graph, fixture)


class TestPinnedXSpacingWarning:
    """A pinned column pitch too narrow for its labels says so out loud.

    Wrapping stops at the longest word, so a pitch the caller pinned below what
    the content needs ships an honest overlap.  That is preferable to a name
    broken mid-word, but only if it is reported rather than shipped silently.
    """

    def test_pinned_pitch_below_content_warns(self) -> None:
        graph = parse_metro_mermaid(_CROWDED_LABELS_MMD)
        with pytest.warns(
            LayoutGeometryWarning, match=r"x_spacing=60\.0 is too narrow"
        ):
            compute_layout(graph, x_spacing=60.0)

    @pytest.mark.parametrize("x_spacing", [None, 100.0])
    def test_a_pitch_the_labels_fit_is_silent(self, x_spacing: float | None) -> None:
        graph = parse_metro_mermaid(_CROWDED_LABELS_MMD)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_layout(graph, x_spacing=x_spacing)
        assert not [w for w in caught if "x_spacing" in str(w.message)]
