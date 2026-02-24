"""Tests for label placement helpers."""

from nf_metro.layout.labels import _compute_port_label_preference
from nf_metro.parser.model import Edge, MetroGraph, Port, PortSide, Station


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
