"""Fork/join diamond branch label placement.

A diamond branch that isn't at its section's Y extreme falls back to plain
column-parity alternation, which is blind to the diamond's own geometry: it
can point the label at a crowded, unrelated neighbouring track even though
the diamond's own interior (the gap toward its sibling branch, guaranteed
clear by construction) is wide open.
"""

from nf_metro.layout.labels import place_labels
from nf_metro.parser.model import Edge, MetroGraph, Section, Station

_SEC_ID = "pipe"


def _station(station_id, x, y, layer, label=None):
    return Station(
        id=station_id,
        label=label or station_id,
        section_id=_SEC_ID,
        x=x,
        y=y,
        layer=layer,
    )


def _build_graph(stations, edges):
    graph = MetroGraph()
    graph.sections[_SEC_ID] = Section(id=_SEC_ID, name="Pipeline", direction="LR")
    for station in stations:
        graph.stations[station.id] = station
        graph.sections[_SEC_ID].station_ids.append(station.id)
    graph.edges = list(edges)
    return graph


def _above(placements, station_id):
    for placement in placements:
        if placement.station_id == station_id:
            return placement.above
    raise AssertionError(f"no placement for {station_id!r}")


def test_on_trunk_branch_prefers_diamond_interior_over_crowded_exterior():
    """An on-trunk branch squeezed against an unrelated track above it must
    point its label at its sibling branch instead."""
    stations = [
        _station("upper1", 80, 120, layer=0, label="Upper QC"),
        _station("upper2", 180.5, 120, layer=1, label="Upper Report"),
        _station("raw", 80, 160, layer=0, label="Raw Reads"),
        _station("trim_galore", 180.5, 160, layer=1, label="Trim Galore!"),
        _station("fastp", 180.5, 200, layer=1, label="fastp"),
        _station("align", 276.5, 160, layer=2, label="Align"),
    ]
    edges = [
        Edge(source="upper1", target="upper2", line_id="qc"),
        Edge(source="raw", target="trim_galore", line_id="a"),
        Edge(source="raw", target="fastp", line_id="b"),
        Edge(source="trim_galore", target="align", line_id="a"),
        Edge(source="fastp", target="align", line_id="b"),
    ]
    placements = place_labels(_build_graph(stations, edges))

    assert _above(placements, "trim_galore") is False, (
        "trim_galore should point down at its sibling fastp (the diamond's "
        "open interior), not up into the crowded gap against Upper Report"
    )
    # fastp is the section's sole bottom extreme; the pre-existing
    # edge-outward override already handles this correctly.
    assert _above(placements, "fastp") is False


def test_on_trunk_branch_prefers_diamond_interior_mirrored_below():
    """Mirror of the above: the crowding sits below the diamond instead."""
    stations = [
        _station("raw", 80, 160, layer=0, label="Raw Reads"),
        _station("trim_galore", 180.5, 120, layer=0, label="Trim Galore!"),
        _station("fastp", 180.5, 160, layer=0, label="fastp"),
        _station("align", 276.5, 160, layer=2, label="Align"),
        _station("lower1", 80, 200, layer=0, label="Lower QC"),
        _station("lower2", 180.5, 200, layer=0, label="Lower Report"),
    ]
    edges = [
        Edge(source="raw", target="trim_galore", line_id="a"),
        Edge(source="raw", target="fastp", line_id="b"),
        Edge(source="trim_galore", target="align", line_id="a"),
        Edge(source="fastp", target="align", line_id="b"),
        Edge(source="lower1", target="lower2", line_id="qc"),
    ]
    placements = place_labels(_build_graph(stations, edges))

    assert _above(placements, "fastp") is True, (
        "fastp should point up at its sibling trim_galore (the diamond's "
        "open interior), not down into the crowded gap against Lower Report"
    )
    # trim_galore is the section's sole top extreme; already handled
    # correctly by the pre-existing edge-outward override.
    assert _above(placements, "trim_galore") is True


def test_both_branches_crowded_prefer_each_other():
    """Content on both sides: both branches should point at each other."""
    stations = [
        _station("upper1", 80, 120, layer=0, label="Upper QC"),
        _station("upper2", 180.5, 120, layer=1, label="Upper Report"),
        _station("raw", 80, 160, layer=0, label="Raw Reads"),
        _station("on_trunk", 180.5, 160, layer=1, label="On Trunk"),
        _station("off_trunk", 180.5, 220, layer=0, label="Off Trunk"),
        _station("align", 276.5, 160, layer=2, label="Align"),
        _station("lower1", 80, 260, layer=0, label="Lower QC"),
        _station("lower2", 180.5, 260, layer=0, label="Lower Report"),
    ]
    edges = [
        Edge(source="upper1", target="upper2", line_id="qc"),
        Edge(source="raw", target="on_trunk", line_id="a"),
        Edge(source="raw", target="off_trunk", line_id="b"),
        Edge(source="on_trunk", target="align", line_id="a"),
        Edge(source="off_trunk", target="align", line_id="b"),
        Edge(source="lower1", target="lower2", line_id="qc2"),
    ]
    placements = place_labels(_build_graph(stations, edges))

    assert _above(placements, "on_trunk") is False
    assert _above(placements, "off_trunk") is True
