"""Fork/join diamond branch label placement.

Column-parity alternation picks a label's side by column index alone, with
no awareness of diamond geometry: for a 2-way fork/join diamond it can
coincidentally point one or both branch labels at the other branch,
squeezing them inside the bubble between the two routes. The outside of a
two-way bubble is always the better default, so every diamond branch must
point away from its sibling branch -- with no further collision reasoning
attempted.
"""

from nf_metro.layout.labels import _apply_diamond_outward_override, place_labels
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


def test_higher_branch_points_away_from_lower_sibling():
    higher = _station("upper_tool", 180.5, 140, layer=0, label="Upper Tool")
    lower = _station("lower_tool", 180.5, 180, layer=1, label="Lower Tool")
    siblings = {higher.id: lower, lower.id: higher}

    assert _apply_diamond_outward_override(higher, siblings) is True
    assert _apply_diamond_outward_override(lower, siblings) is False


def test_non_diamond_station_returns_none():
    station = _station("solo", 80, 160, layer=0, label="Solo")
    assert _apply_diamond_outward_override(station, {}) is None


def test_full_pipeline_keeps_diamond_branches_outward():
    """End-to-end: even with an unrelated crowded neighbour above the upper
    branch, both diamond branches must land on their own outside."""
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

    assert _above(placements, "trim_galore") is True
    assert _above(placements, "fastp") is False
