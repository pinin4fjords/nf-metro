"""Selection tests for shared-row exit bundle feeders."""

from nf_metro.layout.routing.offsets import _exit_trunk_feeder
from nf_metro.parser.model import Edge, MetroGraph, MetroLine, Station


def test_direct_trunk_wins_over_earlier_partial_candidate() -> None:
    graph = MetroGraph(
        lines={
            "first": MetroLine("first", "First", "#111111"),
            "second": MetroLine("second", "Second", "#222222"),
        },
        stations={
            "partial": Station("partial", "Partial", y=0.0),
            "trunk": Station("trunk", "Trunk", y=10.0),
            "other": Station("other", "Other", y=20.0),
            "upstream": Station("upstream", "Upstream"),
            "port": Station("port", "Port", is_port=True),
        },
        edges=[
            Edge("upstream", "partial", "second"),
            Edge("partial", "port", "first"),
            Edge("trunk", "port", "first"),
            Edge("trunk", "port", "second"),
            Edge("other", "port", "second"),
        ],
    )
    line_feeders = {
        "first": [("partial", 0.0), ("trunk", 10.0)],
        "second": [("trunk", 10.0), ("other", 20.0)],
    }

    station_rank = {sid: rank for rank, sid in enumerate(graph.stations)}
    assert (
        _exit_trunk_feeder(graph, line_feeders, {"first", "second"}, station_rank)
        == "trunk"
    )
