"""Folded LEFT-exit -> RIGHT-entry run into a relocated side-stack.

When ``fold_threshold`` relocates a downstream section to a row below its
feeder, an inter-section bundle leaves the feeder's LEFT exit port and runs
west into the relocated section's RIGHT entry port.  Two things must hold:

* **Bundle order is preserved across the run.** The exit and entry ports stack
  the bundle in the same order, so no line crosses a bundle-mate.  The order is
  set upstream: a reconvergence section fed by a single multi-line feeder whose
  lines originate at *separate* single-line producers has no well-defined
  delivered order (the producers each sit on a local slot 0, so two lines
  collide on one offset), and settling the section on that ambiguous order
  desynchronises its exit port from the relocated section's entry port.

* **The exit aligns to the target's settled entry Y.** When the relocated
  target spans several sub-rows its entry Y keeps descending as those sub-rows
  settle, after the exit was first aligned to it.  The exit follows the entry
  down so the inter-section run stays straight rather than ending a sub-row
  above it (a jog).

``fold_left_exit_right_entry`` is the committed minimal fixture (its
``fold_threshold`` directive bakes the relocation in).  ``epitopeprediction``
at fold 7 -- a real nf-core pipeline whose Reporting section relocates below a
multi-sub-row Binding Prediction -- is the motivating case.
"""

from __future__ import annotations

from nf_metro import api
from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
from nf_metro.layout.routing.invariants import (
    assert_render_curve_invariants,
    check_bundle_order_preserved,
    check_seam_segments_meet_at_port,
)

FIXTURE = "examples/topologies/fold_left_exit_right_entry.mmd"


def _route(text: str):
    graph = api.prepare_graph(text)
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)
    return graph, routes, offsets


def _route_fixture():
    return _route(open(FIXTURE).read())


def test_bundle_order_preserved() -> None:
    _graph, routes, _offsets = _route_fixture()
    flips = check_bundle_order_preserved(routes)
    assert not flips, "\n".join(v.message() for v in flips)


def test_seams_meet_at_port() -> None:
    graph, routes, offsets = _route_fixture()
    gaps = check_seam_segments_meet_at_port(graph, routes, offsets)
    assert not gaps, "\n".join(g.message() for g in gaps)


def test_render_passes_curve_self_check() -> None:
    graph, routes, offsets = _route_fixture()
    assert_render_curve_invariants(graph, routes, offsets)


def test_exit_aligns_to_relocated_target_entry() -> None:
    """The fold relocates ``report`` to a lower row and the exit follows it.

    ``report`` sits in a row below ``middle`` (so the alignment is not
    vacuously satisfied by a same-row connector), and the LEFT exit ends at the
    RIGHT entry's settled Y so the run is straight.
    """
    graph, _routes, _offsets = _route_fixture()
    exit_port = graph.stations.get("middle__exit_left_1")
    entry_port = graph.stations.get("report__entry_right_3")
    assert exit_port is not None and entry_port is not None
    assert graph.sections["report"].grid_row > graph.sections["middle"].grid_row, (
        "report did not relocate to a lower row"
    )
    assert abs(entry_port.y - exit_port.y) < 1.0, (
        f"exit y={exit_port.y:.1f} != entry y={entry_port.y:.1f}: the "
        f"inter-section run is not straight"
    )


def test_motivating_epitopeprediction_is_clean() -> None:
    """The real motivating case: ``epitopeprediction`` at the fold that relocates
    one of its three sections renders its binding_prediction -> reporting run
    straight, in bundle order, with no curve defect."""
    graph = api.prepare_graph(
        open("examples/epitopeprediction.mmd").read(),
        layout_options={"fold_threshold": 7},
    )
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)

    assert_render_curve_invariants(graph, routes, offsets)

    lines = ("vcf", "protein", "peptide")

    def order(port: str) -> list[str]:
        return sorted(lines, key=lambda lid: offsets[(port, lid)])

    assert order("binding_prediction__exit_left_1") == order(
        "reporting__entry_right_3"
    ), "exit and entry ports must stack the bundle in the same order"

    exit_port = graph.stations["binding_prediction__exit_left_1"]
    entry_port = graph.stations["reporting__entry_right_3"]
    assert abs(entry_port.y - exit_port.y) < 1.0, (
        f"exit y={exit_port.y:.1f} != entry y={entry_port.y:.1f}: the "
        f"binding_prediction -> reporting run is not straight"
    )
