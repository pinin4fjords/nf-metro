"""Regression: a true ``direction: RL`` serpentine fold of a multi-line bundle.

Row 0 flows L->R, the corner section drops straight down into the section below,
and the return row flows R->L via ``%%metro direction: RL``.  The fold corner is a
TOP entry port turning into the first station of a horizontal-flow (RL) section.
For a 2+-line bundle that descend->turn corner must nest concentrically and keep
bundle order; the inter-section feeder must land each line at the same per-line X
the intra drop departs from (no boundary reversal).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing.core import route_edges_centred
from nf_metro.layout.routing.invariants import (
    assert_render_curve_invariants,
    check_bundle_order_preserved,
    check_concentric_bundle_corners,
)
from nf_metro.layout.routing.offsets import compute_station_offsets
from nf_metro.parser import parse_metro_mermaid

FIXTURE = (
    Path(__file__).parent.parent
    / "examples"
    / "topologies"
    / "serpentine_rl_bundle.mmd"
)


def _laid_out():
    graph = parse_metro_mermaid(FIXTURE.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)
    return graph, routes, offsets


def test_rl_serpentine_bundle_renders_without_abort():
    """The fold corner must not raise from the render curve invariants."""
    graph, routes, offsets = _laid_out()
    assert_render_curve_invariants(graph, routes, offsets)


def test_rl_serpentine_fold_corner_is_concentric():
    """The descend->turn corner's two lines nest with a shared arc centre."""
    graph, routes, offsets = _laid_out()
    violations = check_concentric_bundle_corners(graph, routes, offsets)
    assert not violations, "\n".join(v.message() for v in violations)


def test_rl_serpentine_fold_corner_keeps_bundle_order():
    """The bundle keeps its order through the descend->turn corner."""
    _graph, routes, _offsets = _laid_out()
    violations = check_bundle_order_preserved(routes)
    assert not violations, "\n".join(v.message() for v in violations)
