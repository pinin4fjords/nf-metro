"""Perpendicular port-crossing geometry: the shared lateral conventions for a
line crossing a TOP/BOTTOM section boundary.

A "perpendicular crossing" is a line leaving or entering a section through a
TOP or BOTTOM port: it rises out of a horizontal-flow section into the
inter-row corridor, or drops into a TB section's trunk.  The two ends of that
shape are routed in different handler families - the exit (up-and-over) in
``inter_section_handlers._route_perp_exit_over`` and the entry drop in
``tb_handlers._route_perp_entry`` / ``_route_perp_entry_from_corridor`` - but
they must seat their bundles on the *same* per-line lateral so the legs stay
parallel across the shared port.  Those conventions live here so they can be
read and verified together.

TOP vs BOTTOM sign convention
-----------------------------
The per-line lateral order flips between rising and dropping.  A TOP riser
keeps the raw per-line offset (``_get_offset``); a BOTTOM riser reflects it
(``reversed_offset``).  ``_perp_riser_lateral`` is the single source of that rule,
so both the up-and-over exit corridor and the matching entry drop pick up the
identical lateral for a given line and side.

``_perp_entry_crossing_x`` expresses the matching X for the *aligned* case: the
single X at which a bundled inter-section feeder and its intra-section drop both
cross the port, so the line passes straight through the boundary instead of
converging on the marker and re-fanning.  It fans to the side the feeder
approaches from (``_feeder_fan_sign``) so the crossing tracks a ``-x`` downward
(TB) feeder and its ``+x`` upward (BT) image alike.
"""

from __future__ import annotations

from nf_metro.layout.geometry import lanes_run_along_x
from nf_metro.layout.routing.context import (
    _get_offset,
    _max_offset_at,
    _RoutingCtx,
    _section_lane_frame,
)
from nf_metro.layout.routing.corners import reversed_offset
from nf_metro.parser.model import (
    PortSide,
)


def _feeder_fan_sign(ctx: _RoutingCtx, source_id: str) -> float:
    """Lateral fan sign of the inter-section feeder leaving *source_id*.

    A vertical-flow (TB/BT) feeder rides its section's lane, so it fans to the
    side its :attr:`~AxisFrame.secondary_sign` names: a downward TB feeder to
    ``-x``, its upward BT image to ``+x``.  A horizontal feeder reaching the
    drop instead arrives via the up-and-over riser, whose reference leg fans to
    ``-x`` (the legacy left-fan).
    """
    station = ctx.graph.stations.get(source_id)
    section = (
        ctx.graph.sections.get(station.section_id)
        if station and station.section_id
        else None
    )
    if section is not None and lanes_run_along_x(section.direction):
        return _section_lane_frame(ctx.graph, section, ctx.positive_fan).secondary_sign
    return -1.0


def _perp_entry_crossing_x(
    ctx: _RoutingCtx, entry_port_id: str, line_id: str, port_x: float
) -> float | None:
    """Per-line X at which *line_id* crosses a TOP/BOTTOM entry port.

    The inter-section approach lands, and the intra-section drop departs, at
    this one X so the line passes straight through the boundary rather than
    converging on the port marker and re-fanning.

    A vertical-flow (TB/BT) feeder drops on its section lane -- the exact X
    :func:`inter_section_handlers._route_tb_bottom_exit` lands at,
    ``port_x + _tb_x_offset(feeder, line)`` -- so the crossing tracks that lane.
    The per-line lane width (one offset step per distinct line) is narrower than
    the feeder's index in the *whole* cross-boundary bundle, which counts every
    converging feeder's every line: anchoring on the bundle index instead would
    splay the few descending lines across the wider fan and straddle the target
    station, pinching the drop's turn-in corner.

    For any other feeder the crossing falls back to the marker X offset by the
    feeder's bundle index times the offset step, fanned to the side the feeder
    approaches from (:func:`_feeder_fan_sign`).  Returns ``None`` when no bundled
    inter-section feeder reaches the port for this line (nothing to align to).
    """
    feeders = [
        (info[0], edge.source)
        for edge in ctx.graph.edges_to(entry_port_id)
        if edge.line_id == line_id
        and (info := ctx.bundle_info.get((edge.source, entry_port_id, line_id)))
        is not None
    ]
    if not feeders:
        return None
    max_index, source = max(feeders)
    feeder_st = ctx.graph.stations.get(source)
    feeder_sec = (
        ctx.graph.sections.get(feeder_st.section_id)
        if feeder_st and feeder_st.section_id
        else None
    )
    if feeder_sec is not None and lanes_run_along_x(feeder_sec.direction):
        # The feeder's own per-line lane width (one step per distinct line),
        # equivalent to _tb_x_offset once the lane sign is applied below.
        fan = _get_offset(ctx, source, line_id)
    else:
        fan = max_index * ctx.offset_step
    return port_x + _feeder_fan_sign(ctx, source) * fan


def _perp_riser_lateral(
    ctx: _RoutingCtx,
    station_id: str,
    line_id: str,
    side: PortSide,
    section_id: str | None,
) -> float:
    """Per-line lateral X continuing a perpendicular riser's convention.

    A TOP riser keeps the raw per-line offset; a BOTTOM riser reflects it via
    ``reversed_offset`` (the lateral order flips between rising and dropping).  Both the
    up-and-over exit corridor and the matching entry drop seat their bundle
    with this lateral so the two legs stay parallel across the shared port.
    """
    if side == PortSide.TOP:
        return _get_offset(ctx, station_id, line_id)
    off = _get_offset(ctx, station_id, line_id)
    return reversed_offset(off, _max_offset_at(ctx, station_id))
