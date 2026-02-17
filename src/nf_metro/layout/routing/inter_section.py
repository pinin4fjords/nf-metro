"""Simplified inter-section edge routing (test-only utility).

This module provides a simpler routing function that only handles edges
crossing section boundaries, without the full bundle offset logic of
route_edges(). Used primarily in tests.
"""

from __future__ import annotations

from nf_metro.layout.constants import (
    COORD_TOLERANCE,
    COORD_TOLERANCE_FINE,
    CURVE_RADIUS,
    DIAGONAL_RUN,
    MIN_STRAIGHT_INTER,
)
from nf_metro.layout.routing.common import RoutedPath
from nf_metro.parser.model import MetroGraph


def route_inter_section_edges(
    graph: MetroGraph,
    diagonal_run: float = DIAGONAL_RUN,
    curve_radius: float = CURVE_RADIUS,
) -> list[RoutedPath]:
    """Route edges that cross section boundaries (port-to-port).

    These are edges where both source and target are ports (or one is a port).
    Uses the same routing style as regular edges.
    """
    routes: list[RoutedPath] = []

    for edge in graph.edges:
        src = graph.stations.get(edge.source)
        tgt = graph.stations.get(edge.target)
        if not src or not tgt:
            continue

        # Only route edges that cross section boundaries:
        # - Both endpoints in different sections, OR
        # - One endpoint is a junction (section_id=None, e.g. divergence point)
        src_sec = src.section_id
        tgt_sec = tgt.section_id
        is_junction_edge = src_sec is None or tgt_sec is None
        if not is_junction_edge and (src_sec == tgt_sec):
            continue

        sx, sy = src.x, src.y
        tx, ty = tgt.x, tgt.y
        dx = tx - sx
        dy = ty - sy

        if abs(dy) < COORD_TOLERANCE_FINE:
            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[(sx, sy), (tx, ty)],
                )
            )
        elif abs(dx) < COORD_TOLERANCE:
            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[(sx, sy), (tx, ty)],
                )
            )
        else:
            sign = 1.0 if dx > 0 else -1.0
            half_diag = diagonal_run / 2

            mid_x = (sx + tx) / 2
            diag_start_x = mid_x - sign * half_diag
            diag_end_x = mid_x + sign * half_diag

            # Ensure minimum straight track at each station so the
            # station sits on a visible horizontal segment, not a curve.
            min_straight = curve_radius + MIN_STRAIGHT_INTER
            if sign > 0:
                diag_start_x = max(diag_start_x, sx + min_straight)
                diag_end_x = min(diag_end_x, tx - min_straight)
            else:
                diag_start_x = min(diag_start_x, sx - min_straight)
                diag_end_x = max(diag_end_x, tx + min_straight)

            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[
                        (sx, sy),
                        (diag_start_x, sy),
                        (diag_end_x, ty),
                        (tx, ty),
                    ],
                )
            )

    return routes
