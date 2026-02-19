"""Animation support: animated balls traveling along metro lines."""

from __future__ import annotations

__all__ = ["render_animation"]

import math
import re

import drawsvg as draw

from nf_metro.layout.routing import RoutedPath
from nf_metro.parser.model import MetroGraph
from nf_metro.render.constants import (
    ANIMATION_CURVE_RADIUS,
    EDGE_CONNECT_TOLERANCE,
    MIN_ANIMATION_DURATION,
)
from nf_metro.render.style import Theme
from nf_metro.render.svg import apply_route_offsets


def render_animation(
    d: draw.Drawing,
    graph: MetroGraph,
    routes: list[RoutedPath],
    station_offsets: dict[tuple[str, str], float],
    theme: Theme,
    curve_radius: float = ANIMATION_CURVE_RADIUS,
) -> None:
    """Add animated balls traveling along each metro line.

    For each metro line, builds a continuous SVG path from its chained
    edges, then injects invisible <path> elements and <circle> elements
    with <animateMotion> to create the traveling ball effect.
    """
    line_paths = _build_line_motion_paths(
        graph,
        routes,
        station_offsets,
        theme,
        curve_radius,
    )

    for idx, (line_id, d_attr) in enumerate(line_paths):
        path_id = f"motion-path-{line_id}-{idx}"

        # Invisible path for animateMotion to follow
        d.append(
            draw.Raw(f'<path id="{path_id}" d="{d_attr}" fill="none" stroke="none"/>')
        )

        # Compute duration from approximate path length
        path_length = _compute_path_length(d_attr)
        dur = max(path_length / theme.animation_speed, MIN_ANIMATION_DURATION)

        n_balls = theme.animation_balls_per_line
        for i in range(n_balls):
            begin_offset = -i * dur / n_balls
            d.append(
                draw.Raw(
                    f'<circle r="{theme.animation_ball_radius}" '
                    f'fill="{theme.animation_ball_color}" opacity="0.9">'
                    f'<animateMotion dur="{dur:.2f}s" '
                    f'repeatCount="indefinite" '
                    f'begin="{begin_offset:.2f}s">'
                    f'<mpath href="#{path_id}"/>'
                    f"</animateMotion>"
                    f"</circle>"
                )
            )


def _build_line_motion_paths(
    graph: MetroGraph,
    routes: list[RoutedPath],
    station_offsets: dict[tuple[str, str], float],
    theme: Theme,
    curve_radius: float = ANIMATION_CURVE_RADIUS,
) -> list[tuple[str, str]]:
    """Build continuous SVG motion paths for each metro line.

    At diamond/bubble patterns (fork-join), produces separate paths for
    each branch so balls travel both alternatives (e.g., FastP and
    TrimGalore). Returns list of (line_id, d_attr) pairs -- a line_id
    may appear multiple times when it has forking branches.
    """
    # Index routes by (source, target, line_id) for lookup
    route_by_edge: dict[tuple[str, str, str], RoutedPath] = {}
    for route in routes:
        key = (route.edge.source, route.edge.target, route.line_id)
        route_by_edge[key] = route

    # Group edges by line
    edges_by_line: dict[str, list] = {}
    for edge in graph.edges:
        edges_by_line.setdefault(edge.line_id, []).append(edge)

    result: list[tuple[str, str]] = []

    for line_id, edges in edges_by_line.items():
        if line_id not in graph.lines:
            continue

        # Build adjacency: source -> list of (target, edge)
        adj: dict[str, list] = {}
        incoming: set[str] = set()
        for edge in edges:
            adj.setdefault(edge.source, []).append((edge.target, edge))
            incoming.add(edge.target)

        # Find root nodes (no incoming edges for this line)
        all_sources = set(adj.keys())
        roots = all_sources - incoming
        if not roots:
            continue

        # Find all distinct root-to-sink paths (covers both branches
        # of diamonds/bubbles)
        all_paths: list[list] = []
        for root in sorted(roots):
            _find_all_paths(root, adj, [], all_paths)

        if not all_paths:
            continue

        for path_edges in all_paths:
            all_points = _chain_edge_points(
                path_edges,
                route_by_edge,
                station_offsets,
            )
            if len(all_points) < 2:
                continue

            d_attr = _points_to_svg_path(all_points, curve_radius)
            if d_attr:
                result.append((line_id, d_attr))

    return result


def _find_all_paths(
    current: str,
    adj: dict[str, list],
    path_so_far: list,
    results: list[list],
) -> None:
    """DFS to find all root-to-sink paths through the adjacency map."""
    if current not in adj:
        # Sink node: save the accumulated path
        if path_so_far:
            results.append(list(path_so_far))
        return

    for target, edge in adj[current]:
        path_so_far.append(edge)
        _find_all_paths(target, adj, path_so_far, results)
        path_so_far.pop()


def _chain_edge_points(
    edges: list,
    route_by_edge: dict[tuple[str, str, str], RoutedPath],
    station_offsets: dict[tuple[str, str], float],
) -> list[tuple[float, float]]:
    """Chain edge routes into one continuous list of waypoints."""
    all_points: list[tuple[float, float]] = []

    for edge in edges:
        route = route_by_edge.get(
            (edge.source, edge.target, edge.line_id),
        )
        if not route:
            continue

        pts = apply_route_offsets(route, station_offsets)

        if not all_points:
            all_points.extend(pts)
        elif pts:
            last = all_points[-1]
            first = pts[0]
            if (
                abs(last[0] - first[0]) < EDGE_CONNECT_TOLERANCE
                and abs(last[1] - first[1]) < EDGE_CONNECT_TOLERANCE
            ):
                all_points.extend(pts[1:])
            else:
                all_points.extend(pts)

    return all_points


def _points_to_svg_path(
    pts: list[tuple[float, float]],
    curve_radius: float = ANIMATION_CURVE_RADIUS,
    route_curve_radii: list[float] | None = None,
) -> str:
    """Convert a list of waypoints to an SVG path 'd' attribute.

    Replicates the curve logic from _render_edges in svg.py:
    straight lines with quadratic Bezier curves at direction changes.
    """
    if len(pts) < 2:
        return ""

    if len(pts) == 2:
        return f"M {pts[0][0]:.2f} {pts[0][1]:.2f} L {pts[1][0]:.2f} {pts[1][1]:.2f}"

    parts = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]

    for i in range(1, len(pts) - 1):
        prev = pts[i - 1]
        curr = pts[i]
        nxt = pts[i + 1]

        dx1 = curr[0] - prev[0]
        dy1 = curr[1] - prev[1]
        len1 = math.hypot(dx1, dy1)

        dx2 = nxt[0] - curr[0]
        dy2 = nxt[1] - curr[1]
        len2 = math.hypot(dx2, dy2)

        max_len1 = len1 / 2 if i > 1 else len1
        max_len2 = len2 / 2 if i < len(pts) - 2 else len2

        effective_r = curve_radius
        r = min(effective_r, max_len1, max_len2)

        if len1 > 0 and len2 > 0:
            before_x = curr[0] - (dx1 / len1) * r
            before_y = curr[1] - (dy1 / len1) * r
            after_x = curr[0] + (dx2 / len2) * r
            after_y = curr[1] + (dy2 / len2) * r

            parts.append(
                f"L {before_x:.2f} {before_y:.2f} "
                f"Q {curr[0]:.2f} {curr[1]:.2f} {after_x:.2f} {after_y:.2f}"
            )
        else:
            parts.append(f"L {curr[0]:.2f} {curr[1]:.2f}")

    parts.append(f"L {pts[-1][0]:.2f} {pts[-1][1]:.2f}")

    return " ".join(parts)


def _compute_path_length(d_attr: str) -> float:
    """Approximate the length of an SVG path from its commands.

    Parses M, L, and Q commands and sums segment lengths.
    For Q (quadratic Bezier), approximates with the chord length.
    """
    # Extract all numbers from the path
    tokens = re.findall(r"[MLQ]|[-+]?\d*\.?\d+", d_attr)

    total = 0.0
    cx, cy = 0.0, 0.0  # current position
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token == "M":
            cx = float(tokens[i + 1])
            cy = float(tokens[i + 2])
            i += 3
        elif token == "L":
            nx = float(tokens[i + 1])
            ny = float(tokens[i + 2])
            total += math.hypot(nx - cx, ny - cy)
            cx, cy = nx, ny
            i += 3
        elif token == "Q":
            # Q cx cy ex ey - approximate with control point polygon
            qcx = float(tokens[i + 1])
            qcy = float(tokens[i + 2])
            ex = float(tokens[i + 3])
            ey = float(tokens[i + 4])
            # Sum of legs through control point (overestimates slightly)
            leg1 = math.hypot(qcx - cx, qcy - cy)
            leg2 = math.hypot(ex - qcx, ey - qcy)
            chord = math.hypot(ex - cx, ey - cy)
            # Average of chord and polygon for a decent approximation
            total += (chord + leg1 + leg2) / 2
            cx, cy = ex, ey
            i += 5
        else:
            i += 1

    return total
