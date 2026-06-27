"""Animation support: animated balls traveling along metro lines."""

from __future__ import annotations

__all__ = ["render_animation"]

import math
import re

import drawsvg as draw

from nf_metro.layout.routing import RoutedPath
from nf_metro.layout.routing.common import point_on_polyline
from nf_metro.layout.routing.corners import curve_tangents, resolve_curve_radii
from nf_metro.parser.model import Edge, MetroGraph
from nf_metro.render.constants import (
    ANIMATION_BALL_OPACITY,
    ANIMATION_CURVE_RADIUS,
    EDGE_CONNECT_TOLERANCE,
    MIN_ANIMATION_DURATION,
)
from nf_metro.render.ns import ns
from nf_metro.render.style import Theme
from nf_metro.render.svg import apply_route_offsets

# A line whose travel fraction is within this of a full cycle gets a plain
# two-stop keyframe (no terminus hold), avoiding a degenerate hold of ~0s.
_FULL_CYCLE_EPSILON = 0.001


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
    edges, then emits a <circle> per ball driven along that path by a CSS
    ``offset-path`` animation.

    CSS animation is used rather than SMIL ``<animateMotion>`` because SMIL
    does not run when an SVG is injected into a host page via ``innerHTML``
    (the playground preview, the embeddable inline snippet, any host that
    inlines an exported map): the SMIL timeline advances but the motion is
    never sampled onto the element, freezing every ball at its path start.
    CSS ``offset-path`` animates reliably whether the SVG is opened
    standalone, referenced from ``<img>``, or inlined into a document.
    """
    line_paths = _build_line_motion_paths(
        graph,
        routes,
        station_offsets,
        theme,
        curve_radius,
    )

    # All balls share one cycle (the longest line's at-speed duration) so they
    # stay in sync; a shorter line covers its path in the first part of the
    # cycle and holds at the terminus for the rest (see _travel_keyframes),
    # rather than restarting mid-track while a longer line runs on.
    durations = [
        max(
            _compute_path_length(d_attr) / theme.animation_speed, MIN_ANIMATION_DURATION
        )
        for _, d_attr in line_paths
    ]
    max_dur = max(durations, default=MIN_ANIMATION_DURATION)

    stroke_attr = ""
    if theme.animation_ball_stroke:
        stroke_attr = (
            f' stroke="{theme.animation_ball_stroke}"'
            f' stroke-width="{theme.animation_ball_stroke_width}"'
        )
    ball_prefix = (
        f'<circle r="{theme.animation_ball_radius}" '
        f'fill="{theme.animation_ball_color}" '
        f'opacity="{ANIMATION_BALL_OPACITY}"{stroke_attr} '
    )

    n_balls = theme.animation_balls_per_line
    keyframes: dict[str, str] = {}
    balls: list[str] = []

    for (_, d_attr), natural_dur in zip(line_paths, durations):
        move_frac = min(natural_dur / max_dur, 1.0) if max_dur > 0 else 1.0
        kf_name = _travel_keyframes(move_frac, keyframes)
        for i in range(n_balls):
            delay = -i * max_dur / n_balls
            motion = (
                f"offset-path: path('{d_attr}'); offset-rotate: 0deg; "
                f"animation: {kf_name} {max_dur:.2f}s linear infinite; "
                f"animation-delay: {delay:.2f}s;"
            )
            balls.append(f'{ball_prefix}style="{motion}"/>')

    # @keyframes must precede the inline animations that reference them.
    if keyframes:
        d.append(draw.Raw("<style>" + "".join(keyframes.values()) + "</style>"))
    for ball in balls:
        d.append(draw.Raw(ball))


def _travel_keyframes(move_frac: float, registry: dict[str, str]) -> str:
    """Return the @keyframes name for a ball that travels for *move_frac* of
    the shared cycle then holds at the terminus, registering its CSS in
    *registry*.  Lines sharing a *move_frac* reuse one block; the name is
    namespaced (``ns``) so maps inlined on the same page don't collide.
    """
    name = ns("nfm-travel-" + f"{move_frac:.4f}".replace(".", "_"))
    if name not in registry:
        if move_frac < 1.0 - _FULL_CYCLE_EPSILON:
            registry[name] = (
                f"@keyframes {name}{{"
                f"0%{{offset-distance:0%}}"
                f"{move_frac * 100:.2f}%{{offset-distance:100%}}"
                f"100%{{offset-distance:100%}}}}"
            )
        else:
            registry[name] = (
                f"@keyframes {name}{{"
                f"from{{offset-distance:0%}}to{{offset-distance:100%}}}}"
            )
    return name


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
    # Single offset-applied polyline per route, reused below.
    route_polylines: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for route in routes:
        key = (route.edge.source, route.edge.target, route.line_id)
        route_polylines[key] = apply_route_offsets(route, station_offsets)

    # Group edges by line
    edges_by_line: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        edges_by_line.setdefault(edge.line_id, []).append(edge)

    result: list[tuple[str, str]] = []

    for line_id, edges in edges_by_line.items():
        if line_id not in graph.lines:
            continue

        # Build adjacency: source -> list of (target, edge)
        adj: dict[str, list[tuple[str, Edge]]] = {}
        incoming: set[str] = set()
        for edge in edges:
            adj.setdefault(edge.source, []).append((edge.target, edge))
            incoming.add(edge.target)

        # Find root nodes (no incoming edges for this line)
        all_sources = set(adj.keys())
        roots = all_sources - incoming
        if not roots:
            continue

        # Build edge-disjoint paths: one greedy root-to-sink path
        # first, then short paths for remaining diamond branches.
        # This avoids combinatorial explosion (N diamonds -> 2^N paths)
        # and ensures each edge is traversed by exactly one ball.
        all_paths = _find_edge_disjoint_paths(roots, adj)

        if not all_paths:
            continue

        line_polylines = [
            pts for key, pts in route_polylines.items() if key[2] == line_id
        ]

        for path_edges in all_paths:
            chunks = _chain_edge_points(
                path_edges,
                route_polylines,
                line_polylines,
            )
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                d_attr = _points_to_svg_path(chunk, curve_radius)
                if d_attr:
                    result.append((line_id, d_attr))

    return result


def _find_edge_disjoint_paths(
    roots: set[str],
    adj: dict[str, list[tuple[str, Edge]]],
) -> list[list[Edge]]:
    """Build one full root-to-sink path per unique branch.

    Instead of the cartesian product of all diamonds (which explodes
    combinatorially), this produces:

    1. One canonical path following the first branch at every fork.
    2. For each alternative branch at each fork, one full root-to-sink
       path that diverges only at that specific fork and follows the
       canonical (first) branch everywhere else.

    Result: for N forks with B_i branches each, produces
    1 + sum(B_i - 1) paths instead of product(B_i).
    E.g. 2 binary + 1 ternary fork -> 1+1+1+2 = 5 instead of 12.
    """
    # First find the canonical path (first branch at every fork)
    canonical = _first_path(sorted(roots)[0] if roots else "", adj)
    if not canonical:
        return []

    paths: list[list[Edge]] = [canonical]

    # Build a set of canonical edge choices at each fork for quick lookup
    canonical_set: set[int] = {id(e) for e in canonical}

    # Find fork points: nodes in adj with >1 outgoing edge
    for node, targets in adj.items():
        if len(targets) <= 1:
            continue
        # The canonical path takes targets[0] (first branch).
        # Create a variant path for each alternative branch.
        for alt_target, alt_edge in targets:
            if id(alt_edge) in canonical_set:
                continue
            # Build a full root-to-sink path that follows canonical
            # everywhere except at this fork, where it takes alt_edge.
            variant = _variant_path(
                sorted(roots)[0] if roots else "",
                adj,
                fork_node=node,
                forced_edge=alt_edge,
                forced_target=alt_target,
            )
            if variant:
                paths.append(variant)

    return paths


def _first_path(start: str, adj: dict[str, list[tuple[str, Edge]]]) -> list[Edge]:
    """Follow the first outgoing edge at every node from start to sink."""
    path: list[Edge] = []
    current = start
    visited: set[str] = set()
    while current in adj and current not in visited:
        visited.add(current)
        target, edge = adj[current][0]
        path.append(edge)
        current = target
    return path


def _variant_path(
    start: str,
    adj: dict[str, list[tuple[str, Edge]]],
    fork_node: str,
    forced_edge: Edge,
    forced_target: str,
) -> list[Edge]:
    """Build a root-to-sink path that takes forced_edge at fork_node.

    At every other fork, follows the first (canonical) branch.
    """
    path: list[Edge] = []
    current = start
    visited: set[str] = set()
    while current in adj and current not in visited:
        visited.add(current)
        if current == fork_node:
            path.append(forced_edge)
            current = forced_target
        else:
            target, edge = adj[current][0]
            path.append(edge)
            current = target
    return path


def _chain_edge_points(
    edges: list[Edge],
    route_polylines: dict[tuple[str, str, str], list[tuple[float, float]]],
    line_polylines: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Chain edge route polylines into contiguous waypoint chunks.

    When consecutive edges' route endpoints don't coincide -- a
    merge-junction branch route terminates on the trunk bundle rather
    than at the junction station -- the gap is bridged using
    sibling-route geometry on the same line so the motion path stays
    on rendered geometry instead of cutting an off-piste diagonal.
    The bridge may consume the next edge as well, since the stub from
    merge junction to entry port is often already covered by the trunk.
    """
    chunks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    i = 0
    while i < len(edges):
        edge = edges[i]
        pts = route_polylines.get((edge.source, edge.target, edge.line_id))
        if not pts:
            i += 1
            continue

        if not current:
            current = list(pts)
            i += 1
            continue

        if _points_match(current[-1], pts[0]):
            current.extend(pts[1:])
            i += 1
            continue

        bridge = _find_bridge(current[-1], pts[0], line_polylines)
        if bridge is not None:
            current.extend(bridge[1:])
            current.extend(pts[1:])
            i += 1
            continue

        # Try skipping a stub edge whose geometry the trunk already covered.
        if i + 1 < len(edges):
            n = edges[i + 1]
            next_pts = route_polylines.get((n.source, n.target, n.line_id))
            if next_pts:
                if _points_match(current[-1], next_pts[0]):
                    current.extend(next_pts[1:])
                    i += 2
                    continue
                bridge = _find_bridge(current[-1], next_pts[0], line_polylines)
                if bridge is not None:
                    current.extend(bridge[1:])
                    current.extend(next_pts[1:])
                    i += 2
                    continue

        chunks.append(current)
        current = list(pts)
        i += 1

    if current:
        chunks.append(current)
    return chunks


def _points_match(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return (
        abs(a[0] - b[0]) < EDGE_CONNECT_TOLERANCE
        and abs(a[1] - b[1]) < EDGE_CONNECT_TOLERANCE
    )


def _find_bridge(
    from_pt: tuple[float, float],
    to_pt: tuple[float, float],
    polylines: list[list[tuple[float, float]]],
    tol: float = 2.0,
) -> list[tuple[float, float]] | None:
    """Return a sub-polyline from from_pt to to_pt along a sibling route."""
    for pts in polylines:
        from_loc = point_on_polyline(from_pt, pts, tol)
        if from_loc is None:
            continue
        to_loc = point_on_polyline(to_pt, pts, tol)
        if to_loc is None:
            continue
        from_idx, from_t = from_loc
        to_idx, to_t = to_loc
        if to_idx < from_idx or (to_idx == from_idx and to_t < from_t):
            continue
        bridge: list[tuple[float, float]] = [from_pt]
        for j in range(from_idx + 1, to_idx + 1):
            bridge.append(pts[j])
        bridge.append(to_pt)
        cleaned: list[tuple[float, float]] = [bridge[0]]
        for p in bridge[1:]:
            if not _points_match(cleaned[-1], p):
                cleaned.append(p)
        if len(cleaned) >= 2:
            return cleaned
    return None


def _points_to_svg_path(
    pts: list[tuple[float, float]],
    curve_radius: float = ANIMATION_CURVE_RADIUS,
    route_curve_radii: list[float] | None = None,
) -> str:
    """Convert a list of waypoints to an SVG path 'd' attribute.

    Shares ``resolve_curve_radii`` and ``curve_tangents`` with the static SVG
    renderer so animation corners round identically to the drawn map.
    """
    if len(pts) < 2:
        return ""

    if len(pts) == 2:
        return f"M {pts[0][0]:.2f} {pts[0][1]:.2f} L {pts[1][0]:.2f} {pts[1][1]:.2f}"

    parts = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    resolved = resolve_curve_radii(pts, route_curve_radii, default_radius=curve_radius)

    for tan in curve_tangents(pts, resolved):
        if tan.curved:
            parts.append(
                f"L {tan.before[0]:.2f} {tan.before[1]:.2f} "
                f"Q {tan.corner[0]:.2f} {tan.corner[1]:.2f} "
                f"{tan.after[0]:.2f} {tan.after[1]:.2f}"
            )
        else:
            parts.append(f"L {tan.corner[0]:.2f} {tan.corner[1]:.2f}")

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
