"""SVG generation for metro maps using drawsvg."""

from __future__ import annotations

import drawsvg as draw

from nf_metro.layout.labels import LabelPlacement, place_labels
from nf_metro.layout.routing import RoutedPath, compute_station_offsets, route_edges
from nf_metro.parser.model import MetroGraph
from nf_metro.render.legend import render_legend
from nf_metro.render.style import Theme


def render_svg(
    graph: MetroGraph,
    theme: Theme,
    width: int | None = None,
    height: int | None = None,
    padding: float = 60.0,
) -> str:
    """Render a metro map graph to an SVG string."""
    if not graph.stations:
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    # Filter out port stations for dimension calculation
    visible_stations = [s for s in graph.stations.values() if not s.is_port]
    all_stations_for_bounds = visible_stations if visible_stations else list(graph.stations.values())

    max_x = max(s.x for s in all_stations_for_bounds)
    max_y = max(s.y for s in all_stations_for_bounds)

    # Also consider section bounding boxes
    for section in graph.sections.values():
        if section.bbox_w > 0:
            max_x = max(max_x, section.bbox_x + section.bbox_w)
            max_y = max(max_y, section.bbox_y + section.bbox_h)

    # Reserve space for legend
    legend_height = 24.0 * len(graph.lines) + 40.0 if graph.lines else 0

    auto_width = max_x + padding * 2
    auto_height = max_y + padding * 2 + legend_height + 60

    svg_width = width or int(auto_width)
    svg_height = height or int(auto_height)

    d = draw.Drawing(svg_width, svg_height)

    # Background
    d.append(draw.Rectangle(0, 0, svg_width, svg_height, fill=theme.background_color))

    # Title
    if graph.title:
        d.append(draw.Text(
            graph.title,
            theme.title_font_size,
            padding, 30,
            fill=theme.title_color,
            font_family=theme.label_font_family,
            font_weight="bold",
        ))

    # Sections
    if graph.sections:
        _render_first_class_sections(d, graph, theme)
    elif graph.legacy_sections:
        _render_legacy_sections(d, graph, theme)

    # Route edges
    routes = route_edges(graph)
    station_offsets = compute_station_offsets(graph)

    # Draw edges (lines) behind stations
    _render_edges(d, graph, routes, station_offsets, theme)

    # Draw stations (all circles, skip ports)
    _render_stations(d, graph, theme, station_offsets)

    # Draw labels (horizontal, skip ports)
    labels = place_labels(graph)
    _render_labels(d, labels, theme)

    # Legend
    legend_x = padding
    legend_y = max_y + padding + 30
    render_legend(d, graph, theme, legend_x, legend_y)

    return d.as_svg()


def _render_first_class_sections(
    d: draw.Drawing,
    graph: MetroGraph,
    theme: Theme,
) -> None:
    """Render first-class sections using pre-computed bounding boxes."""
    for section in graph.sections.values():
        if section.bbox_w <= 0 or section.bbox_h <= 0:
            continue

        d.append(draw.Rectangle(
            section.bbox_x, section.bbox_y,
            section.bbox_w, section.bbox_h,
            rx=8, ry=8,
            fill=theme.section_fill,
            stroke=theme.section_stroke,
            stroke_width=1.0,
        ))

        # Numbered circle above the box, left-aligned
        circle_r = 9
        cx = section.bbox_x + circle_r
        cy = section.bbox_y - circle_r - 4

        d.append(draw.Circle(
            cx, cy, circle_r,
            fill=theme.station_stroke,
        ))
        d.append(draw.Text(
            str(section.number),
            9,
            cx, cy,
            fill=theme.station_fill,
            font_family=theme.label_font_family,
            font_weight="bold",
            text_anchor="middle",
            dominant_baseline="central",
        ))

        # Section name to the right of the circle
        d.append(draw.Text(
            section.name,
            theme.section_label_font_size,
            cx + circle_r + 5, cy,
            fill=theme.section_label_color,
            font_family=theme.label_font_family,
            dominant_baseline="central",
        ))


def _section_station_ids(
    graph: MetroGraph,
    section,
) -> set[str]:
    """Find stations belonging to a legacy section via flood-fill within the layer range."""
    from collections import defaultdict, deque

    start = graph.stations.get(section.start_node)
    end = graph.stations.get(section.end_node)
    if not start or not end:
        return set()

    min_layer = min(start.layer, end.layer)
    max_layer = max(start.layer, end.layer)

    adj: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        src = graph.stations.get(edge.source)
        tgt = graph.stations.get(edge.target)
        if src and tgt:
            if min_layer <= src.layer <= max_layer and min_layer <= tgt.layer <= max_layer:
                adj[edge.source].add(edge.target)
                adj[edge.target].add(edge.source)

    visited: set[str] = set()
    queue: deque[str] = deque([section.start_node])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        s = graph.stations.get(node)
        if not s or s.layer < min_layer or s.layer > max_layer:
            continue
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                queue.append(nb)

    return visited


def _render_legacy_sections(
    d: draw.Drawing,
    graph: MetroGraph,
    theme: Theme,
    x_padding: float = 35.0,
    y_padding: float = 35.0,
) -> None:
    """Render legacy section boxes as rounded rectangles around station groups."""
    # First pass: compute all section bounding boxes
    section_boxes: list[tuple] = []
    for section in graph.legacy_sections:
        station_ids = _section_station_ids(graph, section)
        if not station_ids:
            continue

        stations = [graph.stations[sid] for sid in station_ids]

        x_min = min(s.x for s in stations)
        x_max = max(s.x for s in stations)
        y_min = min(s.y for s in stations)
        y_max = max(s.y for s in stations)

        bx = x_min - x_padding
        by = y_min - y_padding
        bw = (x_max - x_min) + x_padding * 2
        bh = (y_max - y_min) + y_padding * 2
        section_boxes.append((section, bx, by, bw, bh))

    # Align top edges of sections with overlapping Y ranges
    if section_boxes:
        station_y_ranges = []
        for section in graph.legacy_sections:
            ids = _section_station_ids(graph, section)
            if ids:
                ys = [graph.stations[sid].y for sid in ids]
                station_y_ranges.append((min(ys), max(ys)))
            else:
                station_y_ranges.append(None)

        n = len(section_boxes)
        group = list(range(n))

        def find(i):
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i

        valid_ranges = [(i, station_y_ranges[i]) for i in range(n) if i < len(station_y_ranges) and station_y_ranges[i]]
        for ai, (a_ymin, a_ymax) in valid_ranges:
            for bi, (b_ymin, b_ymax) in valid_ranges:
                if ai < bi and a_ymin <= b_ymax and b_ymin <= a_ymax:
                    group[find(ai)] = find(bi)

        from collections import defaultdict as _defaultdict
        groups: dict[int, list[int]] = _defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        aligned = list(section_boxes)
        for members in groups.values():
            if len(members) > 1:
                min_top = min(section_boxes[i][2] for i in members)
                for i in members:
                    sec, bx, by, bw, bh = aligned[i]
                    new_bh = bh + (by - min_top)
                    aligned[i] = (sec, bx, min_top, bw, new_bh)

        section_boxes = aligned

    # Second pass: render
    for section, bx, by, bw, bh in section_boxes:
        d.append(draw.Rectangle(
            bx, by, bw, bh,
            rx=8, ry=8,
            fill=theme.section_fill,
            stroke=theme.section_stroke,
            stroke_width=1.0,
        ))

        circle_r = 9
        cx = bx + circle_r
        cy = by - circle_r - 4

        d.append(draw.Circle(
            cx, cy, circle_r,
            fill=theme.station_stroke,
        ))
        d.append(draw.Text(
            str(section.number),
            9,
            cx, cy,
            fill=theme.station_fill,
            font_family=theme.label_font_family,
            font_weight="bold",
            text_anchor="middle",
            dominant_baseline="central",
        ))

        d.append(draw.Text(
            section.name,
            theme.section_label_font_size,
            cx + circle_r + 5, cy,
            fill=theme.section_label_color,
            font_family=theme.label_font_family,
            dominant_baseline="central",
        ))


def _render_edges(
    d: draw.Drawing,
    graph: MetroGraph,
    routes: list[RoutedPath],
    station_offsets: dict[tuple[str, str], float],
    theme: Theme,
    curve_radius: float = 10.0,
) -> None:
    """Render metro line edges with smooth curves at direction changes."""
    for route in routes:
        line = graph.lines.get(route.line_id)
        color = line.color if line else "#888888"

        src_off = station_offsets.get((route.edge.source, route.line_id), 0.0)
        tgt_off = station_offsets.get((route.edge.target, route.line_id), 0.0)

        orig_sy = route.points[0][1]
        orig_ty = route.points[-1][1]
        pts = []
        for i, (x, y) in enumerate(route.points):
            if i == 0:
                pts.append((x, y + src_off))
            elif i == len(route.points) - 1:
                pts.append((x, y + tgt_off))
            elif abs(y - orig_sy) <= abs(y - orig_ty):
                pts.append((x, y + src_off))
            else:
                pts.append((x, y + tgt_off))

        if len(pts) == 2:
            d.append(draw.Line(
                pts[0][0], pts[0][1],
                pts[1][0], pts[1][1],
                stroke=color,
                stroke_width=theme.line_width,
                stroke_linecap="round",
            ))
        elif len(pts) >= 3:
            path = draw.Path(
                stroke=color,
                stroke_width=theme.line_width,
                fill="none",
                stroke_linecap="round",
                stroke_linejoin="round",
            )
            path.M(*pts[0])

            for i in range(1, len(pts) - 1):
                prev = pts[i - 1]
                curr = pts[i]
                nxt = pts[i + 1]

                dx1 = curr[0] - prev[0]
                dy1 = curr[1] - prev[1]
                len1 = (dx1**2 + dy1**2) ** 0.5

                dx2 = nxt[0] - curr[0]
                dy2 = nxt[1] - curr[1]
                len2 = (dx2**2 + dy2**2) ** 0.5

                # Only halve segment length when the adjacent point also
                # has a curve; endpoints (first/last points) never do.
                max_len1 = len1 / 2 if i > 1 else len1
                max_len2 = len2 / 2 if i < len(pts) - 2 else len2
                corner_idx = i - 1
                if route.curve_radii and corner_idx < len(route.curve_radii):
                    effective_r = route.curve_radii[corner_idx]
                else:
                    effective_r = curve_radius
                r = min(effective_r, max_len1, max_len2)

                if len1 > 0 and len2 > 0:
                    before_x = curr[0] - (dx1 / len1) * r
                    before_y = curr[1] - (dy1 / len1) * r
                    after_x = curr[0] + (dx2 / len2) * r
                    after_y = curr[1] + (dy2 / len2) * r

                    path.L(before_x, before_y)
                    path.Q(curr[0], curr[1], after_x, after_y)
                else:
                    path.L(*curr)

            path.L(*pts[-1])
            d.append(path)


def _render_stations(
    d: draw.Drawing,
    graph: MetroGraph,
    theme: Theme,
    station_offsets: dict[tuple[str, str], float] | None = None,
) -> None:
    """Render stations as vertical oblongs (pill shapes).

    Skips port stations (is_port=True).
    """
    for station in graph.stations.values():
        if station.is_port:
            continue

        r = theme.station_radius
        w = r * 2

        if station_offsets:
            line_offsets = [
                station_offsets.get((station.id, lid), 0.0)
                for lid in graph.station_lines(station.id)
            ]
            if line_offsets:
                min_off = min(line_offsets)
                max_off = max(line_offsets)
            else:
                min_off = max_off = 0.0
        else:
            min_off = max_off = 0.0

        span = max_off - min_off
        h = span + r * 2
        cy = station.y + (min_off + max_off) / 2
        d.append(draw.Rectangle(
            station.x - w / 2, cy - h / 2,
            w, h,
            rx=r, ry=r,
            fill=theme.station_fill,
            stroke=theme.station_stroke,
            stroke_width=theme.station_stroke_width,
        ))


def _render_labels(
    d: draw.Drawing,
    labels: list[LabelPlacement],
    theme: Theme,
) -> None:
    """Render horizontal station name labels."""
    for label in labels:
        baseline = "auto" if label.above else "hanging"

        d.append(draw.Text(
            label.text,
            theme.label_font_size,
            label.x, label.y,
            fill=theme.label_color,
            font_family=theme.label_font_family,
            text_anchor="middle",
            dominant_baseline=baseline,
        ))
