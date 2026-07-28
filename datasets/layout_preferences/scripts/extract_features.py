"""Extract geometry feature vectors for every fixture at one engine revision.

Runs INSIDE a worktree parked at some historical SHA, but is itself version-
independent: it touches only the four engine symbols that have been stable
across the project's history (parse, layout, offsets, centred routing) and
derives every feature from raw coordinates. Engine-side detectors are
deliberately not reused, so a feature's definition cannot drift underneath the
dataset as the engine changes.

Geometry is measured on ``route_edges_centred``, the path ``render/svg.py``
draws. ``route_edges`` output is captured alongside it purely to quantify how
far the two disagree.

Emits JSON: {fixture: {feature: value}} plus a per-fixture .mmd content hash so
a preference pair can be restricted to fixtures whose INPUT was identical at
both revisions, isolating engine-caused geometry change from authoring change.

Usage:
    python extract_features.py --worktree DIR --sha SHA --out FILE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

EPS = 1e-6

# Smallest height or width a rendered map can occupy: one station marker plus
# its stroke, from STATION_RADIUS_APPROX (5.0) and STATION_STROKE_APPROX (1.5).
# Held as a literal rather than imported: feature definitions must be identical
# at every replayed revision, so this floor cannot track a constant that the
# engine may retune.
MARKER_EXTENT = 11.5


def seg_len(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def angle_of(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def turn_angle(
    p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
) -> float:
    d = abs(angle_of(q, r) - angle_of(p, q)) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def segments(points: list[tuple[float, float]]) -> list[tuple]:
    return [
        (points[i], points[i + 1])
        for i in range(len(points) - 1)
        if seg_len(points[i], points[i + 1]) > EPS
    ]


def seg_intersect(s1: tuple, s2: tuple) -> bool:
    """Proper crossing only: shared endpoints and collinear overlap excluded."""
    (x1, y1), (x2, y2) = s1
    (x3, y3), (x4, y4) = s2
    for p in s1:
        for q in s2:
            if seg_len(p, q) < 0.5:
                return False
    d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(d) < EPS:
        return False
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
    return 0.01 < t < 0.99 and 0.01 < u < 0.99


def point_seg_dist(p: tuple[float, float], s: tuple) -> float:
    (x1, y1), (x2, y2) = s
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < EPS:
        return seg_len(p, s[0])
    t = max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / L2))
    return math.hypot(p[0] - (x1 + t * dx), p[1] - (y1 + t * dy))


def path_points(route: object) -> list[tuple[float, float]]:
    pts = getattr(route, "points", None) or getattr(route, "waypoints", None) or []
    out = []
    for p in pts:
        if isinstance(p, (tuple, list)) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
        elif hasattr(p, "x") and hasattr(p, "y"):
            out.append((float(p.x), float(p.y)))
    return out


def station_lines(graph: object) -> dict[str, set]:
    """Station -> line ids serving it, derived from edges.

    ``Station`` carries no line membership; the edge list is the only source,
    and its (source, target, line_id) shape is stable across the history being
    replayed.
    """
    out: dict[str, set] = {}
    for e in graph.edges:
        for end in (e.source, e.target):
            out.setdefault(end, set()).add(e.line_id)
    return out


def features(graph: object, routes: list) -> dict[str, float]:
    stations = list(graph.stations.values())
    real = [s for s in stations if not getattr(s, "is_port", False)]
    ports = [s for s in stations if getattr(s, "is_port", False)]
    coords = [
        (float(s.x), float(s.y)) for s in real if s.x is not None and s.y is not None
    ]
    lines_of = station_lines(graph)

    all_segs: list[tuple] = []
    per_path_bends, per_path_turn, detours = [], [], []
    non45 = near_horiz = lone_diag = 0
    total_len = 0.0
    corners_total = 0

    for r in routes:
        pts = path_points(r)
        segs = segments(pts)
        if not segs:
            continue
        for i in range(1, len(pts) - 1):
            if turn_angle(pts[i - 1], pts[i], pts[i + 1]) > math.radians(5):
                corners_total += 1
        all_segs.extend((s, getattr(r, "line_id", None)) for s in segs)
        plen = sum(seg_len(*s) for s in segs)
        total_len += plen
        span = seg_len(pts[0], pts[-1])
        if span > EPS:
            detours.append(plen / span)

        bends = turns = 0.0
        for i in range(1, len(pts) - 1):
            t = turn_angle(pts[i - 1], pts[i], pts[i + 1])
            if t > math.radians(5):
                bends += 1
                turns += t
        per_path_bends.append(bends)
        per_path_turn.append(turns)

        diag_count = 0
        for a, b in segs:
            dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
            if dx > EPS and dy > EPS:
                diag_count += 1
                ratio = dy / dx
                if abs(ratio - 1.0) > 0.08:
                    non45 += 1
                if ratio < 0.2:
                    near_horiz += 1
        if diag_count == 1 and len(segs) > 1:
            lone_diag += 1

    crossings = 0
    for i in range(len(all_segs)):
        s1, l1 = all_segs[i]
        for j in range(i + 1, len(all_segs)):
            s2, l2 = all_segs[j]
            if l1 is not None and l1 == l2:
                continue
            if seg_intersect(s1, s2):
                crossings += 1

    # A line passing through a station that is not on that line: the visual
    # defect of a marker appearing to sit on an unrelated route.
    strikes = 0
    min_marker_gap = float("inf")
    for s in real:
        if s.x is None or s.y is None:
            continue
        p = (float(s.x), float(s.y))
        own = lines_of.get(s.id, set())
        for seg, line in all_segs:
            if line in own:
                continue
            d = point_seg_dist(p, seg)
            min_marker_gap = min(min_marker_gap, d)
            if d < 4.0:
                strikes += 1

    # Extent spans drawn ink: route waypoints as well as station centres, since
    # offset lines on a single grid row occupy height that the station
    # coordinates alone do not express. Floored at the marker's drawn size,
    # because no rendered map is thinner than one station.
    ink = coords + [p for seg, _ in all_segs for p in seg]
    xs = [c[0] for c in ink] or [0.0]
    ys = [c[1] for c in ink] or [0.0]
    w = max(max(xs) - min(xs), MARKER_EXTENT)
    h = max(max(ys) - min(ys), MARKER_EXTENT)
    n_st = max(len(real), 1)
    n_rt = max(len(routes), 1)
    n_seg = max(len(all_segs), 1)
    n_sec_raw = len(getattr(graph, "sections", {}) or {})
    n_sec = max(n_sec_raw, 1)

    return {
        "n_stations": float(len(real)),
        "n_routes": float(len(routes)),
        "n_sections": float(n_sec_raw),
        "bbox_w": w,
        "bbox_h": h,
        "aspect_log": math.log10(w / h),
        "path_len_per_station": total_len / n_st,
        "path_len_per_route": total_len / n_rt,
        "crossings": float(crossings),
        "crossings_per_route": crossings / n_rt,
        "bends_per_route": sum(per_path_bends) / n_rt,
        "turn_angle_per_route": sum(per_path_turn) / n_rt,
        "max_bends_one_route": max(per_path_bends or [0.0]),
        "non_45_segments": float(non45),
        "non_45_frac": non45 / n_seg,
        "near_horizontal": float(near_horiz),
        "near_horizontal_frac": near_horiz / n_seg,
        "lone_diagonals": float(lone_diag),
        "lone_diagonals_per_route": lone_diag / n_rt,
        "detour_mean": sum(detours) / len(detours) if detours else 1.0,
        "detour_max": max(detours or [1.0]),
        "marker_strikes": float(strikes),
        "marker_strikes_per_station": strikes / n_st,
        "min_marker_gap": -1.0 if min_marker_gap == float("inf") else min_marker_gap,
        "corners_total": float(corners_total),
        "stations_per_route": n_st / n_rt,
        "n_ports": float(len(ports)),
        "ports_per_section": len(ports) / n_sec,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", type=Path, required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wt = args.worktree
    sys.path.insert(0, str(wt / "src"))

    from nf_metro.layout import compute_layout
    from nf_metro.parser import parse_metro_mermaid

    try:
        from nf_metro.layout.routing import route_edges
    except ImportError:
        route_edges = None
    try:
        from nf_metro.layout.routing import compute_station_offsets
    except ImportError:
        compute_station_offsets = None

    # `route_edges_centred` is the render path but postdates 2026-06-19; older
    # revisions only have `route_edges`. Which one produced a vector is recorded
    # so pairs straddling the boundary can be excluded rather than silently
    # comparing geometry from two different entrypoints.
    try:
        from nf_metro.layout.routing import route_edges_centred

        router, entrypoint = route_edges_centred, "route_edges_centred"
    except ImportError:
        router, entrypoint = route_edges, "route_edges"
    if router is None:
        args.out.write_text(
            json.dumps({"sha": args.sha, "error": "no_routing_entrypoint"})
        )
        print(f"{args.sha[:9]}  no routing entrypoint", flush=True)
        return

    def route(graph: object) -> tuple[dict, list]:
        if compute_station_offsets is None:
            return {}, router(graph)
        offsets = compute_station_offsets(graph)
        try:
            return offsets, router(graph, station_offsets=offsets)
        except TypeError:
            return offsets, router(graph)

    out: dict[str, dict] = {}
    for mmd in sorted(wt.glob("examples/**/*.mmd")) + sorted(
        wt.glob("tests/fixtures/**/*.mmd")
    ):
        name = mmd.stem
        if name in out:
            continue
        text = mmd.read_text()
        rec: dict = {"input_sha1": hashlib.sha1(text.encode()).hexdigest()[:12]}
        try:
            graph = parse_metro_mermaid(text)
            compute_layout(graph)
            offsets, routes = route(graph)
            rec["features"] = features(graph, routes)
            if route_edges is not None and entrypoint != "route_edges":
                try:
                    rec["render_path_divergence"] = _divergence(
                        routes, route_edges(graph, station_offsets=offsets)
                    )
                except Exception:
                    rec["render_path_divergence"] = None
            rec["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rec["traceback_tail"] = traceback.format_exc()[-300:]
        out[name] = rec

    args.out.write_text(
        json.dumps({"sha": args.sha, "routing_entrypoint": entrypoint, "fixtures": out})
    )
    ok = sum(1 for r in out.values() if r["status"] == "ok")
    print(f"{args.sha[:9]}  {ok}/{len(out)} ok via {entrypoint}", flush=True)


def _divergence(centred: list, plain: list) -> float | None:
    """Mean per-route endpoint displacement between the two routing entrypoints."""
    if not centred or not plain or len(centred) != len(plain):
        return None
    tot = n = 0.0
    for a, b in zip(centred, plain):
        pa, pb = path_points(a), path_points(b)
        if not pa or not pb:
            continue
        tot += sum(seg_len(x, y) for x, y in zip(pa, pb)) / max(
            min(len(pa), len(pb)), 1
        )
        n += 1
    return tot / n if n else None


if __name__ == "__main__":
    main()
