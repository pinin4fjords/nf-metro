#!/usr/bin/env python3
"""Corpus-wide check: live-render features vs extract_features.py's own pass.

``scored_objective.learned_features`` (the render-diff's shadow-mode scorer)
reads a render's actual *drawn* geometry -- station offsets applied, and
label-growth-adjusted extents when ``graph.rendered_geometry`` has been
published. ``extract_features.features`` as run by ``replay.py`` across
historical revisions resolves its own routing pass via
``route_edges_centred``, which settles marker centring but never applies
per-station bundle offsets, so two lines sharing a lane are not drawn as
parallel, separated tracks in the coordinates it measures.

The two were never guaranteed identical; this quantifies the gap across the
whole fixture corpus before the render-diff's numbers are read as "the score
iter2 was fitted against."

Usage:
    python check_feature_parity.py [--out FILE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(HERE))

import extract_features  # noqa: E402
import scored_objective  # noqa: E402
from layout_metrics import drawn_polylines, measured_geometry  # noqa: E402

from nf_metro.layout import compute_layout  # noqa: E402
from nf_metro.layout.phases._common import _restoring_layout_geometry  # noqa: E402
from nf_metro.layout.routing import (  # noqa: E402
    compute_station_offsets,
    route_edges_centred,
)
from nf_metro.parser import parse_metro_mermaid  # noqa: E402


def legacy_features(graph: object) -> dict[str, float]:
    """What ``extract_features.py`` would have measured for this graph today.

    Mirrors its ``route()`` closure exactly: ``compute_station_offsets`` then
    ``route_edges_centred(graph, station_offsets=offsets)``. Wrapped in
    ``_restoring_layout_geometry`` so the bubble-centring mutation this
    entrypoint applies to ``graph.stations`` does not leak into the
    ``live_features`` measurement of the same graph object.
    """
    with _restoring_layout_geometry(graph):
        offsets = compute_station_offsets(graph)
        try:
            routes = route_edges_centred(graph, station_offsets=offsets)
        except TypeError:
            routes = route_edges_centred(graph)
        return extract_features.features(graph, routes)


def live_features(graph: object) -> dict[str, float]:
    """What the render-diff's shadow-mode scorer reports for this graph."""
    offsets, routes = measured_geometry(graph)
    polylines = drawn_polylines(routes, offsets)
    return scored_objective.learned_features(graph, polylines)


def _fixtures() -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for mmd in sorted(ROOT.glob("examples/**/*.mmd")) + sorted(
        ROOT.glob("tests/fixtures/**/*.mmd")
    ):
        if mmd.stem in seen:
            continue
        seen.add(mmd.stem)
        out.append(mmd)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    keys = scored_objective.feature_keys()
    ok: list[tuple[str, dict[str, tuple[float | None, float | None]]]] = []
    errors: list[tuple[str, str]] = []

    for mmd in _fixtures():
        try:
            graph = parse_metro_mermaid(mmd.read_text())
            compute_layout(graph)
            legacy = legacy_features(graph)
            live = live_features(graph)
        except Exception as exc:  # noqa: BLE001 - a bad fixture is a data point
            errors.append((mmd.stem, f"{type(exc).__name__}: {exc}"))
            continue
        ok.append((mmd.stem, {k: (legacy.get(k), live.get(k)) for k in keys}))

    lines: list[str] = []
    lines.append(
        f"fixtures: {len(ok) + len(errors)}  ok: {len(ok)}  errored: {len(errors)}"
    )
    for stem, err in errors:
        lines.append(f"  ERROR {stem}: {err}")

    lines.append("")
    lines.append(
        "per-feature absolute divergence (legacy vs live), over fixtures "
        "where both computed"
    )
    lines.append(f"{'feature':<28}{'max |diff|':>12}{'mean |diff|':>12}{'n moved':>10}")
    for key in keys:
        diffs = [
            abs(a - b)
            for _stem, feats in ok
            for a, b in (feats[key],)
            if a is not None and b is not None
        ]
        if not diffs:
            lines.append(f"{key:<28}{'n/a':>12}{'n/a':>12}{'0':>10}")
            continue
        moved = sum(1 for d in diffs if d > 1e-6)
        lines.append(
            f"{key:<28}{max(diffs):>12.3f}{(sum(diffs) / len(diffs)):>12.3f}{moved:>10}"
        )

    lines.append("")
    lines.append("worst 10 fixtures by total absolute divergence across the 8 features")
    totals = sorted(
        (
            (
                sum(
                    abs(a - b)
                    for a, b in feats.values()
                    if a is not None and b is not None
                ),
                stem,
            )
            for stem, feats in ok
        ),
        reverse=True,
    )
    for total, stem in totals[:10]:
        lines.append(f"  {stem:<40}{total:>10.3f}")

    weights = scored_objective.load_weights()["weights"]
    lines.append("")
    lines.append(
        "weighted score divergence per fixture: |sum(weight * (legacy - live))|, "
        "i.e. how much the reported delta would be thrown off by using one "
        "geometry source over the other on a self-pair"
    )
    weighted = sorted(
        (
            (
                abs(
                    sum(
                        w * (feats[k][0] - feats[k][1])
                        for k, w in weights.items()
                        if feats[k][0] is not None and feats[k][1] is not None
                    )
                ),
                stem,
            )
            for stem, feats in ok
        ),
        reverse=True,
    )
    lines.append(f"max: {weighted[0][0]:.3f} ({weighted[0][1]})")
    lines.append(f"mean: {sum(w for w, _ in weighted) / len(weighted):.3f}")
    lines.append("worst 10:")
    for total, stem in weighted[:10]:
        lines.append(f"  {stem:<40}{total:>10.3f}")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
