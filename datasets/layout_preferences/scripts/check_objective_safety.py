#!/usr/bin/env python3
"""Check a fitted objective cannot be improved without bound, and demonstrate it.

Two halves, because either alone is weak evidence:

**Structural.** Every feature the objective reads is looked up in
``terms.ADMISSIBILITY``, and every weight that needs pinning for boundedness is
checked against its sign. An objective whose features are all non-negative and
whose weights are all non-negative is bounded below by zero, so no input can
drive it arbitrarily low. This is a proof about the weights, and it is cheap.

**Empirical.** The proof only binds if the classification is right, so the growth
transform is also *run*, through the real engine. ``x_spacing`` and ``y_spacing``
are ordinary knobs -- a CLI flag and a ``%%metro`` directive -- so uniformly
inflating a drawing is a reachable input rather than a thought experiment. Each
fixture is laid out at rising multiples of the default spacing and rescored. An
unsafe objective's score falls as the multiple rises, without limit; a safe one's
cannot fall.

Usage:

    python scripts/check_objective_safety.py --out ../safety_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import extract_features  # noqa: E402
import terms  # noqa: E402

X_SPACING_DEFAULT = 60.0
Y_SPACING_DEFAULT = 40.0
"""The spacing a plain ``nf-metro render`` uses, and the base of the multiples."""

MULTIPLES = (1, 2, 4, 8)
"""Uniform growth factors applied to both spacings.

Near-similar drawings: the same map with every lane pitch and column stride
multiplied, so the extent and length features rise while the topology holds. It
is not *exactly* a similarity transform, because label boxes and station markers
keep their absolute size while the grid around them grows, so the angle terms
drift by a little and a crossing or two can resolve. What isolates the growth
terms is that the extent and length features are the only ones that rise in
proportion.
"""

PROBE_FIXTURES = (
    "examples/rnaseq_sections.mmd",
    "examples/rnaseq_auto.mmd",
    "examples/genomic_pipeline.mmd",
    "examples/topologies/fan_in_merge.mmd",
    "examples/topologies/convergence_fold_diamond.mmd",
)
"""Maps to inflate. A mix of hand-gridded, auto-laid-out and synthetic."""


def load_artifact(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------------- #


def structural_findings(weights: dict[str, float]) -> list[str]:
    """Every reason this weight vector is not bounded below, or an empty list."""
    return [
        f"{key}: {reason}"
        for key, reason in sorted(terms.unbounded_below(weights).items())
    ]


# --------------------------------------------------------------------------- #
# Empirical
# --------------------------------------------------------------------------- #


def vector_at(text: str, multiple: int) -> dict[str, float]:
    """One fixture's feature vector, laid out at ``multiple`` times the spacing."""
    from nf_metro.layout import compute_layout
    from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
    from nf_metro.parser import parse_metro_mermaid

    graph = parse_metro_mermaid(text)
    compute_layout(
        graph,
        x_spacing=X_SPACING_DEFAULT * multiple,
        y_spacing=Y_SPACING_DEFAULT * multiple,
    )
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)
    return extract_features.features(graph, routes)


def score(weights: dict[str, float], vector: dict[str, float]) -> float:
    """Absolute score of one layout: what a minimising search would descend."""
    readable = terms.readable(vector)
    return sum(
        weight * value
        for key, weight in weights.items()
        if (value := readable.get(key)) is not None
    )


def growth_probe(fixtures: tuple[str, ...], arms: dict[str, dict[str, float]]) -> dict:
    """Score every fixture at every growth multiple, under every arm."""
    out: dict[str, dict] = {}
    for rel in fixtures:
        path = REPO / rel
        if not path.exists():
            out[rel] = {"error": "missing"}
            continue
        text = path.read_text()
        rows: dict[int, dict[str, float]] = {}
        try:
            for multiple in MULTIPLES:
                vector = vector_at(text, multiple)
                rows[multiple] = {
                    "bbox_h": vector["bbox_h"],
                    "path_len_per_route": vector["path_len_per_route"],
                    "bends_per_route": vector["bends_per_route"],
                    "crossings": vector["crossings"],
                    **{f"score:{arm}": score(w, vector) for arm, w in arms.items()},
                }
        except Exception as exc:  # noqa: BLE001 - a fixture that will not lay out
            out[rel] = {"error": f"{type(exc).__name__}: {exc}"[:160]}
            continue
        out[rel] = {"rows": rows}
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def report(arms: dict[str, dict[str, float]], probe: dict) -> str:
    out: list[str] = []
    add = out.append

    add("=== structural check ===")
    add("a score whose features are all non-negative and whose weights are all")
    add("non-negative is bounded below by zero, so no input improves it without")
    add("bound. Anything listed under an arm is a way to break that.")
    for name, weights in arms.items():
        findings = structural_findings(weights)
        add(f"-- {name}: {'UNSAFE' if findings else 'SAFE'}")
        for finding in findings:
            add(f"     {finding}")

    add("")
    add("=== growth probe: the same map at rising spacing ===")
    add("x_spacing and y_spacing are a CLI flag and a %%metro directive, so this")
    add("is a reachable input. Every row below is the SAME map with the grid")
    add("multiplied: bends per route hold while extent and path length rise in")
    add("proportion. A score that falls down a column is one a search could")
    add("minimise by inflating the drawing, and nothing about 8x ends the fall --")
    add("the growth terms are linear in the multiple.")
    arm_names = list(arms)
    for rel, record in probe.items():
        add(f"-- {rel}")
        if "error" in record:
            add(f"     skipped: {record['error']}")
            continue
        head = (
            f"{'x':>4}{'bbox_h':>10}{'path/route':>12}{'bends':>8}{'cross':>7}"
            + "".join(f"{'score:' + n:>18}" for n in arm_names)
        )
        add("   " + head)
        first: dict[str, float] = {}
        for multiple, row in record["rows"].items():
            cells = (
                f"{str(multiple) + 'x':>4}{row['bbox_h']:>10.0f}"
                f"{row['path_len_per_route']:>12.0f}{row['bends_per_route']:>8.2f}"
                f"{row['crossings']:>7.0f}"
            )
            for name in arm_names:
                value = row[f"score:{name}"]
                first.setdefault(name, value)
                cells += f"{value:>18.2f}"
            add("   " + cells)
        last = record["rows"][MULTIPLES[-1]]
        moved = " ".join(
            f"{name} {last[f'score:{name}'] - first[name]:+.2f}" for name in arm_names
        )
        add(f"     {MULTIPLES[0]}x -> {MULTIPLES[-1]}x change: {moved}")

    add("")
    add("=== verdict ===")
    for name, weights in arms.items():
        findings = structural_findings(weights)
        moved = [
            rec["rows"][MULTIPLES[-1]][f"score:{name}"]
            - rec["rows"][MULTIPLES[0]][f"score:{name}"]
            for rec in probe.values()
            if "rows" in rec
        ]
        fell = sum(1 for d in moved if d < -1e-9)
        add(
            f"{name:<10}{'UNSAFE' if findings else 'SAFE':<8}"
            f"score fell under growth on {fell}/{len(moved)} fixtures"
        )

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="a weights artifact to check; repeatable. Defaults to the two "
        "committed ones.",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    specs = args.weights or [
        f"iter2={HERE.parent / 'iter2_weights.json'}",
        f"safe={HERE.parent / 'safe_weights.json'}",
    ]
    arms = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        arms[name] = load_artifact(Path(path))["weights"]

    text = report(arms, growth_probe(PROBE_FIXTURES, arms))
    print(text, end="")
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
