"""Auto-layout minimiser: search section arrangements that lower the quality metrics.

Holds a pipeline's ``.mmd`` content fixed and explores the levers auto-layout
exposes to an author who writes NO explicit ``%%metro grid:``:

  * ``fold_threshold`` (row-wrap width / ``max_station_columns``)
  * within-topo-column section row order (== subgraph declaration order)
  * per-section ``direction`` (LR vs TB)
  * ``center_ports``

For each candidate it re-parses (auto-layout runs at parse time) and scores the
laid-out graph with the same metrics the CI render-diff reports
(``tests/layout_metrics.py``). It reports the best-scoring arrangement against
the pure-auto-layout baseline and the directives that reproduce it.

IMPORTANT: the metrics are a proxy, not ground truth, and the weights over them
are binned measurements rather than a fitted model. A lower weighted score is
evidence for an arrangement, not proof of one -- always eyeball a suggestion
before adopting it. ``excessive_gaps`` in particular fires on the healthy
vertical gap between two parallel processing tracks, which is why it carries
almost no weight.

Usage:
    python scripts/optimize_layout.py examples/genomic_pipeline.mmd [more.mmd ...]
    python scripts/optimize_layout.py --all
    python scripts/optimize_layout.py            # the complex real pipelines
"""

from __future__ import annotations

import itertools
import re
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
warnings.filterwarnings("ignore")

from layout_metrics import compute_metrics  # noqa: E402

from nf_metro.layout import compute_layout  # noqa: E402
from nf_metro.layout.constants import Y_SPACING  # noqa: E402
from nf_metro.parser import parse_metro_mermaid  # noqa: E402

# The bins encode three distinct states of knowledge, and conflating them is
# what made an earlier version of this objective wrong: measured-with-signal
# outranks not-measured, which outranks measured-without-signal. Plausibility on
# its own buys nothing. Agreement is the share of the fixtures a metric moves on
# where it moves the same way as human layout judgement, over the preference
# pairs in `datasets/layout_preferences` (fixture-grouped, since a raw count
# lets one repetitive map speak for the corpus).
WEIGHTS = {
    # Measured, and agree with the judgement on ~95% of the fixtures they move.
    "single_diagonals": 3.0,
    "bends_per_route": 3.0,
    # Measured at ~75%.
    "turn_angle_per_route": 2.0,
    # Invisible to the instrument, not absent from renders: a strike needs
    # render-time label boxes the replayed pairs never captured. Frequent and
    # complained about, and a line through a label is a defect nobody argues
    # about, so it is weighted on that rather than on a measurement.
    "label_strikes": 2.0,
    # Visible to the instrument and almost never happens, because ERROR-level
    # guards reject it upstream: it has a non-zero delta on 3 of 192 pairs, too
    # few to measure. Weighted below the measured terms and kept non-zero as
    # cover for a future change that starts producing it.
    "marker_crowding": 2.0,
    # Too rare in the corpus to measure at all (`near_horizontal` moves on two
    # pairs; `wasted_canvas` needs a canvas size the pairs lack).
    "near_horizontal": 0.5,
    "wasted_canvas": 0.5,
    # Measured, with the largest sample of any term here (`crossings` moves on
    # 42 pairs across 25 fixtures) and no signal to show for it: 44.9%
    # agreement. Read that as a coin flip with a wobble, NOT as evidence that
    # crossings are desirable: a negative weight would instruct the search to
    # add them, which is the failure mode that keeps this objective out of the
    # optimiser.
    "crossings": 0.25,
    # Measured, and points the wrong way for a knowable reason: it fires on the
    # healthy vertical gap between two parallel processing tracks.
    "excessive_gaps": 0.25,
}
# `corners_total` is deliberately absent: it is `bends_per_route` times the
# route count, so weighting both would re-count one signal in proportion to map
# size, penalising big maps for being big.

PERM_CAP = 24
"""Row-order permutations kept per stacked grid column.

Permuting several stacked columns together multiplies out, and every candidate
costs a full parse plus layout, so the search is bounded per column and reports
what it left unexplored.
"""

GRID_RE = re.compile(
    r"^\s*%%metro\s+(grid|direction|fold_threshold)\s*:", re.IGNORECASE
)
SUBGRAPH_RE = re.compile(r"^(\s*)subgraph\s+([A-Za-z0-9_]+)")
GRAPH_RE = re.compile(r"^\s*graph\s+(LR|RL|TB|BT|TD)", re.IGNORECASE)
END_RE = re.compile(r"^\s*end\s*$")


def marker_crowding(clearance: float | None) -> float:
    """How far the nearest line intrudes on a foreign marker, as a 0..1 fraction.

    A penalty for tight clearance, never a reward for loose clearance: a term
    that kept paying for more room would be minimised by spreading the map out.
    One lane pitch is room enough, so clearance beyond it scores nothing.
    """
    if clearance is None:
        return 0.0
    return max(0.0, Y_SPACING - clearance) / Y_SPACING


def objective(m: dict[str, float | None]) -> float:
    """The weighted score to minimise. A metric this map cannot define is skipped."""
    scores = dict(m)
    scores["marker_crowding"] = marker_crowding(scores.get("marker_clearance"))
    total = 0.0
    for key, weight in WEIGHTS.items():
        value = scores.get(key)
        if value is not None:
            total += weight * value
    return total


def strip_layout_directives(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not GRID_RE.match(line))


def inject_direction(text: str, sec_id: str, direction: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        out.append(line)
        m = SUBGRAPH_RE.match(line)
        if m and m.group(2) == sec_id:
            out.append(f"{m.group(1)}    %%metro direction: {direction}")
    return "\n".join(out)


def set_global(text: str, key: str, val: str) -> str:
    lines = text.splitlines()
    gi = next(i for i, line in enumerate(lines) if GRAPH_RE.match(line))
    lines.insert(gi, f"%%metro {key}: {val}")
    return "\n".join(lines)


def split_subgraphs(
    text: str,
) -> tuple[list[str], list[tuple[str, list[str]]], list[str]]:
    """Return (head_lines, [(section_id, block_lines)...], loose_lines)."""
    lines = text.splitlines()
    gi = next((i for i, line in enumerate(lines) if GRAPH_RE.match(line)), None)
    if gi is None:
        return lines, [], []
    head = lines[: gi + 1]
    blocks: list[tuple[str, list[str]]] = []
    loose: list[str] = []
    depth = 0
    cur: list[str] | None = None
    cur_id = ""
    for line in lines[gi + 1 :]:
        msg = SUBGRAPH_RE.match(line)
        if msg and depth == 0:
            depth = 1
            cur = [line]
            cur_id = msg.group(2)
        elif cur is not None:
            cur.append(line)
            if SUBGRAPH_RE.match(line):
                depth += 1
            elif END_RE.match(line):
                depth -= 1
                if depth == 0:
                    blocks.append((cur_id, cur))
                    cur = None
        else:
            loose.append(line)
    return head, blocks, loose


def reassemble(head, blocks, loose) -> str:
    out = list(head)
    for _id, blk in blocks:
        out.extend(blk)
    out.extend(loose)
    return "\n".join(out)


def score(text: str, fold: int | None = None) -> dict[str, float | None] | None:
    try:
        g = parse_metro_mermaid(text, max_station_columns=fold)
        compute_layout(g)
        return compute_metrics(g)
    except Exception:  # noqa: BLE001 - a crash is itself a result we report
        return None


def _fmt(label: str, o: float, m: dict[str, float | None]) -> str:
    gap = m["marker_clearance"]
    return (
        f"  {label:30} obj={o:6.1f}  bends={m['bends_per_route']:.2f} "
        f"turn={m['turn_angle_per_route']:.2f} "
        f"gap={'n/a' if gap is None else format(gap, '.0f')} "
        f"diag={int(m['single_diagonals'])} strike={int(m['label_strikes'])} "
        f"cross={int(m['crossings'])} near={int(m['near_horizontal'])} "
        f"exc={int(m['excessive_gaps'])} waste={m['wasted_canvas']:.2f}"
    )


def optimize(
    path: Path, fold_values=(8, 10, 12, 15, 20, 30, None), max_perm_col: int = 4
) -> None:
    text = path.read_text()
    auto_text = strip_layout_directives(text)

    base = score(auto_text)
    if base is None:
        print(f"\n=== {path.stem} ===  BASELINE CRASHES under pure auto-layout")
        cur = score(text)
        if cur is not None:
            print(_fmt("curated (as authored)", objective(cur), cur))
        return
    base_obj = objective(base)

    head, blocks, loose = split_subgraphs(auto_text)
    block_by_id = {bid: blk for bid, blk in blocks}
    order = [bid for bid, _ in blocks]

    # Auto-layout runs at parse time, so one parse yields both the section list
    # and the grid columns it inferred.
    auto_graph = parse_metro_mermaid(auto_text)
    sids = list(auto_graph.sections.keys())
    cols: dict[int, list[str]] = {}
    for sid, sec in auto_graph.sections.items():
        cols.setdefault(sec.grid_col, []).append(sid)

    stacked = {c: ids for c, ids in cols.items() if 2 <= len(ids) <= max_perm_col}
    stacked_ids = {x for ids in stacked.values() for x in ids}

    # Within-column row-order permutations (the crossing-relevant lever).
    candidate_orders = [order]
    dropped = 0
    for ids in stacked.values():
        present = [s for s in order if s in ids]
        if len(present) < 2:
            continue
        more: list[list[str]] = []
        for base_order in candidate_orders:
            pos = [i for i, s in enumerate(base_order) if s in ids]
            for perm in itertools.permutations(present):
                if list(perm) == present:
                    continue
                no = list(base_order)
                for slot, sid in zip(pos, perm):
                    no[slot] = sid
                more.append(no)
        candidate_orders.extend(more[:PERM_CAP])
        dropped += max(0, len(more) - PERM_CAP)

    best = (base_obj, "baseline (pure auto-layout, default order)", base)
    # The baseline is the default order at the parser's own fold, already scored.
    seen: set[tuple] = {("fold", None, tuple(order))}

    # Axis 1: fold x row-order.
    for fold in fold_values:
        for co in candidate_orders:
            key = ("fold", fold, tuple(co))
            if key in seen:
                continue
            seen.add(key)
            m = score(reassemble(head, [(b, block_by_id[b]) for b in co], loose), fold)
            if m is None:
                continue
            o = objective(m)
            if o < best[0] - 1e-6:
                desc = f"fold={fold or 'none'}"
                if co != order:
                    desc += f", reorder {[s for s in co if s in stacked_ids]}"
                best = (o, desc, m)

    # Axis 2: per-section TB + center_ports (each against default order).
    extra = [
        (f"direction: TB | {s}", inject_direction(auto_text, s, "TB")) for s in sids
    ]
    extra.append(("center_ports: true", set_global(auto_text, "center_ports", "true")))
    for desc, cand in extra:
        m = score(cand)
        if m is None:
            continue
        o = objective(m)
        if o < best[0] - 1e-6:
            best = (o, desc, m)

    b_obj, b_desc, b_m = best
    improved = b_obj < base_obj - 1e-6
    tag = "  ** lower score found **" if improved else "  (default auto-layout is best)"
    print(f"\n=== {path.stem} ==={tag}")
    print(_fmt("baseline", base_obj, base))
    if dropped:
        print(
            f"  searched {len(candidate_orders)} row orders, {dropped} left unexplored"
        )
    if improved:
        print(_fmt("best", b_obj, b_m))
        print(f"  via: {b_desc}")
        print("  NOTE: verify by eye before adopting -- the metric is a proxy.")
    cur = score(text)
    if cur is not None and path.read_text() != auto_text:
        print(_fmt("curated (as authored)", objective(cur), cur))


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--all":
        paths = sorted((REPO / "examples").glob("*.mmd"))
    elif argv:
        paths = [Path(a) if Path(a).is_absolute() else REPO / a for a in argv]
    else:
        names = [
            "genomic_pipeline",
            "longread_variant_calling",
            "differentialabundance",
            "variantbenchmarking_auto",
            "variantbenchmarking",
            "genomeassembly",
            "genomeassembly_staggered",
            "rnaseq_auto",
            "sarek_metro",
            "hlatyping",
            "epitopeprediction",
        ]
        paths = [REPO / "examples" / f"{n}.mmd" for n in names]
    for p in paths:
        if p.exists():
            optimize(p)
        else:
            print(f"{p}: not found")


if __name__ == "__main__":
    main(sys.argv[1:])
