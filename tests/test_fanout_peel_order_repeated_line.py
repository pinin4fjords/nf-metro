"""Peel order for fans where one line leaves on several branches.

A line occupies a single slot in the source bundle regardless of how many
branches carry it, so the per-branch opening order has to collapse onto one
slot per line.  Slots are handed out by index and a target's entry port
inherits the lane its own branch holds, so merging two slots renumbers every
slot beyond them.  The merge is therefore only sound at the trailing slots, for
branches that want the same lane (same column reach, same band classification)
and that all leave the source row -- a branch running on the source row is flat
only while it keeps the lane its target's entry port inherited, so it cannot be
moved onto a shared slot.

Also covers the near-band classification: a cross-row branch whose entry port
sits on the far side of its target cannot descend into the box, so it traverses
the gap band beside its source row alongside the same-row branches rather than
at its own target row's depth.  The band it turns onto is on the side the fan
travels, so the classification has to hold for a fan that climbs as well as one
that descends, and a branch that does not leave its column at all has no such
band to turn onto.
"""

from __future__ import annotations

from pathlib import Path

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.fan_ordering import (
    _traverses_near_band,
    fanout_divergence_peel_order,
)
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph, PortSide

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_72 = REPO_ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_72.mmd"
BRACKETED = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "curve_invariant_repros"
    / "rl_return_row_convergence.mmd"
)
REPEAT_ON_SOURCE_ROW = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "regressions"
    / "fanout_repeat_same_row_continuation.mmd"
)


def _laid_out(path: Path) -> tuple[MetroGraph, dict[str, int]]:
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    return graph, {lid: i for i, lid in enumerate(graph.lines)}


def test_repeated_line_collapses_to_one_slot() -> None:
    """seed-72's five-branch / four-line fan opens in its crossing-free order.

    ``l3`` leaves on two branches (rows 1 and 2) and descends past the gap band
    below the source row, so it must sit innermost.  ``l0`` targets row 3 but
    enters on the far (RIGHT) side, so it crosses that same gap band and opens
    with the same-row branches -- below ``l6``, whose run returns to the source
    row and would otherwise be pierced by ``l0``'s continuing riser.
    """
    graph, line_priority = _laid_out(SEED_72)

    order = fanout_divergence_peel_order(graph, "__junction_14", line_priority)

    assert order == ["l5", "l6", "l0", "l3"], order


def test_bracketed_repeated_line_declines_to_reorder() -> None:
    """A distinct line ranked between two branches of one line yields no order.

    ``bam`` reaches a LEFT entry on the return row and a RIGHT entry a column
    further out, and ``other`` sorts between them; no single bundle slot for
    ``bam`` serves both branches, so the fan keeps declaration order.
    """
    graph, line_priority = _laid_out(BRACKETED)

    assert fanout_divergence_peel_order(graph, "__junction_16", line_priority) is None


def _fan(rows: dict[str, tuple[int, int]], branches: dict[str, str]) -> str:
    """A one-hub fan: *rows* places each section, *branches* names each target.

    Each target section holds a two-station run on the line that reaches it, so
    every branch lands on a real entry port and the fan resolves as a clean
    divergence.  ``branches`` maps target section id to the line id that feeds
    it; a section named twice in the values is a line leaving on two branches.
    """
    lines = sorted(set(branches.values()) | {"hub"})
    text = ["%%metro title: fan"]
    text += [f"%%metro line: {lid} | {lid.upper()} | #6ef362" for lid in lines]
    text += [f"%%metro grid: {sid} | {col},{row}" for sid, (col, row) in rows.items()]
    text += [
        "",
        "graph LR",
        "    subgraph src [Source]",
        "        s_in[In]",
        "        s_hub[Hub]",
        "        s_in -->|hub| s_hub",
        "    end",
    ]
    for sid, line_id in branches.items():
        text += [f"    subgraph {sid} [{sid}]"]
        if sid.endswith("_far"):
            text += [f"        %%metro entry: right | {line_id}"]
        text += [
            f"        {sid}_a[{sid} a]",
            f"        {sid}_b[{sid} b]",
            f"        {sid}_a -->|{line_id}| {sid}_b",
            "    end",
        ]
    text += [f"    s_hub -->|{lid}| {sid}_a" for sid, lid in branches.items()]
    return "\n".join(text) + "\n"


def _fan_order(text: str) -> list[str] | None:
    """The peel order of the single fan-out junction in an inline fan."""
    graph = parse_metro_mermaid(text)
    compute_layout(graph)
    line_priority = {lid: i for i, lid in enumerate(graph.lines)}
    jid = next(
        sid
        for sid in graph.stations
        if sid.startswith("__junction") and len(list(graph.edges_from(sid))) > 1
    )
    return fanout_divergence_peel_order(graph, jid, line_priority)


def test_repeat_carrying_the_source_row_declines_to_reorder() -> None:
    """A repeated line with one branch on the source row keeps declaration order.

    ``pair`` reaches the section on the source row and the one a row above it.
    The source-row branch is flat only while it holds the lane its target's entry
    port inherited, so it cannot move onto a slot shared with the branch that
    climbs; the fan has no single slot for ``pair`` and must decline.
    """
    graph, line_priority = _laid_out(REPEAT_ON_SOURCE_ROW)

    assert fanout_divergence_peel_order(graph, "__junction_5", line_priority) is None


def test_repeat_carrying_the_source_row_renders_within_invariants() -> None:
    """The same fan settles without violating the engine's own final guards."""
    graph = parse_metro_mermaid(REPEAT_ON_SOURCE_ROW.read_text())

    compute_layout(graph, validate=True)


def test_repeat_ranked_inside_another_line_declines_to_reorder() -> None:
    """A repeat with a line ranked outside it declines: that line's lane moves.

    ``mid`` climbs one and two rows while ``far`` climbs three, so ``far`` sorts
    outside ``mid``'s block.  Merging ``mid``'s two slots renumbers every slot
    beyond them, which would leave ``far`` and the source-row ``near`` on lanes
    their entry ports never inherited.
    """
    order = _fan_order(
        _fan(
            {
                "src": (0, 3),
                "near": (1, 3),
                "one": (1, 2),
                "two": (1, 1),
                "deep": (1, 0),
            },
            {"near": "near", "one": "mid", "two": "mid", "deep": "far"},
        )
    )

    assert order is None, order


SHARED_DEPTH_FAN = _fan(
    {
        "src": (0, 0),
        "onward": (2, 0),
        "one": (1, 2),
        "beyond": (2, 2),
        "two": (1, 3),
    },
    {"onward": "onward", "one": "pair", "beyond": "beyond", "two": "pair"},
)


def test_repeat_sharing_a_depth_with_another_line_declines_to_reorder() -> None:
    """A distinct line stacking at a merged branch's depth yields no order.

    ``beyond`` and one branch of ``pair`` both target row 2, so ``beyond``
    reaches its own column by running past that row while ``pair``'s single slot
    also has to serve row 3.  The depth they tie on says nothing about which of
    the two runs deeper, so neither ranking keeps ``beyond``'s riser clear of
    ``pair``'s flat run into row 2.
    """
    order = _fan_order(SHARED_DEPTH_FAN)

    assert order is None, order


def test_repeat_sharing_a_depth_renders_within_invariants() -> None:
    """The same fan settles without a crossing between the two of them."""
    graph = parse_metro_mermaid(SHARED_DEPTH_FAN)

    compute_layout(graph, validate=True)


def test_repeat_spanning_two_columns_declines_to_reorder() -> None:
    """A repeat whose branches hop different column counts declines.

    ``mid`` reaches one section a column out and another two columns out, so its
    branches ride different corridors: one lane cannot carry both.
    """
    order = _fan_order(
        _fan(
            {
                "src": (0, 0),
                "near": (1, 0),
                "one": (1, 1),
                "two_far": (2, 2),
                "deep_far": (1, 3),
            },
            {"near": "near", "one": "mid", "two_far": "mid", "deep_far": "far"},
        )
    )

    assert order is None, order


def test_upward_near_band_branch_stacks_on_the_ascent_side() -> None:
    """A climbing fan's near-band branch sorts above the source row, not below.

    ``far`` climbs three rows but enters on the far (RIGHT) side, so it cannot
    cross into its target row: it turns onto the gap band half a row *above* the
    source and must rank between the deeper climbers and the source-row pair.  A
    near-band depth that ignored the fan's direction would rank ``far`` outside
    the deepest climber and drive its riser through the source-row runs.
    """
    order = _fan_order(
        _fan(
            {
                "src": (0, 3),
                "near": (1, 3),
                "onward": (2, 3),
                "two": (1, 1),
                "one": (1, 2),
                "deep_far": (1, 0),
            },
            {
                "near": "near",
                "onward": "onward",
                "two": "climb2",
                "one": "climb1",
                "deep_far": "far",
            },
        )
    )

    assert order == ["climb2", "climb1", "far", "near", "onward"], order


def test_same_column_far_side_branch_is_not_near_band() -> None:
    """A branch that stays in its column has no gap band beside it to turn onto.

    The near-band idiom is a single-column hop whose far-side entry forces the
    branch to run on outside the target box.  With no column hop the branch has
    no such traverse, whichever side its entry port sits on, so it stacks at its
    own target row's depth.
    """
    assert not _traverses_near_band(PortSide.RIGHT, 0, 2)
    assert not _traverses_near_band(PortSide.LEFT, 0, 2)
    assert not _traverses_near_band(PortSide.RIGHT, 0, -2)
    assert not _traverses_near_band(PortSide.LEFT, 0, -2)


def test_near_band_holds_for_either_travel_direction() -> None:
    """The classification reads the entry side against the hop, not the row sign."""
    assert _traverses_near_band(PortSide.RIGHT, 1, 3)
    assert _traverses_near_band(PortSide.RIGHT, 1, -3)
    assert _traverses_near_band(PortSide.LEFT, -1, 3)
    assert _traverses_near_band(PortSide.LEFT, -1, -3)
    assert not _traverses_near_band(PortSide.LEFT, 1, 3)
    assert not _traverses_near_band(PortSide.RIGHT, -1, 3)
    assert not _traverses_near_band(PortSide.RIGHT, 1, 0)
