"""A fold-relocated section's flow-axis port faces its connecting sections (#1167).

When a lowered ``fold_threshold`` relocates sections onto a return row, a
directive authored for the unfolded layout (e.g. ``%%metro exit: right`` written
when the consumer sat to the right) can end up facing *away* from the grid side
its connecting section occupies.  The connecting leg then wraps back across the
section's own box, which the routing self-check rejects.

The invariant: for a section the fold relocated, a left/right entry/exit port on
its flow axis sits on the same horizontal side as the sections it connects to,
whenever those all sit strictly to one side.  ``genomeassembly`` under a lowered
fold is the real-pipeline case (#1167, residual of #1085); the parametrised
sweep exercises the property across several gallery fixtures so it generalises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _port_side(port_id: str) -> str:
    # Port ids are "{section}__{entry|exit}_{side}_{n}".
    return port_id.split("__", 1)[1].split("_")[1]


def _connecting_cols(graph, sec_id: str, *, as_exit: bool) -> set[int]:
    """Grid columns of the sections this section's exit/entry connects to."""
    cols: set[int] = set()
    for e in graph.edges:
        src = graph.section_for_station(e.source)
        tgt = graph.section_for_station(e.target)
        if src == tgt:
            continue
        other = None
        if as_exit and src == sec_id and tgt is not None:
            other = tgt
        elif not as_exit and tgt == sec_id and src is not None:
            other = src
        if other is not None:
            cols.add(graph.sections[other].grid_col)
    return cols


def _assert_flow_ports_face_connections(graph) -> None:
    relocated = graph._fold_compressed_sections
    for sec_id in relocated:
        section = graph.sections[sec_id]
        if section.direction not in ("LR", "RL"):
            continue
        col = section.grid_col
        for ports, as_exit in (
            (section.exit_ports, True),
            (section.entry_ports, False),
        ):
            other_cols = _connecting_cols(graph, sec_id, as_exit=as_exit)
            if not other_cols:
                continue
            all_left = all(c < col for c in other_cols)
            all_right = all(c > col for c in other_cols)
            if not (all_left or all_right):
                continue  # connections straddle the section: no single side
            expected = "left" if all_left else "right"
            for pid in ports:
                side = _port_side(pid)
                if side not in ("left", "right"):
                    continue  # cross-axis port, not on the flow axis
                assert side == expected, (
                    f"{sec_id}: flow-axis {'exit' if as_exit else 'entry'} "
                    f"port {pid!r} faces {side} but its connecting sections all "
                    f"sit to the {expected} (cols {sorted(other_cols)} vs {col})"
                )


def _layout(name: str, fold: int, *, validate: bool = False):
    graph = parse_metro_mermaid(
        (EXAMPLES / f"{name}.mmd").read_text(), max_station_columns=fold
    )
    compute_layout(graph, validate=validate)
    return graph


@pytest.mark.xfail(
    strict=True,
    reason="same-side hairpin fold-back in polishing/longranger/freebayes (#1182)",
)
@pytest.mark.parametrize("fold", [3, 5, 7, 9, 11])
def test_genomeassembly_renders_clean_under_lowered_fold(fold: int) -> None:
    # Ports face the convergence sink inward, but the same-side hairpin
    # fold-back (#1182) still aborts the full validate pass.
    _layout("genomeassembly", fold, validate=True)


def test_genomeassembly_sink_exit_and_entry_face_inward() -> None:
    graph = _layout("genomeassembly", 5)
    scaffolding = graph.sections["scaffolding"]
    genome_statistics = graph.sections["genome_statistics"]
    # genome_statistics is the leftmost sink; scaffolding sits to its right.
    assert genome_statistics.grid_col < scaffolding.grid_col
    # scaffolding's only consumer (genome_statistics) is to its left.
    assert all(_port_side(p) == "left" for p in scaffolding.exit_ports)
    # genome_statistics' feeders all sit to its right.
    assert all(_port_side(p) == "right" for p in genome_statistics.entry_ports)


@pytest.mark.parametrize(
    "name, fold",
    [
        ("genomeassembly", 5),
        ("rnaseq_auto", 4),
        ("rnaseq_sections", 4),
    ],
)
def test_fold_relocated_flow_ports_face_connections(name: str, fold: int) -> None:
    graph = _layout(name, fold)
    _assert_flow_ports_face_connections(graph)
