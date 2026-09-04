"""A sibling section's off-track growth must not reorder an unrelated fan.

Regression lock for #1929.  ``orf_calling`` and ``psite_id`` share one
authored grid row.  Adding a single edge inside ``psite_id`` that routes a
line across a new file sink legitimately off-tracks that sink and makes
``psite_id`` taller.  Stage 4.7 then top-aligns the whole row, growing
``orf_calling``'s bbox above its content.  Stage 6.11's fan-balance pass
must not read that bbox growth as room to lift a below-trunk sibling, and
Stage 6.1's top-slack fan must not lift a fan-in branch off the row grid:
``orf_calling``'s content is unchanged, so its five-way fan order and its
reconvergence's position on the entry-port centreline must not move.

The two fixtures inject a ``psite_orf_out`` sink into the shared riboseq map
and differ by exactly one edge line inside ``psite_id``.  The invariant:
``orf_calling``'s internal station order is identical across the two, its
reconvergence stays at the vertical centre of the fan, and its trunk stays an
integer slot count from its row siblings.
"""

from __future__ import annotations

import warnings

import pytest
from conftest import parse_and_layout
from riboseq_map import RIBOSEQ_MMD

from nf_metro.api import prepare_graph
from nf_metro.layout.constants import SAME_COORD_TOLERANCE
from nf_metro.layout.phases._common import _section_lr_port_anchor_y

# The extra edge routes ``riboseq`` across the new ``psite_orf_out`` file sink
# inside the sibling ``psite_id`` section, legitimately off-tracking that sink
# and growing the shared grid row.  ``@EXTRA_EDGE@`` marks where it goes.
_EXTRA_EDGE = "        quantify_orf_psite -->|riboseq| psite_orf_out\n"

_WITH_SINK_SCAFFOLD = (
    RIBOSEQ_MMD.replace(
        "%%metro file: counts_out | TSV | Gene counts\n",
        "%%metro file: counts_out | TSV | Gene counts\n"
        "%%metro file: psite_orf_out | TSV | ORF P-site counts\n",
    )
    .replace(
        "        psite_counts_gene[Gene in-frame\\nP-sites]\n",
        "        psite_counts_gene[Gene in-frame\\nP-sites]\n"
        "        psite_orf_out[ ]\n",
    )
    .replace(
        "        plastid_wiggle -->|riboseq| psite_counts_gene\n",
        "        plastid_wiggle -->|riboseq| psite_counts_gene\n@EXTRA_EDGE@",
    )
)

WITHOUT_SINK = _WITH_SINK_SCAFFOLD.replace("@EXTRA_EDGE@", "")
WITH_SINK = _WITH_SINK_SCAFFOLD.replace("@EXTRA_EDGE@", _EXTRA_EDGE)

# The five-way fan-out column of ``orf_calling`` and its reconvergence.
_FAN_COLUMN = ("star_hybrid", "ribotish", "ribotricer", "rpbp", "price")
_RECONVERGENCE = "orf_merge"


def _orf_calling_internal_order(text: str) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(text)
    section = graph.sections["orf_calling"]
    rows = sorted(
        (round(graph.stations[sid].y, 1), round(graph.stations[sid].x, 1), sid)
        for sid in section.station_ids
        if not graph.stations[sid].is_port
        and not graph.stations[sid].is_hidden
        and not graph.stations[sid].off_track
    )
    return [sid for _, _, sid in rows]


def _fan_is_trunk_centred(text: str) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(text)
    fan_ys = [graph.stations[sid].y for sid in _FAN_COLUMN]
    fan_mid = (min(fan_ys) + max(fan_ys)) / 2
    return fan_mid, graph.stations[_RECONVERGENCE].y


def test_sibling_off_track_growth_preserves_orf_calling_order() -> None:
    """The extra ``psite_id`` sink must not reorder ``orf_calling``'s fan.

    The sink also off-tracks the ``orf_catalogue`` output in one variant (a
    legitimate crossing-avoidance that drops it from the on-track order), so
    the relative order is asserted over the stations common to both.
    """
    without = _orf_calling_internal_order(WITHOUT_SINK)
    with_sink = _orf_calling_internal_order(WITH_SINK)
    common = set(without) & set(with_sink)
    assert [sid for sid in without if sid in common] == [
        sid for sid in with_sink if sid in common
    ]


@pytest.mark.parametrize(
    "text", [WITHOUT_SINK, WITH_SINK], ids=["without_sink", "with_sink"]
)
def test_orf_calling_reconvergence_is_fan_centred(text: str) -> None:
    """``Merge ORF catalogue`` stays at the vertical centre of the fan."""
    fan_mid, reconvergence_y = _fan_is_trunk_centred(text)
    assert abs(fan_mid - reconvergence_y) <= SAME_COORD_TOLERANCE


@pytest.mark.parametrize(
    "text", [WITHOUT_SINK, WITH_SINK], ids=["without_sink", "with_sink"]
)
def test_orf_calling_trunk_stays_on_row_grid(text: str) -> None:
    """``orf_calling``'s trunk sits an integer slot count from its row siblings.

    The off-track P-site sink grows ``psite_id`` and opens top slack in the
    row-mate ``orf_calling``.  Fanning a fan-in branch into that slack would
    drag the reconvergence join a half slot off the row grid; the join must
    instead stay an exact multiple of ``y_spacing`` from the sibling trunk.
    """
    y_spacing = 55.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = parse_and_layout(text, y_spacing=y_spacing)
    orf_port = _section_lr_port_anchor_y(graph, graph.sections["orf_calling"])
    sibling_port = _section_lr_port_anchor_y(graph, graph.sections["psite_id"])
    slots = (orf_port - sibling_port) / y_spacing
    assert abs(slots - round(slots)) * y_spacing <= SAME_COORD_TOLERANCE
