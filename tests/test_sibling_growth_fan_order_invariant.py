"""A sibling section's off-track growth must not reorder or de-grid a fan.

Regression lock for #1929.  ``orf_calling`` and ``psite_id`` share one
authored grid row in the nf-core/riboseq map.  ``psite_id``'s two P-site file
sinks are reached by lines that cross them, so the sinks off-track and make
``psite_id`` taller.  Stage 4.7 then top-aligns the whole row, growing
``orf_calling``'s bbox above its content and opening top slack.  Two passes
must stay robust to that:

- Stage 6.11's fan-balance must not read the grown bbox as room to lift a
  below-trunk sibling (``price`` jumping to the top of the fan), and
- Stage 6.1's top-slack fan must not lift a fan-in branch (``ribotish``) into
  that slack, which would drag the ``orf_merge`` reconvergence a half slot off
  the row grid.

The fixtures are the shipped ``examples/riboseq_metro.mmd`` (WITH both sinks
connected) and the same map with the two sink edges removed (WITHOUT), so the
lock tracks the exact map the bug was found on.  ``orf_calling``'s content is
identical between them, so its fan order, its reconvergence's fan-centre
position, and its trunk's row-grid alignment must not move.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from conftest import parse_and_layout

from nf_metro.layout.constants import SAME_COORD_TOLERANCE
from nf_metro.layout.phases._common import _section_lr_port_anchor_y

_MAP = Path(__file__).resolve().parents[1] / "examples" / "riboseq_metro.mmd"
WITH_SINK = _MAP.read_text()
# Drop only the two edges that route a line across each P-site sink; the sink
# stations stay declared but unconnected, so nothing crosses them, they stay
# on-track, and ``psite_id`` keeps its ungrown height.  This one difference is
# what drives ``orf_calling``'s fan.
WITHOUT_SINK = WITH_SINK.replace(
    "        quantify_orf_psite -->|riboseq| psite_orf_out\n", ""
).replace("        psite_counts_gene -->|riboseq| psite_gene_out\n", "")

assert WITHOUT_SINK != WITH_SINK, "sink edges not found in the shipped map"

# The five-way fan-out column of ``orf_calling`` and its reconvergence.
_FAN_COLUMN = ("star_hybrid", "ribotish", "ribotricer", "rpbp", "price")
_RECONVERGENCE = "orf_merge"


def _orf_calling_internal_order(text: str) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = parse_and_layout(text)
    section = graph.sections["orf_calling"]
    rows = sorted(
        (round(graph.stations[sid].y, 1), round(graph.stations[sid].x, 1), sid)
        for sid in section.station_ids
        if not graph.stations[sid].is_port
        and not graph.stations[sid].is_hidden
        and not graph.stations[sid].off_track
    )
    return [sid for _, _, sid in rows]


def test_sibling_off_track_growth_preserves_orf_calling_order() -> None:
    """The P-site sinks must not reorder ``orf_calling``'s fan (Stage 6.11).

    The sinks also off-track the ``orf_catalogue`` output in one variant (a
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = parse_and_layout(text)
    fan_ys = [graph.stations[sid].y for sid in _FAN_COLUMN]
    fan_mid = (min(fan_ys) + max(fan_ys)) / 2
    assert abs(fan_mid - graph.stations[_RECONVERGENCE].y) <= SAME_COORD_TOLERANCE


def test_orf_calling_trunk_stays_on_row_grid() -> None:
    """``orf_calling``'s trunk sits an integer slot count from its row siblings.

    With both P-site sinks connected, the grown ``psite_id`` opens top slack in
    the row-mate ``orf_calling``.  Fanning a fan-in branch into that slack
    (Stage 6.1) would drag the reconvergence join a half slot off the row grid;
    the join must instead stay an exact multiple of ``y_spacing`` from the
    sibling trunk.  Exercised at ``y_spacing=55``, where the half slot the sink
    heights open is not an integer number of slots.
    """
    y_spacing = 55.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = parse_and_layout(WITH_SINK, y_spacing=y_spacing)
    orf_port = _section_lr_port_anchor_y(graph, graph.sections["orf_calling"])
    sibling_port = _section_lr_port_anchor_y(graph, graph.sections["psite_id"])
    slots = (orf_port - sibling_port) / y_spacing
    assert abs(slots - round(slots)) * y_spacing <= SAME_COORD_TOLERANCE
