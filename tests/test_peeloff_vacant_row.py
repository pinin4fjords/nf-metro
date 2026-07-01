"""Regression: a terminal bundle peel-off lands on its carrier's row.

When a single line peels off a multi-line carrier station to visit its own
short chain of stations that ends inside the section, the row it lands on must
come from scanning for a free row near the carrier, not from the line's
declaration-order index alone. The formula-only choice dropped the peel-off two
rows below a passing through-corridor instead of onto the vacant row the carrier
already occupies.

On ``examples/rnaseq_sections.mmd``'s ``genome_align`` section, ``salmon_quant``
sheds ``bowtie2_salmon`` to ``multiqc_bowtie2`` -> ``report_bowtie2`` (a terminal
spur ending in an ``HTML`` file icon) while ``hisat2`` passes through on its own
row and exits. The peel-off must land on the row already shared with
``salmon_quant`` -- flat, so its file-icon successor inherits the same row and
their connector runs straight -- rather than below the ``hisat2`` corridor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.parser import parse_metro_mermaid

EXAMPLES = Path(__file__).parent.parent / "examples"

TOL = 1.0

REPORT_PAIRS = [
    ("multiqc_bowtie2", "report_bowtie2"),
    ("multiqc_quant", "report_quant"),
    ("multiqc_final", "report_final"),
]


def _rnaseq():
    graph = parse_metro_mermaid((EXAMPLES / "rnaseq_sections.mmd").read_text())
    compute_layout(graph)
    return graph


def test_peeloff_lands_on_carrier_row() -> None:
    """The ``bowtie2_salmon`` peel-off lands on ``salmon_quant``'s row.

    The spur must reclaim the vacant row shared with its carrier rather than
    dropping below the ``hisat2`` through-corridor.
    """
    graph = _rnaseq()
    salmon = graph.stations["salmon_quant"]
    multiqc = graph.stations["multiqc_bowtie2"]
    report = graph.stations["report_bowtie2"]
    assert abs(multiqc.y - salmon.y) < TOL, (
        f"peel-off multiqc_bowtie2 (y={multiqc.y}) did not land on the "
        f"salmon_quant row (y={salmon.y})"
    )
    assert abs(report.y - salmon.y) < TOL, (
        f"report_bowtie2 (y={report.y}) did not land on the "
        f"salmon_quant row (y={salmon.y})"
    )


@pytest.mark.parametrize(("station_id", "report_id"), REPORT_PAIRS)
def test_station_and_file_icon_successor_share_row(
    station_id: str, report_id: str
) -> None:
    """Each ``<station> -> <its own file-icon successor>`` pair is level.

    A file-icon leaf sink inherits its single predecessor's track, so the two
    render on one row with a straight connector.
    """
    graph = _rnaseq()
    station = graph.stations[station_id]
    report = graph.stations[report_id]
    assert abs(station.y - report.y) < TOL, (
        f"{station_id} (y={station.y}) and its file-icon successor "
        f"{report_id} (y={report.y}) are not on the same row"
    )
