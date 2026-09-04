"""The ``validate`` and ``render`` commands accept the same set of maps.

``validate`` exists to tell an author whether a map is good before rendering
it, so a map it rejects must not render and a map it accepts must not fail to
parse.  Each case below goes through both commands: their acceptance is
compared, then pinned against the expected verdict so the pair cannot agree on
the wrong answer.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from nf_metro.cli import cli

_DECLARED = "%%metro line: main | Main | #0570b0\n"

_GRAPH = """graph LR
    subgraph sec [Section]
        a[A]
        b[B]
        {edge}
    end
"""


def _map(edge: str, prologue: str = "") -> str:
    return prologue + _GRAPH.format(edge=edge)


# ``offender`` is the line id both commands must name in their diagnostic, or
# ``None`` when the case carries no undeclared line.
CASES = [
    pytest.param(_map("a -->|main| b", _DECLARED), True, None, id="clean"),
    # No ``%%metro line:`` directive at all, so the annotation names nothing.
    # The render falls back to an anonymous stroke colour and an empty legend.
    pytest.param(_map("a -->|main| b"), False, "main", id="no-declarations"),
    pytest.param(
        _map("a -->|other| b", _DECLARED),
        False,
        "other",
        id="undeclared-alongside-declared",
    ),
    # A ``%%metro line:`` directive too short to carry a name and a colour, so
    # its id never becomes usable.
    pytest.param(
        _map("a -->|main| b", "%%metro line: main\n"),
        False,
        "main",
        id="rejected-declaration",
    ),
    pytest.param(_map("a --> b", _DECLARED), False, None, id="unannotated-edge"),
]


@pytest.mark.parametrize(("source", "accepted", "offender"), CASES)
def test_validate_and_render_agree(
    tmp_path: Path, source: str, accepted: bool, offender: str | None
) -> None:
    path = tmp_path / "map.mmd"
    path.write_text(source)
    runner = CliRunner()

    validated = runner.invoke(cli, ["validate", str(path)])
    rendered = runner.invoke(cli, ["render", str(path), "-o", str(tmp_path / "o.svg")])

    assert (validated.exit_code == 0) == (rendered.exit_code == 0), (
        f"validate exited {validated.exit_code} and render exited "
        f"{rendered.exit_code}\nvalidate: {validated.output}\n"
        f"render: {rendered.output}"
    )
    assert (validated.exit_code == 0) is accepted, validated.output
    if offender is not None:
        assert offender in validated.output
        assert offender in rendered.output
