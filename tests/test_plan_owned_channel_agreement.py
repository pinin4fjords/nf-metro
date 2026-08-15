"""A member channel seated on its convergence claim stays on the plan's column.

The seed_15 fixture drives one junction opening (``__junction_24``, line l1)
whose column is named by both a convergence-plan gap channel and the member
route that draws the stroke.  Whether or not the render completes, its failure
text must never report the defect classes that column disagreement produces:

- the closing feasibility validator's ``ambiguous same-line member channel``
  (plan and member naming different columns for one carrier),
- a ``collinear overlay`` of the two entry tails sharing one lane in the
  gap between columns 5 and 6, and
- sub-radius jogs from a trunk relocated onto a lane it cannot reach with
  formed curves.
"""

from pathlib import Path

import pytest

from nf_metro.api import render_string

SEED_15 = Path(__file__).parent / "fixtures" / "hash_seed_determinism" / "seed_15.mmd"


@pytest.fixture(scope="module")
def seed_15_failure() -> str:
    try:
        render_string(SEED_15.read_text())
    except Exception as error:
        return str(error)
    return ""


def test_no_ambiguous_same_line_member_channel(seed_15_failure: str) -> None:
    assert "ambiguous same-line member channel" not in seed_15_failure


def test_entry_tails_keep_their_own_columns(seed_15_failure: str) -> None:
    assert (
        "line 'l2' (__junction_23->s5__entry_right_16) and line 'l1'"
        not in seed_15_failure
    )


def test_trunk_relocation_keeps_formed_curves(seed_15_failure: str) -> None:
    starved = [
        line
        for line in seed_15_failure.splitlines()
        if "starved below a formed curve" in line
        and "'__junction_24'->'__merge_12'" in line
    ]
    assert starved == []
