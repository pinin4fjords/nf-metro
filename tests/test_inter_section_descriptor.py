"""Tests for the inter-section routing descriptor catalogue.

Pin down :class:`TurnSequence`'s parity-counting contract and
sanity-check the structure of every :data:`WRAP_TABLE` entry.
"""

from __future__ import annotations

from nf_metro.layout.routing.common import Direction
from nf_metro.layout.routing.inter_section import (
    WRAP_TABLE,
    ChannelKind,
    Corner,
    CornerHandedness,
    RouteKind,
    TurnSequence,
    WrapDescriptor,
)
from nf_metro.parser.model import PortSide

# ---------------------------------------------------------------------------
# TurnSequence.parity unit tests
# ---------------------------------------------------------------------------


def test_left_entry_wrap_parity_is_true():
    """The 4-corner left-entry wrap has odd handedness changes.

    Shape: right -> down -> left -> down -> right.

    Corner handednesses: ``[CW, CW, CCW, CCW]``.  Adjacent pairs:
    CW->CW (no change), CW->CCW (change), CCW->CCW (no change).
    One change overall -> parity = True.
    """
    seq = TurnSequence(
        corners=(
            Corner(Direction.R, Direction.D),
            Corner(Direction.D, Direction.L),
            Corner(Direction.L, Direction.D),
            Corner(Direction.D, Direction.R),
        )
    )
    assert seq.parity is True
    # Sanity: len works and iteration yields the corners in order
    assert len(seq) == 4
    assert [c.handedness for c in seq] == [
        CornerHandedness.CW,
        CornerHandedness.CW,
        CornerHandedness.CCW,
        CornerHandedness.CCW,
    ]


def test_l_shape_parity_is_true():
    """Standard L-shape (right -> down -> right) has one handedness change.

    Shape: ``[CW, CCW]`` -> exactly one change -> parity = True.
    """
    seq = TurnSequence(
        corners=(
            Corner(Direction.R, Direction.D),
            Corner(Direction.D, Direction.R),
        )
    )
    assert seq.parity is True


def test_empty_turn_sequence_has_parity_false():
    """A degenerate straight route has no corners and parity = False."""
    seq = TurnSequence(corners=())
    assert seq.parity is False
    assert len(seq) == 0


def test_same_handedness_pair_parity_is_false():
    """Two consecutive corners with the same handedness: zero changes.

    A spiral-like ``[CW, CW]`` (the front half of a four-corner wrap)
    has zero handedness changes -> parity False.
    """
    seq = TurnSequence(
        corners=(
            Corner(Direction.R, Direction.D),
            Corner(Direction.D, Direction.L),
        )
    )
    assert seq.parity is False


def test_wrap_descriptor_parity_delegates_to_turn_sequence():
    """WrapDescriptor.parity is a thin shim over TurnSequence.parity."""
    seq = TurnSequence(
        corners=(
            Corner(Direction.R, Direction.D),
            Corner(Direction.D, Direction.R),
        )
    )
    desc = WrapDescriptor(
        kind=RouteKind.L_SHAPE,
        turn_sequence=seq,
        channel_kind=ChannelKind.L_SHAPE,
    )
    assert desc.parity == seq.parity is True


# ---------------------------------------------------------------------------
# WRAP_TABLE structural checks
# ---------------------------------------------------------------------------


def test_wrap_table_keys_are_well_formed():
    """Sanity: keys use legal exit/entry sides and -1/0/+1 signs."""
    for key in WRAP_TABLE:
        exit_side, entry_side, drow_sign, dcol_sign = key
        assert exit_side is None or isinstance(exit_side, PortSide), (
            f"bad exit_side: {key}"
        )
        assert isinstance(entry_side, PortSide), f"bad entry_side: {key}"
        assert drow_sign in (-1, 0, 1), f"bad drow_sign: {key}"
        assert dcol_sign in (-1, 0, 1), f"bad dcol_sign: {key}"


def test_wrap_table_is_non_empty():
    """WRAP_TABLE must cover at least the genuine wrap cases.

    Catches accidental deletions during refactors.  If you legitimately
    want to remove every entry (e.g. moving the table elsewhere), update
    this test to reflect the new home.
    """
    assert len(WRAP_TABLE) >= 4, "WRAP_TABLE shrank below the wrap-handler floor"
