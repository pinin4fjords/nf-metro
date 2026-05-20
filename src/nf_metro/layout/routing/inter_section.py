"""Declarative inter-section routing descriptors.

:data:`WRAP_TABLE` maps ``(exit_side, entry_side, drow_sign,
dcol_sign)`` to a :class:`WrapDescriptor` recording the corner
sequence and its handedness-change parity.  ``exit_side`` is
``"JUNCTION"`` when the source has no port side; the other three are
``sign()`` values in ``{-1, 0, +1}``.

Parity contract: a route whose corner sequence has an odd number of
CW <-> CCW transitions reverses the bundle's outer/inner ordering
between source and target.  ``_propagate_wrap_flip_parity`` in
``engine.py`` propagates the same parity along the section DAG.  If
the table and the propagator disagree, routes visibly cross at the
entry-port quarter-circles; the alignment test in
``tests/test_inter_section_descriptor.py`` keeps them in sync.

Same-row L-shapes and the TB ``("BOTTOM", "TOP", 1, 0)`` straight
drop are deliberately absent: their geometric parity disagrees with
the propagator's row-delta ``is_wrap`` flag, and reconciling that is
left to a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from nf_metro.layout.routing.common import Direction

# ---------------------------------------------------------------------------
# Corner / turn-sequence primitives
# ---------------------------------------------------------------------------


class CornerHandedness(Enum):
    """Direction of rotation at a corner.

    ``CW`` (clockwise): a route going RIGHT that turns DOWN, or DOWN
    that turns LEFT, etc.  ``CCW`` (counter-clockwise) is the mirror.

    Used to compute :attr:`TurnSequence.parity`: an odd number of
    handedness *changes* (CW->CCW or CCW->CW between consecutive
    corners) inverts bundle ordering across the route.
    """

    CW = "CW"
    CCW = "CCW"


@dataclass(frozen=True)
class Corner:
    """A single right-angle corner in a routed path.

    Attributes
    ----------
    in_tangent:
        Direction the route is travelling as it enters the corner.
    out_tangent:
        Direction it travels as it leaves the corner.  Must be
        perpendicular to ``in_tangent`` (no straight-through corners).
    handedness:
        Whether the turn is clockwise (CW) or counter-clockwise (CCW)
        as seen on screen (y grows downward).
    concentric:
        ``True`` when this corner is part of a concentric bundle - i.e.
        all lines in the bundle share the same arc centre and only
        differ in radius.  ``False`` for an isolated turn.
    """

    in_tangent: Direction
    out_tangent: Direction
    handedness: CornerHandedness
    concentric: bool = True


@dataclass(frozen=True)
class TurnSequence:
    """An ordered sequence of corners along a routed inter-section path.

    The sequence represents the geometric "shape" of the route ignoring
    straight runs.  For example, a standard L-shape (right->down->right)
    is a 2-corner sequence ``[CW, CCW]``; a left-entry wrap
    (right->down->left->down->right) is a 4-corner sequence
    ``[CW, CW, CCW, CCW]``.

    The cached :attr:`parity` property captures whether the route ends
    up flipping the bundle's outer/inner ordering: ``True`` iff there
    is an odd number of handedness changes between consecutive corners.
    """

    corners: tuple[Corner, ...]

    def __len__(self) -> int:
        return len(self.corners)

    def __iter__(self):
        return iter(self.corners)

    def __getitem__(self, idx):
        return self.corners[idx]

    @cached_property
    def parity(self) -> bool:
        """``True`` iff the sequence has an odd number of handedness changes.

        A "change" is a pair of consecutive corners whose handedness
        differs (CW->CCW or CCW->CW).  When parity is True, the
        bundle's outer/inner ordering reverses between the route's
        first and last corner - the same condition that
        ``_propagate_wrap_flip_parity`` propagates as
        ``Section.flip_lines``.

        Examples
        --------
        * Standard L-shape ``[CW, CCW]``: one change -> parity = True.
        * 4-corner wrap ``[CW, CW, CCW, CCW]``: one change -> parity = True.
        * Straight line (no corners): zero changes -> parity = False.
        * 2-corner same-handedness ``[CW, CW]``: zero changes -> parity = False.
        """
        changes = 0
        for prev, curr in zip(self.corners, self.corners[1:]):
            if prev.handedness != curr.handedness:
                changes += 1
        return changes % 2 == 1


# ---------------------------------------------------------------------------
# Wrap descriptor (table populated in a follow-up scaffolding commit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WrapDescriptor:
    """Describes the corner sequence a routing handler produces.

    Attributes
    ----------
    kind:
        Short identifier naming which handler function in ``core.py``
        produces this corner sequence (e.g. ``"l_shape"``,
        ``"left_entry_wrap"``, ``"top_entry_l_shape"``).  Used for
        cross-referencing the if-cascade.
    turn_sequence:
        The ordered list of corners the handler emits.  Its
        :attr:`TurnSequence.parity` is what the section DAG
        propagator must agree with.
    channel_kind:
        Coarse classification of the route's main vertical/horizontal
        channel.  One of ``"L_SHAPE"`` (single vertical channel in
        the inter-column gap), ``"WRAP"`` (route exits then re-enters
        the same column or wraps around the target section),
        ``"BYPASS"`` (route goes around an intervening section),
        ``"TB_EXIT"`` (vertical drop from a TB BOTTOM port),
        ``"STRAIGHT"`` (degenerate same-X or same-Y route).
    """

    kind: str
    turn_sequence: TurnSequence
    channel_kind: str

    @property
    def parity(self) -> bool:
        """Convenience: forward to :attr:`TurnSequence.parity`."""
        return self.turn_sequence.parity


# ---------------------------------------------------------------------------
# Pre-built turn sequences keyed by route shape
# ---------------------------------------------------------------------------
# Each constant captures the shape of the route the corresponding
# ``core.py`` handler builds in coordinate space, ignoring straight
# runs between corners.

_L_RIGHT_DOWN_RIGHT = TurnSequence(
    corners=(
        # exit right, then turn down
        Corner(Direction.R, Direction.D, CornerHandedness.CW),
        # arrive going down, then turn right into entry
        Corner(Direction.D, Direction.R, CornerHandedness.CCW),
    )
)

_L_LEFT_DOWN_LEFT = TurnSequence(
    corners=(
        Corner(Direction.L, Direction.D, CornerHandedness.CCW),
        Corner(Direction.D, Direction.L, CornerHandedness.CW),
    )
)

_L_RIGHT_UP_RIGHT = TurnSequence(
    corners=(
        Corner(Direction.R, Direction.U, CornerHandedness.CCW),
        Corner(Direction.U, Direction.R, CornerHandedness.CW),
    )
)

# Left-entry wrap (source row above, target row below, entry on LEFT):
# right -> down -> left -> down -> right.  Four corners with one
# handedness change (parity = True).
_LEFT_ENTRY_WRAP_DOWN = TurnSequence(
    corners=(
        Corner(Direction.R, Direction.D, CornerHandedness.CW),
        Corner(Direction.D, Direction.L, CornerHandedness.CW),
        Corner(Direction.L, Direction.D, CornerHandedness.CCW),
        Corner(Direction.D, Direction.R, CornerHandedness.CCW),
    )
)

# Right-entry wrap mirror.
_RIGHT_ENTRY_WRAP_DOWN = TurnSequence(
    corners=(
        Corner(Direction.L, Direction.D, CornerHandedness.CCW),
        Corner(Direction.D, Direction.R, CornerHandedness.CCW),
        Corner(Direction.R, Direction.D, CornerHandedness.CW),
        Corner(Direction.D, Direction.L, CornerHandedness.CW),
    )
)

# TOP-entry L-shape variants.
_TOP_ENTRY_L_DOWN_FROM_RIGHT = TurnSequence(
    corners=(
        Corner(Direction.R, Direction.D, CornerHandedness.CW),
        Corner(Direction.D, Direction.R, CornerHandedness.CCW),
    )
)
_TOP_ENTRY_L_DOWN_FROM_LEFT = TurnSequence(
    corners=(
        Corner(Direction.L, Direction.D, CornerHandedness.CCW),
        Corner(Direction.D, Direction.L, CornerHandedness.CW),
    )
)


# ---------------------------------------------------------------------------
# WRAP_TABLE: declarative mirror of _route_inter_section's if-cascade
# ---------------------------------------------------------------------------
# Key tuple: (exit_side, entry_side, drow_sign, dcol_sign).
#
# This table covers the **cross-row** cases the current dispatcher
# fires.  Same-row L-shapes and the degenerate same-X TB BOTTOM->TOP
# straight drop are deliberately omitted: their geometric corner
# count disagrees with the propagator's row-based ``is_wrap`` flag.
# Those mismatches are flagged in the module docstring as drift the
# C1/C2 unification work needs to resolve before integration.
#
# Sources/sinks that are junctions without a clear port side use the
# literal ``"JUNCTION"`` for ``exit_side``.  Entry sides are always
# known (every inter-section edge terminates at a port).
#
# This table is forward-declared scaffolding; it is NOT called from
# the dispatcher yet.  See module docstring for the integration plan.
WRAP_TABLE: dict[tuple[str, str, int, int], WrapDescriptor] = {
    # ------- Cross-row L-shapes (dispatcher fallback line 635) ----------
    # Descending L: source exits RIGHT (LR section), target entry on
    # LEFT, one row down.  The standard right-down-right L is what the
    # dispatcher's terminal ``return _route_l_shape(...)`` produces.
    ("RIGHT", "LEFT", 1, 1): WrapDescriptor(
        kind="l_shape",
        turn_sequence=_L_RIGHT_DOWN_RIGHT,
        channel_kind="L_SHAPE",
    ),
    # Ascending L (serpentine return).
    ("RIGHT", "LEFT", -1, 1): WrapDescriptor(
        kind="l_shape",
        turn_sequence=_L_RIGHT_UP_RIGHT,
        channel_kind="L_SHAPE",
    ),
    # ------- TOP entry L-shape (dispatcher line 528-529) ----------------
    # TB section reached from an LR predecessor on the LEFT.
    ("RIGHT", "TOP", 1, 1): WrapDescriptor(
        kind="top_entry_l_shape",
        turn_sequence=_TOP_ENTRY_L_DOWN_FROM_RIGHT,
        channel_kind="L_SHAPE",
    ),
    # TB section reached from an RL predecessor on the RIGHT.
    ("LEFT", "TOP", 1, -1): WrapDescriptor(
        kind="top_entry_l_shape",
        turn_sequence=_TOP_ENTRY_L_DOWN_FROM_LEFT,
        channel_kind="L_SHAPE",
    ),
    # ------- LEFT-entry cross-row wrap (dispatcher line 608-617) --------
    # Source row above, target row below, entry on LEFT, source column
    # to the RIGHT of the target column (dx < 0).  Four-corner zigzag.
    ("RIGHT", "LEFT", 1, -1): WrapDescriptor(
        kind="left_entry_wrap",
        turn_sequence=_LEFT_ENTRY_WRAP_DOWN,
        channel_kind="WRAP",
    ),
    # Junction source (exit-chain wraps around): same shape.
    ("JUNCTION", "LEFT", 1, -1): WrapDescriptor(
        kind="left_entry_wrap",
        turn_sequence=_LEFT_ENTRY_WRAP_DOWN,
        channel_kind="WRAP",
    ),
    # ------- RIGHT-entry cross-row wrap (dispatcher line 598-599) -------
    # Source to the LEFT of the target column, entry on RIGHT.  Wraps
    # over the top of the target section and drops in from the right.
    ("LEFT", "RIGHT", 1, 1): WrapDescriptor(
        kind="right_entry_wrap",
        turn_sequence=_RIGHT_ENTRY_WRAP_DOWN,
        channel_kind="WRAP",
    ),
    # ------- TB BOTTOM exit to side-entry (dispatcher line 521-522) -----
    # TB section exits via BOTTOM, target entry on LEFT side.  Vertical
    # drop with X offsets terminating in an L-shape into the side port.
    ("BOTTOM", "LEFT", 1, 1): WrapDescriptor(
        kind="tb_bottom_exit",
        turn_sequence=_L_RIGHT_DOWN_RIGHT,
        channel_kind="TB_EXIT",
    ),
    ("BOTTOM", "LEFT", 1, -1): WrapDescriptor(
        kind="tb_bottom_exit",
        turn_sequence=_L_LEFT_DOWN_LEFT,
        channel_kind="TB_EXIT",
    ),
}


__all__ = [
    "Corner",
    "CornerHandedness",
    "TurnSequence",
    "WrapDescriptor",
    "WRAP_TABLE",
]
