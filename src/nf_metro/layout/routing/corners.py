"""Concentric corner geometry for bundled metro lines.

When multiple metro lines travel as a parallel bundle and turn a corner,
each line follows a different-radius arc to maintain the bundle's visual
ordering without crossings.  Lines on the OUTSIDE of the turn get larger
radii (wider arcs), lines on the INSIDE get smaller radii (tighter arcs).

Key invariant
-------------
A line on the left of a downward-going bundle must be on TOP of the
following horizontal if the bundle turns left, but on the BOTTOM if it
turns right.  Equivalently, the line on the outside of every corner
always gets the largest radius.

All radii are computed as::

    radius = base_radius + k * offset_step

where *k* ranges from 0 (innermost) to n-1 (outermost).  The radius
is NEVER variable beyond the line's position within the bundle.
"""

from __future__ import annotations

from nf_metro.layout.constants import CURVE_RADIUS, OFFSET_STEP

# ---------------------------------------------------------------------------
# Primitive: reversed (inner/outer) offset
# ---------------------------------------------------------------------------


def reversed_offset(offset: float, max_offset: float) -> float:
    """Flip a line's offset within a bundle.

    Maps the outermost line (offset == max_offset) to 0 and the
    innermost line (offset == 0) to max_offset.  Used whenever a
    concentric corner swaps the spatial ordering of lines in a bundle.
    """
    return max_offset - offset


# ---------------------------------------------------------------------------
# Standard inter-section L-shape (horizontal -> vertical -> horizontal)
# ---------------------------------------------------------------------------


def l_shape_radii(
    i: int,
    n: int,
    going_down: bool,
    offset_step: float = OFFSET_STEP,
    base_radius: float = CURVE_RADIUS,
) -> tuple[float, float, float]:
    """Compute offset and radii for a standard inter-section L-shape.

    An L-shape routes ``horizontal -> vertical -> horizontal`` with two
    corners.  The bundle of *n* parallel lines fans out in the vertical
    channel, and each line gets a different radius at each corner so
    the arcs are concentric (nested) rather than overlapping.

    Parameters
    ----------
    i : int
        This line's index within the bundle (0 to n-1), as assigned by
        ``compute_bundle_info()``.
    n : int
        Total number of lines in the bundle.
    going_down : bool
        ``True`` if the vertical segment goes downward (dy > 0).
    offset_step : float
        Spacing between adjacent lines in the bundle.
    base_radius : float
        Minimum curve radius (innermost line).

    Returns
    -------
    delta : float
        X offset from the vertical channel center for this line.
    r_first : float
        Corner radius at the first turn (horizontal -> vertical).
    r_second : float
        Corner radius at the second turn (vertical -> horizontal).

    Geometry
    --------
    Going DOWN (right -> down -> right):
        * Corner 1 is a CW turn.  i=0 is placed rightmost (positive
          delta), on the outside, so it gets the largest radius.
        * Corner 2 is a CCW turn.  The rightmost line is now on the
          inside, so it gets the smallest radius.

    Going UP (right -> up -> right):
        * Corner 1 is a CCW turn.  i=0 is placed leftmost (negative
          delta), on the inside, so it gets the smallest radius.
        * Corner 2 is a CW turn.  The leftmost line is now on the
          outside, so it gets the largest radius.
    """
    if going_down:
        # i=0 -> rightmost (positive delta)
        delta = ((n - 1) / 2 - i) * offset_step
        # Corner 1 (CW):  rightmost = outside -> largest radius
        r_first = base_radius + (n - 1 - i) * offset_step
        # Corner 2 (CCW): rightmost = inside  -> smallest radius
        r_second = base_radius + i * offset_step
    else:
        # i=0 -> leftmost (negative delta)
        delta = (i - (n - 1) / 2) * offset_step
        # Corner 1 (CCW): leftmost = inside  -> smallest radius
        r_first = base_radius + i * offset_step
        # Corner 2 (CW):  leftmost = outside -> largest radius
        r_second = base_radius + (n - 1 - i) * offset_step

    return delta, r_first, r_second


# ---------------------------------------------------------------------------
# TB section LEFT/RIGHT exit L-shape (vertical drop -> horizontal)
# ---------------------------------------------------------------------------


def tb_exit_corner(
    src_off: float,
    max_src_off: float,
    exit_right: bool,
    base_radius: float = CURVE_RADIUS,
) -> tuple[float, float, float]:
    """Compute offsets and radius for a TB section exit L-shape.

    Routes: vertical drop from last station -> corner -> horizontal to
    the LEFT or RIGHT exit port.

    Parameters
    ----------
    src_off : float
        This line's X offset within the TB section.
    max_src_off : float
        Maximum X offset across all lines at this station.
    exit_right : bool
        ``True`` for a RIGHT exit port, ``False`` for LEFT.
    base_radius : float
        Minimum curve radius (innermost line).

    Returns
    -------
    vert_x_off : float
        X offset for the vertical segment.
    horiz_y_off : float
        Y offset for the horizontal segment.
    corner_radius : float
        Concentric arc radius at the corner.

    Geometry
    --------
    The horizontal Y offset always uses the reversed offset so that the
    outermost vertical line (furthest from center) maps to the largest
    radius.

    RIGHT exit (DOWN -> RIGHT, CCW turn):
        Vertical X uses the non-reversed offset.  The line with the
        largest non-reversed offset is on the outside of the CCW turn.

    LEFT exit (DOWN -> LEFT, CW turn):
        Vertical X uses the reversed offset.  The line with the largest
        reversed offset is on the outside of the CW turn.
    """
    rev = reversed_offset(src_off, max_src_off)
    horiz_y_off = rev
    corner_radius = base_radius + rev

    if exit_right:
        vert_x_off = src_off
    else:
        vert_x_off = rev

    return vert_x_off, horiz_y_off, corner_radius


# ---------------------------------------------------------------------------
# TB section LEFT/RIGHT entry L-shape (horizontal -> vertical drop)
# ---------------------------------------------------------------------------


def tb_entry_corner(
    tgt_off: float,
    max_tgt_off: float,
    entry_right: bool,
    base_radius: float = CURVE_RADIUS,
) -> tuple[float, float, float]:
    """Compute offsets and radius for a TB section entry L-shape.

    Routes: horizontal from LEFT or RIGHT entry port -> corner ->
    vertical drop to the first internal station.

    Parameters
    ----------
    tgt_off : float
        This line's X offset at the target station in the TB section.
    max_tgt_off : float
        Maximum X offset across all lines at the target station.
    entry_right : bool
        ``True`` for a RIGHT entry port, ``False`` for LEFT.
    base_radius : float
        Minimum curve radius (innermost line).

    Returns
    -------
    vert_x_off : float
        X offset for the vertical segment.
    corner_radius : float
        Concentric arc radius at the corner.

    Geometry
    --------
    RIGHT entry (LEFT -> DOWN, CW turn):
        Vertical X uses the non-reversed offset.

    LEFT entry (RIGHT -> DOWN, CCW turn):
        Vertical X uses the reversed offset.

    The corner radius always uses the reversed target offset so that
    the outermost vertical line gets the largest radius.
    """
    rev = reversed_offset(tgt_off, max_tgt_off)
    corner_radius = base_radius + rev

    if entry_right:
        vert_x_off = tgt_off
    else:
        vert_x_off = rev

    return vert_x_off, corner_radius
