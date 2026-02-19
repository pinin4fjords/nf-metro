"""Tests for the concentric corner geometry helpers.

These tests directly verify the invariants that are easy to accidentally
break when modifying routing logic:

1. Radii are always ``base_radius + k * offset_step`` (never variable).
2. The outermost line at every corner gets the largest radius.
3. Bundle ordering is preserved through L-shapes (no crossings).
4. ``going_down=True`` and ``going_down=False`` are mirror-symmetric.
"""

from __future__ import annotations

import pytest

from nf_metro.layout.constants import CURVE_RADIUS, OFFSET_STEP
from nf_metro.layout.routing.corners import (
    l_shape_radii,
    reversed_offset,
    tb_entry_corner,
    tb_exit_corner,
)

# ---------------------------------------------------------------------------
# reversed_offset
# ---------------------------------------------------------------------------


class TestReversedOffset:
    def test_zero_becomes_max(self):
        assert reversed_offset(0.0, 6.0) == 6.0

    def test_max_becomes_zero(self):
        assert reversed_offset(6.0, 6.0) == 0.0

    def test_middle_stays(self):
        assert reversed_offset(3.0, 6.0) == 3.0

    def test_involutory(self):
        """Reversing twice gives back the original offset."""
        for off in [0.0, 1.5, 3.0, 4.5, 6.0]:
            assert reversed_offset(reversed_offset(off, 6.0), 6.0) == pytest.approx(off)

    def test_zero_bundle(self):
        """Single line: offset and max are both 0."""
        assert reversed_offset(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# l_shape_radii: invariant tests
# ---------------------------------------------------------------------------


class TestLShapeRadii:
    """Test the standard inter-section L-shape (horiz -> vert -> horiz)."""

    @pytest.mark.parametrize("going_down", [True, False])
    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_radii_are_concentric(self, n: int, going_down: bool):
        """All radii must be base_radius + k * offset_step for integer k."""
        for i in range(n):
            _delta, r1, r2 = l_shape_radii(i, n, going_down)
            # Check r1 is an exact multiple of offset_step above base
            k1 = (r1 - CURVE_RADIUS) / OFFSET_STEP
            assert k1 == pytest.approx(round(k1)), (
                f"r_first={r1} is not base + k*step for i={i}, n={n}, down={going_down}"
            )
            k2 = (r2 - CURVE_RADIUS) / OFFSET_STEP
            assert k2 == pytest.approx(round(k2)), (
                f"r_second={r2} not base+k*step i={i} n={n} down={going_down}"
            )

    @pytest.mark.parametrize("going_down", [True, False])
    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_radii_cover_full_range(self, n: int, going_down: bool):
        """The set of radii for a bundle must span [base, base + (n-1)*step]."""
        r1s = []
        r2s = []
        for i in range(n):
            _, r1, r2 = l_shape_radii(i, n, going_down)
            r1s.append(r1)
            r2s.append(r2)
        expected_min = CURVE_RADIUS
        expected_max = CURVE_RADIUS + (n - 1) * OFFSET_STEP
        assert min(r1s) == pytest.approx(expected_min)
        assert max(r1s) == pytest.approx(expected_max)
        assert min(r2s) == pytest.approx(expected_min)
        assert max(r2s) == pytest.approx(expected_max)

    @pytest.mark.parametrize("going_down", [True, False])
    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_all_radii_distinct(self, n: int, going_down: bool):
        """Each line in the bundle must get a distinct radius at each corner."""
        r1s = set()
        r2s = set()
        for i in range(n):
            _, r1, r2 = l_shape_radii(i, n, going_down)
            r1s.add(round(r1, 6))
            r2s.add(round(r2, 6))
        assert len(r1s) == n
        assert len(r2s) == n

    @pytest.mark.parametrize("going_down", [True, False])
    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_r_first_and_r_second_are_complementary(self, n: int, going_down: bool):
        """For each line, r_first + r_second must equal 2*base + (n-1)*step.

        This ensures that the line on the outside of corner 1 is on the
        inside of corner 2 (and vice versa), which prevents crossings.
        """
        expected_sum = 2 * CURVE_RADIUS + (n - 1) * OFFSET_STEP
        for i in range(n):
            _, r1, r2 = l_shape_radii(i, n, going_down)
            assert r1 + r2 == pytest.approx(expected_sum), (
                f"r1+r2={r1 + r2} != {expected_sum} for i={i}, n={n}, down={going_down}"
            )

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_mirror_symmetry(self, n: int):
        """going_down and going_up must produce mirror-symmetric results.

        going_down line i=0 is rightmost; going_up line n-1 is also
        rightmost.  So going_down[i] must match going_up[n-1-i] with
        the same delta, same r_first, and same r_second - they occupy
        the same spatial position in the vertical channel.
        """
        for i in range(n):
            d_down, r1_down, r2_down = l_shape_radii(i, n, going_down=True)
            d_up, r1_up, r2_up = l_shape_radii(n - 1 - i, n, going_down=False)
            assert d_down == pytest.approx(d_up), (
                f"delta mismatch: down[{i}]={d_down}, up[{n - 1 - i}]={d_up}"
            )
            assert r1_down == pytest.approx(r1_up)
            assert r2_down == pytest.approx(r2_up)

    @pytest.mark.parametrize("going_down", [True, False])
    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_no_crossing_in_vertical_channel(self, n: int, going_down: bool):
        """Lines must not cross in the vertical channel.

        The delta offsets must be strictly monotonic (either all
        increasing or all decreasing with i).
        """
        deltas = [l_shape_radii(i, n, going_down)[0] for i in range(n)]
        diffs = [deltas[j + 1] - deltas[j] for j in range(n - 1)]
        # All diffs must have the same sign (strictly monotonic)
        assert all(d > 0 for d in diffs) or all(d < 0 for d in diffs), (
            f"Deltas not monotonic: {deltas} for n={n}, down={going_down}"
        )

    @pytest.mark.parametrize("going_down", [True, False])
    def test_outermost_gets_largest_radius_corner1(self, going_down: bool):
        """At corner 1, the spatially outermost line gets the largest radius.

        Going DOWN: rightmost (largest delta) should have largest r_first.
        Going UP: leftmost (smallest delta) should have... smallest r_first
        (because leftmost is on the inside of the CCW turn).
        """
        n = 4
        results = [l_shape_radii(i, n, going_down) for i in range(n)]
        deltas = [r[0] for r in results]
        r_firsts = [r[1] for r in results]

        if going_down:
            # CW turn: the line with the largest (most positive) delta is
            # outermost and should have the largest r_first.
            outermost_idx = deltas.index(max(deltas))
            assert r_firsts[outermost_idx] == max(r_firsts)
        else:
            # CCW turn: the line with the most negative delta (leftmost)
            # is on the inside and should have the smallest r_first.
            innermost_idx = deltas.index(min(deltas))
            assert r_firsts[innermost_idx] == min(r_firsts)

    def test_single_line(self):
        """A single-line bundle should get base_radius at both corners."""
        delta, r1, r2 = l_shape_radii(0, 1, going_down=True)
        assert delta == 0.0
        assert r1 == CURVE_RADIUS
        assert r2 == CURVE_RADIUS


# ---------------------------------------------------------------------------
# tb_exit_corner
# ---------------------------------------------------------------------------


class TestTbExitCorner:
    """Test the TB section LEFT/RIGHT exit L-shape."""

    @pytest.mark.parametrize("exit_right", [True, False])
    def test_single_line(self, exit_right: bool):
        """Single line: all offsets zero, radius = base."""
        vx, hy, r = tb_exit_corner(0.0, 0.0, exit_right)
        assert vx == 0.0
        assert hy == 0.0
        assert r == CURVE_RADIUS

    @pytest.mark.parametrize("exit_right", [True, False])
    def test_radius_uses_reversed_offset(self, exit_right: bool):
        """Radius is always base + reversed_offset, never the raw offset."""
        for src_off in [0.0, 3.0, 6.0]:
            max_off = 6.0
            _, _, r = tb_exit_corner(src_off, max_off, exit_right)
            expected = CURVE_RADIUS + (max_off - src_off)
            assert r == pytest.approx(expected)

    @pytest.mark.parametrize("exit_right", [True, False])
    def test_horiz_y_is_reversed(self, exit_right: bool):
        """Horizontal Y offset is always the reversed source offset."""
        for src_off in [0.0, 3.0, 6.0]:
            max_off = 6.0
            _, hy, _ = tb_exit_corner(src_off, max_off, exit_right)
            assert hy == pytest.approx(max_off - src_off)

    def test_right_exit_vert_x_is_raw(self):
        """RIGHT exit: vertical X offset = raw source offset."""
        vx, _, _ = tb_exit_corner(3.0, 6.0, exit_right=True)
        assert vx == pytest.approx(3.0)

    def test_left_exit_vert_x_is_reversed(self):
        """LEFT exit: vertical X offset = reversed source offset."""
        vx, _, _ = tb_exit_corner(3.0, 6.0, exit_right=False)
        assert vx == pytest.approx(3.0)  # reversed_offset(3, 6) = 3

        # More telling: check an asymmetric case
        vx, _, _ = tb_exit_corner(0.0, 6.0, exit_right=False)
        assert vx == pytest.approx(6.0)

        vx, _, _ = tb_exit_corner(6.0, 6.0, exit_right=False)
        assert vx == pytest.approx(0.0)

    @pytest.mark.parametrize("exit_right", [True, False])
    def test_outermost_line_gets_largest_radius(self, exit_right: bool):
        """In a 3-line bundle, the outermost line at the corner must have
        the largest radius."""
        offsets = [0.0, 3.0, 6.0]
        max_off = 6.0
        radii = [tb_exit_corner(o, max_off, exit_right)[2] for o in offsets]
        # offset 0.0 -> reversed 6.0 -> largest radius
        assert radii[0] == max(radii)
        # offset 6.0 -> reversed 0.0 -> smallest radius
        assert radii[2] == min(radii)

    @pytest.mark.parametrize("exit_right", [True, False])
    def test_radii_are_concentric(self, exit_right: bool):
        """All radii must be base + k * step (not arbitrary values)."""
        offsets = [i * OFFSET_STEP for i in range(4)]
        max_off = max(offsets)
        for off in offsets:
            _, _, r = tb_exit_corner(off, max_off, exit_right)
            k = (r - CURVE_RADIUS) / OFFSET_STEP
            assert k == pytest.approx(round(k))


# ---------------------------------------------------------------------------
# tb_entry_corner
# ---------------------------------------------------------------------------


class TestTbEntryCorner:
    """Test the TB section LEFT/RIGHT entry L-shape."""

    @pytest.mark.parametrize("entry_right", [True, False])
    def test_single_line(self, entry_right: bool):
        """Single line: offset zero, radius = base."""
        vx, r = tb_entry_corner(0.0, 0.0, entry_right)
        assert vx == 0.0
        assert r == CURVE_RADIUS

    @pytest.mark.parametrize("entry_right", [True, False])
    def test_radius_uses_reversed_offset(self, entry_right: bool):
        """Radius is always base + reversed_offset."""
        for tgt_off in [0.0, 3.0, 6.0]:
            max_off = 6.0
            _, r = tb_entry_corner(tgt_off, max_off, entry_right)
            expected = CURVE_RADIUS + (max_off - tgt_off)
            assert r == pytest.approx(expected)

    def test_right_entry_vert_x_is_raw(self):
        """RIGHT entry: vertical X offset = raw target offset."""
        vx, _ = tb_entry_corner(3.0, 6.0, entry_right=True)
        assert vx == pytest.approx(3.0)

    def test_left_entry_vert_x_is_reversed(self):
        """LEFT entry: vertical X offset = reversed target offset."""
        vx, _ = tb_entry_corner(0.0, 6.0, entry_right=False)
        assert vx == pytest.approx(6.0)

        vx, _ = tb_entry_corner(6.0, 6.0, entry_right=False)
        assert vx == pytest.approx(0.0)

    @pytest.mark.parametrize("entry_right", [True, False])
    def test_mirrors_exit(self, entry_right: bool):
        """Entry and exit should produce the same radius for the same offset.

        The vertical X offset direction matches (both use reversed for
        LEFT, raw for RIGHT), and the radius is the same reversed-offset
        formula.
        """
        for off in [0.0, 3.0, 6.0]:
            max_off = 6.0
            vx_exit, _, r_exit = tb_exit_corner(off, max_off, exit_right=entry_right)
            vx_entry, r_entry = tb_entry_corner(off, max_off, entry_right=entry_right)
            assert r_exit == pytest.approx(r_entry)
            assert vx_exit == pytest.approx(vx_entry)
