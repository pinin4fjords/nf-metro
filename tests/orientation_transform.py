"""Dihedral transforms of a metro map's ``.mmd`` source.

A map's geometry is carried by three direction-bearing directives: each
section's ``%%metro direction:`` flow, the ``%%metro grid:`` cell it occupies,
and the ``%%metro entry:``/``exit:`` port sides.  Rewriting all three together
rotates or reflects the whole map, so the engine should lay the result out as
the same picture under the same transform.

The section meta-graph is always left-to-right (grid columns run along X, rows
along Y; the ``graph`` header's direction is ignored), so the grid remap is what
carries a rotation at the meta level while ``direction:`` carries it inside each
section.

:class:`Orientation` is the group element; :func:`transform_source` applies it.
The eight elements form the dihedral group of the square: four quarter turns,
each optionally preceded by a reflection on the vertical axis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from nf_metro.parser.model import MetroGraph

# A quarter turn clockwise sends a rightward flow downward, a downward flow
# leftward, and so on around the cycle.
_TURN_DIRECTION = {"LR": "TB", "TB": "RL", "RL": "BT", "BT": "LR"}
# Reflecting on the vertical axis reverses a horizontal flow and leaves a
# vertical one alone.
_MIRROR_DIRECTION = {"LR": "RL", "RL": "LR", "TB": "TB", "BT": "BT"}

# Port sides follow their outward normal through the same maps.
_TURN_SIDE = {"left": "top", "top": "right", "right": "bottom", "bottom": "left"}
_MIRROR_SIDE = {"left": "right", "right": "left", "top": "top", "bottom": "bottom"}


@dataclass(frozen=True)
class Orientation:
    """One element of the square's dihedral group.

    *quarter_turns* counts 90-degree clockwise turns; *mirrored* reflects on the
    vertical axis first.  ``Orientation(0, False)`` is the identity, so
    transforming a source under it must reproduce it byte for byte.
    """

    quarter_turns: int
    mirrored: bool

    def __post_init__(self) -> None:
        if not 0 <= self.quarter_turns <= 3:
            raise ValueError(f"quarter_turns must be 0..3, got {self.quarter_turns}")

    @property
    def name(self) -> str:
        return f"{'m' if self.mirrored else 'r'}{self.quarter_turns * 90}"

    @property
    def is_identity(self) -> bool:
        return self.quarter_turns == 0 and not self.mirrored

    @property
    def swaps_axes(self) -> bool:
        """``True`` when the transform exchanges the X and Y axes."""
        return self.quarter_turns % 2 == 1

    @property
    def flips_handedness(self) -> bool:
        """``True`` for the reflections, which reverse the sense of a turn."""
        return self.mirrored

    def direction(self, direction: str) -> str:
        """Map a section flow direction through the transform."""
        result = _MIRROR_DIRECTION[direction] if self.mirrored else direction
        for _ in range(self.quarter_turns):
            result = _TURN_DIRECTION[result]
        return result

    def side(self, side: str) -> str:
        """Map a port side (lowercase ``left``/``right``/``top``/``bottom``)."""
        result = _MIRROR_SIDE[side] if self.mirrored else side
        for _ in range(self.quarter_turns):
            result = _TURN_SIDE[result]
        return result

    def cell(
        self, cell: tuple[int, int, int, int], dims: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Map a ``(col, row, rowspan, colspan)`` cell within a ``(cols, rows)`` grid.

        The tuple order matches the ``%%metro grid:`` directive's
        ``col,row,rowspan,colspan``.
        """
        col, row, rowspan, colspan = cell
        cols, rows = dims
        if self.mirrored:
            col = cols - col - colspan
        for _ in range(self.quarter_turns):
            # Clockwise: the leftmost column becomes the top row, so the new
            # column index counts back from the bottom of the old grid.
            col, row, rowspan, colspan = (rows - row - rowspan, col, colspan, rowspan)
            cols, rows = rows, cols
        return (col, row, rowspan, colspan)

    def grid_dims(self, dims: tuple[int, int]) -> tuple[int, int]:
        """Map the grid's ``(cols, rows)`` extent through the transform."""
        return (dims[1], dims[0]) if self.swaps_axes else dims


def all_orientations() -> Iterator[Orientation]:
    """The eight dihedral elements, identity first."""
    for mirrored in (False, True):
        for turns in range(4):
            yield Orientation(turns, mirrored)


def non_identity_orientations() -> Iterator[Orientation]:
    """The seven elements that actually move the map."""
    return (o for o in all_orientations() if not o.is_identity)


_DIRECTIVE = re.compile(r"^(\s*%%metro\s+)(\w+)(\s*:\s*)(.*?)(\s*)$")


def _parse_grid_value(value: str) -> tuple[str, tuple[int, int, int, int]] | None:
    """Split a ``grid:`` value into its section-id field and numeric cell."""
    fields = value.split("|")
    if len(fields) < 2:
        return None
    coords = [c.strip() for c in fields[1].split(",")]
    if len(coords) < 2:
        return None
    try:
        numbers = [int(c) for c in coords]
    except ValueError:
        return None
    col, row = numbers[0], numbers[1]
    rowspan = numbers[2] if len(numbers) >= 3 else 1
    colspan = numbers[3] if len(numbers) >= 4 else 1
    return fields[0], (col, row, rowspan, colspan)


def grid_dims(source: str) -> tuple[int, int]:
    """The ``(cols, rows)`` extent spanned by *source*'s ``grid:`` directives.

    A source with no ``grid:`` directives spans a single cell, so the identity
    is the only transform that can be checked against it meaningfully.
    """
    cols = rows = 1
    for line in source.splitlines():
        m = _DIRECTIVE.match(line)
        if not m or m.group(2) != "grid":
            continue
        parsed = _parse_grid_value(m.group(4))
        if parsed is None:
            continue
        col, row, rowspan, colspan = parsed[1]
        cols = max(cols, col + colspan)
        rows = max(rows, row + rowspan)
    return (cols, rows)


def transformable_reason(graph: MetroGraph) -> str | None:
    """Why *graph*'s source cannot be transformed, or ``None`` when it can.

    Rewriting the directives only carries a map's geometry when the author
    stated all of it: a section whose flow or cell was inferred is re-inferred
    from the rotated source, and inference answering differently is not the
    engine laying one geometry out two ways.

    Effective directions carry independent author-ownership and lock state, so
    a resolve-time pin never makes an inferred direction transformable.
    """
    sections = set(graph.sections)
    if len(sections) < 2:
        return "single section: every transform is the identity"
    authored = {
        sid for sid in sections if graph.layout_provenance.author_owns_direction(sid)
    }
    if inferred := sections - authored:
        return f"inferred flow direction: {sorted(inferred)}"
    gridded = {sid for sid in sections if graph.layout_provenance.author_owns_grid(sid)}
    if ungridded := sections - gridded:
        return f"inferred grid cell: {sorted(ungridded)}"
    return None


def _format_cell(cell: tuple[int, int, int, int]) -> str:
    """Render a cell as ``col,row``, adding spans only when they carry meaning."""
    col, row, rowspan, colspan = cell
    if colspan != 1:
        return f"{col},{row},{rowspan},{colspan}"
    if rowspan != 1:
        return f"{col},{row},{rowspan}"
    return f"{col},{row}"


def transform_source(source: str, orientation: Orientation) -> str:
    """Rewrite *source*'s direction-bearing directives under *orientation*.

    Everything else -- titles, line definitions, stations, edges -- is carried
    through untouched, so the transformed map describes the same pipeline.
    """
    if orientation.is_identity:
        return source

    dims = grid_dims(source)
    out: list[str] = []
    for line in source.splitlines(keepends=True):
        m = _DIRECTIVE.match(line.rstrip("\n"))
        if m is None:
            out.append(line)
            continue
        indent, key, sep, value, trailing = m.groups()
        newline = line[len(line.rstrip("\n")) :]
        if key == "direction":
            value = orientation.direction(value.strip().upper())
        elif key in ("entry", "exit"):
            fields = value.split("|")
            if len(fields) >= 2:
                mapped = orientation.side(fields[0].strip().lower())
                fields[0] = f"{mapped} "
                value = "|".join(fields)
        elif key == "grid":
            parsed = _parse_grid_value(value)
            if parsed is not None:
                ids, cell = parsed
                mapped = orientation.cell(cell, dims)
                value = f"{ids}| {_format_cell(mapped)}"
        out.append(f"{indent}{key}{sep}{value}{trailing}{newline}")
    return "".join(out)
