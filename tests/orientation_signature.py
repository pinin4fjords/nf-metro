"""Rotation-invariant summaries of a laid-out map, for orientation equivalence.

Rotating a map does not rotate its text: a label is as wide as its glyphs
whichever way the flow runs, so the *distances* a layout puts between stations
are not expected to survive a quarter turn.  What must survive is everything the
engine decides structurally -- which station comes next along the flow, which
lane it sits in, which side a port is pinned to, how far a port sits from the
box edge it is pinned to, and which sections' box edges are made to agree.

So a signature is split in two:

:func:`ordinal_signature`
    Text-free integers and ordinals (``Station.layer``/``track``, port sides,
    grid cells).  Compared for exact equality after mapping the reference
    through the transform.

:func:`relational_signature`
    Constant-driven clearances and boolean relationships (a port's clearance to
    its pinned edge, whether an entry port precedes its section's stations in
    flow order, whether cell-mates' leading box edges agree).  Also compared for
    exact equality, because none of it is a function of glyph extents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from orientation_transform import Orientation

from nf_metro.layout.geometry import AxisFrame, perpendicular_port_sides
from nf_metro.parser.model import MetroGraph, PortSide, Section

_PORT_ID = re.compile(
    r"^(?P<section>.+)__(?P<kind>entry|exit)_(?P<side>\w+)_(?P<n>\d+)$"
)


# Ports are named for the side they land on, so the same port carries a
# different id in each orientation.  The trailing counter is assigned in edge
# order, which the transform leaves alone, so re-keying on it is a bijection.
def port_key(port_id: str) -> str:
    """A port id with its side token removed, stable across orientations."""
    m = _PORT_ID.match(port_id)
    if m is None:
        return port_id
    return f"{m['section']}__{m['kind']}_{m['n']}"


@dataclass(frozen=True)
class SectionOrdinals:
    direction: str
    cell: tuple[int, int, int, int]
    # station id -> (layer, track): its rank along the flow axis and its lane.
    stations: tuple[tuple[str, int, float], ...]
    # port key -> (side, is_entry)
    ports: tuple[tuple[str, str, bool], ...]


def _frame(section: Section) -> AxisFrame:
    return AxisFrame.for_direction(section.direction, 1.0, 1.0)


def ordinal_signature(graph: MetroGraph) -> dict[str, SectionOrdinals]:
    """Per-section text-free ordinals: flow ranks, lanes, port sides, grid cell."""
    out: dict[str, SectionOrdinals] = {}
    for sid, section in graph.sections.items():
        stations = tuple(
            sorted(
                (st.id, st.layer, st.track)
                for st in graph.stations.values()
                if st.section_id == sid and not st.is_port
            )
        )
        ports = tuple(
            sorted(
                (port_key(p.id), p.side.value, p.is_entry)
                for p in graph.ports.values()
                if p.section_id == sid
            )
        )
        out[sid] = SectionOrdinals(
            direction=section.direction,
            cell=(
                section.grid_col,
                section.grid_row,
                section.grid_row_span,
                section.grid_col_span,
            ),
            stations=stations,
            ports=ports,
        )
    return out


def _box_edges(section: Section) -> dict[str, float]:
    return {
        "left": section.bbox_x,
        "right": section.bbox_x + section.bbox_w,
        "top": section.bbox_y,
        "bottom": section.bbox_y + section.bbox_h,
    }


def _port_coord_on(axis: str, port_x: float, port_y: float) -> float:
    return port_x if axis == "x" else port_y


# The side a port is pinned to fixes it on one axis; the other axis is the one it
# is free to slide along.
_SIDE_AXIS = {
    PortSide.LEFT: "x",
    PortSide.RIGHT: "x",
    PortSide.TOP: "y",
    PortSide.BOTTOM: "y",
}


@dataclass(frozen=True)
class PortRelation:
    """One port's frame-relative relationship to its section."""

    side: str
    is_entry: bool
    is_perpendicular: bool
    # Distance from the box edge the port's side names. Zero when the port sits
    # exactly on the edge; the X-axis inset makes it non-zero.
    edge_clearance: float
    # Does the port sit at or before every internal station in flow order?
    # Only meaningful for an entry port; ``None`` for an exit.
    precedes_stations: bool | None
    # Does the port sit at or after every internal station in flow order?
    follows_stations: bool | None


@dataclass(frozen=True)
class RelationalSignature:
    ports: dict[str, PortRelation] = field(default_factory=dict)
    # (axis, index) -> whether the sections sharing that grid row/column agree on
    # one of the two box edges perpendicular to the group's own axis: top or
    # bottom for a row of side-by-side sections, left or right for a column of
    # stacked ones.
    #
    # Which of the pair is deliberately aligned is not itself rotation-invariant
    # (a quarter turn carries a row's top onto a column's right), and the other
    # edge of the pair agrees only when the members happen to be equally deep --
    # an accident of glyph extents. That the group agrees on *an* edge is the
    # part the engine decides, so that is what the signature records.
    #
    # This family is weaker than the rest of the signature, and a divergence in
    # it is a lead rather than a proof.  A pass that levels a group's edges can
    # be followed by one that reclaims each box back to its own content, and
    # content extents are glyph-driven: two boxes then share an edge only where
    # their content offsets happen to coincide.  So a group_alignment difference
    # between two orientations can come from label metrics rather than from two
    # code paths, and each one needs its mechanism identified before it is
    # treated as a defect.
    #
    # The other families do not have this weakness: they are ordinals or
    # constant-driven clearances, with no dependence on text.
    aligned_groups: dict[tuple[str, int], bool] = field(default_factory=dict)


def _flow_extent(graph: MetroGraph, section: Section) -> tuple[float, float] | None:
    """The section's first and last internal-station coordinates in flow order."""
    frame = _frame(section)
    axis = frame.primary.name
    coords = [
        (st.x if axis == "x" else st.y)
        for st in graph.stations.values()
        if st.section_id == section.id and not st.is_port
    ]
    if not coords:
        return None
    sign = frame.primary_sign
    ordered = sorted(coords, key=lambda c: c * sign)
    return (ordered[0], ordered[-1])


def relational_signature(graph: MetroGraph) -> RelationalSignature:
    """Constant-driven clearances and boolean edge/order relationships."""
    ports: dict[str, PortRelation] = {}
    for port in graph.ports.values():
        section = graph.sections.get(port.section_id)
        if section is None:
            continue
        frame = _frame(section)
        edges = _box_edges(section)
        pinned_axis = _SIDE_AXIS[port.side]
        clearance = abs(
            _port_coord_on(pinned_axis, port.x, port.y) - edges[port.side.value]
        )
        extent = _flow_extent(graph, section)
        precedes = follows = None
        if extent is not None:
            flow_axis = frame.primary.name
            here = _port_coord_on(flow_axis, port.x, port.y) * frame.primary_sign
            first, last = (c * frame.primary_sign for c in extent)
            # A tolerance keeps a port seated level with the first station from
            # reading as behind it through float noise.
            precedes = here <= first + 0.5
            follows = here >= last - 0.5
        ports[port_key(port.id)] = PortRelation(
            side=port.side.value,
            is_entry=port.is_entry,
            is_perpendicular=port.side in perpendicular_port_sides(section.direction),
            edge_clearance=round(clearance, 3),
            precedes_stations=precedes,
            follows_stations=follows,
        )

    aligned: dict[tuple[str, int], bool] = {}
    for axis, index_of, edges in (
        ("row", lambda s: s.grid_row, ("top", "bottom")),
        ("col", lambda s: s.grid_col, ("left", "right")),
    ):
        groups: dict[int, list[Section]] = {}
        for section in graph.sections.values():
            groups.setdefault(index_of(section), []).append(section)
        for index, members in groups.items():
            if len(members) < 2:
                continue
            aligned[(axis, index)] = any(
                len({round(_box_edges(s)[edge], 1) for s in members}) == 1
                for edge in edges
            )
    return RelationalSignature(ports=ports, aligned_groups=aligned)


DIVERGENCE_FAMILIES = frozenset(
    {
        "sections",
        "direction",
        "cell",
        "station_ordinal",
        "port_side",
        "port_perpendicular",
        "port_clearance",
        "port_flow_end",
        "group_alignment",
    }
)
"""Every property :func:`divergences` reports on.

A caller excepting a residual keys on one of these, so the set is shared rather
than restated: a misspelled family would otherwise match nothing and read as an
unexcepted divergence in one test and a stale exception in another.
"""


@dataclass(frozen=True)
class Divergence:
    """One way a transformed layout failed to be the reference's image.

    *family* names the property that broke, so a residual can be excepted by
    the kind of defect behind it rather than by listing every orbit member.
    """

    family: str
    detail: str

    def __post_init__(self) -> None:
        if self.family not in DIVERGENCE_FAMILIES:
            raise ValueError(f"unknown divergence family {self.family!r}")

    def __str__(self) -> str:
        return f"[{self.family}] {self.detail}"


def _map_group_key(
    key: tuple[str, int], orientation: Orientation, dims: tuple[int, int]
) -> tuple[str, int]:
    """Carry a grid-group key through *orientation*.

    A quarter turn sends a row onto a column and vice versa; the reflection
    renumbers columns.  Mirrors the cell remap in
    :meth:`Orientation.cell` at the level of whole rows and columns.
    """
    axis, index = key
    cols, rows = dims
    if orientation.mirrored and axis == "col":
        index = cols - 1 - index
    for _ in range(orientation.quarter_turns):
        axis, index = ("col", rows - 1 - index) if axis == "row" else ("row", index)
        cols, rows = rows, cols
    return (axis, index)


def _grid_dims(ordinals: dict[str, SectionOrdinals]) -> tuple[int, int]:
    """The ``(cols, rows)`` extent the sections occupy, spans included."""
    cols = rows = 1
    for section in ordinals.values():
        col, row, rowspan, colspan = section.cell
        cols = max(cols, col + colspan)
        rows = max(rows, row + rowspan)
    return (cols, rows)


def divergences(
    reference: MetroGraph,
    image: MetroGraph,
    orientation: Orientation,
) -> list[Divergence]:
    """Every way *image* differs from *reference*'s image under *orientation*."""
    found: list[Divergence] = []
    ref_o, img_o = ordinal_signature(reference), ordinal_signature(image)
    if set(ref_o) != set(img_o):
        return [
            Divergence("sections", f"section set differs: {set(ref_o) ^ set(img_o)}")
        ]
    dims = _grid_dims(ref_o)

    for sid, ref in ref_o.items():
        img = img_o[sid]
        want_direction = orientation.direction(ref.direction)
        if img.direction != want_direction:
            found.append(
                Divergence(
                    "direction",
                    f"{sid}: flow {img.direction}, expected {want_direction}",
                )
            )
        want_cell = orientation.cell(ref.cell, dims)
        if img.cell != want_cell:
            found.append(
                Divergence("cell", f"{sid}: cell {img.cell}, expected {want_cell}")
            )
        ref_stations = {s[0]: s[1:] for s in ref.stations}
        img_stations = {s[0]: s[1:] for s in img.stations}
        for st in sorted(set(ref_stations) | set(img_stations)):
            if ref_stations.get(st) != img_stations.get(st):
                found.append(
                    Divergence(
                        "station_ordinal",
                        f"{sid}.{st}: (layer, track) {img_stations.get(st)}, "
                        f"expected {ref_stations.get(st)}",
                    )
                )
        want_ports = {k: (orientation.side(s), e) for k, s, e in ref.ports}
        img_ports = {k: (s, e) for k, s, e in img.ports}
        for pk in sorted(set(want_ports) | set(img_ports)):
            if want_ports.get(pk) != img_ports.get(pk):
                found.append(
                    Divergence(
                        "port_side",
                        f"{sid}.{pk}: (side, is_entry) {img_ports.get(pk)}, "
                        f"expected {want_ports.get(pk)}",
                    )
                )

    ref_r, img_r = relational_signature(reference), relational_signature(image)
    for pk, ref_p in ref_r.ports.items():
        img_p = img_r.ports.get(pk)
        if img_p is None:
            found.append(Divergence("port_side", f"port {pk} absent from the image"))
            continue
        if img_p.is_perpendicular != ref_p.is_perpendicular:
            found.append(
                Divergence(
                    "port_perpendicular",
                    f"{pk}[{img_p.side}]: perpendicular {img_p.is_perpendicular}, "
                    f"expected {ref_p.is_perpendicular}",
                )
            )
        if img_p.edge_clearance != ref_p.edge_clearance:
            found.append(
                Divergence(
                    "port_clearance",
                    f"{pk}[{img_p.side}]: edge clearance {img_p.edge_clearance}, "
                    f"expected {ref_p.edge_clearance}",
                )
            )
        for prop in ("precedes_stations", "follows_stations"):
            if getattr(img_p, prop) != getattr(ref_p, prop):
                found.append(
                    Divergence(
                        "port_flow_end",
                        f"{pk}[{img_p.side}]: {prop} {getattr(img_p, prop)}, "
                        f"expected {getattr(ref_p, prop)}",
                    )
                )

    for key, ref_aligned in ref_r.aligned_groups.items():
        want_key = _map_group_key(key, orientation, dims)
        img_aligned = img_r.aligned_groups.get(want_key)
        if img_aligned is None:
            found.append(Divergence("group_alignment", f"grid group {want_key} absent"))
        elif img_aligned != ref_aligned:
            found.append(
                Divergence(
                    "group_alignment",
                    f"grid group {key} -> {want_key}: shares an edge "
                    f"{img_aligned}, expected {ref_aligned}",
                )
            )
    return found
