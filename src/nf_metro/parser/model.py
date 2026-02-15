"""Data model for metro map graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PortSide(Enum):
    """Side of a section boundary where a port is located."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class MetroLine:
    """A metro line (colored route through the graph)."""

    id: str
    display_name: str
    color: str


@dataclass
class Station:
    """A node/station in the metro map."""

    id: str
    label: str
    section_id: str | None = None
    is_port: bool = False
    # Populated by layout engine
    x: float = 0.0
    y: float = 0.0
    layer: int = 0
    track: float = 0.0


@dataclass
class Edge:
    """A directed edge between stations, belonging to a metro line."""

    source: str
    target: str
    line_id: str


@dataclass
class Port:
    """A synthetic entry/exit point on a section boundary.

    Ports are created from inter-section edges and explicit %%metro entry/exit
    directives. They become invisible stations that participate in layout but
    are skipped during rendering.
    """

    id: str
    section_id: str
    side: PortSide
    line_ids: list[str] = field(default_factory=list)
    is_entry: bool = True
    x: float = 0.0
    y: float = 0.0


@dataclass
class Section:
    """A first-class visual grouping of stations (subgraph).

    Used with the new subgraph-based format where sections are explicit
    Mermaid subgraphs with entry/exit port directives.
    """

    id: str
    name: str
    number: int = 0
    station_ids: list[str] = field(default_factory=list)
    internal_edges: list[Edge] = field(default_factory=list)
    entry_ports: list[str] = field(default_factory=list)  # port IDs
    exit_ports: list[str] = field(default_factory=list)  # port IDs
    # Hints from %%metro entry/exit directives: list of (side, [line_ids])
    exit_hints: list[tuple[PortSide, list[str]]] = field(default_factory=list)
    entry_hints: list[tuple[PortSide, list[str]]] = field(default_factory=list)
    # Internal flow direction ("LR" = left-to-right, "TB" = top-to-bottom)
    direction: str = "LR"
    # Bounding box (set by layout engine)
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    # Grid position (for section placement)
    grid_col: int = -1  # -1 means auto
    grid_row: int = -1
    grid_row_span: int = 1
    grid_col_span: int = 1
    # Global offset (set by section placement)
    offset_x: float = 0.0
    offset_y: float = 0.0



@dataclass
class RouteSegment:
    """A segment of a routed edge path (populated by routing engine)."""

    x1: float
    y1: float
    x2: float
    y2: float
    line_id: str
    edge: Edge | None = None


@dataclass
class MetroGraph:
    """Complete metro map graph definition."""

    title: str = ""
    style: str = "dark"
    lines: dict[str, MetroLine] = field(default_factory=dict)
    stations: dict[str, Station] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    sections: dict[str, Section] = field(default_factory=dict)
    ports: dict[str, Port] = field(default_factory=dict)
    junctions: list[str] = field(default_factory=list)
    grid_overrides: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    legend_position: str = "bottom"

    def add_line(self, line: MetroLine) -> None:
        self.lines[line.id] = line

    def add_station(self, station: Station) -> None:
        self.stations[station.id] = station

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def add_section(self, section: Section) -> None:
        self.sections[section.id] = section

    def add_port(self, port: Port) -> None:
        self.ports[port.id] = port
        # Also add as a station so it participates in layout
        self.stations[port.id] = Station(
            id=port.id,
            label="",
            section_id=port.section_id,
            is_port=True,
        )
        # Register with the section
        section = self.sections.get(port.section_id)
        if section:
            section.station_ids.append(port.id)
            if port.is_entry:
                section.entry_ports.append(port.id)
            else:
                section.exit_ports.append(port.id)

    def station_lines(self, station_id: str) -> list[str]:
        """Return line IDs that pass through a station."""
        line_ids = set()
        for edge in self.edges:
            if edge.source == station_id or edge.target == station_id:
                line_ids.add(edge.line_id)
        return sorted(line_ids)

    def line_stations(self, line_id: str) -> list[str]:
        """Return station IDs on a line, in edge order."""
        stations = []
        seen = set()
        for edge in self.edges:
            if edge.line_id == line_id:
                if edge.source not in seen:
                    stations.append(edge.source)
                    seen.add(edge.source)
                if edge.target not in seen:
                    stations.append(edge.target)
                    seen.add(edge.target)
        return stations

    def inter_section_edges(self) -> list[Edge]:
        """Return edges that cross section boundaries."""
        result = []
        for edge in self.edges:
            src_section = self.section_for_station(edge.source)
            tgt_section = self.section_for_station(edge.target)
            if src_section and tgt_section and src_section != tgt_section:
                result.append(edge)
        return result

    def section_for_station(self, station_id: str) -> str | None:
        """Return the section ID containing a station, or None."""
        station = self.stations.get(station_id)
        if station:
            return station.section_id
        return None
