"""Immutable settled geometry consumed by render artifact emitters."""
# ruff: noqa: ANN401

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from nf_metro.parser.model import LineSpread, MetroGraph


@dataclass(frozen=True)
class FrozenMap(Mapping[Any, Any]):
    """Insertion-ordered immutable mapping used inside a :class:`RenderPlan`."""

    entries: tuple[tuple[Any, Any], ...]

    def __getitem__(self, key: Any) -> Any:
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[Any]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class FrozenRecord:
    """A named immutable record copied from a parser or layout value."""

    kind: str
    values: FrozenMap

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "values")
        try:
            return values[name]
        except KeyError as exc:
            if self.kind == "Station":
                if name == "is_terminus":
                    return bool(self.terminus_labels)
                if name == "is_blank_terminus":
                    return bool(self.terminus_labels) and not self.label.strip()
                if name == "is_captioned_terminus":
                    return (
                        bool(self.terminus_labels)
                        and not self.label.strip()
                        and any(self.terminus_names)
                    )
            if self.kind == "Section" and name == "port_ids":
                return frozenset((*self.entry_ports, *self.exit_ports))
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class FrozenGraph(FrozenRecord):
    """Read-only render view of a settled ``MetroGraph``."""

    def station_lines(self, station_id: str) -> list[str]:
        lines = {
            edge.line_id
            for edge in self.edges
            if edge.source == station_id or edge.target == station_id
        }
        return sorted(lines)

    def station_lines_ordered(self, station_id: str) -> list[str]:
        served = set(self.station_lines(station_id))
        return [line_id for line_id in self.lines if line_id in served]

    def edges_from(self, station_id: str) -> list[FrozenRecord]:
        return [edge for edge in self.edges if edge.source == station_id]

    def edges_to(self, station_id: str) -> list[FrozenRecord]:
        return [edge for edge in self.edges if edge.target == station_id]

    def station_for_edge_source(self, edge: FrozenRecord) -> FrozenRecord:
        return self.stations[edge.source]

    def station_for_edge_target(self, edge: FrozenRecord) -> FrozenRecord:
        return self.stations[edge.target]

    def station_is_rail(self, station_id: str) -> bool:
        station = self.stations.get(station_id)
        section_id = station.section_id if station is not None else None
        return self.section_line_spread(section_id) is LineSpread.RAILS

    def section_line_spread(self, section_id: str | None) -> LineSpread:
        if section_id is not None and section_id in self.line_spread_overrides:
            return self.line_spread_overrides[section_id]
        return self.line_spread

    @property
    def has_rail_sections(self) -> bool:
        return self.line_spread is LineSpread.RAILS or any(
            mode is LineSpread.RAILS for mode in self.line_spread_overrides.values()
        )

    @property
    def real_sections(self) -> dict[str, FrozenRecord]:
        return {
            section_id: section
            for section_id, section in self.sections.items()
            if not section.is_implicit
        }


@dataclass(frozen=True)
class RenderPlan:
    """Complete immutable render state, in SVG user-space pixels."""

    graph: FrozenGraph
    station_offsets: FrozenMap
    routes: tuple[FrozenRecord, ...]
    edge_routes: tuple[FrozenRecord, ...]
    bridge_breaks: tuple[tuple[FrozenRecord, ...], ...]
    labels: tuple[FrozenRecord, ...]
    header_placements: FrozenMap
    group_bands: tuple[FrozenRecord, ...]
    positive_fan_sections: frozenset[str]
    svg_width: int
    svg_height: int
    padding: float
    legend_x: float
    legend_y: float
    legend_w: float
    legend_h: float
    show_legend: bool
    show_logo: bool
    logo_in_legend: bool
    adaptive_logo: bool
    effective_logo: str
    resolved_logo_light: str
    resolved_logo_dark: str
    logo_x: float
    logo_y: float
    logo_w: float
    logo_h: float
    legend_logo_size: tuple[float, float] | None
    manifest: FrozenMap | None

    def offset_polylines(
        self,
    ) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
        """Return the exact per-line polylines intended for SVG emission."""
        from nf_metro.layout.routing.common import apply_route_offsets

        return tuple(
            (
                route.line_id,
                tuple(
                    apply_route_offsets(
                        route,  # type: ignore[arg-type]
                        self.station_offsets,  # type: ignore[arg-type]
                    )
                ),
            )
            for route in self.routes
        )


def freeze_render_value(value: Any) -> Any:
    """Recursively copy render input into immutable scalar and tuple records."""
    if isinstance(value, (str, bytes, int, float, bool, type(None), Enum)):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(
            tuple(
                (freeze_render_value(key), freeze_render_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        if hasattr(value, "_fields"):
            return FrozenRecord(
                type(value).__name__,
                FrozenMap(
                    tuple(
                        (name, freeze_render_value(getattr(value, name)))
                        for name in value._fields
                    )
                ),
            )
        return tuple(freeze_render_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_render_value(item) for item in value)
    if is_dataclass(value):
        record_type = FrozenGraph if isinstance(value, MetroGraph) else FrozenRecord
        return record_type(
            type(value).__name__,
            FrozenMap(
                tuple(
                    (field.name, freeze_render_value(getattr(value, field.name)))
                    for field in fields(value)
                    if field.name
                    not in {
                        "_station_lines_cache",
                        "_edges_from_cache",
                        "_edges_to_cache",
                        "_junction_ids_cache",
                    }
                )
            ),
        )
    raise TypeError(f"RenderPlan cannot freeze {type(value).__name__}")


def thaw_render_value(value: Any) -> Any:
    """Convert immutable plan containers to ordinary serialization values."""
    if isinstance(value, FrozenMap):
        return {
            thaw_render_value(key): thaw_render_value(item)
            for key, item in value.entries
        }
    if isinstance(value, FrozenRecord):
        return thaw_render_value(value.values)
    if isinstance(value, tuple | frozenset):
        return [thaw_render_value(item) for item in value]
    return value


def contains_mutable_model_reference(value: Any) -> bool:
    """Whether a plan value retains a parser/layout model instance."""
    from nf_metro.layout.routing.common import RoutedPath
    from nf_metro.parser.model import Edge, Port, Section, Station

    model_types = (MetroGraph, Station, Section, Edge, Port, RoutedPath)
    if isinstance(value, model_types):
        return True
    if isinstance(value, FrozenMap):
        return any(
            contains_mutable_model_reference(key)
            or contains_mutable_model_reference(item)
            for key, item in value.entries
        )
    if isinstance(value, FrozenRecord):
        return contains_mutable_model_reference(value.values)
    if isinstance(value, RenderPlan):
        return any(
            contains_mutable_model_reference(getattr(value, field.name))
            for field in fields(value)
        )
    if isinstance(value, tuple | frozenset):
        return any(contains_mutable_model_reference(item) for item in value)
    return False
