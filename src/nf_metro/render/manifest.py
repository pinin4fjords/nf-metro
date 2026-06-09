"""Self-describing JSON manifest embedded in every rendered SVG.

The rendered SVG is meant to be a durable, machine-readable contract: a single
committed file that downstream consumers can drive without re-running the
layout engine or holding a :class:`~nf_metro.parser.model.MetroGraph` in
memory.  Two redundant mechanisms carry the data, and this module owns the
first:

1. A JSON manifest embedded in a ``<metadata>`` element (built here).
2. ``data-metro-*`` attributes on each station's ``<g>`` element (emitted by
   :mod:`nf_metro.render.svg`).

The contract between the two halves: a station's ``id`` in the manifest is the
join key and **equals** ``data-metro-station="<id>"`` on its ``<g>`` element, so
a consumer can go manifest->element and element->manifest without guessing.

Coordinate space
-----------------
``x``/``y``/``r`` are absolute SVG user units inside the ``viewBox`` declared by
the manifest's ``width``/``height`` (the renderer emits ``viewBox="0 0 width
height"`` with no outer transform), so an overlay sharing that viewBox lines up
exactly.  Coordinates are rounded to one decimal place, matching the live
server's overlay geometry so the two never drift apart.

Process matching
----------------
``station.processes`` are regular expressions matched **case-insensitively**
against the **fully-qualified** Nextflow process name (the ``match`` block makes
this explicit for non-Python consumers).  Patterns must stay within a portable
regex subset common to Python ``re`` and JavaScript ``RegExp`` -- plain
character classes, anchors, ``.``/``*``/``+``/``?``, bounded quantifiers
``{m,n}``, alternation, and groups -- so the two implementations cannot diverge.
Avoid Python-only constructs (named groups ``(?P<>)``, inline flags ``(?i)``,
possessive quantifiers, ``\\Z``, etc.).  A process may legitimately match more
than one station; resolving that is a consumer-side policy decision, not a
schema error (``check-mapping`` flags it as ``ambiguous``).

Forward compatibility
----------------------
Consumers MUST ignore unknown fields so the schema can grow within a major
``version``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nf_metro.parser.model import MetroGraph

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_ELEMENT_ID",
    "build_manifest",
    "manifest_json",
    "manifest_metadata_svg",
    "read_manifest",
    "match_station_ids",
]

# Bump the major part for a breaking change; minor for additive fields (which
# consumers tolerate via the "ignore unknown fields" rule).
MANIFEST_SCHEMA_VERSION = "1.0"

# The id of the <metadata> element carrying the manifest; the stable anchor a
# consumer searches for in the SVG.
MANIFEST_ELEMENT_ID = "nf-metro-manifest"

# How station.processes are matched, made explicit for non-Python consumers.
_MATCH_SPEC = {"target": "fqProcessName", "type": "regex", "flags": "i"}


def _round1(value: float) -> float:
    """Round a coordinate to one decimal place, matching the live overlay."""
    return round(float(value), 1)


def build_manifest(
    graph: MetroGraph,
    *,
    width: int,
    height: int,
    station_radius: float,
) -> dict[str, Any]:
    """Build the manifest dict from the graph the renderer just laid out.

    Coordinates and the process mapping are read from the same fields the live
    server uses (``graph.stations[].x/.y`` and ``graph.process_mapping``), so
    the embedded data cannot drift from the live behaviour.  ``width`` and
    ``height`` are the final canvas dimensions; ``station_radius`` is the
    single nominal marker radius (an overlay needs a point and a radius, not
    the per-station pill geometry).

    Ports and hidden nodes are excluded.  Every other station is included --
    unmapped ones simply carry an empty ``processes`` list -- so the manifest
    is a complete, future-proof inventory of addressable stations rather than
    only the subset that lights up today.
    """
    real_sections = {
        sid: sec for sid, sec in graph.sections.items() if not sec.is_implicit
    }

    lines = [
        {"id": line.id, "label": line.display_name, "color": line.color}
        for line in graph.lines.values()
    ]

    sections = [{"id": sid, "label": sec.name} for sid, sec in real_sections.items()]

    stations: list[dict[str, Any]] = []
    for station in graph.stations.values():
        if station.is_port or station.is_hidden:
            continue
        entry: dict[str, Any] = {
            "id": station.id,
            "label": station.label or station.id,
            "x": _round1(station.x),
            "y": _round1(station.y),
            "r": _round1(station_radius),
            "lines": graph.station_lines(station.id),
            "processes": list(graph.process_mapping.get(station.id, [])),
        }
        if station.section_id in real_sections:
            entry["section"] = station.section_id
        stations.append(entry)

    return {
        "version": MANIFEST_SCHEMA_VERSION,
        "match": dict(_MATCH_SPEC),
        "title": graph.title,
        "width": int(width),
        "height": int(height),
        "lines": lines,
        "sections": sections,
        "stations": stations,
    }


def manifest_json(manifest: dict[str, Any]) -> str:
    """Serialize a manifest deterministically (sorted keys, compact)."""
    return json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def manifest_metadata_svg(manifest: dict[str, Any]) -> str:
    """Return the ``<metadata>`` element carrying the manifest as CDATA JSON.

    CDATA keeps the JSON pristine (no entity escaping of the many quote
    characters).  The only sequence CDATA cannot contain is ``]]>``; on the
    rare chance a regex includes it, split it with the standard idiom so the
    document stays well-formed.  :func:`read_manifest` reverses the split.
    """
    payload = manifest_json(manifest).replace("]]>", "]]]]><![CDATA[>")
    return (
        f'<metadata id="{MANIFEST_ELEMENT_ID}" '
        f'data-nf-metro-schema="{manifest.get("version", MANIFEST_SCHEMA_VERSION)}">'
        f"<![CDATA[{payload}]]></metadata>"
    )


_METADATA_RE = re.compile(
    rf'<metadata id="{re.escape(MANIFEST_ELEMENT_ID)}"[^>]*>(.*?)</metadata>',
    re.DOTALL,
)
_CDATA_RE = re.compile(r"\s*<!\[CDATA\[(.*)\]\]>\s*\Z", re.DOTALL)


def read_manifest(svg: str) -> dict[str, Any] | None:
    """Parse the embedded manifest back out of a rendered SVG string.

    The canonical reader for the contract: returns the manifest dict, or
    ``None`` if the SVG carries no nf-metro manifest.  Parser-independent (a
    plain regex extract) so a consumer needn't load an XML library.
    """
    m = _METADATA_RE.search(svg)
    if m is None:
        return None
    inner = m.group(1)
    cdata = _CDATA_RE.match(inner)
    text = cdata.group(1) if cdata else inner
    text = text.replace("]]]]><![CDATA[>", "]]>")
    return json.loads(text)


def match_station_ids(manifest: dict[str, Any], process_name: str) -> list[str]:
    """Station ids whose patterns match ``process_name`` (case-insensitive).

    The reference matcher for the manifest: it mirrors the live server's
    ``stations_for_process`` against the embedded ``processes`` regexes, so a
    consumer (in any language) can reproduce the documented semantics exactly.
    """
    return [
        station["id"]
        for station in manifest.get("stations", [])
        if any(
            re.search(pattern, process_name, re.IGNORECASE)
            for pattern in station.get("processes", [])
        )
    ]
