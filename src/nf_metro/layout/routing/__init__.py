"""Edge routing subpackage for metro map layout.

Public API:
- route_edges: Main edge routing dispatcher
- RoutedPath: Routed path dataclass
- compute_station_offsets: Per-station Y offset computation
- route_inter_section_edges: Simplified inter-section routing (tests)
"""

from nf_metro.layout.routing.common import RoutedPath
from nf_metro.layout.routing.core import route_edges
from nf_metro.layout.routing.inter_section import route_inter_section_edges
from nf_metro.layout.routing.offsets import compute_station_offsets

__all__ = [
    "RoutedPath",
    "compute_station_offsets",
    "route_edges",
    "route_inter_section_edges",
]
