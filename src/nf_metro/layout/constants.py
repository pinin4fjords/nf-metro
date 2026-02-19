"""Layout constants used across layout modules.

Centralizes magic numbers from engine.py, routing.py, labels.py,
section_placement.py, and ordering.py.
"""

# ---------------------------------------------------------------------------
# Font / text metrics
# ---------------------------------------------------------------------------
CHAR_WIDTH: float = 7.0
"""Approximate pixel width of a single character at default font size."""

FONT_HEIGHT: float = 14.0
"""Approximate pixel height of default font."""

LABEL_PAD: float = 6.0
"""Padding added to label width when computing section bounds."""

# ---------------------------------------------------------------------------
# Global spacing defaults (used as function parameter defaults)
# ---------------------------------------------------------------------------
X_SPACING: float = 60.0
"""Horizontal spacing between layers."""

Y_SPACING: float = 40.0
"""Vertical spacing between tracks."""

X_OFFSET: float = 80.0
"""Left padding from canvas edge to first layer."""

Y_OFFSET: float = 120.0
"""Top padding from canvas edge to first track."""

ROW_GAP: float = 120.0
"""Vertical gap between fold rows."""

# ---------------------------------------------------------------------------
# Section sizing / padding (engine defaults)
# ---------------------------------------------------------------------------
SECTION_GAP: float = 3.0
"""Spacing between stations within a section."""

SECTION_X_PADDING: float = 50.0
"""Horizontal padding around section content."""

SECTION_Y_PADDING: float = 35.0
"""Vertical padding around section content."""

SECTION_X_GAP: float = 50.0
"""Horizontal gap between section columns (engine-level)."""

SECTION_Y_GAP: float = 40.0
"""Vertical gap between section rows (engine-level)."""

# ---------------------------------------------------------------------------
# Section placement defaults
# ---------------------------------------------------------------------------
PLACEMENT_X_GAP: float = 80.0
"""Horizontal gap between section columns in meta-graph placement."""

PLACEMENT_Y_GAP: float = 60.0
"""Vertical gap between section rows in meta-graph placement."""

PORT_MIN_GAP: float = 15.0
"""Minimum spacing between adjacent ports on a section boundary."""

MIN_INTER_SECTION_GAP: float = 40.0
"""Minimum physical gap between adjacent section bboxes.

Ensures the gap midpoint is at least 2*CURVE_RADIUS from each section
edge, giving enough horizontal run for smooth curves at bypass route
corners.  Value: 40px (4 * 10px CURVE_RADIUS).
"""

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
DIAGONAL_RUN: float = 30.0
"""Length of the diagonal segment in direction changes."""

CURVE_RADIUS: float = 10.0
"""Default corner radius for routed paths."""

OFFSET_STEP: float = 3.0
"""Per-line offset increment for parallel lines in bundles."""

COORD_TOLERANCE: float = 1.0
"""Tolerance for coordinate comparison (same X or same Y)."""

COORD_TOLERANCE_FINE: float = 0.01
"""Fine tolerance for detecting nearly identical Y coordinates."""

CROSS_ROW_THRESHOLD: float = 80.0
"""Y gap threshold for detecting cross-row (fold) edges."""

FOLD_MARGIN: float = 30.0
"""Offset from fold edge for cross-row routing."""

MIN_STRAIGHT_INTER: float = 15.0
"""Minimum straight track length for inter-section routing."""

MIN_STRAIGHT_PORT: float = 5.0
"""Curve radius offset for port-adjacent edges."""

MIN_STRAIGHT_EDGE: float = 10.0
"""Minimum straight track for non-port edges."""

BYPASS_CLEARANCE: float = 25.0
"""Vertical clearance below the lowest intervening section for bypass routes."""

BYPASS_NEST_STEP: float = 8.0
"""Per-line vertical offset for stacking multiple bypass routes."""

# ---------------------------------------------------------------------------
# Engine: entry/exit alignment
# ---------------------------------------------------------------------------
TB_LINE_Y_OFFSET: float = 3.0
"""Per-line Y offset increment in TB sections."""

ENTRY_SHIFT_TB: float = 0.6
"""Entry shift multiplier for TB sections with perpendicular entry."""

ENTRY_SHIFT_TB_CROSS: float = 1.0
"""Entry shift multiplier for TB sections with cross-column TOP entry."""

ENTRY_INSET_LR: float = 0.3
"""Entry inset multiplier for LR/RL sections with perpendicular entry."""

EXIT_GAP_MULTIPLIER: float = 0.4
"""Exit gap multiplier for flow-side exits."""

JUNCTION_MARGIN: float = 10.0
"""Margin for positioning junctions in inter-section gaps."""

MIN_PORT_STATION_GAP: float = 16.0
"""Minimum gap between entry port and internal stations (TB perpendicular)."""

STATION_ELBOW_TOLERANCE: float = 12.0
"""Tolerance for station-as-elbow detection."""

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL_MARGIN: float = 2.0
"""Overlap detection margin for labels."""

LABEL_OFFSET: float = 16.0
"""Vertical distance from station center to label."""

TB_PILL_EDGE_OFFSET: float = 5.0
"""Pill edge offset for TB vertical station labels."""

TB_LABEL_H_SPACING: float = 6.0
"""Horizontal spacing for TB vertical station labels."""

COLLISION_MULTIPLIER: float = 2.2
"""Label offset multiplier when resolving collisions."""

LABEL_BBOX_MARGIN: float = 4.0
"""Margin for clamping labels within section bounding box."""

# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------
LINE_GAP: float = 1.0
"""Fixed gap between line base tracks."""

DIAMOND_COMPRESSION: float = 0.25
"""Compression factor toward trunk for diamond (fork-join) paths."""

SIDE_BRANCH_NUDGE: float = 1.0
"""Nudge amount for side-branch tracking."""

FANOUT_SPACING: float = 1.5
"""Spacing multiplier for fan-out node layout."""
