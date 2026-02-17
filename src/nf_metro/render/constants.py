"""Render constants used across render modules.

Centralizes magic numbers from svg.py, legend.py, animate.py, and icons.py.
Theme-dependent values remain in style.py.
"""

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
CANVAS_PADDING: float = 60.0
"""Default padding around the entire SVG canvas."""

LEGEND_GAP: float = 30.0
"""Gap between content area and legend (bottom/right positions)."""

LEGEND_INSET: float = 10.0
"""Inset from content edge for corner legend positions (tl/tr/bl/br)."""

LOGO_Y_STANDALONE: float = 5.0
"""Y offset for standalone logo (no legend)."""

LOGO_HEIGHT_DEFAULT: float = 80.0
"""Default logo display height."""

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
LEGEND_LINE_HEIGHT: float = 24.0
"""Vertical height per line entry in legend."""

LEGEND_PADDING: float = 12.0
"""Internal padding of legend box."""

LEGEND_SWATCH_WIDTH: float = 24.0
"""Width of color swatch line in legend."""

LEGEND_TEXT_GAP: float = 12.0
"""Gap between swatch end and label text."""

LEGEND_CHAR_WIDTH_RATIO: float = 0.48
"""Character width as a fraction of font size for legend text sizing."""

LOGO_SCALE_FACTOR: float = 0.6
"""Logo scale factor relative to content height."""

LOGO_GAP: float = 12.0
"""Gap between logo and line entries in legend."""

LEGEND_BORDER_RADIUS: int = 6
"""Corner radius for legend background rectangle."""

# ---------------------------------------------------------------------------
# SVG drawing
# ---------------------------------------------------------------------------
SVG_CURVE_RADIUS: float = 10.0
"""Default corner radius for edge path smoothing."""

SECTION_NUM_CIRCLE_R: int = 8
"""Radius of section number circle background (small variant)."""

SECTION_NUM_CIRCLE_R_LARGE: int = 9
"""Radius of section number circle background (large variant)."""

SECTION_NUM_Y_OFFSET: int = 4
"""Y offset of section number circle from section top."""

SECTION_LABEL_TEXT_OFFSET: int = 5
"""Text X offset from section number circle."""

ICON_STATION_GAP: float = 6.0
"""Gap between terminus station pill and file icon."""

ICON_BBOX_MARGIN: float = 2.0
"""Margin around icon bounding box for clamping."""

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
ANIMATION_CURVE_RADIUS: float = 10.0
"""Default curve radius for animation motion paths."""

MIN_ANIMATION_DURATION: float = 2.0
"""Minimum duration in seconds for ball animation."""

EDGE_CONNECT_TOLERANCE: float = 1.0
"""Tolerance for detecting connected edge endpoints."""

# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------
TRAIN_ICON_SIZE: float = 12.0
"""Default size of train icon placeholder."""

# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------
DEBUG_FONT_SIZE: int = 7
"""Font size for debug overlay labels."""

DEBUG_DIAMOND_RADIUS: int = 5
"""Radius of diamond markers in debug overlay."""

DEBUG_STROKE_WIDTH: float = 1.5
"""Stroke width for hidden station markers in debug mode."""
