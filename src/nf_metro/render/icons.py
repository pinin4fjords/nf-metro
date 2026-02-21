"""Icon helpers for metro map rendering."""

from __future__ import annotations

__all__ = ["render_file_icon"]

import drawsvg as draw

from nf_metro.render.constants import TEXT_VCENTER_DY


def train_icon_path(x: float, y: float, size: float = 12.0) -> str:
    """Generate an SVG path string for a small train icon. Placeholder for future."""
    # Simple diamond shape as placeholder
    hs = size / 2
    return f"M {x} {y - hs} L {x + hs} {y} L {x} {y + hs} L {x - hs} {y} Z"


def render_file_icon(
    d: draw.Drawing,
    cx: float,
    cy: float,
    width: float,
    height: float,
    fold_size: float,
    fill: str,
    stroke: str,
    stroke_width: float,
    corner_radius: float,
    label: str,
    font_size: float,
    font_color: str,
    font_family: str,
) -> None:
    """Render a file/document icon with a dog-ear fold at top-right.

    The icon is centered on (cx, cy). The shape is a rectangle with the
    top-right corner replaced by a diagonal fold.
    """
    hw = width / 2
    hh = height / 2
    x0 = cx - hw
    y0 = cy - hh
    x1 = cx + hw
    y1 = cy + hh
    r = corner_radius
    f = fold_size

    # Main document shape: rectangle with top-right dog-ear
    # Start at top-left + corner radius, go clockwise
    path = draw.Path(
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
        stroke_linejoin="round",
    )
    # Top edge: from top-left corner to fold start
    path.M(x0 + r, y0)
    path.L(x1 - f, y0)
    # Diagonal fold
    path.L(x1, y0 + f)
    # Right edge down to bottom-right corner
    path.L(x1, y1 - r)
    # Bottom-right corner
    path.Q(x1, y1, x1 - r, y1)
    # Bottom edge
    path.L(x0 + r, y1)
    # Bottom-left corner
    path.Q(x0, y1, x0, y1 - r)
    # Left edge
    path.L(x0, y0 + r)
    # Top-left corner
    path.Q(x0, y0, x0 + r, y0)
    path.Z()
    d.append(path)

    # Fold triangle (slightly darker overlay)
    fold_path = draw.Path(
        fill=stroke,
        opacity=0.15,
        stroke="none",
    )
    fold_path.M(x1 - f, y0)
    fold_path.L(x1 - f, y0 + f)
    fold_path.L(x1, y0 + f)
    fold_path.Z()
    d.append(fold_path)

    # Fold crease line
    crease = draw.Path(
        fill="none",
        stroke=stroke,
        stroke_width=stroke_width * 0.6,
    )
    crease.M(x1 - f, y0)
    crease.L(x1 - f, y0 + f)
    crease.L(x1, y0 + f)
    d.append(crease)

    # Extension label centered in the body (shifted down slightly to
    # account for fold taking up top-right space)
    text_y = cy + f * 0.15
    d.append(
        draw.Text(
            label,
            font_size,
            cx,
            text_y,
            fill=font_color,
            font_family=font_family,
            font_weight="bold",
            text_anchor="middle",
            dy=TEXT_VCENTER_DY,
        )
    )
