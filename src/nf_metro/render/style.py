"""Theme and style constants for metro map rendering."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Theme:
    """Visual theme for a metro map."""

    name: str
    background_color: str
    station_fill: str
    station_stroke: str
    station_radius: float
    station_stroke_width: float
    line_width: float
    label_color: str
    label_font_family: str
    label_font_size: float
    title_color: str
    title_font_size: float
    section_fill: str
    section_stroke: str
    section_label_color: str
    section_label_font_size: float
    legend_background: str
    legend_text_color: str
    legend_font_size: float
    # Animation settings
    animation_ball_radius: float = 3.0
    animation_ball_color: str = "#ffffff"
    animation_balls_per_line: int = 3
    animation_speed: float = 80.0  # pixels per second
