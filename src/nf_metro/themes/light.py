"""Light theme."""

from nf_metro.render.style import Theme

LIGHT_THEME = Theme(
    name="light",
    background_color="none",
    station_fill="#ffffff",
    station_stroke="#333333",
    station_radius=6.0,
    station_stroke_width=2.0,
    line_width=4.0,
    label_color="#333333",
    label_font_family="'Helvetica Neue', Helvetica, Arial, sans-serif",
    label_font_size=14.0,
    title_color="#111111",
    title_font_size=26.0,
    section_fill="rgba(0, 0, 0, 0.03)",
    section_stroke="rgba(0, 0, 0, 0.1)",
    section_label_color="#666666",
    section_label_font_size=17.0,
    legend_background="rgba(255, 255, 255, 0.8)",
    legend_text_color="#333333",
    legend_font_size=15.0,
    animation_ball_color="#333333",
)
