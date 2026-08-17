"""Theme definitions for metro maps.

Brand and mode are orthogonal axes. ``THEMES`` is the flat by-name registry
(every brand and every concrete variant, for ``--theme``/``style:`` lookup).
``THEME_MODES`` groups brands into their ``{light, dark}`` pairs so a brand can
be resolved against an independently chosen mode, and so the renderer can emit
both palettes from a single render.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nf_metro.render.style import Theme
from nf_metro.themes.light import LIGHT_THEME
from nf_metro.themes.nfcore import (
    NFCORE_DARK_THEME,
    NFCORE_LIGHT_THEME,
    NFCORE_THEME,
)
from nf_metro.themes.seqera import (
    SEQERA_DARK_THEME,
    SEQERA_LIGHT_THEME,
    SEQERA_THEME,
)

if TYPE_CHECKING:
    from nf_metro.parser.model import MetroGraph

# ``style: dark`` predates theme names; alias it onto the nfcore brand.
_STYLE_THEME_ALIASES = {"dark": "nfcore"}

# Mode used when a single concrete palette must be baked and none was chosen
# (e.g. PNG raster). Applies equally to every brand - no brand is intrinsically
# light or dark. SVG output carries both palettes and adapts at view time, so
# this only governs raster/standalone fallback.
DEFAULT_MODE = "dark"

# Brand -> mode -> Theme. The renderer reads a resolved theme's ``brand`` here to
# recover both mode palettes for ``light-dark()`` emission; the resolver reads it
# to combine a brand with an independently chosen mode.
THEME_MODES: dict[str, dict[str, Theme]] = {
    "nfcore": {"dark": NFCORE_DARK_THEME, "light": NFCORE_LIGHT_THEME},
    "seqera": {"dark": SEQERA_DARK_THEME, "light": SEQERA_LIGHT_THEME},
}

# Flat by-name registry for direct ``--theme`` / ``style:`` selection. Bare brand
# names resolve to the brand at ``DEFAULT_MODE``; the suffixed names pin a mode.
THEMES = {
    "nfcore": THEME_MODES["nfcore"][DEFAULT_MODE],
    "nfcore-light": NFCORE_LIGHT_THEME,
    "nfcore-dark": NFCORE_DARK_THEME,
    "seqera": THEME_MODES["seqera"][DEFAULT_MODE],
    "seqera-light": SEQERA_LIGHT_THEME,
    "seqera-dark": SEQERA_DARK_THEME,
    "light": LIGHT_THEME,
}


def resolve_theme(
    theme: str | None, graph: MetroGraph, mode: str | None = None
) -> Theme:
    """Resolve a concrete theme from independent brand and mode axes.

    Brand comes from the explicit ``theme`` name or the graph's style. Mode
    comes from the explicit argument, the graph directive, or ``DEFAULT_MODE``.
    """
    if theme is not None:
        brand = theme
    else:
        name = graph.style.strip().lower()
        brand = _STYLE_THEME_ALIASES.get(name, name)

    resolved_mode = (mode or graph.mode).strip().lower() or DEFAULT_MODE
    family = THEME_MODES.get(brand)
    if family and resolved_mode in family:
        return family[resolved_mode]

    return THEMES.get(brand, THEMES["nfcore"])


def mode_pair(theme: Theme) -> tuple[Theme, Theme] | None:
    """Return ``(light_theme, dark_theme)`` for *theme*'s brand family.

    ``None`` when the theme has no registered light/dark family (e.g. the
    transparent ``light`` theme), so callers fall back to a single palette.
    """
    family = THEME_MODES.get(theme.brand)
    if family is None or "light" not in family or "dark" not in family:
        return None
    return family["light"], family["dark"]


__all__ = [
    "THEMES",
    "THEME_MODES",
    "DEFAULT_MODE",
    "resolve_theme",
    "mode_pair",
    "LIGHT_THEME",
    "NFCORE_THEME",
    "NFCORE_DARK_THEME",
    "NFCORE_LIGHT_THEME",
    "SEQERA_THEME",
    "SEQERA_LIGHT_THEME",
    "SEQERA_DARK_THEME",
]
