"""Theme definitions for metro maps."""

from nf_metro.themes.light import LIGHT_THEME
from nf_metro.themes.nfcore import NFCORE_THEME

THEMES = {
    "nfcore": NFCORE_THEME,
    "light": LIGHT_THEME,
}

__all__ = ["THEMES", "NFCORE_THEME", "LIGHT_THEME"]
