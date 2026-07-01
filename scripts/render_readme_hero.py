#!/usr/bin/env python3
"""Regenerate the README hero renders from examples/rnaseq_auto.mmd.

Produces the two theme-adaptive animated SVGs the README's <picture> block
embeds directly (rnaseq_{dark,light}_animated.svg) and a static PNG fallback
for clients that can't render <picture>/CSS animation (rnaseq_auto_dark.png).
Run manually to preview a hero update, or via the release workflow to keep it
in lockstep with engine changes.

Must run from the repo root: the source's `%%metro logo:` directive names
paths relative to the working directory.

Usage:
    python scripts/render_readme_hero.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nf_metro.api import resolve_theme  # noqa: E402
from nf_metro.layout.engine import compute_layout  # noqa: E402
from nf_metro.parser.mermaid import parse_metro_mermaid  # noqa: E402
from nf_metro.render.svg import render_svg  # noqa: E402

EXAMPLES_DIR = project_root / "examples"
SOURCE = EXAMPLES_DIR / "rnaseq_auto.mmd"


def _laid_out_graph():
    graph = parse_metro_mermaid(SOURCE.read_text())
    compute_layout(graph)
    return graph


def render_animated(mode: str, out_path: Path) -> None:
    """Render the theme-adaptive animated SVG the README embeds directly."""
    graph = _laid_out_graph()
    theme = resolve_theme("nfcore", graph, mode=mode)
    out_path.write_text(render_svg(graph, theme))


def render_png_fallback(out_path: Path) -> None:
    """Render a baked-colour, chrome-less SVG and rasterize it to PNG.

    cairosvg can't parse the live var()/light-dark() chrome CSS, so this bakes
    concrete colours (chrome_css=False) the same way the CLI's --no-chrome-css
    raster workflow does.
    """
    import cairosvg

    graph = _laid_out_graph()
    theme = resolve_theme("nfcore", graph, mode="dark")
    svg = render_svg(graph, theme, chrome_css=False, baked_mode="dark")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out_path), scale=2)


def main() -> None:
    render_animated("dark", EXAMPLES_DIR / "rnaseq_dark_animated.svg")
    render_animated("light", EXAMPLES_DIR / "rnaseq_light_animated.svg")
    render_png_fallback(EXAMPLES_DIR / "rnaseq_auto_dark.png")
    print("Regenerated README hero renders from rnaseq_auto.mmd.")


if __name__ == "__main__":
    main()
