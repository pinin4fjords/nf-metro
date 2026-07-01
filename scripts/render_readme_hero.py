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

from nf_metro.api import render_string  # noqa: E402

EXAMPLES_DIR = project_root / "examples"
SOURCE_TEXT = (EXAMPLES_DIR / "rnaseq_auto.mmd").read_text()


def main() -> None:
    (EXAMPLES_DIR / "rnaseq_dark_animated.svg").write_text(
        render_string(SOURCE_TEXT, theme="nfcore", mode="dark")
    )
    (EXAMPLES_DIR / "rnaseq_light_animated.svg").write_text(
        render_string(SOURCE_TEXT, theme="nfcore", mode="light")
    )

    # cairosvg can't parse the live var()/light-dark() chrome CSS, so this
    # bakes concrete colours (chrome_css=False) the same way the CLI's
    # --no-chrome-css raster workflow does.
    import cairosvg

    baked_svg = render_string(
        SOURCE_TEXT, theme="nfcore", mode="dark", chrome_css=False
    )
    cairosvg.svg2png(
        bytestring=baked_svg.encode(),
        write_to=str(EXAMPLES_DIR / "rnaseq_auto_dark.png"),
        scale=2,
    )

    print("Regenerated README hero renders from rnaseq_auto.mmd.")


if __name__ == "__main__":
    main()
