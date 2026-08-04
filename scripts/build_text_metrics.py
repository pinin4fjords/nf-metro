#!/usr/bin/env python3
"""Generate deterministic metrics tables from nf-metro's bundled Inter fonts."""

from __future__ import annotations

from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "src" / "nf_metro" / "fonts"
OUTPUT = ROOT / "src" / "nf_metro" / "_inter_metrics.py"


def _font_metrics(path: Path) -> tuple[int, int, int, int, dict[int, tuple[int, ...]]]:
    font = TTFont(path)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    upem = int(font["head"].unitsPerEm)
    ascent = int(font["hhea"].ascent)
    descent = int(font["hhea"].descent)
    line_gap = int(font["hhea"].lineGap)
    metrics: dict[int, tuple[int, ...]] = {}
    for codepoint, glyph_name in sorted(cmap.items()):
        glyph = glyphs[glyph_name]
        pen = BoundsPen(glyphs)
        glyph.draw(pen)
        advance = int(round(glyph.width))
        if pen.bounds is None:
            metrics[codepoint] = (advance,)
        else:
            metrics[codepoint] = (advance, *(int(round(v)) for v in pen.bounds))
    return upem, ascent, descent, line_gap, metrics


def _format_table(name: str, metrics: dict[int, tuple[int, ...]]) -> str:
    rows = [f"    {codepoint}: {values!r}," for codepoint, values in metrics.items()]
    return f"{name}: dict[int, tuple[int, ...]] = {{\n" + "\n".join(rows) + "\n}\n"


def main() -> None:
    regular = _font_metrics(FONTS / "Inter-Regular.woff2")
    bold = _font_metrics(FONTS / "Inter-Bold.woff2")
    if regular[:4] != bold[:4]:
        raise RuntimeError("bundled Inter faces disagree on vertical metrics")
    upem, ascent, descent, line_gap = regular[:4]
    content = "\n".join(
        [
            '"""Generated Inter metrics.',
            "",
            "Rebuild with scripts/build_text_metrics.py.",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            f"INTER_UNITS_PER_EM = {upem}",
            f"INTER_ASCENT = {ascent}",
            f"INTER_DESCENT = {descent}",
            f"INTER_LINE_GAP = {line_gap}",
            "INTER_REPLACEMENT_CODEPOINT = 63",
            "",
            _format_table("INTER_REGULAR_METRICS", regular[4]),
            _format_table("INTER_BOLD_METRICS", bold[4]),
        ]
    )
    OUTPUT.write_text(content)


if __name__ == "__main__":
    main()
