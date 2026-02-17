#!/usr/bin/env python3
"""Batch render all topology fixtures and the rnaseq example to SVG and PNG.

Outputs go to /tmp/nf_metro_topology_renders/.

Usage:
    python scripts/render_topologies.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nf_metro.layout.engine import compute_layout  # noqa: E402
from nf_metro.parser.mermaid import parse_metro_mermaid  # noqa: E402
from nf_metro.render.svg import render_svg  # noqa: E402
from nf_metro.themes import THEMES  # noqa: E402

OUTPUT_DIR = Path("/tmp/nf_metro_topology_renders")
TOPOLOGIES_DIR = project_root / "tests" / "fixtures" / "topologies"
EXAMPLES_DIR = project_root / "examples"

# Files to render
FIXTURE_FILES = sorted(TOPOLOGIES_DIR.glob("*.mmd"))
EXTRA_FILES = [EXAMPLES_DIR / "rnaseq_sections.mmd"]


def render_file(
    mmd_path: Path, output_dir: Path, *, debug: bool = False
) -> tuple[str, list[str]]:
    """Parse, layout, and render a .mmd file to SVG (and optionally PNG).

    Returns (name, list_of_issues).
    """
    name = mmd_path.stem
    issues: list[str] = []

    try:
        text = mmd_path.read_text()
        graph = parse_metro_mermaid(text)
    except Exception as e:
        return name, [f"PARSE ERROR: {e}"]

    try:
        compute_layout(graph)
    except Exception as e:
        return name, [f"LAYOUT ERROR: {e}"]

    theme_name = graph.style if graph.style in THEMES else "nfcore"
    theme = THEMES[theme_name]

    try:
        svg_str = render_svg(graph, theme, debug=debug)
    except Exception as e:
        return name, [f"RENDER ERROR: {e}"]

    svg_path = output_dir / f"{name}.svg"
    svg_path.write_text(svg_str)

    # Try PNG conversion via cairosvg (optional)
    try:
        import cairosvg

        png_path = output_dir / f"{name}.png"
        cairosvg.svg2png(bytestring=svg_str.encode(), write_to=str(png_path), scale=2)
    except ImportError:
        issues.append("cairosvg not available, skipping PNG")
    except Exception as e:
        issues.append(f"PNG conversion error: {e}")

    return name, issues


def main():
    parser = argparse.ArgumentParser(description="Batch render topology fixtures")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug overlay (ports, hidden stations, waypoints)"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = list(FIXTURE_FILES) + EXTRA_FILES
    print(f"Rendering {len(all_files)} files to {OUTPUT_DIR}/")
    if args.debug:
        print("Debug overlay: ON")
    print()

    max_name_len = max(len(f.stem) for f in all_files)
    any_errors = False

    for mmd_path in all_files:
        name, issues = render_file(mmd_path, OUTPUT_DIR, debug=args.debug)
        status = "OK" if not issues else "ISSUES"
        if any("ERROR" in i for i in issues):
            status = "FAIL"
            any_errors = True

        print(f"  {name:<{max_name_len}}  [{status}]")
        for issue in issues:
            print(f"    - {issue}")

    print(f"\nOutputs in: {OUTPUT_DIR}/")

    if any_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
