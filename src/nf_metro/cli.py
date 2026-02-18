"""CLI for nf-metro."""

from __future__ import annotations

from pathlib import Path

import click

from nf_metro import __version__
from nf_metro.layout import compute_layout
from nf_metro.parser import parse_metro_mermaid
from nf_metro.render import render_svg
from nf_metro.themes import THEMES


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """nf-metro: Generate metro-map-style SVG diagrams from Mermaid definitions."""


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output SVG file path. Defaults to <input>.svg",
)
@click.option(
    "--theme",
    type=click.Choice(list(THEMES.keys())),
    default="nfcore",
    help="Visual theme (default: nfcore)",
)
@click.option("--width", type=int, default=None, help="SVG width in pixels")
@click.option("--height", type=int, default=None, help="SVG height in pixels")
@click.option(
    "--x-spacing",
    type=float,
    default=60.0,
    help="Horizontal spacing between layers (default: 60)",
)
@click.option(
    "--y-spacing",
    type=float,
    default=40.0,
    help="Vertical spacing between tracks (default: 40)",
)
@click.option(
    "--max-layers-per-row",
    type=int,
    default=None,
    help="Max layers before folding to next row (default: auto)",
)
@click.option(
    "--animate/--no-animate",
    default=False,
    help="Add animated balls traveling along lines",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Show debug overlay (ports, hidden stations, edge waypoints)",
)
@click.option(
    "--logo",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Logo image path (overrides %%metro logo: directive)",
)
def render(
    input_file: Path,
    output: Path | None,
    theme: str,
    width: int | None,
    height: int | None,
    x_spacing: float,
    y_spacing: float,
    max_layers_per_row: int | None,
    animate: bool,
    debug: bool,
    logo: Path | None,
) -> None:
    """Render a Mermaid metro map definition to SVG."""
    text = input_file.read_text()
    try:
        graph = parse_metro_mermaid(
            text,
            max_station_columns=max_layers_per_row or 15,
        )
    except ValueError as e:
        raise click.ClickException(str(e))

    if logo is not None:
        graph.logo_path = str(logo)

    compute_layout(
        graph,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
    )

    theme_obj = THEMES[theme]
    svg = render_svg(
        graph, theme_obj, width=width, height=height, animate=animate, debug=debug
    )

    if output is None:
        output = input_file.with_suffix(".svg")

    output.write_text(svg)
    click.echo(
        f"Rendered {len(graph.stations)} stations, "
        f"{len(graph.edges)} edges, "
        f"{len(graph.lines)} lines -> {output}"
    )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
def validate(input_file: Path) -> None:
    """Validate a Mermaid metro map definition."""
    text = input_file.read_text()
    try:
        graph = parse_metro_mermaid(text)
    except Exception as e:
        click.echo(f"Parse error: {e}", err=True)
        raise SystemExit(1)

    errors = []

    # Check that all edge line_ids reference defined lines
    for edge in graph.edges:
        if edge.line_id != "default" and edge.line_id not in graph.lines:
            errors.append(
                f"Edge {edge.source} -> {edge.target} references "
                f"undefined line '{edge.line_id}'"
            )

    # Check that section station IDs exist
    for section in graph.sections.values():
        for sid in section.station_ids:
            if sid not in graph.stations:
                errors.append(
                    f"Section '{section.name}' references unknown station '{sid}'"
                )

    if errors:
        click.echo("Validation errors:", err=True)
        for err in errors:
            click.echo(f"  - {err}", err=True)
        raise SystemExit(1)

    click.echo(
        f"Valid: {len(graph.stations)} stations, "
        f"{len(graph.edges)} edges, "
        f"{len(graph.lines)} lines, "
        f"{len(graph.sections)} sections"
    )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
def info(input_file: Path) -> None:
    """Show information about a Mermaid metro map definition."""
    text = input_file.read_text()
    try:
        graph = parse_metro_mermaid(text)
    except ValueError as e:
        raise click.ClickException(str(e))

    click.echo(f"Title: {graph.title or '(none)'}")
    click.echo(f"Style: {graph.style}")
    click.echo(f"Stations: {len(graph.stations)}")
    click.echo(f"Edges: {len(graph.edges)}")
    click.echo(f"Lines: {len(graph.lines)}")
    for lid, line in graph.lines.items():
        stations = graph.line_stations(lid)
        click.echo(f"  {line.display_name} ({line.color}): {len(stations)} stations")
    click.echo(f"Sections: {len(graph.sections)}")
    for section in graph.sections.values():
        click.echo(
            f"  [{section.number}] {section.name}: {len(section.station_ids)} stations"
        )
