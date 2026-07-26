"""Orientation equivalence: one geometry must reach one code path (#1545).

Rotating or reflecting a whole map is a meaning-preserving rewrite of three
directive families (see :mod:`orientation_transform`), so the engine should lay
the result out as the reference's image under the same transform.  Where it does
not, the same geometric requirement is being met by two orientation-keyed code
paths, and a defect fixed in one of them survives in the other.

The oracle compares only what a transform genuinely preserves.  Rotating a map
does not rotate its text, so the *distances* between stations are free to change
with glyph extents; what must not change is anything the engine decides -- flow
ranks and lanes, port sides, a port's clearance to the edge it is pinned to,
which end of the flow a port is seated at, and which grid groups are aligned.
:mod:`orientation_signature` defines those measures.

``KNOWN_DIVERGENCES`` records the residuals that do not hold yet, each naming the
defect behind it.  Entries are strict xfails, so completing a fix reds this suite
until the entry is removed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from orientation_signature import divergences
from orientation_transform import (
    Orientation,
    grid_dims,
    non_identity_orientations,
    transform_source,
    transformable_reason,
)

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIRS = (
    REPO_ROOT / "examples" / "topologies",
    REPO_ROOT / "examples",
    REPO_ROOT / "tests" / "fixtures",
)

# Each entry is (fixture stem, divergence family) -> why it does not hold yet.
# A residual is excepted by the *kind* of defect behind it, so one entry covers
# every orbit member the same defect breaks.
KNOWN_DIVERGENCES: dict[tuple[str, str], str] = {
    # Grid rows are aligned to a common bbox top (_top_align_row_bboxes_only,
    # _align_row_trunk_ys, _fit_bboxes_to_content_top) with no counterpart for
    # grid columns, so a quarter turn carries an aligned row onto an unaligned
    # column.
    **{
        (stem, "group_alignment"): "no column counterpart to row top-alignment (#1545)"
        for stem in (
            "bt_exit_top_above",
            "bt_exit_top_above_2line",
            "bt_perp_left_entry_right_exit",
            "bt_to_tb",
            "lr_to_tb_top_cross_col",
            "lr_to_tb_top_drop_two_lines",
            "lr_top_entry_cross_column",
            "lr_top_entry_cross_column_two_line",
            "orbit_perp_exit_back_row_entry",
            "tb_two_line_vert_seam",
        )
    },
    # _align_perp_entry_port_y aligns a vertical-flow section's perpendicular
    # entry port to its feeder's exit coordinate without checking the result
    # against the section's own stations, so the port can land on the flow-END
    # station instead of before the flow-start one.
    **{
        (stem, "port_flow_end"): "perpendicular entry seated at the flow end (#1545)"
        for stem in (
            "bt_exit_top_above",
            "bt_exit_top_above_2line",
            "lr_top_entry_cross_column",
            "lr_top_entry_cross_column_two_line",
        )
    },
    # A folded flow-axis port is resolved either by reversing the section's flow
    # or by re-anchoring the port, but _FLIP_HORIZONTAL makes the reversal
    # reachable only for LR/RL, so a vertical flow takes the other remedy.
    **{
        (stem, family): "flow reversal unavailable to vertical flows (#1545)"
        for stem in ("lr_to_tb_top_drop", "top_entry_header_clash")
        for family in ("port_side", "port_perpendicular")
    },
}


def _corpus() -> list[Path]:
    return sorted({p for d in CORPUS_DIRS for p in d.glob("*.mmd")})


def _laid_out(source: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = parse_metro_mermaid(source)
        compute_layout(graph)
        return graph


def _transformable_fixtures() -> list[Path]:
    """Corpus fixtures whose geometry is stated explicitly enough to transform."""
    out = []
    for path in _corpus():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                graph = parse_metro_mermaid(path.read_text())
        except Exception:
            continue
        if transformable_reason(graph) is None:
            out.append(path)
    return out


TRANSFORMABLE = _transformable_fixtures()

ORBIT = [
    (path, orientation)
    for path in TRANSFORMABLE
    for orientation in non_identity_orientations()
]


def test_corpus_offers_transformable_fixtures() -> None:
    """The oracle has real inputs, so a discovery break cannot pass as success."""
    assert len(TRANSFORMABLE) >= 15, (
        f"only {len(TRANSFORMABLE)} fully-explicit multi-section fixtures found; "
        "orientation equivalence is unverified without them"
    )


@pytest.mark.parametrize(
    ("source_path", "orientation"),
    ORBIT,
    ids=[f"{p.stem}-{o.name}" for p, o in ORBIT],
)
def test_layout_is_congruent_under_orientation(
    source_path: Path, orientation: Orientation
) -> None:
    """A transformed map lays out as the reference's image under the transform."""
    source = source_path.read_text()
    reference = _laid_out(source)
    image = _laid_out(transform_source(source, orientation))

    found = divergences(reference, image, orientation, grid_dims(source))
    excepted = [d for d in found if (source_path.stem, d.family) in KNOWN_DIVERGENCES]
    live = [d for d in found if d not in excepted]

    if live:
        detail = "\n  ".join(str(d) for d in live)
        pytest.fail(
            f"{source_path.name} under {orientation.name} is not its reference's "
            f"image:\n  {detail}\n"
            "The same geometry is reaching two orientation-keyed code paths. Fix "
            "the shared path (or state the residual in KNOWN_DIVERGENCES with the "
            "defect behind it)."
        )
    if excepted:
        families = sorted({d.family for d in excepted})
        pytest.xfail(
            f"known divergence in {families}: "
            + "; ".join(
                sorted({KNOWN_DIVERGENCES[(source_path.stem, f)] for f in families})
            )
        )


def test_known_divergences_all_still_bite() -> None:
    """Every stated exception names a divergence the corpus actually exhibits.

    An entry whose divergence the corpus does not exhibit would stand in for an
    enforced invariant, so closing a defect requires deleting its entry.
    """
    live: set[tuple[str, str]] = set()
    for path in TRANSFORMABLE:
        source = path.read_text()
        reference = _laid_out(source)
        dims = grid_dims(source)
        for orientation in non_identity_orientations():
            image = _laid_out(transform_source(source, orientation))
            for d in divergences(reference, image, orientation, dims):
                live.add((path.stem, d.family))

    stale = sorted(set(KNOWN_DIVERGENCES) - live)
    assert not stale, (
        f"{len(stale)} stated exception(s) no longer diverge: {stale}. "
        "Remove them from KNOWN_DIVERGENCES so the invariant is enforced."
    )


def test_identity_transform_is_a_no_op() -> None:
    """The identity element rewrites nothing, so the orbit's base is the fixture."""
    for path in TRANSFORMABLE:
        source = path.read_text()
        assert transform_source(source, Orientation(0, False)) == source, path


@pytest.mark.parametrize("source_path", TRANSFORMABLE, ids=lambda p: p.stem)
def test_four_quarter_turns_restore_the_source(source_path: Path) -> None:
    """The transform is a group action: four turns and two mirrors are identities."""
    source = source_path.read_text()
    turned = source
    for _ in range(4):
        turned = transform_source(turned, Orientation(1, False))
    assert turned == source, (
        f"{source_path.name}: four quarter turns changed the source"
    )

    mirrored = transform_source(
        transform_source(source, Orientation(0, True)), Orientation(0, True)
    )
    assert mirrored == source, f"{source_path.name}: two reflections changed the source"
