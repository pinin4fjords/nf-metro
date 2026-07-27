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
defect behind it.  Which families an orbit member exhibits is only known once it
is laid out, so the exception is applied in the test body rather than as a
collection-time marker; a fixed defect therefore turns its orbit member into a
plain pass rather than an XPASS.  ``test_known_divergences_all_still_bite`` is
what reds the suite in that case, by failing on any entry whose divergence the
corpus does not exhibit.
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from pathlib import Path

import pytest
from conftest import content_corpus
from orientation_signature import DIVERGENCE_FAMILIES, divergences
from orientation_transform import (
    Orientation,
    non_identity_orientations,
    transform_source,
    transformable_reason,
)

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph

# Each entry is (fixture stem, divergence family) -> why it does not hold yet.
# A residual is excepted by the *kind* of defect behind it, so one entry covers
# every orbit member the same defect breaks.
KNOWN_DIVERGENCES: dict[tuple[str, str], str] = {
    # Whether a grid group shares a box edge is decided by a content-hug policy
    # on both axes: a row's bbox top hugs its content (_fit_bboxes_to_content_top)
    # and a column's box edges are levelled only across mates whose content
    # nearest that edge stands at one X (Stage 3.6).  Hugged edges coincide only
    # when the members' content extents happen to match, and a rotation does not
    # carry text extents with it, so the boolean is not rotation-invariant where
    # the hug wins over the flush.
    **{
        (stem, "group_alignment"): (
            "hugged row tops coincide only when the row-mates' content-top "
            "padding matches, which a quarter turn does not preserve (#1545)"
        )
        for stem in (
            "bt_perp_left_entry_right_exit",
            "lr_to_tb_top_drop_two_lines",
            "tb_two_line_vert_seam",
        )
    },
    # A column mate whose content starts at a different X is held out of the
    # Stage 3.6 levelling, because levelling it would trade the shared edge for a
    # widened runway spread.  The Stage 3.3 perpendicular-entry runway is what
    # puts the content at a different X here: it re-wraps the box around the
    # shifted run, moving the box's own edge with it.
    **{
        (stem, "group_alignment"): (
            "column mate held out of the levelling because the perpendicular-entry "
            "runway moved its content off the column's content X (#1545)"
        )
        for stem in (
            "bt_exit_top_above",
            "bt_exit_top_above_2line",
            "bt_to_tb",
            "lr_top_entry_cross_column",
            "lr_top_entry_cross_column_two_line",
        )
    },
    # Both levelling passes act on maximal runs of *adjacent* rows / columns, so
    # a group with a hole in it is levelled as separate runs and any shared edge
    # across the hole is incidental.
    **{
        (stem, "group_alignment"): (
            "grid group's members sit in non-adjacent cells, so no single run "
            "levels them (#1545)"
        )
        for stem in ("lr_to_tb_top_cross_col",)
    },
    # A column of same-flow sections lands on one edge only when their leftward
    # box overhangs match: Stage 1.5 absorbs the largest overhang globally, so a
    # member with a smaller one is seated further in, and its content moves with
    # it out of the levelling's reach.
    **{
        (stem, "group_alignment"): (
            "column mates seated apart by their differing box overhangs, which "
            "the global Stage 1.5 absorption does not equalise (#1545)"
        )
        for stem in ("orbit_perp_exit_back_row_entry",)
    },
    # _infer_flow_exit_hints_with_drops's perpendicular-drop exception tests only
    # whether the TARGET section is vertical, so a vertical-flow source feeding a
    # same-row horizontal target keeps a flow-aligned exit instead of turning
    # toward its neighbour.
    **{
        (stem, family): (
            "auto-inferred exit has no drop exception for a vertical-flow "
            "source facing a same-row horizontal target (#1545)"
        )
        for stem in ("lr_to_tb_top_drop", "top_entry_header_clash")
        for family in ("port_side", "port_perpendicular")
    },
}


def _corpus() -> list[Path]:
    """Every ``.mmd`` in the shared render corpus that this oracle can transform.

    The Nextflow-DAG fixtures are dropped: they need converting before they are
    metro sources at all, so there are no directives to rewrite.  Rail-mode
    fixtures are already absent from ``content_corpus``, which suits the oracle
    for the same reason it suits the declarative tests -- a rail section's
    geometry comes from its own pipeline rather than the phases under test.
    """
    return [path for _id, path, is_nextflow in content_corpus() if not is_nextflow]


@lru_cache(maxsize=None)
def _laid_out(source: str) -> MetroGraph:
    """Lay *source* out, memoised on the text.

    Each fixture's reference is compared against all seven of its images, and the
    staleness check walks the same orbit again, so the same few hundred sources
    are laid out repeatedly.  The signatures only read the result.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = parse_metro_mermaid(source)
        compute_layout(graph)
        return graph


_SUBGRAPH = re.compile(r"^\s*subgraph\s+\S+", re.MULTILINE)
_DIRECTION_DIRECTIVE = re.compile(r"^\s*%%metro\s+direction\s*:", re.MULTILINE)


def _could_be_transformable(source: str) -> bool:
    """Whether *source* is worth parsing to check transformability.

    ``%%metro direction:`` is section-scoped, so a source declaring fewer of them
    than it has subgraphs must be leaving at least one flow to inference.
    Rejecting those on the text alone keeps collection from parsing the whole
    corpus to find the handful that qualify; it never rejects a fixture the full
    check would accept.
    """
    sections = len(_SUBGRAPH.findall(source))
    return sections >= 2 and len(_DIRECTION_DIRECTIVE.findall(source)) >= sections


def _transformable_fixtures() -> list[Path]:
    """Corpus fixtures whose geometry is stated explicitly enough to transform."""
    out = []
    for path in _corpus():
        source = path.read_text()
        if not _could_be_transformable(source):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                graph = parse_metro_mermaid(source)
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

    found = divergences(reference, image, orientation)
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
        for orientation in non_identity_orientations():
            image = _laid_out(transform_source(source, orientation))
            for d in divergences(reference, image, orientation):
                live.add((path.stem, d.family))

    stale = sorted(set(KNOWN_DIVERGENCES) - live)
    assert not stale, (
        f"{len(stale)} stated exception(s) no longer diverge: {stale}. "
        "Remove them from KNOWN_DIVERGENCES so the invariant is enforced."
    )


def test_stated_exceptions_name_real_families() -> None:
    """Each exception keys on a family the oracle reports.

    A misspelled family matches no divergence, which surfaces as an unexcepted
    failure in one test and a stale entry in another; neither names the typo.
    """
    unknown = sorted(
        key for key in KNOWN_DIVERGENCES if key[1] not in DIVERGENCE_FAMILIES
    )
    assert not unknown, (
        f"unknown divergence family in KNOWN_DIVERGENCES: {unknown}. "
        f"Known families: {sorted(DIVERGENCE_FAMILIES)}"
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
