"""Lock the property that makes a fitted layout objective safe to minimise.

A score is safe to descend when it cannot be improved without bound. The
committed `safe_weights.json` is built to have that property structurally --
every feature it reads is non-negative on every layout, and every weight on it
is non-negative, so the score is bounded below by zero. These tests check both
halves against real geometry rather than against the arithmetic that produced
them, and pin the failure mode they exist to prevent: `iter2_weights.json` is
kept alongside as a live counter-example, so the growth probe is known to be
capable of catching an unsafe objective.

See `datasets/layout_preferences/README.md` for the measurement this rests on
and for why nothing here gates CI on a score value.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "datasets" / "layout_preferences"
SCRIPTS = DATASET_DIR / "scripts"

GROWTH_MULTIPLE = 4
"""Spacing multiple the monotonicity check inflates a map by.

Enough that a linear growth term moves by far more than any incidental drift in
the angle terms, and small enough to stay one extra layout pass per fixture.
"""

PROBE = (
    "examples/rnaseq_sections.mmd",
    "examples/genomic_pipeline.mmd",
    "examples/topologies/convergence_fold_diamond.mmd",
)


def _load(stem: str):
    """Import a dataset script by path; `datasets/` is not an importable package."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def terms():
    return _load("terms")


@pytest.fixture(scope="module")
def safety():
    return _load("check_objective_safety")


@pytest.fixture(scope="module")
def safe_weights() -> dict[str, float]:
    return json.loads((DATASET_DIR / "safe_weights.json").read_text())["weights"]


@pytest.fixture(scope="module")
def iter2_weights() -> dict[str, float]:
    return json.loads((DATASET_DIR / "iter2_weights.json").read_text())["weights"]


@pytest.fixture(scope="module")
def corpus_vectors() -> list[dict[str, float]]:
    """Every feature vector committed in the corpus, from both sides of a pair."""
    vectors = []
    for name in ("dataset_pairs.jsonl", "dataset_anchors.jsonl"):
        for line in (DATASET_DIR / name).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("features", "features_before", "features_after"):
                if row.get(key):
                    vectors.append(row[key])
    assert vectors
    return vectors


# --- The sentinel, which inverts the term it appears in --------------------- #


def test_an_absent_clearance_is_the_cleanest_state_not_the_worst(terms):
    """`-1.0` means no line came near a marker it does not serve, which is the
    best a map can do. Fed to the crowding formula as a gap it reads as 1.025 --
    past the ceiling and above every real value -- so a non-negative weight on
    the term would penalise exactly the fixtures with the most room."""
    assert terms.marker_crowding(terms.UNDEFINED) == 0.0
    assert terms.marker_crowding(None) == 0.0


def test_crowding_is_confined_to_the_unit_interval(terms):
    """The term stands for "how much of one lane pitch has been eaten", so a gap
    of zero saturates it. Values outside [0, 1] would let one term dominate a
    weighted sum by an unbounded margin."""
    gaps = [terms.UNDEFINED, -5.0, 0.0, 1.0, 20.0, 39.9, 40.0, 400.0]
    assert all(0.0 <= terms.marker_crowding(g) <= 1.0 for g in gaps)
    assert terms.marker_crowding(0.0) == 1.0
    assert terms.marker_crowding(terms.CROWDING_PITCH) == 0.0


def test_a_map_with_no_foreign_segment_scores_no_crowding(terms):
    """The end-to-end version of the two checks above, over a real layout: a
    single-line map has no line passing a marker it does not serve, so the
    engine reports no clearance and the term has to read zero."""
    sys.path.insert(0, str(REPO / "tests"))
    from layout_metrics import compute_metrics

    from nf_metro.layout import compute_layout
    from nf_metro.parser import parse_metro_mermaid

    graph = parse_metro_mermaid(
        """%%metro line: only | Only | #ff0000
graph LR
    subgraph s [S]
        a[A] -->|only| b[B]
        b -->|only| c[C]
    end
"""
    )
    compute_layout(graph)

    assert compute_metrics(graph)["marker_clearance"] is None
    assert terms.marker_crowding(compute_metrics(graph)["marker_clearance"]) == 0.0


# --- The admissibility table, against the data ----------------------------- #


def test_every_feature_in_the_corpus_has_an_admissibility_verdict(
    terms, corpus_vectors
):
    """A feature with no verdict has not been reasoned about, which is not the
    same as being safe. Adding a feature to the extractor without classifying it
    would otherwise let it into a minimisable objective unexamined."""
    seen = {key for vector in corpus_vectors for key in vector}
    assert not (seen - set(terms.ADMISSIBILITY))


def test_admissible_features_are_non_negative_across_the_corpus(terms, corpus_vectors):
    """The bound `score >= 0` rests on the features themselves being
    non-negative, so the classification is checked against every vector the
    corpus holds rather than asserted from the extractor's source."""
    offenders: dict[str, float] = {}
    for vector in corpus_vectors:
        for key, value in vector.items():
            if terms.ADMISSIBILITY.get(key) == terms.ADMISSIBLE and value < 0.0:
                offenders[key] = min(value, offenders.get(key, 0.0))
    assert not offenders


# --- The committed artifact ------------------------------------------------- #


def test_the_safe_artifact_is_structurally_safe(safety, safe_weights):
    assert safety.structural_findings(safe_weights) == []


def test_the_safe_artifact_reads_only_admissible_features(terms, safe_weights):
    assert terms.inadmissible(safe_weights) == {}


def test_the_safe_artifact_has_no_negative_weight(safe_weights):
    """The half of the bound that the fit itself has to deliver: a negative
    weight on a feature that grows without bound is what makes a score
    minimisable to minus infinity."""
    assert {k: v for k, v in safe_weights.items() if v < 0.0} == {}


def test_the_unconstrained_artifact_is_still_detected_as_unsafe(safety, iter2_weights):
    """`iter2` is the live counter-example. If this ever passes as safe, the
    detector has stopped working rather than the weights having become safe --
    the arm is fitted with no constraint at all."""
    findings = safety.structural_findings(iter2_weights)
    assert findings
    assert any("path_len_per_route" in f for f in findings)


# --- Growth monotonicity, through the real engine --------------------------- #


@pytest.mark.parametrize("rel", PROBE)
def test_inflating_a_map_cannot_improve_the_safe_score(safety, safe_weights, rel):
    """`x_spacing` and `y_spacing` are a CLI flag and a `%%metro` directive, so a
    uniformly inflated drawing is a reachable input and not a hypothetical. The
    same map on a larger grid must not score better, or a search could buy score
    with nothing but space."""
    text = (REPO / rel).read_text()
    base = safety.score(safe_weights, safety.vector_at(text, 1))
    grown = safety.score(safe_weights, safety.vector_at(text, GROWTH_MULTIPLE))

    assert grown >= base - 1e-9


@pytest.mark.parametrize("rel", PROBE)
def test_inflating_a_map_does_improve_the_unconstrained_score(
    safety, iter2_weights, rel
):
    """The counter-example that gives the check above its teeth: on the same
    fixtures, the unconstrained weights reward the inflation outright."""
    text = (REPO / rel).read_text()
    base = safety.score(iter2_weights, safety.vector_at(text, 1))
    grown = safety.score(iter2_weights, safety.vector_at(text, GROWTH_MULTIPLE))

    assert grown < base
