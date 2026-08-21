"""Identical input must settle and render identically across hash seeds."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest
from hash_seed_oracle import (
    _freeze_settled_graph,
    observation_differences,
    observe_paths_across_hash_seeds,
)

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.parser.model import MetroGraph
from nf_metro.render.svg import render_svg

ROOT = Path(__file__).resolve().parents[1]
REGRESSIONS = ROOT / "tests" / "fixtures" / "hash_seed_determinism"

FROZEN_SOURCE_SHA256 = {
    "seed_15.mmd": "c0f514bcd7109a8c5b83e8aa1b8964feee5f759b0ffaef0b0d66c0b7f716b49b",
    "seed_41.mmd": "aa7240e9a8bca5309a8435c06eeb3a1fdde4fc5f63a0647866a14a5ac9680a1b",
    "seed_72.mmd": "2aa378b381e1af863cfe82b6b1f492cf28a3899650787430ec00f597138dcc35",
    "seed_77.mmd": "600d869e312cd7afd4f7ccee8cc533f1a81c47f69f4e393d48059c2655fe8397",
}

REPRESENTATIVE_CORPUS = (
    "examples/rnaseq_auto.mmd",
    "examples/variant_calling.mmd",
    "examples/topologies/convergence_fold_diamond.mmd",
    "examples/topologies/fan_in_merge.mmd",
    "examples/topologies/lr_to_tb_top_two_lines.mmd",
    "examples/topologies/packed_cell_cellmate_bypass.mmd",
    "examples/topologies/u_turn_fold.mmd",
    "examples/topologies/wide_fan_out.mmd",
)


def _assert_identical(paths: tuple[Path, ...]) -> None:
    observations = observe_paths_across_hash_seeds(paths)
    differences = observation_differences(observations)
    assert not differences, json.dumps(differences, indent=2, sort_keys=True)


def test_settled_graph_snapshot_includes_semantic_route_state() -> None:
    field_names = {
        name for name, _value in _freeze_settled_graph(MetroGraph()).values.entries
    }
    assert {"route_topology", "route_resolution", "layout_provenance"} <= field_names
    assert (
        not {
            "_station_lines_cache",
            "_edges_from_cache",
            "_edges_to_cache",
            "_junction_ids_cache",
        }
        & field_names
    )


@pytest.mark.parametrize("name", sorted(FROZEN_SOURCE_SHA256))
def test_frozen_regression_source_bytes(name: str) -> None:
    assert (
        hashlib.sha256((REGRESSIONS / name).read_bytes()).hexdigest()
        == (FROZEN_SOURCE_SHA256[name])
    )


def test_regressions_and_representative_corpus_are_hash_seed_deterministic() -> None:
    paths = tuple(REGRESSIONS / name for name in sorted(FROZEN_SOURCE_SHA256)) + tuple(
        ROOT / relative_path for relative_path in REPRESENTATIVE_CORPUS
    )
    _assert_identical(paths)


EXPECTED_DEFECT_CLASSES = {
    "seed_15.mmd": (
        "bundle order (line crosses its bundle-mate)",
        "undeclared gap channel",
        "peel-off bundle braids into port",
        "merge fan-out branches split off one fork corner",
    ),
    "seed_41.mmd": (
        "non-concentric bundle corner",
        "merge fan-out branches split off one fork corner",
    ),
    "seed_77.mmd": (
        "undeclared gap channel",
        "peel-off bundle braids into port",
    ),
}

EXPECTED_EXCEPTION_SHA256 = {
    "seed_15.mmd": "e6f1c8e5faf9f11e2501ed47dd508e5d077ba9cb1812c32c6c3a96bc9e43979e",
    "seed_41.mmd": "b8cab47b9ffb38a1e96961daf2dd21132eb8a4d1435122f194da294d7fdf397f",
    "seed_77.mmd": "40597beab7da0cc311a70e6012896d1a493a854999499eea700c172aac652643",
}


def _defect_classes(message: str) -> tuple[str, ...]:
    """The bracketed defect classes a routing-guard abort message enumerates."""
    return tuple(re.findall(r"^\s*\[([^]]+)\]", message, re.MULTILINE))


def test_seed_render_outcomes_are_stable_across_hash_seeds() -> None:
    """Each seed's abort names exactly the defect classes frozen for it.

    The classes are frozen by name as well as inside the message digest: a
    digest alone accepts a newly *added* defect as just another expected value,
    which is how a gained defect can ride along invisibly. Naming them makes any
    change to the set legible in the diff and any addition a named failure.
    """
    paths = tuple(REGRESSIONS / name for name in sorted(FROZEN_SOURCE_SHA256))
    observations = observe_paths_across_hash_seeds(paths)

    for by_path in observations.values():
        for path, observation in by_path.items():
            name = Path(path).name
            if name == "seed_72.mmd":
                assert observation["outcome"] == "success"
                assert observation["exception"] is None
                continue

            assert observation["outcome"] == "exception"
            exception = observation["exception"]
            assert exception is not None
            assert (
                _defect_classes(exception["message"]) == EXPECTED_DEFECT_CLASSES[name]
            ), name
            contract = {
                "phase": exception["phase"],
                "qualified_class": exception["qualified_class"],
                "message": exception["message"],
            }
            digest = hashlib.sha256(
                json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            assert digest == EXPECTED_EXCEPTION_SHA256[name]


def _pinned_cairosvg_version() -> str:
    """The cairosvg version pyproject pins, which the frozen PNG hash assumes."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pinned = {
        requirement.removeprefix("cairosvg==")
        for group in pyproject["project"]["optional-dependencies"].values()
        for requirement in group
        if requirement.startswith("cairosvg==")
    }
    assert len(pinned) == 1, f"cairosvg is pinned to several versions: {pinned}"
    return pinned.pop()


def test_seed_72_linux_cairosvg_png_is_frozen() -> None:
    if sys.platform != "linux":
        pytest.skip("authoritative PNG lock runs on Linux")
    import cairosvg

    assert cairosvg.__version__ == _pinned_cairosvg_version()

    path = REGRESSIONS / "seed_72.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    svg = render_svg(
        graph,
        resolve_theme(None, graph),
        chrome_css=False,
    )
    png = cairosvg.svg2png(bytestring=svg.encode(), scale=2)

    assert hashlib.sha256(png).hexdigest() == (
        "c951485a7fe6338db39691c68116eb104312c1d9e545bd049c986dbc0b9083b9"
    )


def test_same_destination_topologies_render_and_plan_across_hash_seeds() -> None:
    paths = tuple(
        ROOT / "examples" / "topologies" / f"{stem}.mmd"
        for stem in (
            "same_destination_short_overlap",
            "same_destination_vertical_convergence",
        )
    )
    observations = observe_paths_across_hash_seeds(paths)

    for by_path in observations.values():
        for observation in by_path.values():
            assert observation["outcome"] == "success"
            assert observation["render_plan_sha256"] is not None
            assert observation["svg_sha256"] is not None
            assert observation["findings"] == []
    differences = observation_differences(observations)
    assert not differences, json.dumps(differences, indent=2, sort_keys=True)
