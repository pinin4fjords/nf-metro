"""Identical input must settle and render identically across hash seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from hash_seed_oracle import (
    _freeze_settled_graph,
    observation_differences,
    observe_paths_across_hash_seeds,
)

from nf_metro.parser.model import MetroGraph

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
    assert {"route_topology", "route_resolution"} <= field_names
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
