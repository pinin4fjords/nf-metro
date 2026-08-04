"""Tests for section numbering by visual reading order.

After layout, automatic sections are numbered top-to-bottom.  Sections within
a row follow its horizontal flow direction, while authored numbers stay fixed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    """Parse and lay out an example pipeline."""
    text = (EXAMPLES_DIR / f"{name}.mmd").read_text()
    g = parse_metro_mermaid(text)
    compute_layout(g)
    return g


# Module-scoped fixtures to avoid redundant compute_layout calls.


@pytest.fixture(scope="module")
def variantprioritization():
    return _load("variantprioritization")


@pytest.fixture(scope="module")
def variantbenchmarking():
    return _load("variantbenchmarking")


@pytest.fixture(scope="module")
def rnaseq_auto():
    return _load("rnaseq_auto")


@pytest.fixture(scope="module")
def longread_variant_calling():
    return _load("longread_variant_calling")


@pytest.fixture(scope="module")
def leftward_bypass():
    return _load("topologies/bypass_left_entry_from_right")


@pytest.fixture(scope="module")
def asymmetric_tree():
    return _load("topologies/asymmetric_tree")


class TestSectionNumberingOrder:
    """Section numbers should follow visual reading order."""

    def test_numbers_are_sequential(self, variantprioritization):
        """Section numbers should be 1..N with no gaps."""
        numbers = sorted(s.number for s in variantprioritization.sections.values())
        assert numbers == list(range(1, len(variantprioritization.sections) + 1))

    def test_all_examples_sequential(self):
        """Every example with sections should have sequential numbering."""
        for mmd_path in sorted(EXAMPLES_DIR.glob("*.mmd")):
            text = mmd_path.read_text()
            g = parse_metro_mermaid(text)
            if not g.sections:
                continue
            compute_layout(g)
            numbers = sorted(s.number for s in g.sections.values())
            assert numbers == list(range(1, len(g.sections) + 1)), (
                f"{mmd_path.name}: section numbers not sequential: {numbers}"
            )

    def test_rows_are_numbered_top_to_bottom(
        self, variantprioritization, variantbenchmarking, rnaseq_auto
    ):
        for graph in (variantprioritization, variantbenchmarking, rnaseq_auto):
            rows = [
                section.grid_row
                for section in sorted(
                    graph.sections.values(), key=lambda section: section.number
                )
            ]
            assert rows == sorted(rows)

    def test_lr_rows_are_numbered_left_to_right(self, variantprioritization):
        ordered_ids = [
            section.id
            for section in sorted(
                variantprioritization.sections.values(),
                key=lambda section: section.number,
            )
        ]
        assert ordered_ids == [
            "preprocessing",
            "format_files",
            "run_pcgr",
            "get_reference",
            "run_cpsr",
        ]

    def test_rl_rows_are_numbered_right_to_left(self, longread_variant_calling):
        return_row = sorted(
            (
                section
                for section in longread_variant_calling.sections.values()
                if section.grid_row == 1
            ),
            key=lambda section: section.number,
        )
        assert [section.grid_col for section in return_row] == [5, 4, 3, 2, 1, 0]

    def test_row_edge_sets_flow_when_section_directions_are_mixed(
        self, leftward_bypass
    ):
        ordered_ids = [
            section.id
            for section in sorted(
                (
                    section
                    for section in leftward_bypass.sections.values()
                    if section.grid_row == 0
                ),
                key=lambda section: section.number,
            )
        ]
        assert ordered_ids == ["source", "blocker", "target"]

    def test_authored_number_is_reserved_while_automatic_numbers_fill_gaps(self):
        text = (
            "%%metro line: main | Main | #ff0000\n"
            "graph LR\n"
            "    subgraph first [First]\n"
            "        a[A]\n"
            "    end\n"
            "    subgraph second [Second]\n"
            "        %%metro number: 7\n"
            "        b[B]\n"
            "    end\n"
            "    subgraph third [Third]\n"
            "        c[C]\n"
            "    end\n"
            "    a -->|main| b\n"
            "    b -->|main| c\n"
        )
        graph = parse_metro_mermaid(text)
        compute_layout(graph)
        assert {sid: section.number for sid, section in graph.sections.items()} == {
            "first": 1,
            "second": 7,
            "third": 2,
        }

    def test_fold_return_row_numbered_after_forward_row(self, rnaseq_auto):
        """RL sections after a fold should have higher numbers than all
        LR sections in the preceding sweep."""
        lr_nums = [
            s.number
            for s in rnaseq_auto.sections.values()
            if s.direction in ("LR", "TB")
        ]
        rl_nums = [
            s.number for s in rnaseq_auto.sections.values() if s.direction == "RL"
        ]
        if lr_nums and rl_nums:
            assert min(rl_nums) > max(lr_nums), (
                f"RL sections {rl_nums} should all be > LR/TB sections {lr_nums}"
            )

    def test_asymmetric_top_row_sequential(self, asymmetric_tree):
        top_row = sorted(
            (s for s in asymmetric_tree.sections.values() if s.grid_row == 0),
            key=lambda s: s.grid_col,
        )
        nums = [s.number for s in top_row]
        for i in range(len(nums) - 1):
            assert nums[i] < nums[i + 1], f"Top row numbers not increasing: {nums}"
