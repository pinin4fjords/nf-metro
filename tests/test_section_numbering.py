"""Tests for dependency-aware section numbering.

After layout, automatic sections are numbered by dependency wave.  Visual row
and horizontal flow break ties, while authored numbers stay fixed.
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
    """Automatic numbers should follow dependencies, then visual order."""

    def test_numbers_are_sequential(self, variantprioritization):
        """Section numbers should be 1..N with no gaps."""
        numbers = sorted(s.number for s in variantprioritization.sections.values())
        assert numbers == list(range(1, len(variantprioritization.sections) + 1))

    def test_all_examples_sequential_and_dependency_ordered(self):
        """Every example should number producers before their consumers."""
        for mmd_path in sorted(EXAMPLES_DIR.rglob("*.mmd")):
            text = mmd_path.read_text()
            g = parse_metro_mermaid(text)
            if not g.sections:
                continue
            compute_layout(g)
            numbers = sorted(s.number for s in g.sections.values())
            assert numbers == list(range(1, len(g.sections) + 1)), (
                f"{mmd_path.name}: section numbers not sequential: {numbers}"
            )
            section_edges = g.section_dag.section_edges if g.section_dag else set()
            for source, target in section_edges:
                assert g.sections[source].number < g.sections[target].number, (
                    f"{mmd_path.name}: producer {source!r} is numbered after "
                    f"consumer {target!r}"
                )

    def test_dependency_waves_precede_visual_row_order(self, variantprioritization):
        ordered_ids = [
            section.id
            for section in sorted(
                variantprioritization.sections.values(),
                key=lambda section: section.number,
            )
        ]
        assert ordered_ids == [
            "preprocessing",
            "get_reference",
            "format_files",
            "run_cpsr",
            "run_pcgr",
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

    def test_connected_flow_precedes_a_disconnected_rowmate(self, leftward_bypass):
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
        assert ordered_ids == ["source", "target", "blocker"]

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (
                "guide/03_fan_out",
                [
                    "preprocessing",
                    "wgs_analysis",
                    "wes_analysis",
                    "panel_analysis",
                    "annotation",
                ],
            ),
            (
                "guide/04_directions",
                [
                    "preprocessing",
                    "rna_analysis",
                    "dna_analysis",
                    "postprocessing",
                    "reporting",
                ],
            ),
            (
                "rnaseq_auto",
                [
                    "preprocessing",
                    "genome_align",
                    "pseudo_align",
                    "postprocessing",
                    "qc_report",
                ],
            ),
            (
                "topologies/around_below_ep_col_gt0",
                ["source", "middle", "target"],
            ),
            (
                "topologies/around_section_below",
                ["source", "middle", "target"],
            ),
            (
                "topologies/corridor_narrow_gap_fallback",
                ["source", "tall", "target"],
            ),
        ],
    )
    def test_reviewed_render_order(self, name, expected):
        graph = _load(name)
        ordered_ids = [
            section.id
            for section in sorted(
                graph.sections.values(), key=lambda section: section.number
            )
        ]
        assert ordered_ids == expected

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
