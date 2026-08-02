"""Semantic fan plans own complete structural objects or no geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from nf_metro.api import prepare_graph
from nf_metro.layout.constants import (
    INTER_ROW_EDGE_CLEARANCE,
    SECTION_Y_PADDING,
    X_SPACING,
)
from nf_metro.layout.engine import compute_layout, compute_min_y_spacing
from nf_metro.layout.fan_geometry import fan_lane_offsets, symmetric_lane_offsets
from nf_metro.layout.fan_plans import (
    FanPlanExecution,
    FanPlanQuery,
    FanTopologyQuery,
    build_fan_plan_execution,
    install_fan_plan_execution,
    validate_fan_route_emissions,
)
from nf_metro.layout.labels import place_labels
from nf_metro.layout.phases.guards import (
    PhaseInvariantError,
    _guard_planned_fan_frame_realised,
)
from nf_metro.layout.phases.planned_fans import (
    _apply_planned_fan_geometry,
    _apply_planned_fan_port_geometry,
    _snapshot_planned_fan_centrelines,
)
from nf_metro.layout.route_plan import (
    CoordinateRegime,
    DemandAxis,
    DemandKind,
    FanAppearancePolicy,
    FanCentrelineAnchor,
    FanOffsetAssignment,
    FanOffsetCarrier,
    FanPlanDisposition,
    KeepOutClass,
    SharedReferenceKind,
    build_route_plan_query,
)
from nf_metro.layout.routing import (
    compute_station_offsets,
    observe_route_edges,
    route_edges,
)
from nf_metro.layout.routing.offsets import _apply_planned_fan_offsets
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import Edge, MetroGraph, Port, PortSide, Section, Station
from nf_metro.parser.route_topology import (
    AuthoredEdgeFact,
    AuthoredEdgeKey,
    BundleId,
    ConnectorId,
    ConvergenceId,
    ResolvedEdge,
    build_route_topology_query,
)

ROOT = Path(__file__).parents[1]


def _fact(
    source: str,
    target: str,
    line_id: str,
    rank: int,
    *,
    section: str = "section",
) -> AuthoredEdgeFact:
    return AuthoredEdgeFact(
        key=AuthoredEdgeKey(source, target, line_id, 0),
        rank=rank,
        source_line=rank + 1,
        source_section=section,
        target_section=section,
    )


@dataclass
class _Topology:
    authored_edges: tuple[AuthoredEdgeFact, ...]
    paths: dict[ConnectorId, tuple[tuple[ResolvedEdge, ...], ...]]
    connector_bundles: dict[ConnectorId, BundleId]
    convergences: tuple[object, ...]

    @classmethod
    def direct(cls, facts: list[AuthoredEdgeFact]) -> _Topology:
        return cls(
            authored_edges=tuple(reversed(facts)),
            paths={
                fact.id: (
                    (
                        ResolvedEdge(
                            fact.key.source,
                            fact.key.target,
                            fact.key.line_id,
                        ),
                    ),
                )
                for fact in facts
            },
            connector_bundles={},
            convergences=(),
        )

    def authored_edge(self, edge_id: ConnectorId) -> AuthoredEdgeFact:
        return next(fact for fact in self.authored_edges if fact.id == edge_id)

    def resolved_paths(
        self, edge_id: ConnectorId
    ) -> tuple[tuple[ResolvedEdge, ...], ...]:
        return self.paths.get(edge_id, ())

    def authored_edge_ids_for_edge(self, edge: ResolvedEdge) -> tuple[ConnectorId, ...]:
        return tuple(
            edge_id
            for edge_id, paths in self.paths.items()
            if any(edge in path for path in paths)
        )

    def connector(self, edge_id: ConnectorId) -> object:
        try:
            bundle_id = self.connector_bundles[edge_id]
        except KeyError as error:
            raise KeyError(edge_id) from error
        return SimpleNamespace(bundle_id=bundle_id)

    def convergence_for_junction(self, junction_id: str) -> object | None:
        return next(
            (
                convergence
                for convergence in self.convergences
                if getattr(convergence, "junction_id", None) == junction_id
            ),
            None,
        )


def _graph(direction: str = "LR") -> MetroGraph:
    graph = MetroGraph()
    graph.add_section(Section(id="section", name="Section", direction=direction))
    return graph


def test_topology_test_double_implements_planner_contract() -> None:
    assert isinstance(_Topology.direct([]), FanTopologyQuery)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (2, (-5.0, 5.0)),
        (3, (-10.0, 0.0, 10.0)),
        (4, (-15.0, -5.0, 5.0, 15.0)),
        (6, (-25.0, -15.0, -5.0, 5.0, 15.0, 25.0)),
    ],
)
def test_lane_offsets_straddle_one_centreline(
    count: int, expected: tuple[float, ...]
) -> None:
    assert symmetric_lane_offsets(count, 10.0) == expected


def test_branch_rank_comes_from_authored_order() -> None:
    targets = ("zeta", "alpha", "mu", "beta")
    facts = [
        _fact("fork", target, f"line_{rank}", rank)
        for rank, target in enumerate(targets)
    ]

    execution = build_fan_plan_execution(
        _graph(),
        _Topology.direct(facts),
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=24.0,
    )

    assert len(execution.plans) == 1
    plan = execution.plans[0]
    assert plan.disposition is FanPlanDisposition.PLANNED
    assert tuple(branch.root_station_id for branch in plan.branches) == targets
    assert plan.appearance_centreline_branch_id is None
    assert tuple(branch.lane_offset for branch in plan.branches) == (
        -15.0,
        -5.0,
        5.0,
        15.0,
    )
    assert tuple(branch.diagonal_runway for branch in plan.branches) == (
        24.0,
        34.0,
        44.0,
        54.0,
    )
    assert plan.entry_runway == 24.0
    assert plan.exit_runway == 24.0
    assert plan.centreline_reference_id is not None
    assert len(plan.demand_ids) == 6
    assert execution.query.planned_for_fork("fork") is plan


def test_unique_exit_branch_keeps_trunk_on_centreline() -> None:
    facts = [
        _fact("fork", "spur", "spur", 0),
        _fact("fork", "trunk", "trunk", 1),
        _fact("trunk", "downstream", "trunk", 2),
    ]
    topology = _Topology.direct(facts)
    topology.paths[facts[-1].id] = (
        (
            ResolvedEdge("trunk", "exit_port", "trunk"),
            ResolvedEdge("exit_port", "downstream", "trunk"),
        ),
    )
    graph = _graph()
    for station_id in ("spur", "trunk"):
        graph.register_station(
            Station(id=station_id, label=station_id.title(), section_id="section")
        )
    graph.ports["fork"] = Port(
        id="fork",
        section_id="section",
        side=PortSide.LEFT,
        is_entry=True,
    )
    graph.ports["exit_port"] = Port(
        id="exit_port",
        section_id="section",
        side=PortSide.RIGHT,
        is_entry=False,
    )

    plan = build_fan_plan_execution(
        graph,
        topology,
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    ).plans[0]

    assert tuple(branch.is_trunk_continuation for branch in plan.branches) == (
        False,
        True,
    )
    assert plan.appearance_centreline_branch_id == plan.branches[1].id
    assert tuple(branch.lane_offset for branch in plan.branches) == (10.0, 0.0)
    assert plan.local_frame_anchor == FanCentrelineAnchor("trunk")


def test_local_full_bundle_continuation_owns_the_fan_centreline() -> None:
    path = ROOT / "examples" / "topologies" / "render_labelwrap_row_gap.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "star")

    trunk_branches = tuple(
        branch for branch in plan.branches if branch.is_trunk_continuation
    )
    assert plan.disposition is FanPlanDisposition.PLANNED
    assert tuple(branch.root_station_id for branch in trunk_branches) == ("cram_out",)
    assert trunk_branches[0].lane_offset == 0.0
    assert all(
        not branch.is_trunk_continuation
        for branch in plan.branches
        if branch.landing_port_ids
    )
    assert (
        len(
            {
                graph.stations[station_id].y
                for station_id in ("reads_in", "bbduk", "star", "cram_out")
            }
        )
        == 1
    )


def test_foreign_merge_frame_keeps_the_complete_fan_on_legacy_layout() -> None:
    path = ROOT / "examples" / "topologies" / "tb_passthrough_continuation.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "starN")

    assert plan.disposition is FanPlanDisposition.LEGACY
    assert plan.legacy_reason == "local-layout-has-foreign-owner"
    assert graph.stations["starN"].x == graph.stations["leftchild"].x
    assert (
        len(
            {graph.stations[station_id].x for station_id in ("hisatN", "merge", "tail")}
        )
        == 1
    )


def test_straight_diamond_keeps_established_layout_ownership() -> None:
    path = ROOT / "examples" / "topologies" / "shared_cell_fork_trunk_align.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "p_hub")

    assert graph.diamond_style == "straight"
    assert plan.authored_join_station_id == "p_merge"
    assert plan.join_station_id == "p_merge"
    assert plan.appearance_policy is FanAppearancePolicy.STRAIGHT
    assert plan.disposition is FanPlanDisposition.LEGACY
    assert plan.legacy_reason == "straight-diamond-layout-owns-geometry"
    assert plan.layout_station_ids == ()


@pytest.mark.parametrize(
    "fixture,source_id",
    [
        ("wide_label_fan.mmd", "hub"),
        ("bypass_v_tight.mmd", "m1"),
        ("junction_entry_collision.mmd", "pre2"),
    ],
)
def test_straight_open_fan_keeps_top_branch_on_centreline(
    fixture: str, source_id: str
) -> None:
    """Straight fans keep their first authored branch on the main track."""
    path = ROOT / "examples" / "topologies" / fixture
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(
        item for item in graph.fan_plans if item.authored_source_id == source_id
    )

    assert plan.appearance_policy is FanAppearancePolicy.STRAIGHT
    assert plan.disposition is FanPlanDisposition.PLANNED
    assert plan.frame is not None
    assert plan.appearance_centreline_branch_id == plan.branches[0].id
    assert tuple(branch.lane_offset for branch in plan.branches) == pytest.approx(
        tuple(rank * plan.frame.secondary.step for rank in range(len(plan.branches)))
    )
    fork = graph.stations[plan.fork_station_id]
    first_branch = graph.stations[plan.branches[0].lane_station_ids[0]]
    assert plan.frame.secondary.get(first_branch) == pytest.approx(
        plan.frame.secondary.get(fork)
    )


@pytest.mark.parametrize(
    "fixture,source_id",
    [
        ("wide_label_fan.mmd", "hub"),
        ("bypass_v_tight.mmd", "m1"),
        ("junction_entry_collision.mmd", "pre2"),
    ],
)
def test_symmetric_open_fan_straddles_centreline_only_when_requested(
    fixture: str, source_id: str
) -> None:
    """The symmetric directive gives the same fan an evenly centred frame."""
    path = ROOT / "examples" / "topologies" / fixture
    graph = parse_metro_mermaid(path.read_text())
    graph.diamond_style = "symmetric"
    compute_layout(graph, validate=True)
    plan = next(
        item for item in graph.fan_plans if item.authored_source_id == source_id
    )

    assert plan.appearance_policy is FanAppearancePolicy.SYMMETRIC
    assert plan.appearance_centreline_branch_id is None
    assert plan.frame is not None
    assert tuple(branch.lane_offset for branch in plan.branches) == pytest.approx(
        symmetric_lane_offsets(len(plan.branches), plan.frame.secondary.step)
    )


@pytest.mark.parametrize(
    ("fixture", "hub_id", "branch_ids", "axis"),
    [
        ("file_icons.mmd", "align", ("ref_in", "reads_in"), "y"),
        (
            "file_icons.mmd",
            "collect",
            ("aln_out", "report_out", "results_out"),
            "y",
        ),
        (
            "tb_file_termini.mmd",
            "report",
            ("multiqc", "bundle", "report_html"),
            "x",
        ),
    ],
)
def test_balanced_file_examples_explicitly_request_symmetric_fans(
    fixture: str,
    hub_id: str,
    branch_ids: tuple[str, ...],
    axis: str,
) -> None:
    path = ROOT / "examples" / fixture
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)

    assert graph.diamond_style == "symmetric"
    centreline = getattr(graph.stations[hub_id], axis)
    offsets = sorted(
        getattr(graph.stations[station_id], axis) - centreline
        for station_id in branch_ids
    )
    assert offsets == pytest.approx(tuple(-offset for offset in reversed(offsets)))


def test_tb_file_termini_symmetric_plan_preserves_complete_branch_bundles() -> None:
    path = ROOT / "examples" / "tb_file_termini.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "report")

    assert plan.disposition is FanPlanDisposition.PLANNED
    assert plan.appearance_policy is FanAppearancePolicy.SYMMETRIC
    assert all(branch.line_ids == ("rna", "dna") for branch in plan.branches)
    assert tuple(branch.lane_offset for branch in plan.branches) == pytest.approx(
        symmetric_lane_offsets(len(plan.branches), plan.appearance_lane_pitch)
    )
    assert graph.stations["bundle"].x == pytest.approx(graph.stations["bundle_zip"].x)
    assert graph.stations["multiqc"].x == pytest.approx(
        graph.stations["multiqc_html"].x
    )


def test_runtime_guard_rejects_symmetric_straight_open_fan_plan() -> None:
    """A straight plan cannot silently realise symmetric lane geometry."""
    path = ROOT / "examples" / "topologies" / "wide_label_fan.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=False)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "hub")
    assert plan.frame is not None
    symmetric_offsets = symmetric_lane_offsets(
        len(plan.branches), plan.frame.secondary.step
    )
    bad_branches = tuple(
        replace(branch, lane_offset=offset)
        for branch, offset in zip(plan.branches, symmetric_offsets, strict=True)
    )
    with pytest.raises(
        ValueError,
        match="straight local fan must have one non-negative centreline lane",
    ):
        replace(plan, branches=bad_branches)
    object.__setattr__(
        plan,
        "branches",
        bad_branches,
    )

    with pytest.raises(
        PhaseInvariantError,
        match="straight planned fan .* does not keep its top branch on the centreline",
    ):
        _guard_planned_fan_frame_realised(
            graph,
            "test",
            offsets=compute_station_offsets(graph),
        )


def test_runtime_guard_accepts_content_expanded_appearance_lane_pitch() -> None:
    """A fan may freeze a content-safe pitch larger than its nominal axis step."""
    path = ROOT / "examples" / "topologies" / "tb_internal_diagonal.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=False)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "hub")
    assert plan.frame is not None
    expanded_pitch = plan.frame.secondary.step + 20.0
    expanded_branches = tuple(
        replace(
            branch,
            lane_offset=rank * expanded_pitch,
            diagonal_runway=max(
                branch.diagonal_runway or 0.0,
                rank * expanded_pitch,
            ),
        )
        for rank, branch in enumerate(plan.branches)
    )
    with pytest.raises(
        ValueError,
        match="fan lane offsets disagree with appearance pitch",
    ):
        replace(plan, appearance_lane_pitch=expanded_pitch)
    expanded_plan = replace(
        plan,
        branches=expanded_branches,
        appearance_lane_pitch=expanded_pitch,
    )
    install_fan_plan_execution(
        graph,
        FanPlanExecution(
            query=FanPlanQuery.build((expanded_plan,)),
        ),
    )
    centreline = expanded_plan.frame.secondary.get(
        graph.stations[expanded_plan.fork_station_id]
    )
    _apply_planned_fan_geometry(graph, {expanded_plan.id: centreline})

    _guard_planned_fan_frame_realised(
        graph,
        "test",
        offsets=compute_station_offsets(graph),
    )


def test_same_line_open_boundary_fan_keeps_established_layout_ownership() -> None:
    path = ROOT / "examples" / "topologies" / "section_trunk_short_output_branch.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "s3")

    assert plan.disposition is FanPlanDisposition.LEGACY
    assert plan.legacy_reason == "same-line-open-fan-layout-owns-geometry"
    assert plan.resolved_member_edges
    assert plan.layout_station_ids == ()
    assert (graph.stations["s3"].x, graph.stations["s3"].y) != (
        graph.stations["sink"].x,
        graph.stations["sink"].y,
    )


def test_branches_keep_line_identity_after_partial_convergence() -> None:
    facts = [
        _fact("fork", "via", "b", 0),
        _fact("fork", "merge", "a", 1),
        _fact("fork", "other", "c", 2),
        _fact("via", "merge", "b", 3),
        _fact("merge", "join", "a", 4),
        _fact("merge", "join", "b", 5),
        _fact("other", "join", "c", 6),
    ]

    graph = _graph()
    graph.diamond_style = "symmetric"
    execution = build_fan_plan_execution(
        graph,
        _Topology.direct(facts),
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    )

    plan = execution.plans[0]
    assert plan.disposition is FanPlanDisposition.PLANNED
    assert tuple(
        tuple(
            edge.line_id for path in branch.continuation_resolved_paths for edge in path
        )
        for branch in plan.branches
    ) == (("b", "b", "b"), ("a", "a"), ("c", "c"))
    assert execution.query.owner_for_authored_edge(facts[4].id) is plan
    assert execution.query.owner_for_authored_edge(facts[5].id) is plan


def test_shared_same_line_suffix_has_plan_but_no_unique_branch_owner() -> None:
    facts = [
        _fact("fork", "a", "rna", 0),
        _fact("fork", "b", "rna", 1),
        _fact("fork", "dead", "qc", 2),
        _fact("a", "merge", "rna", 3),
        _fact("b", "merge", "rna", 4),
        _fact("merge", "tail", "rna", 5),
    ]

    execution = build_fan_plan_execution(
        _graph(),
        _Topology.direct(facts),
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    )

    plan = execution.plans[0]
    shared = ResolvedEdge("merge", "tail", "rna")
    assert plan.disposition is FanPlanDisposition.PLANNED
    assert execution.query.structural_owner_for_resolved_edge(shared) is plan
    assert execution.query.structural_branch_for_resolved_edge(shared) is None
    assert plan.offset_line_order == ("rna", "qc")


def test_duplicated_bundle_fan_explicitly_preserves_incoming_line_order() -> None:
    facts = [
        _fact("fork", "a", "rna", 0),
        _fact("fork", "a", "qc", 1),
        _fact("fork", "b", "rna", 2),
        _fact("fork", "b", "qc", 3),
    ]

    plan = build_fan_plan_execution(
        _graph(),
        _Topology.direct(facts),
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    ).plans[0]

    assert plan.disposition is FanPlanDisposition.PLANNED
    assert plan.offset_line_order == ()
    assert plan.offset_carriers == ()


@pytest.mark.parametrize(
    (
        "direction",
        "primary",
        "secondary",
        "primary_sign",
        "secondary_sign",
        "lane_pitch",
    ),
    [
        ("LR", DemandAxis.X, DemandAxis.Y, 1.0, 1.0, 14.0),
        ("RL", DemandAxis.X, DemandAxis.Y, -1.0, 1.0, 14.0),
        ("TB", DemandAxis.Y, DemandAxis.X, 1.0, -1.0, 30.0),
        ("BT", DemandAxis.Y, DemandAxis.X, -1.0, 1.0, 30.0),
    ],
)
def test_fan_frame_rotates_without_changing_branch_order(
    direction: str,
    primary: DemandAxis,
    secondary: DemandAxis,
    primary_sign: float,
    secondary_sign: float,
    lane_pitch: float,
) -> None:
    facts = [_fact("fork", "a", "one", 0), _fact("fork", "b", "two", 1)]

    plan = build_fan_plan_execution(
        _graph(direction),
        _Topology.direct(facts),
        x_spacing=30.0,
        y_spacing=14.0,
        minimum_runway=20.0,
    ).plans[0]

    assert plan.frame is not None
    assert plan.frame.primary.name == primary.value
    assert plan.frame.secondary.name == secondary.value
    assert plan.frame.primary_sign == primary_sign
    assert plan.frame.secondary_sign == secondary_sign
    assert plan.frame.secondary.step == lane_pitch
    assert tuple(branch.lane_offset for branch in plan.branches) == (
        -lane_pitch / 2,
        lane_pitch / 2,
    )


def test_vertical_fan_pitch_keeps_same_layer_labels_clear_of_markers() -> None:
    path = ROOT / "examples" / "topologies" / "tb_internal_diagonal.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "hub")
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    labels = {
        placement.station_id: placement.text
        for placement in place_labels(graph, station_offsets=offsets, routes=routes)
    }

    assert plan.frame is not None
    assert plan.appearance_lane_pitch == pytest.approx(78.0)
    assert labels["left"] == "Lane A"
    assert labels["right"] == "Lane B"


def test_vertical_fan_pitch_stays_canonical_when_labels_already_clear() -> None:
    path = ROOT / "examples" / "topologies" / "tb_trunk_through_fan.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "hub")

    assert plan.frame is not None
    assert plan.appearance_lane_pitch == X_SPACING


@pytest.mark.parametrize(
    ("fixture", "bad_pitch"),
    [
        ("tb_internal_diagonal.mmd", X_SPACING),
        ("tb_trunk_through_fan.mmd", X_SPACING - 2.0),
    ],
)
def test_runtime_guard_rejects_vertical_fan_pitch_under_reservation(
    fixture: str,
    bad_pitch: float,
) -> None:
    path = ROOT / "examples" / "topologies" / fixture
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "hub")
    bad_offsets = fan_lane_offsets(
        tuple(branch.id for branch in plan.branches),
        bad_pitch,
        plan.appearance_centreline_branch_id,
    )
    bad_branches = tuple(
        replace(
            branch,
            lane_offset=lane_offset,
            diagonal_runway=max(branch.diagonal_runway or 0.0, abs(lane_offset)),
        )
        for branch, lane_offset in zip(plan.branches, bad_offsets, strict=True)
    )
    bad_plan = replace(
        plan,
        branches=bad_branches,
        appearance_lane_pitch=bad_pitch,
    )
    install_fan_plan_execution(
        graph,
        FanPlanExecution(
            query=FanPlanQuery.build((bad_plan,)),
        ),
    )

    with pytest.raises(PhaseInvariantError, match="under-reserves vertical label"):
        _guard_planned_fan_frame_realised(
            graph,
            "test",
            offsets=compute_station_offsets(graph),
        )


def test_resolved_port_fork_uses_its_section_direction() -> None:
    facts = [_fact("source", "a", "one", 0), _fact("source", "b", "two", 1)]
    topology = _Topology.direct(facts)
    topology.paths[facts[0].id] = (
        (
            ResolvedEdge("source", "entry_port", "one"),
            ResolvedEdge("entry_port", "a", "one"),
        ),
    )
    topology.paths[facts[1].id] = (
        (
            ResolvedEdge("source", "entry_port", "two"),
            ResolvedEdge("entry_port", "b", "two"),
        ),
    )
    graph = _graph("LR")
    graph.add_section(Section(id="vertical", name="Vertical", direction="TB"))
    for station_id in ("a", "b"):
        graph.register_station(
            Station(id=station_id, label=station_id.upper(), section_id="vertical")
        )
    graph.ports["entry_port"] = Port(
        id="entry_port",
        section_id="vertical",
        side=PortSide.TOP,
        is_entry=True,
    )

    plan = build_fan_plan_execution(
        graph,
        topology,
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    ).plans[0]

    assert plan.fork_station_id == "entry_port"
    assert plan.frame is not None
    assert plan.direction == "TB"
    assert plan.frame.primary.name == DemandAxis.Y.value
    assert plan.frame.secondary.step == 30.0
    assert tuple(branch.lane_offset for branch in plan.branches) == (0.0, 30.0)
    assert plan.appearance_centreline_branch_id == plan.branches[0].id
    assert plan.local_frame_anchor == FanCentrelineAnchor("a")


def test_common_resolved_approach_is_owned_as_one_fan_seam() -> None:
    facts = [_fact("source", "a", "one", 0), _fact("source", "b", "two", 1)]
    topology = _Topology.direct(facts)
    for fact in facts:
        topology.paths[fact.id] = (
            (
                ResolvedEdge("source", "exit_port", fact.key.line_id),
                ResolvedEdge("exit_port", "junction", fact.key.line_id),
                ResolvedEdge("junction", fact.key.target, fact.key.line_id),
            ),
        )
    graph = _graph()
    graph.ports["exit_port"] = Port(
        id="exit_port",
        section_id="section",
        side=PortSide.BOTTOM,
        is_entry=False,
    )
    graph.add_junction("junction")

    execution = build_fan_plan_execution(
        graph,
        topology,
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    )

    plan = execution.plans[0]
    assert plan.fork_station_id == "junction"
    assert len(plan.entry_seam_paths) == 2
    assert all(
        tuple((edge.source, edge.target) for edge in seam)
        == (
            ("source", "exit_port"),
            ("exit_port", "junction"),
        )
        for seam in plan.entry_seam_paths
    )
    assert len(plan.resolved_seam_edges) == 4
    assert all(edge in plan.resolved_member_edges for edge in plan.resolved_seam_edges)
    assert all(
        execution.query.structural_owner_for_resolved_edge(seam[-1]) is plan
        for seam in plan.entry_seam_paths
    )


def test_diamond_keeps_ports_handoffs_and_extra_output_in_one_plan() -> None:
    facts = [
        _fact("upstream", "fork", "trunk", 0),
        _fact("fork", "a", "one", 1),
        _fact("fork", "b", "two", 2),
        _fact("a", "join", "one", 3),
        _fact("a", "extra", "report", 4),
        _fact("b", "join", "two", 5),
        _fact("join", "downstream", "trunk", 6),
    ]
    topology = _Topology.direct(facts)
    topology.paths[facts[0].id] = (
        (
            ResolvedEdge("upstream", "entry_port", "trunk"),
            ResolvedEdge("entry_port", "fork", "trunk"),
        ),
    )
    topology.paths[facts[-1].id] = (
        (
            ResolvedEdge("join", "exit_port", "trunk"),
            ResolvedEdge("exit_port", "downstream", "trunk"),
        ),
    )
    topology.connector_bundles[facts[1].id] = BundleId("branch-bundle")
    topology.convergences = (
        SimpleNamespace(
            group=SimpleNamespace(
                id=ConvergenceId("join-handoff"),
                connector_ids=(facts[3].id,),
            )
        ),
    )
    graph = _graph()
    graph.diamond_style = "symmetric"
    graph.ports["entry_port"] = Port(
        id="entry_port",
        section_id="section",
        side=PortSide.LEFT,
        is_entry=True,
    )
    graph.ports["exit_port"] = Port(
        id="exit_port",
        section_id="section",
        side=PortSide.RIGHT,
        is_entry=False,
    )

    execution = build_fan_plan_execution(
        graph,
        topology,
        x_spacing=40.0,
        y_spacing=20.0,
        minimum_runway=30.0,
    )

    assert len(execution.plans) == 1
    plan = execution.plans[0]
    assert plan.disposition is FanPlanDisposition.PLANNED
    assert plan.join_station_id == "join"
    assert plan.entry_handoff_edge_ids == (facts[0].id,)
    assert plan.exit_handoff_edge_ids == (facts[-1].id,)
    assert plan.entry_port_ids == ("entry_port",)
    assert plan.exit_port_ids == ("exit_port",)
    assert plan.trunk_follower_ids == ("upstream", "downstream")
    assert plan.bundle_handoff_ids == (BundleId("branch-bundle"),)
    assert plan.convergence_handoff_ids == (ConvergenceId("join-handoff"),)
    first = plan.branches[0]
    assert first.continuation_edge_ids == (facts[1].id, facts[3].id)
    assert first.extra_output_edge_ids == (facts[4].id,)
    assert facts[4].id in plan.authored_edge_ids
    assert facts[0].id not in plan.authored_edge_ids
    assert execution.query.owner_for_authored_edge(facts[4].id) is plan


def test_missing_resolved_member_falls_back_as_one_complete_group() -> None:
    facts = [_fact("fork", "a", "one", 0), _fact("fork", "b", "two", 1)]
    topology = _Topology.direct(facts)
    del topology.paths[facts[1].id]

    execution = build_fan_plan_execution(
        _graph(),
        topology,
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    )

    plan = execution.plans[0]
    assert plan.disposition is FanPlanDisposition.LEGACY
    assert plan.legacy_reason == "missing-resolved-member-path"
    assert plan.frame is None
    assert plan.centreline_reference_id is None
    assert plan.centreline_anchor is None
    assert plan.demand_ids == ()
    assert all(branch.lane_offset is None for branch in plan.branches)
    assert execution.query.owner_for_authored_edge(facts[0].id) is None


def test_off_track_member_falls_back_as_one_complete_group() -> None:
    facts = [_fact("fork", "a", "one", 0), _fact("fork", "b", "two", 1)]
    graph = _graph()
    for station_id in ("fork", "a", "b"):
        graph.register_station(
            Station(
                id=station_id,
                label=station_id.upper(),
                section_id="section",
                off_track=station_id == "b",
            )
        )

    execution = build_fan_plan_execution(
        graph,
        _Topology.direct(facts),
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    )

    plan = execution.plans[0]
    assert plan.disposition is FanPlanDisposition.LEGACY
    assert plan.legacy_reason == "off-track-layout-owns-fan-geometry"
    assert plan.layout_station_ids == ()
    assert execution.query.planned_for_fork("fork") is None


def test_overlapping_fans_are_both_rejected() -> None:
    facts = [
        _fact("outer", "a", "one", 0),
        _fact("outer", "b", "two", 1),
        _fact("a", "shared", "one", 2),
        _fact("b", "shared", "two", 3),
        _fact("shared", "c", "three", 4),
        _fact("shared", "d", "four", 5),
    ]

    execution = build_fan_plan_execution(
        _graph(),
        _Topology.direct(facts),
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    )

    assert len(execution.plans) == 2
    assert {plan.authored_source_id for plan in execution.plans} == {"outer", "shared"}
    assert all(
        plan.disposition is FanPlanDisposition.LEGACY
        and plan.legacy_reason == "overlapping-fan-ownership"
        for plan in execution.plans
    )
    assert execution.query.planned_for_fork("outer") is None
    assert execution.query.planned_for_fork("shared") is None


def test_install_publishes_matching_immutable_query() -> None:
    facts = [_fact("fork", "a", "one", 0), _fact("fork", "b", "two", 1)]
    graph = _graph()
    execution = build_fan_plan_execution(
        graph,
        _Topology.direct(facts),
        x_spacing=30.0,
        y_spacing=10.0,
        minimum_runway=20.0,
    )

    install_fan_plan_execution(graph, execution)

    assert graph.fan_plans is execution.plans
    assert graph.fan_plan_query is execution.query


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("dogleg_twoline_fanout.mmd", ("to_new", "to_src")),
        ("fanout_intersection_shared_channel.mmd", ("l1", "l2")),
        ("seed72_cross_family_fan.mmd", ("through", "normal", "exempt")),
    ],
)
def test_cross_family_opening_order_is_planned_before_canvas_placement(
    fixture: str, expected: tuple[str, ...]
) -> None:
    path = ROOT / "examples" / "topologies" / fixture
    graph = parse_metro_mermaid(path.read_text())
    topology = build_route_topology_query(graph)
    assert topology is not None

    execution = build_fan_plan_execution(
        graph,
        topology,
        x_spacing=60.0,
        y_spacing=40.0,
        minimum_runway=20.0,
    )
    plan = next(
        item for item in execution.plans if item.fork_station_id in graph.junction_ids
    )

    assert (
        tuple(
            branch.line_ids[0]
            for branch in sorted(plan.branches, key=lambda item: item.opening_rank)
        )
        == expected
    )


def test_runtime_guard_rejects_planned_branch_coordinate_drift() -> None:
    path = ROOT / "examples" / "topologies" / "port_fed_three_branch_diamond.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(item for item in graph.fan_plans if item.layout_station_ids)
    station_id = plan.branches[0].lane_station_ids[0]

    graph.stations[station_id].y += 5.0

    with pytest.raises(PhaseInvariantError, match="expected .* from its frame"):
        _guard_planned_fan_frame_realised(graph, "test", offsets=offsets)


def test_planned_geometry_requires_every_frozen_centreline() -> None:
    path = ROOT / "examples" / "topologies" / "wide_label_fan.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)

    with pytest.raises(
        PhaseInvariantError,
        match="has no frozen placement centreline",
    ):
        _apply_planned_fan_geometry(graph, {})


def test_planned_straight_diamond_is_invalid_at_construction() -> None:
    path = ROOT / "examples" / "topologies" / "port_fed_three_branch_diamond.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.join_station_id is not None)

    with pytest.raises(
        ValueError,
        match="straight-diamond geometry requires established layout",
    ):
        replace(plan, appearance_policy=FanAppearancePolicy.STRAIGHT)


def test_runtime_guard_rejects_corrupted_straight_diamond_policy() -> None:
    path = ROOT / "examples" / "topologies" / "port_fed_three_branch_diamond.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(item for item in graph.fan_plans if item.join_station_id is not None)

    object.__setattr__(
        plan,
        "appearance_policy",
        FanAppearancePolicy.STRAIGHT,
    )

    with pytest.raises(
        PhaseInvariantError,
        match="claims geometry for frozen appearance policy 'straight'",
    ):
        _guard_planned_fan_frame_realised(graph, "test", offsets=offsets)


def test_fan_appearance_policy_rejects_string_equivalents() -> None:
    path = ROOT / "examples" / "topologies" / "port_fed_three_branch_diamond.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.join_station_id is not None)

    with pytest.raises(ValueError, match="appearance policy is not canonical"):
        replace(plan, appearance_policy="symmetric")

    object.__setattr__(plan, "appearance_policy", "symmetric")
    with pytest.raises(PhaseInvariantError, match="non-canonical appearance policy"):
        _guard_planned_fan_frame_realised(
            graph,
            "test",
            offsets=compute_station_offsets(graph),
        )


def test_planned_reconvergence_requires_a_resolved_join() -> None:
    path = ROOT / "examples" / "topologies" / "port_fed_three_branch_diamond.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.join_station_id is not None)

    with pytest.raises(ValueError, match="reconvergence has no resolved join"):
        replace(plan, join_station_id=None)

    object.__setattr__(plan, "join_station_id", None)
    with pytest.raises(PhaseInvariantError, match="has no resolved join"):
        _guard_planned_fan_frame_realised(
            graph,
            "test",
            offsets=compute_station_offsets(graph),
        )


def test_runtime_guard_rejects_planned_handoff_offset_drift() -> None:
    path = ROOT / "examples" / "topologies" / "dogleg_twoline_fanout.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(
        item for item in graph.fan_plans if item.fork_station_id in graph.junction_ids
    )
    branch = min(plan.branches, key=lambda item: item.opening_rank)
    line_id = branch.line_ids[0]

    offsets[(plan.fork_station_id, line_id)] = 8.0

    with pytest.raises(PhaseInvariantError, match="expected .* from its plan"):
        _guard_planned_fan_frame_realised(graph, "test", offsets=offsets)


def test_runtime_guard_rejects_missing_planned_carrier_offset() -> None:
    path = ROOT / "examples" / "topologies" / "dogleg_twoline_fanout.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(item for item in graph.fan_plans if item.offset_carriers)
    carrier = plan.offset_carriers[0]
    del offsets[(carrier.station_id, carrier.line_ids[0])]

    with pytest.raises(PhaseInvariantError, match="has no offset"):
        _guard_planned_fan_frame_realised(graph, "test", offsets=offsets)


def test_offset_carrier_rejects_repeated_exact_slot() -> None:
    with pytest.raises(ValueError, match="repeats a slot"):
        FanOffsetCarrier(
            station_id="hub",
            assignments=(
                FanOffsetAssignment("alpha", 0),
                FanOffsetAssignment("beta", 0),
            ),
        )


def test_planned_fan_rejects_assignment_outside_its_offset_frame() -> None:
    path = ROOT / "examples" / "topologies" / "junction_entry_collision.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "pre2")
    carrier = plan.offset_carriers[0]
    bad_carrier = replace(
        carrier,
        assignments=(
            replace(
                carrier.assignments[0],
                slot=len(plan.offset_line_order),
            ),
            *carrier.assignments[1:],
        ),
    )

    with pytest.raises(ValueError, match="slot lies outside its offset frame"):
        replace(plan, offset_carriers=(bad_carrier, *plan.offset_carriers[1:]))


def test_runtime_guard_rejects_legacy_offset_carriers() -> None:
    path = ROOT / "examples" / "topologies" / "tb_passthrough_continuation.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "starN")
    line_id = plan.branches[0].line_ids[0]
    object.__setattr__(
        plan,
        "offset_carriers",
        (
            FanOffsetCarrier(
                station_id=plan.fork_station_id,
                assignments=(FanOffsetAssignment(line_id, 0),),
            ),
        ),
    )

    with pytest.raises(PhaseInvariantError, match="legacy fan .* owns offset carriers"):
        _guard_planned_fan_frame_realised(graph, "test", offsets={})


def test_runtime_guard_rejects_unowned_carrier_line() -> None:
    path = ROOT / "examples" / "topologies" / "exit_turn_frame_filters.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(item for item in graph.fan_plans if item.offset_carriers)
    carrier = plan.offset_carriers[0]
    graph.add_edge(Edge(carrier.station_id, carrier.station_id, "seam_blocker"))

    with pytest.raises(PhaseInvariantError, match="carries unowned lines"):
        _guard_planned_fan_frame_realised(graph, "test", offsets=offsets)


def test_port_only_fan_freezes_only_structurally_shared_offset_carriers() -> None:
    path = ROOT / "examples" / "topologies" / "disjoint_sameline_trunks.mmd"
    graph = parse_metro_mermaid(path.read_text())
    graph.diamond_style = "symmetric"
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(
        item for item in graph.fan_plans if item.fork_station_id in graph.junction_ids
    )

    for carrier in plan.offset_carriers:
        for line_id in carrier.line_ids:
            assert (
                offsets[(carrier.station_id, line_id)]
                == offsets[(plan.fork_station_id, line_id)]
            )

    assert offsets[("secB__entry_left_4", "b")] == 0.0


def test_solo_trunk_branch_offsets_are_frozen_in_the_plan() -> None:
    path = ROOT / "examples" / "topologies" / "junction_entry_align.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "pre2")
    assignments = {
        carrier.station_id: {
            assignment.line_id: assignment.slot for assignment in carrier.assignments
        }
        for carrier in plan.offset_carriers
    }

    assert {
        station_id: assignments[station_id]
        for station_id in ("s_a", "da1", "da2", "dst_a__entry_left_6")
    } == {
        "s_a": {"alpha": 0},
        "da1": {"alpha": 0},
        "da2": {"alpha": 0},
        "dst_a__entry_left_6": {"alpha": 0},
    }


def test_runtime_applies_only_frozen_fan_offset_assignments() -> None:
    path = ROOT / "examples" / "topologies" / "junction_entry_align.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    sentinel = 999.0
    initial = {
        (station_id, line_id): sentinel
        for station_id in graph.stations
        for line_id in graph.station_lines(station_id)
    }
    expected = {
        (carrier.station_id, assignment.line_id): assignment.slot * 4.0
        for plan in graph.fan_plans
        if plan.owns_geometry
        for carrier in plan.offset_carriers
        for assignment in carrier.assignments
    }

    before_mutation = SimpleNamespace(
        graph=graph,
        offsets=initial.copy(),
        offset_step=4.0,
    )
    _apply_planned_fan_offsets(before_mutation)
    assert {
        key: value
        for key, value in before_mutation.offsets.items()
        if value != sentinel
    } == expected

    graph.junction_ids.add("da1")
    after_mutation = SimpleNamespace(
        graph=graph,
        offsets=initial.copy(),
        offset_step=4.0,
    )
    _apply_planned_fan_offsets(after_mutation)
    assert after_mutation.offsets == before_mutation.offsets


def test_partial_offset_carrier_retains_absolute_slots() -> None:
    path = ROOT / "examples" / "topologies" / "junction_entry_collision.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "pre2")
    carrier = next(item for item in plan.offset_carriers if item.station_id == "dbg1")

    assert {
        assignment.line_id: assignment.slot for assignment in carrier.assignments
    } == {"beta": 1, "gamma": 2}
    offsets = compute_station_offsets(graph)
    assert offsets[("dbg1", "beta")] == 4.0
    assert offsets[("dbg1", "gamma")] == 8.0


def test_planned_fan_preserves_branch_local_reversal() -> None:
    path = ROOT / "examples" / "topologies" / "near_vertical_junction_hook.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(
        item for item in graph.fan_plans if item.fork_station_id in graph.junction_ids
    )

    assert "p1" not in {carrier.station_id for carrier in plan.offset_carriers}
    assert offsets[("p1", "a")] == 4.0
    assert offsets[("p1", "b")] == 0.0


def test_planned_fan_preserves_target_local_frame_with_blocker() -> None:
    path = ROOT / "examples" / "topologies" / "exit_turn_frame_filters.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(
        item for item in graph.fan_plans if item.fork_station_id in graph.junction_ids
    )

    carrier_ids = {carrier.station_id for carrier in plan.offset_carriers}
    assert carrier_ids.isdisjoint({"seam_in", "seam_out"})
    assert {
        line_id: offsets[("seam_in", line_id)]
        for line_id in ("seam_blocker", "seam_a", "seam_b")
    } == {"seam_blocker": 0.0, "seam_a": 4.0, "seam_b": 8.0}


def test_stacked_right_landing_route_emission_ownership_is_exact() -> None:
    path = ROOT / "examples" / "topologies" / "bottom_exit_stacked_right_entry_fan.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.route_emissions)
    query = graph.fan_plan_query
    assert query is not None

    for emission in plan.route_emissions:
        binding = query.route_emission_for_resolved_edge(emission.edge)
        assert binding is not None
        owner, branch, indexed = binding
        assert owner is plan
        assert branch.id == emission.branch_id
        assert indexed is emission

    downstream = next(
        edge
        for edge in plan.resolved_member_edges
        if edge not in {item.edge for item in plan.route_emissions}
    )
    assert query.structural_owner_for_resolved_edge(downstream) is plan
    assert query.route_emission_for_resolved_edge(downstream) is None

    assignments = {
        carrier.station_id: {
            assignment.line_id: assignment.slot for assignment in carrier.assignments
        }
        for carrier in plan.offset_carriers
    }
    assert assignments == {
        "split": {"upper": 1, "lower": 0},
        "source__exit_bottom_0": {"upper": 1, "lower": 0},
        "__junction_3": {"upper": 1, "lower": 0},
        "prepare": {"upper": 1, "lower": 0},
        "lower_in": {"lower": 0},
        "lower_done": {"lower": 0},
        "lower_target__entry_right_2": {"lower": 0},
    }

    routes = route_edges(graph, station_offsets=compute_station_offsets(graph))
    tagged = {
        ResolvedEdge(route.edge.source, route.edge.target, route.line_id): route
        for route in routes
        if route.fan_plan_id is not None
    }
    assert set(tagged) == {item.edge for item in plan.route_emissions}
    assert all(route.fan_plan_id == plan.id for route in tagged.values())
    assert all(
        route.fan_route_emitter == "bottom-exit-right-landings"
        for route in tagged.values()
    )

    route = next(iter(tagged.values()))
    route.fan_route_emitter = None
    with pytest.raises(RuntimeError, match="route tag drifted"):
        validate_fan_route_emissions(graph, routes)


def test_stacked_right_multiline_landing_freezes_reflected_screen_order() -> None:
    path = (
        ROOT
        / "examples"
        / "topologies"
        / "bottom_exit_stacked_right_entry_multiline_branch.mmd"
    )
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.route_emissions)

    assert (
        {item.edge.line_id for item in plan.route_emissions}
        == set(plan.offset_line_order)
        == {"upper_a", "upper_b", "lower"}
    )
    assert {
        branch.id: {
            item.edge.line_id
            for item in plan.route_emissions
            if item.branch_id == branch.id
        }
        for branch in plan.branches
    } == {branch.id: set(branch.line_ids) for branch in plan.branches}

    assignments = {
        carrier.station_id: {
            assignment.line_id: assignment.slot for assignment in carrier.assignments
        }
        for carrier in plan.offset_carriers
    }
    expected = {"upper_a": 1, "upper_b": 2, "lower": 0}
    assert assignments == {
        "split": expected,
        "source__exit_bottom_0": expected,
        "__junction_3": expected,
        "prepare": expected,
        "lower_in": {"lower": 0},
        "lower_done": {"lower": 0},
        "lower_target__entry_right_2": {"lower": 0},
    }


def test_centreline_port_membership_is_frozen_before_materialisation() -> None:
    path = (
        ROOT / "examples" / "topologies" / "ported_symmetric_fan_centreline_trunk.mmd"
    )
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "hub")

    assert plan.centreline_port_ids == (
        "fetch__entry_left_2",
        "report__entry_left_3",
        "input__exit_right_0",
        "fetch__exit_right_1",
    )
    port_id = plan.centreline_port_ids[0]
    graph.ports[port_id].side = PortSide.TOP
    graph.stations[port_id].y += 5.0

    with pytest.raises(PhaseInvariantError, match="expected .* from its frame"):
        _guard_planned_fan_frame_realised(graph, "test", offsets=offsets)


def test_absolute_centreline_anchor_is_frozen_before_materialisation() -> None:
    path = ROOT / "examples" / "topologies" / "bypass_v_tight.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "m1")

    assert plan.centreline_anchor == FanCentrelineAnchor("src__exit_right_0")
    assert plan.local_frame_anchor == FanCentrelineAnchor("m1")
    anchor_y = 137.0
    graph.stations[plan.centreline_anchor.station_id].y = anchor_y
    graph.stations["mid__entry_left_2"].y = 263.0

    graph.ports.pop(plan.centreline_anchor.station_id)
    graph.edges.clear()
    graph.sections["src"].grid_row += 3
    graph.sections["mid"].grid_col += 2

    _apply_planned_fan_port_geometry(graph)
    centrelines = _snapshot_planned_fan_centrelines(graph)

    assert centrelines[plan.id] == anchor_y
    assert {
        graph.stations[port_id].y
        for port_id in plan.centreline_port_ids
        if port_id in graph.ports
    } == {anchor_y}


def test_centreline_anchor_is_complete_and_inside_fan_membership() -> None:
    path = ROOT / "examples" / "topologies" / "bypass_v_tight.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "m1")

    with pytest.raises(ValueError, match="centreline anchor is incomplete"):
        replace(plan, centreline_anchor=None)
    with pytest.raises(ValueError, match="outside complete membership"):
        replace(plan, centreline_anchor=FanCentrelineAnchor("unrelated"))

    assert plan.centreline_anchor is not None
    graph.stations.pop(plan.centreline_anchor.station_id)
    with pytest.raises(PhaseInvariantError, match="centreline anchor .* is missing"):
        _apply_planned_fan_port_geometry(graph)


def test_symmetric_style_keeps_planned_two_way_fan_on_shared_centreline() -> None:
    path = ROOT / "examples" / "topologies" / "symmetric_deadend_fanout_exit.mmd"
    graph = parse_metro_mermaid(path.read_text())
    topology = build_route_topology_query(graph)
    assert topology is not None
    execution = build_fan_plan_execution(
        graph,
        topology,
        x_spacing=X_SPACING,
        y_spacing=compute_min_y_spacing(graph),
        minimum_runway=INTER_ROW_EDGE_CLEARANCE,
    )
    plan = next(item for item in execution.plans if item.authored_source_id == "entry")

    lane_offsets = tuple(branch.lane_offset for branch in plan.branches)
    assert lane_offsets == pytest.approx(
        (-plan.frame.secondary.step / 2, plan.frame.secondary.step / 2)
    )

    laid_out = parse_metro_mermaid(path.read_text())
    compute_layout(laid_out)
    fork = laid_out.stations["s1__entry_left_2"]
    branch_ys = [laid_out.stations[station_id].y for station_id in ("split", "salmon")]
    assert fork.y == pytest.approx(sum(branch_ys) / 2)


def test_runtime_guard_rejects_asymmetric_symmetric_fan_plan() -> None:
    path = ROOT / "examples" / "topologies" / "symmetric_deadend_fanout.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    plan = next(item for item in graph.fan_plans if item.authored_source_id == "entry")
    bad_branches = (
        replace(plan.branches[0], lane_offset=0.0),
        plan.branches[1],
    )
    with pytest.raises(
        ValueError,
        match="fan lane offsets disagree with appearance pitch",
    ):
        replace(plan, branches=bad_branches)
    object.__setattr__(plan, "branches", bad_branches)

    with pytest.raises(PhaseInvariantError, match="uses asymmetric lane offsets"):
        _guard_planned_fan_frame_realised(
            graph,
            "test",
            offsets=compute_station_offsets(graph),
        )


def test_planned_fan_does_not_level_unrelated_row_bbox_tops() -> None:
    path = (
        ROOT / "examples" / "topologies" / "ported_symmetric_fan_centreline_trunk.mmd"
    )
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)

    assert graph.sections["input"].bbox_y == pytest.approx(
        graph.stations["identify"].y - SECTION_Y_PADDING
    )
    assert graph.sections["report"].bbox_y == pytest.approx(
        graph.stations["generate"].y - SECTION_Y_PADDING
    )
    assert graph.sections["fetch"].bbox_y < graph.sections["input"].bbox_y


def test_planned_handoff_does_not_reslot_unrelated_same_line_stations() -> None:
    path = ROOT / "examples" / "topologies" / "compact_gap_peer_conflict.mmd"
    text = path.read_text().replace(
        "        s2[Prepare]\n",
        "        s2[Prepare]\n"
        "        peer1[Peer input]\n"
        "        peer2[Peer output]\n"
        "        peer1 -->|beta| peer2\n",
    )
    graph = parse_metro_mermaid(text)
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    plan = next(item for item in graph.fan_plans if item.fork_station_id == "p1")

    assert {carrier.station_id for carrier in plan.offset_carriers}.isdisjoint(
        {"peer1", "peer2"}
    )
    assert offsets[("peer1", "beta")] == offsets[("peer2", "beta")] == 4.0


def test_planned_fan_resources_resolve_through_final_route_plan() -> None:
    path = ROOT / "examples" / "topologies" / "port_fed_three_branch_diamond.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observation = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )
    query = build_route_plan_query(observation.plan)
    planned = tuple(
        item
        for item in graph.fan_plans
        if item.disposition is FanPlanDisposition.PLANNED
    )

    assert planned
    assert tuple(
        item.id
        for item in observation.plan.shared_references
        if item.kind is SharedReferenceKind.CENTRELINE
        and item.coordinate_regime is CoordinateRegime.RELATIVE_FRAME
    ) == tuple(item.centreline_reference_id for item in planned)
    assert tuple(
        item.id
        for item in observation.plan.demands
        if item.minimum_size_regime is CoordinateRegime.RELATIVE_FRAME
    ) == tuple(demand_id for item in planned for demand_id in item.demand_ids)
    for fan_plan in planned:
        assert fan_plan.centreline_reference_id is not None
        reference = query.shared_reference(fan_plan.centreline_reference_id)
        demands = tuple(query.demand(item) for item in fan_plan.demand_ids)
        system = next(
            item for item in observation.plan.systems if item.id == reference.system_id
        )

        assert reference.kind is SharedReferenceKind.CENTRELINE
        assert reference.coordinate_regime is CoordinateRegime.RELATIVE_FRAME
        assert reference.id in system.shared_reference_ids
        assert all(
            branch.continuation_edge_ids[0] in system.connector_ids
            for branch in fan_plan.branches
        )
        assert tuple(item.id for item in demands) == fan_plan.demand_ids
        assert all(item.system_id == system.id for item in demands)
        assert all(item.id in system.demand_ids for item in demands)
        assert all(item.kind is DemandKind.RUNWAY for item in demands)
        assert all(
            item.minimum_size_regime is CoordinateRegime.RELATIVE_FRAME
            for item in demands
        )
        assert all(item.ordered_reference_ids == (reference.id,) for item in demands)
        assert all(
            item.keep_out_classes == (KeepOutClass.SECTION, KeepOutClass.MARKER)
            for item in demands
        )


def test_legacy_fans_publish_no_relative_route_plan_resources() -> None:
    path = ROOT / "examples" / "centered_tracks.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observation = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )

    assert graph.fan_plans
    assert all(
        item.disposition is FanPlanDisposition.LEGACY for item in graph.fan_plans
    )
    assert not any(
        item.kind is SharedReferenceKind.CENTRELINE
        and item.coordinate_regime is CoordinateRegime.RELATIVE_FRAME
        for item in observation.plan.shared_references
    )
    assert not any(
        item.minimum_size_regime is CoordinateRegime.RELATIVE_FRAME
        for item in observation.plan.demands
    )


def test_legacy_fan_disposition_is_visible_in_route_plan_diagnostics() -> None:
    path = ROOT / "examples" / "topologies" / "tb_passthrough_continuation.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observation = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )

    diagnostics = tuple(
        item for item in observation.plan.diagnostics if item.code == "fan-plan-legacy"
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].blocking is False
    assert "local-layout-has-foreign-owner" in diagnostics[0].message
