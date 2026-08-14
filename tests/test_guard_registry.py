"""Structural invariants of the guard + routing-check registries.

The golden-baseline oracle (``test_guard_registry_golden``) pins the *runtime*
guard call sequence.  These tests pin the *classification* registries
themselves: every guard and check is registered, the tiers are well-formed,
and the Tier-A routing-check set provably equals the always-on render
chokepoint, so the tier table in ``docs/dev/guard_tiers.md`` cannot drift from
the code it documents.
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest

from nf_metro.layout.phases import guards
from nf_metro.layout.phases.guards import GUARD_REGISTRY, PhaseInvariantError
from nf_metro.layout.routing import invariants
from nf_metro.layout.routing.common import RoutedPath
from nf_metro.layout.routing.invariants import (
    CHECK_REGISTRY,
    assert_render_curve_invariants,
    check_no_dogleg_crosses_exempt_trunk,
    check_right_entry_drop_in_when_clear,
)
from nf_metro.parser.model import Edge, MetroGraph, Port, PortSide, Section, Station

VALID_TIERS = {"A", "B", "C"}


def test_final_runner_runs_deferred_tier_b_route_guard_on_settled_geometry(
    monkeypatch,
) -> None:
    called: list[str] = []

    def structural_guard(graph, phase):
        called.append("structural")

    def route_guard(graph, phase, *, routes):
        called.append("route")

    monkeypatch.setattr(
        guards,
        "GUARD_REGISTRY",
        (
            guards.GuardSpec(structural_guard, "A"),
            guards.GuardSpec(route_guard, "B", needs=frozenset({"routes"})),
        ),
    )
    monkeypatch.setattr(guards, "_RENDER_LAYOUT_INVARIANT_SPECS", ())
    monkeypatch.setattr(
        "nf_metro.layout.routing.observe_route_edges",
        lambda *args, **kwargs: SimpleNamespace(
            routes=[object()],
            plan=SimpleNamespace(boundary_clearance_requirements=(object(),)),
        ),
    )

    guards.run_validate_guards(object(), "after final", include_final=True, offsets={})

    assert called == ["structural"]
    guards.assert_render_layout_invariants(
        object(), [object()], {}, strict=True, include_deferred_final=True
    )
    assert called == ["structural", "route"]


def test_final_runner_rejects_bad_routes_without_pending_clearance(monkeypatch) -> None:
    def route_guard(graph, phase, *, routes):
        raise PhaseInvariantError("bad final geometry")

    monkeypatch.setattr(
        guards,
        "GUARD_REGISTRY",
        (guards.GuardSpec(route_guard, "A", needs=frozenset({"routes"})),),
    )
    monkeypatch.setattr(
        "nf_metro.layout.routing.observe_route_edges",
        lambda *args, **kwargs: SimpleNamespace(
            routes=[object()],
            plan=SimpleNamespace(boundary_clearance_requirements=()),
        ),
    )

    with pytest.raises(PhaseInvariantError, match="bad final geometry"):
        guards.run_validate_guards(
            object(), "after final", include_final=True, offsets={}
        )


def test_deferred_tier_b_guard_rejects_bad_settled_routes(monkeypatch) -> None:
    def route_guard(graph, phase, *, routes):
        raise PhaseInvariantError("bad settled geometry")

    monkeypatch.setattr(
        guards,
        "GUARD_REGISTRY",
        (guards.GuardSpec(route_guard, "B", needs=frozenset({"routes"})),),
    )
    monkeypatch.setattr(guards, "_RENDER_LAYOUT_INVARIANT_SPECS", ())

    with pytest.raises(guards.LayoutInvariantError, match="bad settled geometry"):
        guards.assert_render_layout_invariants(
            object(), [object()], {}, strict=True, include_deferred_final=True
        )


def test_plan_ownership_does_not_hide_a_dogleg_crossing() -> None:
    exempt = RoutedPath(
        Edge("exempt-source", "exempt-target", "fixed"),
        "fixed",
        [
            (400.0, 298.0),
            (416.0, 298.0),
            (416.0, 196.0),
            (14.0, 196.0),
            (14.0, 120.0),
            (30.0, 120.0),
        ],
        is_inter_section=True,
        normalize_exempt=True,
    )
    movable = RoutedPath(
        Edge("source", "target", "moving"),
        "moving",
        [
            (190.0, 120.0),
            (209.0, 120.0),
            (209.0, 199.0),
            (419.0, 199.0),
            (419.0, 298.0),
            (450.0, 298.0),
        ],
        is_inter_section=True,
    )

    assert check_no_dogleg_crosses_exempt_trunk(None, [exempt, movable], {})
    movable.convergence_plan_id = "convergence"
    movable.convergence_owned_segment_ranks = (2,)
    assert check_no_dogleg_crosses_exempt_trunk(None, [exempt, movable], {})


def test_owned_shared_opening_is_not_a_needless_right_entry_dive(
    monkeypatch,
) -> None:
    graph = MetroGraph()
    graph.sections = {
        "source": Section("source", "Source", grid_row=0),
        "target": Section("target", "Target", grid_row=1, bbox_y=100.0, bbox_h=40.0),
    }
    graph.stations = {
        "source": Station("source", "Source", section_id="source"),
        "entry": Station("entry", "", section_id="target", is_port=True),
    }
    graph.ports = {"entry": Port("entry", "target", PortSide.RIGHT, is_entry=True)}
    route = RoutedPath(
        Edge("source", "entry", "line"),
        "line",
        [(120.0, 20.0), (140.0, 20.0), (140.0, 180.0), (100.0, 180.0)],
        is_inter_section=True,
    )
    monkeypatch.setattr(
        "nf_metro.layout.routing.inter_section_handlers._right_entry_drop_in_is_clear",
        lambda *_args: True,
    )

    assert check_right_entry_drop_in_when_clear(graph, [route])
    route.exit_shared_opening_points = tuple(route.points[:3])
    route.route_system_owned_segment_ranks = (0, 1)
    assert not check_right_entry_drop_in_when_clear(graph, [route])


def test_one_convergence_may_own_opposing_legs_on_its_shared_carrier() -> None:
    first = RoutedPath(
        Edge("first-source", "merge", "line"),
        "line",
        [(0.0, 0.0), (100.0, 0.0)],
    )
    second = RoutedPath(
        Edge("second-source", "merge", "line"),
        "line",
        [(80.0, 0.0), (20.0, 0.0)],
    )

    assert list(
        guards.iter_opposing_line_overlaps(None, offsets={}, routes=[first, second])
    )
    first.convergence_plan_id = second.convergence_plan_id = "convergence"
    first.convergence_owned_segment_ranks = (0,)
    second.convergence_owned_segment_ranks = (0,)
    assert not list(
        guards.iter_opposing_line_overlaps(None, offsets={}, routes=[first, second])
    )


def _defined(module, prefix: str) -> set[str]:
    return {
        name
        for name, obj in vars(module).items()
        if name.startswith(prefix) and inspect.isfunction(obj)
    }


def test_guard_registry_tiers_are_well_formed() -> None:
    assert all(spec.tier in VALID_TIERS for spec in GUARD_REGISTRY)
    names = [spec.name for spec in GUARD_REGISTRY]
    assert len(names) == len(set(names)), "duplicate guard in registry"


def test_registry_bisection_set_is_the_pass_c_prefix() -> None:
    """The bisection-safe specs are a contiguous prefix of the registry: the
    runner relies on this so a Pass C checkpoint and the final block share one
    ordered list."""
    flags = [spec.bisection_safe for spec in GUARD_REGISTRY]
    first_final = flags.index(False)
    assert all(flags[:first_final]), "bisection-safe specs must come first"
    assert not any(flags[first_final:]), "bisection-safe specs must be contiguous"


def test_derived_bisection_first_valid_matches_registry() -> None:
    """``_BISECTION_FIRST_VALID`` is derived from the registry, so the two must
    agree and only bisection-safe specs may carry a threshold."""
    expected = {
        spec.name: spec.first_valid_stage
        for spec in GUARD_REGISTRY
        if spec.bisection_safe and spec.first_valid_stage is not None
    }
    assert guards._BISECTION_FIRST_VALID == expected
    for spec in GUARD_REGISTRY:
        if spec.first_valid_stage is not None:
            assert spec.bisection_safe
            assert spec.first_valid_stage in guards._PASS_C_BISECTION_ORDER


def test_check_registry_classifies_every_check() -> None:
    """Every ``check_*`` invariant must be classified, so a new check cannot
    escape the tier table."""
    registered = {spec.name for spec in CHECK_REGISTRY}
    defined = _defined(invariants, "check_")
    assert registered == defined, (
        f"unclassified checks: {sorted(defined - registered)}; "
        f"stale registry entries: {sorted(registered - defined)}"
    )
    assert all(spec.tier in VALID_TIERS for spec in CHECK_REGISTRY)


def test_tier_a_checks_are_exactly_the_render_chokepoint() -> None:
    """Tier A means 'already always-on'.  For routing checks that is precisely
    the set called by :func:`assert_render_curve_invariants`, so the two must
    match exactly -- a check moved in or out of the chokepoint must move tier."""
    chokepoint = set(
        re.findall(
            r"\bcheck_[a-z_]+", inspect.getsource(assert_render_curve_invariants)
        )
    )
    tier_a = {spec.name for spec in CHECK_REGISTRY if spec.tier == "A"}
    assert tier_a == chokepoint, (
        f"Tier-A checks {sorted(tier_a)} != chokepoint {sorted(chokepoint)}"
    )


def _all_guard_specs() -> list:
    """Every classified guard: the dispatched ``GUARD_REGISTRY`` plus the
    classification-only ``INLINE_GUARD_REGISTRY`` (guards engine.py invokes at
    a specific stage rather than through the Pass C / final runner)."""
    return [*GUARD_REGISTRY, *guards.INLINE_GUARD_REGISTRY]


def test_render_layout_chokepoint_is_tier_a_minus_authoring_guards() -> None:
    """The render-layout chokepoint runs exactly the Tier-A guards that are
    observational postconditions: every Tier-A guard from both registries minus
    the two authoring-error guards, which raise a ``ValueError`` on
    un-renderable input and stay always-on hard fails in the engine."""
    tier_a = {spec.name for spec in _all_guard_specs() if spec.tier == "A"}
    chokepoint = {spec.name for spec in guards.render_layout_invariant_specs()}
    assert chokepoint == tier_a - guards._RENDER_CHOKEPOINT_AUTHORING_GUARDS
    assert guards._RENDER_CHOKEPOINT_AUTHORING_GUARDS <= tier_a


def test_deferred_final_set_is_every_route_dependent_guard() -> None:
    expected = tuple(spec for spec in GUARD_REGISTRY if "routes" in spec.needs)
    assert guards.deferred_final_route_guard_specs() == expected
    assert any(spec.tier == "B" for spec in expected)


def _guards_citing_an_issue() -> dict[str, set[str]]:
    """Map every ``_guard_*`` whose source cites a ``#NNN`` issue to those
    issue tokens, so a guard born of a specific bug cannot silently drop the
    regression trail."""
    out: dict[str, set[str]] = {}
    for name, obj in vars(guards).items():
        if name.startswith("_guard_") and inspect.isfunction(obj):
            issues = set(re.findall(r"#\d{3,}", inspect.getsource(obj)))
            if issues:
                out[name] = issues
    return out


def test_no_registry_guard_duplicates_an_always_on_check() -> None:
    """A ``validate=True`` guard that merely raises around a check already in
    the always-on render chokepoint is pure duplication: the check runs on
    every render regardless of ``validate``.  The check is the single
    authority; the guard wrapper must not re-register it."""
    chokepoint = set(
        re.findall(
            r"\bcheck_[a-z_]+", inspect.getsource(assert_render_curve_invariants)
        )
    )
    offenders = {}
    for spec in GUARD_REGISTRY:
        refs = set(re.findall(r"\bcheck_[a-z_]+", inspect.getsource(spec.fn)))
        dup = refs & chokepoint
        if dup:
            offenders[spec.name] = sorted(dup)
    assert not offenders, (
        "validate-only guards duplicate always-on render-chokepoint checks "
        f"(drop the wrapper; the check already runs on every render): {offenders}"
    )


# Geometric properties checked by both a runtime guard and the offline
# validator oracle, each single-sourced through one shared predicate: the
# guard contributes tier/registry/raise semantics, the oracle contributes
# ``Violation`` packaging, and neither re-implements the geometry.  The tuple
# is ``(guard_name, oracle_name, shared_predicate_name)``.
_SHARED_GEOMETRY_PREDICATES = (
    ("_guard_no_section_overlap", "check_section_overlap", "iter_section_overlaps"),
    (
        "_guard_stations_within_bbox",
        "check_station_containment",
        "iter_stations_outside_bbox",
    ),
    (
        "_guard_no_coincident_station_coords",
        "check_coincident_stations",
        "iter_coincident_stations",
    ),
    (
        "_guard_no_route_through_section",
        "check_edge_section_crossing",
        "routes_through_unrelated_sections",
    ),
    ("_guard_no_label_overlap", "check_label_overlap", "_residual_label_overlaps"),
    (
        "_guard_serpentine_no_backtrack",
        "check_serpentine_no_backtrack",
        "iter_serpentine_backtracks",
    ),
)


def test_guard_and_oracle_share_one_geometry_predicate() -> None:
    """Each duplicated geometric property is single-sourced across the
    guards/oracle file boundary: both the runtime guard and the offline
    validator check must reference the same shared predicate, so a geometry
    rule (tolerance, exemption scope) lives in exactly one place and the two
    cannot silently drift."""
    import layout_validator

    missing: dict[str, list[str]] = {}
    for guard_name, oracle_name, predicate in _SHARED_GEOMETRY_PREDICATES:
        guard_src = inspect.getsource(getattr(guards, guard_name))
        oracle_src = inspect.getsource(getattr(layout_validator, oracle_name))
        absent = [
            f"{side}:{name}"
            for side, name, src in (
                ("guard", guard_name, guard_src),
                ("oracle", oracle_name, oracle_src),
            )
            if predicate not in src
        ]
        if absent:
            missing[predicate] = absent
    assert not missing, (
        "guard/oracle pairs not routed through their shared predicate "
        f"(each side must call it, not re-implement the geometry): {missing}"
    )


def test_every_guard_is_classified_in_exactly_one_registry() -> None:
    """Every defined ``_guard_*`` lives in exactly one of the two guard
    registries, so no guard escapes tier / issue-pin classification."""
    defined = _defined(guards, "_guard_")
    names = [spec.name for spec in _all_guard_specs()]
    duplicated = {n for n in names if names.count(n) > 1}
    assert not duplicated, f"guards in more than one registry: {sorted(duplicated)}"
    unclassified = defined - set(names)
    assert not unclassified, f"unclassified guards: {sorted(unclassified)}"
    stale = set(names) - defined
    assert not stale, f"registry names with no guard: {sorted(stale)}"


def test_issue_pinned_guards_record_their_issue_as_data() -> None:
    """Every guard whose source cites an issue carries that issue in its
    spec's ``issue_pin``, so consolidation cannot lose the regression trail."""
    by_name = {spec.name: spec for spec in _all_guard_specs()}
    missing = {}
    for name, issues in _guards_citing_an_issue().items():
        spec = by_name.get(name)
        pinned = set(spec.issue_pin) if spec else set()
        absent = issues - pinned
        if absent:
            missing[name] = sorted(absent)
    assert not missing, f"guards citing an issue but not pinning it as data: {missing}"


def test_issue_pinned_guards_document_why_they_are_narrow() -> None:
    """A guard kept pinned to a past issue must populate ``narrow_reason``
    saying why it stays narrow rather than expressing a general property, so
    the field is an enforced contract rather than optional documentation."""
    undocumented = [
        spec.name
        for spec in _all_guard_specs()
        if spec.issue_pin and not spec.narrow_reason
    ]
    assert not undocumented, (
        f"issue-pinned guards with no narrow_reason: {sorted(undocumented)}"
    )
