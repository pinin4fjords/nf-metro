"""Pure, deterministic allocation of rigid corridor-lane cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import isfinite


class CorridorAllocationStatus(Enum):
    PLANNED = "planned"
    COMPATIBILITY = "compatibility"
    FAILURE = "failure"


class CorridorAllocationFailureReason(Enum):
    CONTRADICTION = "contradiction"
    INFEASIBLE = "infeasible"
    INVALID = "invalid"


@dataclass(frozen=True)
class CorridorLane:
    member_id: str
    cohort_id: str
    endpoint_owner_id: str
    boundary_coordinate: float
    planned_coordinate: float
    span_start: float
    span_end: float
    semantic_rank: tuple[int, ...]


@dataclass(frozen=True)
class CorridorObstacle:
    obstacle_id: str
    order_coordinate: float
    realised_coordinate: float
    span_start: float
    span_end: float
    semantic_rank: tuple[int, ...]


@dataclass(frozen=True)
class CorridorEquality:
    owner_id: str
    left_member_id: str
    right_member_id: str
    delta: float


@dataclass(frozen=True)
class CorridorAllocationProblem:
    """A complete preclosed endpoint, equality, and clearance component."""

    lanes: tuple[CorridorLane, ...]
    obstacles: tuple[CorridorObstacle, ...] = ()
    equalities: tuple[CorridorEquality, ...] = ()
    clearance: float = 4.0
    witnesses_complete: bool = True
    axis_sign: int = 1


@dataclass(frozen=True)
class CorridorAllocationResult:
    status: CorridorAllocationStatus
    allocations: tuple[tuple[str, float], ...] = ()
    reason: CorridorAllocationFailureReason | None = None
    blocking_member_ids: tuple[str, ...] = ()
    blocking_obstacle_ids: tuple[str, ...] = ()
    blocking_equality_owner_ids: tuple[str, ...] = ()
    blocking_endpoint_owner_ids: tuple[str, ...] = ()


def _q(value: float) -> Fraction:
    return Fraction(str(value))


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _lower_median(values: list[Fraction]) -> Fraction:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def _failure(
    reason: CorridorAllocationFailureReason,
    *,
    members: tuple[str, ...] | list[str] | set[str] = (),
    obstacles: tuple[str, ...] | list[str] | set[str] = (),
    equality_owners: tuple[str, ...] | list[str] | set[str] = (),
    endpoint_owners: tuple[str, ...] | list[str] | set[str] = (),
) -> CorridorAllocationResult:
    return CorridorAllocationResult(
        CorridorAllocationStatus.FAILURE,
        reason=reason,
        blocking_member_ids=tuple(sorted(set(members))),
        blocking_obstacle_ids=tuple(sorted(set(obstacles))),
        blocking_equality_owner_ids=tuple(sorted(set(equality_owners))),
        blocking_endpoint_owner_ids=tuple(sorted(set(endpoint_owners))),
    )


class _WeightedUnionFind:
    """Exact union-find with potentials relative to canonical member roots."""

    def __init__(self, keys: list[tuple[tuple[int, ...], Fraction, Fraction]]) -> None:
        self.parent = list(range(len(keys)))
        self.weight = [Fraction(0)] * len(keys)
        self.keys = keys

    def find(self, item: int) -> tuple[int, Fraction]:
        parent = self.parent[item]
        if parent == item:
            return item, Fraction(0)
        root, parent_weight = self.find(parent)
        self.weight[item] += parent_weight
        self.parent[item] = root
        return root, self.weight[item]

    def union(self, left: int, right: int, delta: Fraction) -> bool:
        """Require coordinate(right) - coordinate(left) == delta."""
        left_root, left_weight = self.find(left)
        right_root, right_weight = self.find(right)
        if left_root == right_root:
            return right_weight - left_weight == delta
        if self.keys[left_root] <= self.keys[right_root]:
            self.parent[right_root] = left_root
            self.weight[right_root] = delta + left_weight - right_weight
        else:
            self.parent[left_root] = right_root
            self.weight[left_root] = right_weight - left_weight - delta
        return True


def _invalid_problem(
    problem: CorridorAllocationProblem,
) -> CorridorAllocationResult | None:
    lanes = problem.lanes
    member_ids = [lane.member_id for lane in lanes]
    numeric = [problem.clearance]
    for lane in lanes:
        numeric.extend(
            (
                lane.boundary_coordinate,
                lane.planned_coordinate,
                lane.span_start,
                lane.span_end,
            )
        )
    for obstacle in problem.obstacles:
        numeric.extend(
            (
                obstacle.order_coordinate,
                obstacle.realised_coordinate,
                obstacle.span_start,
                obstacle.span_end,
            )
        )
    numeric.extend(equality.delta for equality in problem.equalities)
    invalid_members = {
        lane.member_id
        for lane in lanes
        if not lane.member_id
        or not lane.cohort_id
        or not lane.endpoint_owner_id
        or not lane.semantic_rank
    }
    duplicate_members = {
        member_id for member_id in member_ids if member_ids.count(member_id) > 1
    }
    invalid_obstacles = {
        obstacle.obstacle_id
        for obstacle in problem.obstacles
        if not obstacle.obstacle_id or not obstacle.semantic_rank
    }
    cohort_endpoint_owners: dict[str, set[str]] = {}
    for lane in lanes:
        cohort_endpoint_owners.setdefault(lane.cohort_id, set()).add(
            lane.endpoint_owner_id
        )
    split_cohorts = {
        cohort_id
        for cohort_id, endpoint_owners in cohort_endpoint_owners.items()
        if len(endpoint_owners) != 1
    }
    split_members = {
        lane.member_id for lane in lanes if lane.cohort_id in split_cohorts
    }
    split_endpoint_owners = {
        lane.endpoint_owner_id for lane in lanes if lane.cohort_id in split_cohorts
    }
    by_id = set(member_ids)
    invalid_equalities = {
        equality.owner_id
        for equality in problem.equalities
        if not equality.owner_id
        or equality.left_member_id not in by_id
        or equality.right_member_id not in by_id
    }
    invalid_shape = (
        problem.clearance < 0
        or problem.axis_sign not in (-1, 1)
        or not all(isfinite(value) for value in numeric)
        or any(lane.span_start > lane.span_end for lane in lanes)
        or any(
            obstacle.span_start > obstacle.span_end for obstacle in problem.obstacles
        )
    )
    if (
        invalid_members
        or duplicate_members
        or invalid_obstacles
        or invalid_equalities
        or split_cohorts
        or invalid_shape
    ):
        return _failure(
            CorridorAllocationFailureReason.INVALID,
            members=invalid_members | duplicate_members | split_members,
            obstacles=invalid_obstacles,
            equality_owners=invalid_equalities,
            endpoint_owners={
                lane.endpoint_owner_id
                for lane in lanes
                if lane.member_id in invalid_members | duplicate_members
            }
            | split_endpoint_owners,
        )
    return None


def _equality_peer_owners(
    equalities: tuple[CorridorEquality, ...], by_id: dict[str, int]
) -> dict[frozenset[int], set[str]]:
    adjacency: dict[str, dict[int, set[int]]] = {}
    for equality in equalities:
        left = by_id[equality.left_member_id]
        right = by_id[equality.right_member_id]
        owner_graph = adjacency.setdefault(equality.owner_id, {})
        owner_graph.setdefault(left, set()).add(right)
        owner_graph.setdefault(right, set()).add(left)
    peers: dict[frozenset[int], set[str]] = {}
    for owner, graph in adjacency.items():
        unseen = set(graph)
        while unseen:
            pending = [min(unseen)]
            component: set[int] = set()
            while pending:
                member = pending.pop()
                if member in component:
                    continue
                component.add(member)
                pending.extend(graph[member] - component)
            unseen -= component
            for left in component:
                for right in component:
                    if left < right:
                        peers.setdefault(frozenset((left, right)), set()).add(owner)
    return peers


def _contradiction_attribution(
    lanes: tuple[CorridorLane, ...],
    equalities: tuple[CorridorEquality, ...],
    union_find: _WeightedUnionFind,
    equality: CorridorEquality,
) -> CorridorAllocationResult:
    left = next(
        i for i, lane in enumerate(lanes) if lane.member_id == equality.left_member_id
    )
    root, _ = union_find.find(left)
    members = {
        lane.member_id
        for index, lane in enumerate(lanes)
        if union_find.find(index)[0] == root
    }
    owners = {
        item.owner_id
        for item in equalities
        if item.left_member_id in members and item.right_member_id in members
    }
    owners.add(equality.owner_id)
    return _failure(
        CorridorAllocationFailureReason.CONTRADICTION,
        members=members,
        equality_owners=owners,
        endpoint_owners={
            lane.endpoint_owner_id for lane in lanes if lane.member_id in members
        },
    )


def solve_corridor_cohorts(  # noqa: C901, PLR0915
    problem: CorridorAllocationProblem,
) -> CorridorAllocationResult:
    """Return the unique canonical seating for a complete witness set."""
    if not problem.witnesses_complete:
        return CorridorAllocationResult(CorridorAllocationStatus.COMPATIBILITY)
    invalid = _invalid_problem(problem)
    if invalid is not None:
        return invalid
    if not problem.lanes:
        return CorridorAllocationResult(CorridorAllocationStatus.PLANNED)

    sign = problem.axis_sign
    lanes = problem.lanes
    by_id = {lane.member_id: index for index, lane in enumerate(lanes)}
    canonical_member_keys = [
        (
            lane.semantic_rank,
            sign * _q(lane.boundary_coordinate),
            sign * _q(lane.planned_coordinate),
        )
        for lane in lanes
    ]
    union_find = _WeightedUnionFind(canonical_member_keys)

    cohorts: dict[str, list[int]] = {}
    for index, lane in enumerate(lanes):
        cohorts.setdefault(lane.cohort_id, []).append(index)
    for members in cohorts.values():
        anchor = min(members, key=lambda index: canonical_member_keys[index])
        anchor_boundary = sign * _q(lanes[anchor].boundary_coordinate)
        for member in members:
            delta = sign * _q(lanes[member].boundary_coordinate) - anchor_boundary
            if not union_find.union(anchor, member, delta):
                return _failure(
                    CorridorAllocationFailureReason.CONTRADICTION,
                    members={lanes[index].member_id for index in members},
                    endpoint_owners={
                        lanes[index].endpoint_owner_id for index in members
                    },
                )

    processed_equalities: list[CorridorEquality] = []
    for equality in sorted(
        problem.equalities,
        key=lambda item: (
            item.owner_id,
            item.left_member_id,
            item.right_member_id,
            item.delta,
        ),
    ):
        left = by_id[equality.left_member_id]
        right = by_id[equality.right_member_id]
        if not union_find.union(left, right, sign * _q(equality.delta)):
            return _contradiction_attribution(
                lanes,
                tuple((*processed_equalities, equality)),
                union_find,
                equality,
            )
        processed_equalities.append(equality)

    roots: dict[int, list[int]] = {}
    raw_potentials: dict[int, Fraction] = {}
    for index in range(len(lanes)):
        root, potential = union_find.find(index)
        roots.setdefault(root, []).append(index)
        raw_potentials[index] = potential

    # Rebase every component to its canonical semantic member.  This removes
    # union history and input order from preferred bases and fixed-side choices.
    potentials: dict[int, Fraction] = {}
    component_reference: dict[int, int] = {}
    for root, members in roots.items():
        reference = min(members, key=lambda index: canonical_member_keys[index])
        component_reference[root] = reference
        reference_potential = raw_potentials[reference]
        for member in members:
            potentials[member] = raw_potentials[member] - reference_potential

    preferred = {
        root: _lower_median(
            [
                sign * _q(lanes[index].planned_coordinate) - potentials[index]
                for index in members
            ]
        )
        for root, members in roots.items()
    }
    root_key = {
        root: (
            preferred[root],
            lanes[component_reference[root]].semantic_rank,
            tuple(sorted(lanes[index].semantic_rank for index in members)),
        )
        for root, members in roots.items()
    }
    peer_owners = _equality_peer_owners(problem.equalities, by_id)

    lower: dict[int, Fraction | None] = {root: None for root in roots}
    upper: dict[int, Fraction | None] = {root: None for root in roots}
    bound_members: dict[int, set[str]] = {root: set() for root in roots}
    bound_obstacles: dict[int, set[str]] = {root: set() for root in roots}
    edges: dict[tuple[int, int], Fraction] = {}
    edge_members: dict[tuple[int, int], set[str]] = {}

    for root, members in roots.items():
        for obstacle in problem.obstacles:
            overlapping = [
                index
                for index in members
                if _overlaps(
                    lanes[index].span_start,
                    lanes[index].span_end,
                    obstacle.span_start,
                    obstacle.span_end,
                )
            ]
            if not overlapping:
                continue
            obstacle_order_coordinate = sign * _q(obstacle.order_coordinate)
            component_order_key = (
                preferred[root],
                lanes[component_reference[root]].semantic_rank,
            )
            obstacle_order_key = (
                obstacle_order_coordinate,
                obstacle.semantic_rank,
            )
            if component_order_key == obstacle_order_key:
                return _failure(
                    CorridorAllocationFailureReason.INVALID,
                    members={lanes[index].member_id for index in overlapping},
                    obstacles=(obstacle.obstacle_id,),
                    endpoint_owners={
                        lanes[index].endpoint_owner_id for index in overlapping
                    },
                )
            component_before = component_order_key < obstacle_order_key
            obstacle_coordinate = sign * _q(obstacle.realised_coordinate)
            if component_before:
                candidate = min(
                    obstacle_coordinate - _q(problem.clearance) - potentials[index]
                    for index in overlapping
                )
                current_upper = upper[root]
                upper[root] = (
                    candidate
                    if current_upper is None
                    else min(current_upper, candidate)
                )
            else:
                candidate = max(
                    obstacle_coordinate + _q(problem.clearance) - potentials[index]
                    for index in overlapping
                )
                current_lower = lower[root]
                lower[root] = (
                    candidate
                    if current_lower is None
                    else max(current_lower, candidate)
                )
            bound_members[root].update(lanes[index].member_id for index in overlapping)
            bound_obstacles[root].add(obstacle.obstacle_id)

    for left_index, left_lane in enumerate(lanes):
        left_root, _ = union_find.find(left_index)
        for right_index in range(left_index + 1, len(lanes)):
            right_lane = lanes[right_index]
            if not _overlaps(
                left_lane.span_start,
                left_lane.span_end,
                right_lane.span_start,
                right_lane.span_end,
            ):
                continue
            pair = frozenset((left_index, right_index))
            if pair in peer_owners:
                continue
            right_root, _ = union_find.find(right_index)
            if left_root == right_root:
                if abs(potentials[right_index] - potentials[left_index]) < _q(
                    problem.clearance
                ):
                    return _failure(
                        CorridorAllocationFailureReason.INFEASIBLE,
                        members=(left_lane.member_id, right_lane.member_id),
                        endpoint_owners=(
                            left_lane.endpoint_owner_id,
                            right_lane.endpoint_owner_id,
                        ),
                    )
                continue
            if root_key[left_root] == root_key[right_root]:
                return _failure(
                    CorridorAllocationFailureReason.INVALID,
                    members=(left_lane.member_id, right_lane.member_id),
                    endpoint_owners=(
                        left_lane.endpoint_owner_id,
                        right_lane.endpoint_owner_id,
                    ),
                )
            if root_key[left_root] <= root_key[right_root]:
                before, after = left_root, right_root
                separation = (
                    _q(problem.clearance)
                    + potentials[left_index]
                    - potentials[right_index]
                )
            else:
                before, after = right_root, left_root
                separation = (
                    _q(problem.clearance)
                    + potentials[right_index]
                    - potentials[left_index]
                )
            edge = (before, after)
            edges[edge] = max(edges.get(edge, separation), separation)
            edge_members.setdefault(edge, set()).update(
                (left_lane.member_id, right_lane.member_id)
            )

    successors: dict[int, list[tuple[int, Fraction]]] = {root: [] for root in roots}
    predecessors: dict[int, list[tuple[int, Fraction]]] = {root: [] for root in roots}
    for (before, after), separation in edges.items():
        successors[before].append((after, separation))
        predecessors[after].append((before, separation))
    order = sorted(roots, key=lambda root: root_key[root])

    for root in reversed(order):
        for after, separation in successors[root]:
            after_upper = upper[after]
            if after_upper is None:
                continue
            candidate = after_upper - separation
            root_upper = upper[root]
            if root_upper is None or candidate < root_upper:
                upper[root] = candidate
                bound_members[root].update(bound_members[after])
                bound_members[root].update(edge_members[(root, after)])
                bound_obstacles[root].update(bound_obstacles[after])

    coordinates: dict[int, Fraction] = {}
    coordinate_members: dict[int, set[str]] = {root: set() for root in roots}
    for root in order:
        coordinate = preferred[root]
        root_upper = upper[root]
        if root_upper is not None:
            coordinate = min(coordinate, root_upper)
        root_lower = lower[root]
        if root_lower is not None:
            coordinate = max(coordinate, root_lower)
            coordinate_members[root].update(bound_members[root])
        for before, separation in predecessors[root]:
            candidate = coordinates[before] + separation
            if candidate > coordinate:
                coordinate = candidate
                coordinate_members[root].update(coordinate_members[before])
                coordinate_members[root].update(edge_members[(before, root)])
        if root_upper is not None and coordinate > root_upper:
            component_members = {lanes[index].member_id for index in roots[root]}
            equality_owners = {
                owner
                for pair, owners in peer_owners.items()
                if pair <= set(roots[root])
                for owner in owners
            }
            return _failure(
                CorridorAllocationFailureReason.INFEASIBLE,
                members=component_members
                | bound_members[root]
                | coordinate_members[root],
                obstacles=bound_obstacles[root],
                equality_owners=equality_owners,
                endpoint_owners={
                    lanes[index].endpoint_owner_id for index in roots[root]
                },
            )
        coordinates[root] = coordinate

    allocations: list[tuple[str, float]] = []
    for index, lane in enumerate(lanes):
        root, _ = union_find.find(index)
        oriented_coordinate = coordinates[root] + potentials[index]
        output_coordinate = float(sign * oriented_coordinate)
        allocations.append(
            (
                lane.member_id,
                0.0 if output_coordinate == 0 else output_coordinate,
            )
        )
    return CorridorAllocationResult(
        CorridorAllocationStatus.PLANNED,
        tuple(sorted(allocations)),
    )


__all__ = [
    "CorridorAllocationFailureReason",
    "CorridorAllocationProblem",
    "CorridorAllocationResult",
    "CorridorAllocationStatus",
    "CorridorEquality",
    "CorridorLane",
    "CorridorObstacle",
    "solve_corridor_cohorts",
]
