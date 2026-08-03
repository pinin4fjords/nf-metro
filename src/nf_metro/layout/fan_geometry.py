"""Pure relative geometry for authored fan appearances."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import TypeVar

_BranchId = TypeVar("_BranchId", bound=Hashable)


def symmetric_lane_offsets(branch_count: int, lane_pitch: float) -> tuple[float, ...]:
    """Return lane offsets centred on zero in canonical branch order."""
    if branch_count < 1:
        raise ValueError("lane offsets require at least one branch")
    if not math.isfinite(lane_pitch) or lane_pitch <= 0:
        raise ValueError("fan lane pitch must be finite and positive")
    midpoint = (branch_count - 1) / 2.0
    return tuple((rank - midpoint) * lane_pitch for rank in range(branch_count))


def fan_lane_offsets(
    branch_ids: Sequence[_BranchId],
    lane_pitch: float,
    centreline_branch_id: _BranchId | None,
) -> tuple[float, ...]:
    """Return symmetric or straight offsets in canonical branch order.

    A missing centreline branch selects a symmetric frame. A straight frame
    keeps the named branch at zero and seats every other branch at successive
    positive pitches.
    """
    if len(branch_ids) < 2:
        raise ValueError("a fan requires at least two branches")
    if centreline_branch_id is None:
        return symmetric_lane_offsets(len(branch_ids), lane_pitch)
    if not math.isfinite(lane_pitch) or lane_pitch <= 0:
        raise ValueError("fan lane pitch must be finite and positive")
    if centreline_branch_id not in branch_ids:
        raise ValueError("fan appearance centreline names an unknown branch")
    offsets = {centreline_branch_id: 0.0}
    offsets.update(
        (branch_id, slot * lane_pitch)
        for slot, branch_id in enumerate(
            (
                branch_id
                for branch_id in branch_ids
                if branch_id != centreline_branch_id
            ),
            start=1,
        )
    )
    return tuple(offsets[branch_id] for branch_id in branch_ids)
