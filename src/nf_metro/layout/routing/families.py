"""Stable identities for production inter-section route families."""

from enum import Enum


class RouteFamilyId(str, Enum):
    """A dispatcher or rail family that emitted an inter-section route."""

    PERP_EXIT = "perp-exit"
    TB_PERP_EXIT_OVER = "tb-perp-exit-over"
    SAME_Y_STRAIGHT = "same-y-straight"
    PERP_EXIT_FAR_SIDE_WRAP = "perp-exit-far-side-entry-wrap"
    TB_BOTTOM_EXIT_AROUND_STACK = "tb-bottom-exit-around-stack"
    TB_BOTTOM_EXIT = "tb-bottom-exit"
    TOP_ENTRY_L_SHAPE = "top-entry-l-shape"
    BOTTOM_ENTRY_L_SHAPE = "bottom-entry-l-shape"
    SAME_X_VERTICAL_DROP = "same-x-vertical-drop"
    BOTTOM_EXIT_JUNCTION = "bottom-exit-junction"
    MERGE_TRUNK = "merge-trunk"
    MERGE_BRANCH = "merge-branch"
    BYPASS_FAMILY = "bypass-family"
    NEAR_VERTICAL_JUNCTION = "near-vertical-same-col-junction"
    RIGHT_ENTRY_WRAP = "right-entry-wrap"
    LEFT_ENTRY_WRAP = "left-entry-wrap-family"
    SERPENTINE_LEFT = "serpentine-left-exit-left-entry"
    LEFT_EXIT_FAR_SIDE_WRAP = "left-exit-far-side-left-entry-wrap"
    MERGE_ENTRY = "merge-entry-family"
    RIGHT_ENTRY_PLOUGH_BYPASS = "right-entry-plough-bypass"
    RIGHT_ENTRY_CROSS_ROW_WRAP = "right-entry-cross-row-wrap"
    STANDARD_L_SHAPE = "standard-l-shape"
    TB_SECTION_FALLBACK = "tb-section-fallback"
    ENTRY_RUNWAY_FALLBACK = "entry-runway-fallback"
    INTRA_SECTION_FALLBACK = "intra-section-fallback"
    RAIL_INTER_SECTION = "rail-inter-section"
