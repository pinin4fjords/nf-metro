"""SVG rendering for metro maps."""

from nf_metro.render.html import emit_render_plan_html
from nf_metro.render.manifest import (
    MANIFEST_ELEMENT_ID,
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    build_manifest_data,
    inject_manifest,
    manifest_json,
    manifest_metadata_svg,
    manifest_schema,
    match_node_ids,
    matching_node_ids,
    node_data_attrs,
    overlay_svg,
    read_manifest,
)
from nf_metro.render.plan import RenderPlan
from nf_metro.render.svg import build_render_plan, emit_render_plan, render_svg
from nf_metro.render.validate import RenderFinding, validate_render

__all__ = [
    "RenderFinding",
    "RenderPlan",
    "build_render_plan",
    "emit_render_plan",
    "emit_render_plan_html",
    "validate_render",
    "MANIFEST_ELEMENT_ID",
    "MANIFEST_SCHEMA_VERSION",
    "build_manifest",
    "build_manifest_data",
    "inject_manifest",
    "manifest_json",
    "manifest_metadata_svg",
    "manifest_schema",
    "match_node_ids",
    "matching_node_ids",
    "node_data_attrs",
    "overlay_svg",
    "read_manifest",
    "render_svg",
]
